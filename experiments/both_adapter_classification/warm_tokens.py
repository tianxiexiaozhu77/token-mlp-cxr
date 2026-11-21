"""Read-only, bounded-memory warming of existing NFS token caches.

This helper never changes files, ordering, model state or training inputs.
It only brings the already verified token files into the host page cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.baseline_token_classification.common import save,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--models",required=True)
    p.add_argument("--workers",type=int,default=8)
    args=p.parse_args();run=args.run_dir.resolve()
    assert 1<=args.workers<=8
    protocol=json.loads((run/"PROTOCOL.json").read_text())
    paths=[]
    for model in args.models.split(","):
        for spec in protocol["caches"][model+"/"+args.dataset]:
            assert sha256(spec["path"])==spec["sha256"]
            record=json.loads(Path(spec["path"]).read_text())
            files=[t["path"] for t in record["token_shards"]] if "token_shards" in record else [record["files"]["tokens.npy"]["path"]]
            for filename in files:
                expected=spec["files"][filename];stat=Path(filename).stat()
                assert stat.st_size==expected["bytes"] and stat.st_mtime_ns==expected["mtime_ns"]
                paths.append((filename,stat.st_size,stat.st_mtime_ns))
    block=256*1024**2
    spans=sorted([(path,offset,min(block,size-offset)) for path,size,_ in paths for offset in range(0,size,block)],key=lambda x:(x[1],x[0]))
    def read_span(span):
        filename,offset,size=span
        fd=os.open(filename,os.O_RDONLY)
        try:
            end=offset+size
            os.posix_fadvise(fd,offset,size,os.POSIX_FADV_SEQUENTIAL)
            while offset<end:
                data=os.pread(fd,min(8*1024**2,end-offset),offset)
                if not data:raise IOError("Unexpected EOF: "+filename)
                offset+=len(data)
            return size
        finally:os.close(fd)
    start=last=time.time();done=0;total=sum(size for _,size,_ in paths)
    out=run/"io_warming"/(args.dataset+"_"+args.models.replace(",","_")+".json")
    state=dict(status="running",read_only=True,paths=[p for p,_,_ in paths],workers=args.workers,
        helper_sha256=sha256(__file__),total_bytes=total,read_bytes=0,seconds=0.)
    save(out,state)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(read_span,span) for span in spans]):
            done+=future.result()
            if time.time()-last>=20:
                state.update(read_bytes=done,seconds=time.time()-start);save(out,state);last=time.time()
                print(f"WARMED {done/1024**3:.2f}/{total/1024**3:.2f} GiB in {time.time()-start:.1f}s",flush=True)
    for filename,size,mtime in paths:
        stat=Path(filename).stat();assert stat.st_size==size and stat.st_mtime_ns==mtime
    state.update(status="complete",read_bytes=done,seconds=time.time()-start,files_unchanged=True);save(out,state)
    print(json.dumps(state,indent=2))


if __name__=="__main__":main()
