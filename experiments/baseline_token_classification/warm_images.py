"""Read/hash existing image files concurrently without changing any dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from common import DATASETS,read_dataset,sha256,save


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for path in DATASETS.values():
        _,samples=read_dataset(path);rows.extend(samples)
    def check(row):
        path=Path(row["image"]);digest=sha256(path)
        if row.get("file_sha256") and digest!=row["file_sha256"]:
            raise ValueError("Image checksum changed: "+str(path))
        stat=path.stat()
        return dict(path=str(path),sha256=digest,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    started=time.time();done=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for result in pool.map(check,rows):
            done.append(result)
            if len(done)%1000==0:
                print(len(done),len(rows),round(time.time()-started),flush=True)
                save(args.run_dir/"image_audit_progress.json",dict(status="running",completed=len(done),total=len(rows)))
    save(args.run_dir/"image_audit.json",dict(status="complete",images=len(done),files=done,seconds=time.time()-started))


if __name__=="__main__":main()
