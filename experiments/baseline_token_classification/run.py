"""Persistent, resumable cache/training queue restricted to physical GPUs 4--7."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from common import ROOT,MODELS,save,sha256


def alive(pid, script, model):
    path=Path(f"/proc/{pid}/cmdline")
    if not path.exists(): return False
    parts=path.read_bytes().split(b"\0")
    return any(script.encode() in p for p in parts) and model.encode() in parts


def existing_worker(job,run):
    script="baseline_token_classification/"+("cache.py" if job["stage"]=="cache" else "train.py")
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try: parts=[p.decode() for p in path.read_bytes().split(b"\0") if p]
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if not parts or "python" not in Path(parts[0]).name: continue
        if not any(script in p for p in parts) or "--model" not in parts or parts[parts.index("--model")+1]!=job["model"]: continue
        if "--run-dir" not in parts or Path(parts[parts.index("--run-dir")+1]).resolve()!=run: continue
        if job["stage"]=="train" and (parts[parts.index("--dataset")+1]!=job["dataset"] or int(parts[parts.index("--seed")+1])!=job["seed"]): continue
        return int(path.parent.name),int(parts[parts.index("--gpu")+1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve()
    # The queue manages this host's physical GPUs. NFS advisory locks can
    # remain stale after process exit; use a host-local lock for the queue.
    lock_root=Path("/tmp")/f"medical-gpu-locks-{os.getuid()}"
    lock_root.mkdir(parents=True,exist_ok=True)
    lock_name="scheduler-"+hashlib.sha256(str(run).encode()).hexdigest()+".lock"
    scheduler_lock=(lock_root/lock_name).open("a+")
    fcntl.flock(scheduler_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((run/"PROTOCOL.json").read_text())
    jobs=[]
    for model in MODELS:
        jobs.append(dict(id="cache_"+model,stage="cache",model=model,status="pending",
            completion=str(run/"cache"/model/"completion.json")))
    for dataset_position in (0,1,2):
        for seed in protocol["seeds"]:
            for model in MODELS:
                dataset=protocol["model_datasets"][model][dataset_position]
                jobs.append(dict(id=f"train_{model}_{dataset}_{seed}",stage="train",model=model,dataset=dataset,seed=seed,status="pending",
                    completion=str(run/"results"/model/dataset/f"seed{seed}"/"token_mlp/result.json")))
    matrix=run/"matrix.json";old={}
    if matrix.exists(): old={j["id"]:j for j in json.loads(matrix.read_text())["jobs"]}
    active={};logs=run/"logs";logs.mkdir(exist_ok=True)
    for job in jobs:
        previous=old.get(job["id"],{})
        if Path(job["completion"]).exists():
            data=json.loads(Path(job["completion"]).read_text())
            if data["status"]!="complete" or data.get("smoke_only",False): raise ValueError("Invalid completion")
            job.update(previous);job["status"]="complete";continue
        worker=existing_worker(job,run)
        if worker:
            pid,gpu=worker
            if gpu not in (4,5,6,7) or gpu in active: raise ValueError("Conflicting adopted worker")
            job.update(previous);job.update(status="running",pid=pid,gpu=gpu,adopted=True)
            active[gpu]=(None,job,None)
        elif previous.get("status") in ("running","failed"):
            job.update(previous);job["status"]="failed"
    save(run/"scheduler.json",dict(pid=os.getpid(),allowed_physical_gpus=[4,5,6,7],started=time.time()))
    while True:
        for gpu,(proc,job,stream) in list(active.items()):
            code=proc.poll() if proc else (None if alive(job["pid"],job["stage"]+".py",job["model"]) else 0)
            if code is None: continue
            if stream: stream.close()
            complete=code==0 and Path(job["completion"]).exists() and json.loads(Path(job["completion"]).read_text()).get("status")=="complete"
            job.update(status="complete" if complete else "failed",exit_code=code,ended=time.time())
            print(job["status"],job["id"],flush=True);del active[gpu]
        output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
        free=[int(line.split(",")[0]) for line in output.splitlines() if int(line.split(",")[1])<512]
        free=[g for g in free if g in (4,5,6,7) and g not in active]
        # Exercise one complete training/evaluation per model as soon as its
        # CheXpert cache is ready; this catches worker incompatibilities early.
        scheduled = sorted(jobs, key=lambda j: 0 if j["stage"]=="train" and j["dataset"]=="chexpert" and j["seed"]==42 else 1 if j["stage"]=="cache" else 2)
        for job in scheduled:
            if not free or job["status"]!="pending": continue
            if job["stage"]=="train":
                cache=run/"cache"/job["model"]/job["dataset"]/"cache.json"
                if not cache.exists(): continue
                cache_job=next(j for j in jobs if j["stage"]=="cache" and j["model"]==job["model"])
                if cache_job["status"]=="failed": continue
            else:
                root=ROOT/"baselines"/job["model"]/"local_setup"
                validations=[root/"token_mlp_cpu_validation.json",root/"TOKEN_MLP_CPU_VALIDATION.json"]
                if not any(path.exists() for path in validations): continue
            gpu=free.pop(0)
            cmd=["bash",str(ROOT/"baselines"/job["model"]/"local_setup/python_local.sh"),"-u",
                 str(Path(__file__).with_name(job["stage"]+".py")),"--model",job["model"],"--run-dir",str(run),"--gpu",str(gpu)]
            if job["stage"]=="train":cmd.extend(["--dataset",job["dataset"],"--seed",str(job["seed"])])
            log=logs/(job["id"]+".log");stream=log.open("w")
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            job.update(status="running",pid=proc.pid,gpu=gpu,started=time.time(),command=cmd,log=str(log))
            active[gpu]=(proc,job,stream)
            print("LAUNCH",job["id"],"GPU",gpu,flush=True)
        failed=any(j["status"]=="failed" for j in jobs)
        complete=all(j["status"]=="complete" for j in jobs)
        save(matrix,dict(status="complete" if complete else "running_with_failures" if failed else "running",jobs=jobs,updated=time.time()))
        if complete:break
        failed_caches={j["model"] for j in jobs if j["stage"]=="cache" and j["status"]=="failed"}
        pending_runnable=any(j["status"]=="pending" and (j["stage"]=="cache" or (j["model"] not in failed_caches and (run/"cache"/j["model"]/j["dataset"]/"cache.json").exists())) for j in jobs)
        if failed and not active and not pending_runnable: raise RuntimeError("Workers failed; inspect/archive failures before explicit retry")
        time.sleep(10)
    audit=run/"image_audit.json"
    if not audit.exists() or json.loads(audit.read_text())["status"]!="complete":
        raise RuntimeError("Image audit is not complete")
    subprocess.run(["bash",str(ROOT/"AlphaRAD/local_setup/python_local.sh"),str(Path(__file__).with_name("report.py")),"--run-dir",str(run)],check=True,cwd=ROOT)
    files=[run/name for name in ("RESULTS.txt","RESULTS.csv","per_class_results.csv","summary.json","comparison.png","comparison.pdf")]
    save(run/"completion.json",dict(status="complete",training_runs=72,evaluations=108,
        allowed_physical_gpus=[4,5,6,7],completed=time.time(),output_sha256={str(p):sha256(p) for p in files}))


if __name__=="__main__":main()
