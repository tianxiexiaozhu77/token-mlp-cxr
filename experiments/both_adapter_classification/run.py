"""Schedule the 63 Both fits on idle physical GPUs from the authorized 4--7 set."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.baseline_token_classification.common import save, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--gpus",default="4,5,6,7")
    args=p.parse_args()
    run=args.run_dir.resolve();allowed=set(map(int,args.gpus.split(",")))
    assert allowed and allowed <= {4,5,6,7}
    protocol=json.loads((run/"PROTOCOL.json").read_text());digest=sha256(run/"PROTOCOL.json")
    (run/"logs").mkdir(parents=True,exist_ok=True)
    controller=(run/"controller.lock").open("a+")
    fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[dict(id=f"{model}_{dataset}",model=model,dataset=dataset,status="pending")
        for dataset in ("chexpert","openi","chestxray14") for model in protocol["models"]]
    active={}
    while True:
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is not None:
                log.close();job.update(status="complete" if code==0 else "failed",exit_code=code,end=time.time())
                del active[gpu]
        failures=any(j["status"]=="failed" for j in jobs)
        if not failures:
            output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
            free=[int(line.split(",")[0]) for line in output.splitlines()
                if int(line.split(",")[0]) in allowed and int(line.split(",")[1])<512]
            free=[gpu for gpu in free if gpu not in active]
            for job in jobs:
                if job["status"]!="pending":continue
                expected=[run/"results"/job["model"]/job["dataset"]/f"seed{seed}"/"both_mlp/result.json" for seed in protocol["seeds"]]
                if all(path.exists() and (lambda r:r["status"]=="complete" and r["protocol_sha256"]==digest)(json.loads(path.read_text())) for path in expected):
                    job["status"]="complete_reused";continue
                if not free:continue
                gpu=free.pop(0)
                launcher=ROOT/"AlphaRAD/local_setup/python_local.sh" if job["model"] in ("AlphaRAD","RadZero") else ROOT/"baselines"/job["model"]/"local_setup/python_local.sh"
                command=["bash",str(launcher),"-u",str(Path(__file__).with_name("train.py")),"--model",job["model"],
                    "--dataset",job["dataset"],"--run-dir",str(run),"--gpu",str(gpu)]
                log_path=run/"logs"/(job["id"]+".log");log=log_path.open("a")
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job.update(status="running",pid=process.pid,gpu=gpu,command=command,start=time.time(),log=str(log_path))
                active[gpu]=process,job,log
                print("LAUNCHED",job["id"],"GPU",gpu,flush=True)
        complete=all(j["status"].startswith("complete") for j in jobs)
        save(run/"training_matrix.json",dict(status="complete" if complete else "failed" if failures else "running",
            jobs=jobs,allowed_gpus=sorted(allowed),updated=time.time(),new_fits=63,protocol_sha256=digest))
        if complete:break
        if failures and not active:raise RuntimeError("A worker failed; inspect logs before retry")
        time.sleep(5)


if __name__=="__main__":main()
