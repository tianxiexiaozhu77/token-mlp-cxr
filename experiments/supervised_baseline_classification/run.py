"""Run nine independent model/dataset jobs using leased physical GPUs 4--7."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args();run=args.run_dir.resolve();p=json.loads((run/'PROTOCOL.json').read_text())
    lock=(run/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[dict(model=m,dataset=d,status='pending') for d in ('chexpert','openi','chestxray14') for m in MODELS]
    active={};(run/'logs').mkdir(exist_ok=True)
    while True:
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is not None:
                job.update(status='complete' if code==0 else 'failed',exit_code=code);log.close();del active[gpu]
        failed=any(j['status']=='failed' for j in jobs)
        if not failed:
            status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            free=[int(line.split(',')[0]) for line in status.splitlines() if int(line.split(',')[0]) in p['allowed_physical_gpus']
                  and int(line.split(',')[1])<512 and int(line.split(',')[0]) not in active]
            for job in jobs:
                if job['status']!='pending':continue
                expected=[run/'results'/job['model']/job['dataset']/f'seed{s}'/m/'result.json' for s in SEEDS for m in METHODS]
                if all(f.exists() and json.loads(f.read_text())['status']=='complete' for f in expected):
                    job['status']='complete_reused';continue
                if not free:break
                gpu=free.pop(0);log_path=run/'logs'/f'{job["model"]}_{job["dataset"]}.log';log=log_path.open('a')
                cmd=['bash',str(launcher(job['model'])),'-u',str(Path(__file__).with_name('train.py')),
                     '--run-dir',str(run),'--model',job['model'],'--dataset',job['dataset'],'--gpu',str(gpu)]
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job.update(status='running',pid=process.pid,gpu=gpu,command=cmd,log=str(log_path));active[gpu]=process,job,log
                print('LAUNCHED',job['model'],job['dataset'],gpu,process.pid,flush=True)
        complete=all(j['status'].startswith('complete') for j in jobs)
        save(run/'training_matrix.json',dict(status='complete' if complete else 'failed' if failed else 'running',jobs=jobs,updated=time.time()))
        if complete:return
        if failed and not active:raise RuntimeError('Worker failed; inspect before retry')
        time.sleep(5)


if __name__=='__main__':main()
