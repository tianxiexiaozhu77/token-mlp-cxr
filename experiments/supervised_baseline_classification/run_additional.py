"""Schedule source-ready bootstrap and locked transfer after GPU fits release leases."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def main():
    run=DEFAULT_RUN;lock=(run/'additional_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[dict(kind='bootstrap',dataset=d,status='pending') for d in DATASETS]+[dict(kind='transfer',model=m,status='pending') for m in MODELS]
    active=[]
    while True:
        for process,job,log in list(active):
            code=process.poll()
            if code is not None:
                job.update(status='complete' if code==0 else 'failed',exit_code=code);log.close();active.remove((process,job,log))
        failed=any(j['status']=='failed' for j in jobs)
        if not failed:
            training=json.loads((run/'training_matrix.json').read_text())
            occupied={j['gpu'] for j in training['jobs'] if j['status']=='running'}|{j['gpu'] for _,j,_ in active if 'gpu' in j}
            status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            free=[int(line.split(',')[0]) for line in status.splitlines() if int(line.split(',')[0]) in (4,5,6,7)
                  and int(line.split(',')[0]) not in occupied and int(line.split(',')[1])<512]
            for job in jobs:
                if job['status']!='pending':continue
                if job['kind']=='bootstrap':
                    expected=[run/'results'/m/job['dataset']/f'seed{s}'/method/'result.json' for m in MODELS for s in SEEDS for method in METHODS]
                    if not all(f.exists() for f in expected):continue
                    dest=ROOT/'experiments/classification_uncertainty/runs/20261120/supervised'/job['dataset']/'summary.json'
                    command=['bash',str(launcher('AlphaRAD')),'-u',str(ROOT/'experiments/classification_uncertainty/run.py'),'--scope','supervised','--dataset',job['dataset']]
                    name='bootstrap_'+job['dataset']
                else:
                    expected=[run/'results'/job['model']/'chexpert'/f'seed{s}'/method/'result.json' for s in SEEDS for method in METHODS]
                    if not all(f.exists() for f in expected):continue
                    dest=run/'external/results'/job['model']/'summary.json';name='transfer_'+job['model']
                    if not dest.exists():
                        if not free:continue
                        job['gpu']=free.pop(0)
                    command=['bash',str(launcher(job['model'])),'-u',str(Path(__file__).with_name('external.py')),
                        '--model',job['model'],'--gpu',str(job.get('gpu',4))]
                if dest.exists() and json.loads(dest.read_text())['status']=='complete':job['status']='complete_reused';continue
                log=(run/'logs'/(name+'.log')).open('a');process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job.update(status='running',pid=process.pid,command=command);active.append((process,job,log));print('LAUNCHED',name,process.pid,flush=True)
        complete=all(j['status'].startswith('complete') for j in jobs)
        save(run/'additional_matrix.json',dict(status='complete' if complete else 'failed' if failed else 'running',jobs=jobs,updated=time.time()))
        if complete:return
        if failed and not active:raise RuntimeError('Additional worker failed; inspect before retry')
        time.sleep(5)


if __name__=='__main__':main()
