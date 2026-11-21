"""Finish registered statistics and transfer as each source dataset completes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.radzero_supervised_classification.common import *


def main():
    run=DEFAULT_RUN
    lock=(run/'finish.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[dict(name=d,kind='uncertainty',status='pending',dataset=d) for d in DATASETS]
    jobs += [dict(name='rsna',kind='external',status='pending'),
             dict(name='rsna_uncertainty',kind='external_uncertainty',status='pending')]
    for job in jobs:
        path=(run/'uncertainty/supervised'/job['dataset']/'summary.json' if job['kind']=='uncertainty'
              else run/'external/results/RadZero/summary.json' if job['kind']=='external'
              else run/'uncertainty/rsna_transfer/summary.json')
        if path.exists():
            assert json.loads(path.read_text())['status']=='complete'
            job.update(status='complete',reused_completed=True)
    active={}
    def ready(dataset):
        return all((result_folder(run,dataset,m,s)/'result.json').exists() for m in METHODS for s in SEEDS)
    def start(job,args):
        log=(run/'logs'/(job['name']+'_additional.log')).open('a')
        command=['bash',str(launcher()),'-u',str(Path(__file__).with_name(args[0])),*args[1:]]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        job.update(status='running',pid=process.pid,command=command)
        active[job['name']]=(process,log,job)
        print('LAUNCHED',job['name'],process.pid,flush=True)
    while True:
        for name,(process,log,job) in list(active.items()):
            code=process.poll()
            if code is not None:
                job.update(status='complete' if code==0 else 'failed',exit_code=code)
                log.close();del active[name]
        failed=any(j['status']=='failed' for j in jobs)
        matrix=json.loads((run/'training_matrix.json').read_text())
        if matrix['status']=='failed':failed=True
        if not failed:
            for job in jobs:
                if job['status']!='pending':continue
                if job['kind']=='uncertainty' and ready(job['dataset']):
                    start(job,['uncertainty.py','--scope','supervised','--dataset',job['dataset']])
                elif job['kind']=='external' and ready('chexpert'):
                    start(job,['external_entry.py','--model','RadZero','--gpu','7'])
                elif job['kind']=='external_uncertainty' and (run/'external/results/RadZero/summary.json').exists():
                    start(job,['uncertainty_external.py'])
        complete=all(j['status']=='complete' for j in jobs)
        save(run/'additional_matrix.json',dict(status='complete' if complete else 'failed' if failed else 'running',jobs=jobs))
        if complete and matrix['status']=='complete':break
        if failed and not active:raise RuntimeError('Additional stage failed; inspect logs before retry.')
        time.sleep(5)
    with (run/'logs/report.log').open('a') as log:
        subprocess.run(['bash',str(launcher()),'-u',str(Path(__file__).with_name('report.py'))],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',run,flush=True)


if __name__=='__main__':main()
