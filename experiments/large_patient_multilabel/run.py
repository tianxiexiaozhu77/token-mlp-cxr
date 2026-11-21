"""Persistent four-GPU controller: all fitting, selection lock, then evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *


def proc_identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z':return None
        return stat[19]
    except (FileNotFoundError,ProcessLookupError):return None


class Adopted:
    def __init__(self,job):self.job=job
    def poll(self):
        if proc_identity(self.job['pid'])==self.job['process_start_ticks']:
            cmd=Path(f"/proc/{self.job['pid']}/cmdline").read_bytes()
            if self.job['script'].encode() in cmd:return None
        return 0 if Path(self.job['completion_file']).is_file() else 1


def available_gpus(active):
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return [int(line.split(',')[0]) for line in result.splitlines()
        if int(line.split(',')[0]) in (4,5,6,7) and int(line.split(',')[1])<512 and int(line.split(',')[0]) not in active]


def execute_stage(run,jobs,stage):
    path=run/(stage+'_matrix.json');active={}
    if path.exists():
        old={r['id']:r for r in json.loads(path.read_text())['jobs']}
        for i,job in enumerate(jobs):
            previous=old.get(job['id'])
            if previous and previous['status']=='running':
                jobs[i]=previous;active[previous['gpu']]=(Adopted(previous),previous,None)
            elif previous and previous['status']=='failed':
                raise RuntimeError('Failed job requires review; do not replace seeds: '+job['id'])
    while True:
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is None:continue
            if log:log.close()
            success=code==0 and Path(job['completion_file']).exists()
            job.update(status='complete' if success else 'failed',exit_code=code,finished=time.time())
            del active[gpu]
        for job in jobs:
            if job['status']=='pending' and Path(job['completion_file']).exists():
                record=json.loads(Path(job['completion_file']).read_text())
                if record['status']!='complete':raise ValueError('Invalid completed worker')
                job['status']='complete'
        failed=any(j['status']=='failed' for j in jobs)
        free=available_gpus(active) if not failed else []
        for job in jobs:
            if not free or job['status']!='pending':continue
            gpu=free.pop(0)
            cmd=['bash',str(launcher(job['model'])),'-u',str(HERE/job['script']),
                 '--run-dir',str(run),'--model',job['model'],'--gpu',str(gpu),*job['args']]
            log_path=run/'logs'/(job['id']+'.log');log_path.parent.mkdir(parents=True,exist_ok=True)
            log=log_path.open('a')
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            job.update(status='running',pid=process.pid,process_start_ticks=proc_identity(process.pid),gpu=gpu,
                command=cmd,started=time.time(),log=str(log_path))
            active[gpu]=(process,job,log)
            print('Launched',job['id'],'physical GPU',gpu,flush=True)
        complete=all(j['status']=='complete' for j in jobs)
        save(path,dict(status='complete' if complete else 'failed' if failed else 'running',jobs=jobs,updated=time.time()))
        if complete:return
        if failed and not active:raise RuntimeError('A worker failed; inspect its log')
        time.sleep(10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    args=parser.parse_args();run=args.run_dir.resolve()
    load_protocol(run)
    jobs=[]
    # Token fits are longest; schedule these first and interleave both backbones.
    for method in METHODS:
        for seed in SEEDS:
            for model in MODELS:
                jobs.append(dict(id=f'{model}_{method}_{seed}',model=model,script='train.py',status='pending',
                    args=['--method',method,'--seed',str(seed)],completion_file=str(folder(run,model,method,seed)/'fit.json')))
    execute_stage(run,jobs,'training')
    subprocess.run(['bash',str(launcher('AlphaRAD')),str(HERE/'evaluate.py'),'--run-dir',str(run),'--lock-only'],cwd=ROOT,check=True)
    evaluations=[dict(id=f'evaluate_{model}',model=model,script='evaluate.py',status='pending',args=[],
        completion_file=str(folder(run,model,'clip_global',44)/'result.json')) for model in MODELS]
    execute_stage(run,evaluations,'evaluation')
    for script in ('uncertainty.py','report.py'):
        subprocess.run(['bash',str(launcher('AlphaRAD')),'-u',str(HERE/script),'--run-dir',str(run)],cwd=ROOT,check=True)


if __name__=='__main__':main()
