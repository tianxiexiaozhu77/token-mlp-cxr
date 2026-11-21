"""Wait on a verified feature-audit process, then execute the locked campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *
from experiments.large_patient_multilabel.run import proc_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-feature-pid',type=int)
    args=parser.parse_args()
    if args.adopt_feature_pid:
        pid=args.adopt_feature_pid;identity=proc_identity(pid)
        if identity is None:raise ValueError('Feature process is not live')
        if b'large_patient_multilabel/features.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():
            raise ValueError('PID does not belong to the feature audit')
        save(RUN/'workflow.json',dict(status='running',stage='feature_integrity',
            adopted_pid=pid,process_start_ticks=identity,controller_pid=__import__('os').getpid()))
        while proc_identity(pid)==identity:
            time.sleep(10)
    for model in MODELS:
        cache=RUN/'features'/model/'cache.json'
        if not cache.exists() or json.loads(cache.read_text())['status']!='complete':
            raise ValueError('Feature process ended without complete verified caches')
    for script,stage in [('freeze.py','freeze_protocol'),('run.py','training_then_locked_evaluation')]:
        cmd=['bash',str(launcher('AlphaRAD')),'-u',str(HERE/script)]
        process=subprocess.Popen(cmd,cwd=ROOT)
        save(RUN/'workflow.json',dict(status='running',stage=stage,pid=process.pid,
            process_start_ticks=proc_identity(process.pid),controller_pid=__import__('os').getpid(),command=cmd))
        code=process.wait()
        if code:
            save(RUN/'workflow.json',dict(status='failed',stage=stage,exit_code=code))
            raise RuntimeError('Stage failed: '+stage)
    completed=json.loads((RUN/'COMPLETION.json').read_text())
    if completed['status']!='complete':raise ValueError('Final audit incomplete')
    save(RUN/'workflow.json',dict(status='complete',completion_sha256=sha256(RUN/'COMPLETION.json')))


if __name__=='__main__':main()
