"""Run required checks in both model environments and record tested source hashes."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *


def main():
    sources=[HERE/name for name in ('common.py','prepare.py','features.py','train.py','test_experiment.py')]
    fingerprints={str(p):sha256(p) for p in sources}
    target=RUN/'test_checks.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['status']=='passed' and old['source_sha256']==fingerprints and all(sha256(p)==h for p,h in old['logs'].items()):
            print('Both environments already passed for these exact sources; verified existing logs.')
            return
        if (RUN/'PROTOCOL.json').exists():raise ValueError('Checks changed after protocol freeze')
    logs={}
    for model in MODELS:
        path=HERE/('test_checks_alpha.log' if model=='AlphaRAD' else 'test_checks_radzero.log')
        with path.open('w') as stream:
            subprocess.run(['bash',str(launcher(model)),'-m','unittest',
                'experiments.large_patient_multilabel.test_experiment','-v'],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        text=path.read_text()
        if 'Ran 10 tests' not in text or not text.rstrip().endswith('OK'):raise ValueError('Incomplete test report')
        logs[str(path)]=sha256(path)
    save(target,dict(status='passed',tests_per_environment=10,environments=list(MODELS),
        source_sha256=fingerprints,logs=logs))


if __name__=='__main__':main()
