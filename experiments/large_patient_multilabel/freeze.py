"""Freeze the data, feature audit, and causal training implementation."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *


def main():
    meta,rows=load_data()
    plan=json.loads((RUN/'SPLIT_PLAN.json').read_text())
    tests=json.loads((RUN/'test_checks.json').read_text())
    if tests['status']!='passed':raise ValueError('Unit checks have not passed')
    encoder_audit=json.loads((RUN/'ENCODER_AUDIT.json').read_text())
    if encoder_audit['status']!='passed' or {r['model'] for r in encoder_audit['checks']}!=set(MODELS):
        raise ValueError('Fresh encoder parity checks have not passed')
    integrity=json.loads((RUN/'source_integrity.json').read_text())
    historical=ROOT/'experiments/direct_anchor_classification/runs/20261118'
    old_records=json.loads((historical/'token_cache_sha256.json').read_text())
    for path in (ROOT/'experiments/direct_anchor_classification/runs/20261118_alpha_token_mlp/cache_checksums/chestxray14').glob('tokens_*.json'):
        old_records.append(json.loads(path.read_text()))
    checked=[]
    for record in old_records:
        current=integrity[record['path']]
        if current['sha256']!=record['sha256'] or current['bytes']!=record['bytes']:
            raise ValueError('Frozen token cache differs from its historical full hash')
        checked.append(record['path'])
    if len(set(checked))!=4:raise ValueError('Missing historical token checksums')
    save(RUN/'HISTORICAL_CACHE_AUDIT.json',dict(status='passed',unchanged_token_files=checked))
    sources=[HERE/name for name in ('common.py','prepare.py','features.py','train.py','test_experiment.py')]
    sources += [ROOT/'experiments/baseline_token_classification/common.py',
        ROOT/'experiments/direct_anchor_classification/model.py',
        ROOT/'experiments/supervised_baseline_classification/components.py',
        ROOT/'AlphaRAD/local_setup/python_local.sh',ROOT/'AlphaRAD/local_setup/activate.sh',
        ROOT/'RadZero/scripts/python_local.sh']
    for path,expected in tests['source_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Tested implementation changed')
    caches={}
    for model in MODELS:
        path=RUN/'features'/model/'cache.json'
        cache=json.loads(path.read_text())
        if cache['status']!='complete' or cache['dataset_sha256']!=sha256(DATA/'dataset.json'):
            raise ValueError('Features not ready')
        caches[model]=dict(path=str(path),sha256=sha256(path))
    protocol=dict(status='frozen',models=list(MODELS),methods=list(METHODS),seeds=list(SEEDS),
        train=plan['training'],dataset=str(DATA/'dataset.json'),dataset_sha256=sha256(DATA/'dataset.json'),
        samples_sha256=meta['samples_sha256'],class_names=meta['class_names'],split_stats=meta['split_stats'],
        split_plan_sha256=sha256(RUN/'SPLIT_PLAN.json'),split_audit_sha256=sha256(RUN/'SPLIT_AUDIT.json'),
        feature_caches=caches,source_integrity_sha256=sha256(RUN/'source_integrity.json'),
        encoder_audit_sha256=sha256(RUN/'ENCODER_AUDIT.json'),
        historical_cache_audit_sha256=sha256(RUN/'HISTORICAL_CACHE_AUDIT.json'),
        source_sha256={str(p):sha256(p) for p in sources},test_checks_sha256=sha256(RUN/'test_checks.json'),
        registered_fits=24,registered_test_evaluations=28,allowed_physical_gpus=[4,5,6,7],
        test_access=plan['test_access'],initialization=plan['initialization'],uncertainty=plan['uncertainty'],
        limitations=plan['limitations'])
    target=RUN/'PROTOCOL.json'
    if target.exists():
        if json.loads(target.read_text())!=protocol:raise ValueError('Existing frozen protocol differs')
    else:
        for path in sources:
            snapshot=RUN/'source'/path.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,snapshot)
        save(target,protocol)
    print('Frozen: 24 fresh fits, 28 test evaluations; TEST held until ALL fits selected.',flush=True)


if __name__=='__main__':main()
