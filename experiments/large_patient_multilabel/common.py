"""Experiment constants and strict data/protocol checks."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import (
    sha256, save, metrics, setup_gpu, release_gpu, residual_mlp)

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/20261121'
DATA = ROOT / 'data/large_patient_multilabel_20261121/chestxray14'
SOURCE_DATA = ROOT / 'data/direct_anchor_classification_20261118/chestxray14/dataset.json'
SOURCE_CACHE = ROOT / 'experiments/direct_anchor_classification/runs/20261118/cache'
MODELS = ('AlphaRAD', 'RadZero')
METHODS = ('pre_mlp', 'post_mlp', 'linear_probe', 'clip_global')
SEEDS = (42, 43, 44)
SPLITS = ('train', 'val', 'test')


def launcher(model):
    return ROOT / ('AlphaRAD/local_setup/python_local.sh' if model == 'AlphaRAD'
                   else 'RadZero/scripts/python_local.sh')


def check_disjoint(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty data or duplicate sample ID')
    for field in ('patient_id', 'group_id', 'file_sha256', 'pixel_sha256'):
        membership = {}
        for row in rows:
            if row['split'] not in SPLITS or not row[field]:
                raise ValueError('Invalid membership or missing grouping field')
            if membership.setdefault(row[field], row['split']) != row['split']:
                raise ValueError('Cross-split leakage: ' + field)


def load_data():
    meta = json.loads((DATA / 'dataset.json').read_text())
    if sha256(meta['samples_file']) != meta['samples_sha256']:
        raise ValueError('Frozen sample manifest changed')
    rows = json.loads(Path(meta['samples_file']).read_text())
    check_disjoint(rows)
    return meta, rows


def load_protocol(run):
    p = json.loads((run / 'PROTOCOL.json').read_text())
    if sha256(DATA / 'dataset.json') != p['dataset_sha256']:
        raise ValueError('Frozen dataset changed')
    for key, filename in (('source_integrity_sha256','source_integrity.json'),
                          ('split_plan_sha256','SPLIT_PLAN.json'),('split_audit_sha256','SPLIT_AUDIT.json')):
        if sha256(run / filename) != p[key]:
            raise ValueError('Frozen experiment artifact changed: ' + filename)
    for spec in p['feature_caches'].values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('Frozen feature mapping changed')
    for path, expected in p['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen implementation changed: ' + path)
    return p


def folder(run, model, method, seed):
    return run / 'results' / model / method / f'seed{seed}'


def expected_jobs():
    return [(model, method, seed) for model in MODELS for method in METHODS for seed in SEEDS]


def selected_epoch(initial, history):
    best, epoch = initial, 0
    for row in history:
        value = row['val']['macro_auroc']
        if value > best:
            best, epoch = value, row['epoch']
    return epoch
