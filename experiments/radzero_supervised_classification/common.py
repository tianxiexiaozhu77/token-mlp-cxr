from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import (
    DATASETS, read_dataset, save, sha256, metrics, setup_gpu, release_gpu,
    freeze_backend, residual_mlp)
from experiments.both_adapter_classification.common import DIRECT, reference_folder

MODELS = ('RadZero',)
METHODS = ('linear_probe', 'clip_global', 'post_mlp')
SEEDS = (42, 43, 44)
DEFAULT_RUN = ROOT / 'experiments/radzero_supervised_classification/runs/20261120'
PARENT_RUN = ROOT / 'experiments/supervised_baseline_classification/runs/20261120'


def launcher(model='RadZero'):
    assert model == 'RadZero'
    return ROOT / 'RadZero/scripts/python_local.sh'


def result_folder(run, dataset, method, seed):
    if method in ('original', 'pre_mlp'):
        return reference_folder('RadZero', dataset, seed, method)
    return run / 'results/RadZero' / dataset / f'seed{seed}' / method


def verify_sources(protocol):
    for path, digest in protocol['source_sha256'].items():
        assert sha256(path) == digest, 'Frozen source changed: ' + path
