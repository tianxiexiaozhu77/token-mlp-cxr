from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import (
    DATASETS, read_dataset, save, sha256, metrics, setup_gpu, release_gpu,
    load_backend, freeze_backend, query_hashes)
from experiments.both_adapter_classification.common import reference_folder, BASE, DIRECT

MODELS = ('AlphaRAD', 'KAD', 'MaCo')
METHODS = ('linear_probe', 'clip_global')
SEEDS = (42, 43, 44)
DEFAULT_RUN = ROOT / 'experiments/supervised_baseline_classification/runs/20261120'


def launcher(model):
    return ROOT / ('AlphaRAD' if model == 'AlphaRAD' else 'baselines/' + model) / 'local_setup/python_local.sh'


def current_sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'experiments/baseline_token_classification/common.py',
              ROOT / 'experiments/direct_anchor_classification/model.py']
    for model in ('KAD', 'MaCo'):
        paths += list((ROOT / 'baselines' / model / 'local_setup').glob('*.py'))
    return {str(p): sha256(p) for p in paths}
