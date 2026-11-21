"""Fixed transfer scope and checkpoint loading; no training entry point."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import sha256, save, metrics, residual_mlp, setup_gpu, release_gpu

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/20261121'
SOURCE = ROOT / 'experiments/large_patient_multilabel/runs/20261121'
TARGET = ROOT / 'data/alpharad/CheXpert'
MODELS = ('AlphaRAD', 'RadZero')
METHODS = ('original', 'linear_probe', 'clip_global', 'post_mlp', 'pre_mlp')
SEEDS = (42, 43, 44)
MAPPING = (
    ('Atelectasis', 'Atelectasis'), ('Cardiomegaly', 'Cardiomegaly'),
    ('Effusion', 'Pleural Effusion'), ('Pneumonia', 'Pneumonia'),
    ('Pneumothorax', 'Pneumothorax'), ('Consolidation', 'Consolidation'), ('Edema', 'Edema'))


def read(path):
    return json.loads(Path(path).read_text())


def launcher(model):
    return ROOT / ('AlphaRAD/local_setup/python_local.sh' if model == 'AlphaRAD' else 'RadZero/scripts/python_local.sh')


def jobs(model=None):
    return [(m, method, seed) for m in (MODELS if model is None else (model,))
            for method in METHODS for seed in ((42,) if method == 'original' else SEEDS)]


def folder(model, method, seed):
    return RUN / 'results' / model / method / f'seed{seed}'


def source_folder(model, method, seed):
    return SOURCE / 'results' / model / method / f'seed{seed}'


def project(scores, source_classes):
    """Keep the registered semantic mapping, never positional target assumptions."""
    if len(source_classes) != len(set(source_classes)) or scores.shape[-1] != len(source_classes):
        raise ValueError('Invalid source class axis')
    return scores[..., [source_classes.index(source) for source, _ in MAPPING]]


def build_module(method, state, classes):
    from experiments.supervised_baseline_classification.components import LinearProbe, ClipGlobalAdapter
    if method in ('pre_mlp', 'post_mlp'):
        module = residual_mlp(768, 64)
    elif method == 'linear_probe':
        # Normalization buffers come exclusively from the selected NIH checkpoint.
        module = LinearProbe(state['feature_mean'], state['feature_std'], classes)
    elif method == 'clip_global':
        module = ClipGlobalAdapter()
    else:
        raise ValueError(method)
    module.load_state_dict(state, strict=True)
    return module.eval().requires_grad_(False)


def validate_payload(payload, model, method, seed, protocol, selection):
    expected = dict(model=model, method=method, seed=seed, dim=768, bottleneck=64,
        classes=protocol['source_class_names'], best_epoch=selection['best_epoch'],
        protocol_sha256=sha256(SOURCE / 'PROTOCOL.json'),
        head_config=read(SOURCE / 'features' / model / 'head.json'))
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError('Source checkpoint mismatch: ' + key)


def selected_module(protocol, model, method, seed, device):
    import torch
    selection = next(r for r in protocol['source_selection']
                     if (r['model'], r['method'], r['seed']) == (model, method, seed))
    if sha256(selection['checkpoint']) != selection['checkpoint_sha256']:
        raise ValueError('Selected source checkpoint changed')
    payload = torch.load(selection['checkpoint'], map_location='cpu', weights_only=True)
    validate_payload(payload, model, method, seed, protocol, selection)
    return build_module(method, payload['module_state_dict'], len(payload['classes'])).to(device), selection


def check_source():
    from experiments.large_patient_multilabel.common import load_protocol, selected_epoch
    p = load_protocol(SOURCE)
    completion = read(SOURCE / 'COMPLETION.json')
    if completion['status'] != 'complete':
        raise ValueError('Source experiment incomplete')
    for name, digest in completion['output_sha256'].items():
        if sha256(SOURCE / name) != digest:
            raise ValueError('Source completion artifact changed: ' + name)
    seal = read(SOURCE / 'SELECTION_LOCK.json')
    expected = {(m, method, s) for m, method, s in jobs() if method != 'original'}
    if {(r['model'], r['method'], r['seed']) for r in seal['selections']} != expected:
        raise ValueError('Source selection coverage differs')
    for row in seal['selections']:
        out = source_folder(row['model'], row['method'], row['seed'])
        for filename, key in (('adapter.pt', 'checkpoint_sha256'), ('fit.json', 'fit_sha256'),
                              ('training.json', 'history_sha256')):
            if sha256(out / filename) != row[key]:
                raise ValueError('Source selected artifact changed')
        fit, history = read(out / 'fit.json'), read(out / 'training.json')
        if fit['test_accessed'] or not fit['frozen_head_verified']:
            raise ValueError('Source fitting invariant failed')
        if selected_epoch(history['initial_val']['macro_auroc'], history['history']) != row['best_epoch']:
            raise ValueError('Source selection is not NIH validation selection')
    return p, seal


def load_protocol():
    p = read(RUN / 'PROTOCOL.json')
    if p['status'] != 'locked':
        raise ValueError('Transfer protocol is not locked')
    for path, digest in p['artifact_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Locked artifact changed: ' + path)
    for path, digest in p['implementation_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen implementation changed: ' + path)
    return p
