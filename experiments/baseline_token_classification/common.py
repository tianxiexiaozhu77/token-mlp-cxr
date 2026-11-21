"""Shared protocol helpers; model-specific aggregation remains in each backend."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("CARZero", "KAD", "MedKLIP", "MAVL", "BiomedCLIP", "BioViL-T", "CheXzero", "MaCo")
DATASETS = {
    "openi": ROOT / "RadZero/runs/openi_adapter_20261020_152558/inputs/dataset.json",
    "chexpert": ROOT / "data/direct_anchor_classification_20261118/chexpert/dataset.json",
    "chestxray14": ROOT / "data/direct_anchor_classification_20261118/chestxray14/dataset.json",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_dataset(path):
    meta = json.loads(Path(path).read_text())
    if sha256(meta["samples_file"]) != meta["samples_sha256"]:
        raise ValueError("Changed dataset rows")
    rows = json.loads(Path(meta["samples_file"]).read_text())
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image identity")
    for field in ("group_id", "patient_id", "uid"):
        seen = {}
        for r in rows:
            if field in r and seen.setdefault(r[field], r["split"]) != r["split"]:
                raise ValueError("Split overlap: " + field)
    return meta, rows


def load_backend(model, device, variant="native"):
    if model not in MODELS:
        raise ValueError(model)
    root = ROOT / "baselines" / model
    if variant not in ("native", "openi18_query_extension"):
        raise ValueError("Unknown backend variant: " + variant)
    if variant == "openi18_query_extension" and model not in ("MedKLIP", "MAVL"):
        raise ValueError("OpenI query extension is defined only for MedKLIP/MAVL")
    filename = "token_mlp_adapter.py" if variant == "native" else "openi18_token_mlp_adapter.py"
    path = root / "local_setup" / filename
    sys.path[:0] = [str(root / "local_setup"), str(root)]
    spec = importlib.util.spec_from_file_location("token_backend_" + model.replace("-", "_") + "_" + variant, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TokenBackend(device=device)


def residual_mlp(dim, bottleneck=64):
    from torch import nn
    class ResidualTokenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(dim, bottleneck)
            self.activation = nn.GELU()
            self.up = nn.Linear(bottleneck, dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, tokens):
            return tokens + self.up(self.activation(self.down(tokens)))
    return ResidualTokenMLP()


def metrics(labels, scores, classes):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid predictions")
    per_class = {}
    for i, name in enumerate(classes):
        y = labels[:, i]
        valid = len(np.unique(y)) == 2
        per_class[name] = dict(positives=int(y.sum()), negatives=int(len(y)-y.sum()),
            auroc=float(roc_auc_score(y, scores[:, i])) if valid else None,
            auprc=float(average_precision_score(y, scores[:, i])) if valid else None)
    valid = [v for v in per_class.values() if v["auroc"] is not None]
    if len(valid) != len(classes):
        raise ValueError("Full-class metrics undefined")
    return dict(macro_auroc=float(np.mean([v["auroc"] for v in valid])),
                macro_auprc=float(np.mean([v["auprc"] for v in valid])),
                micro_auprc=float(average_precision_score(labels.ravel(), scores.ravel())),
                valid_classes=len(valid), total_classes=len(classes), per_class=per_class)


def setup_gpu(index):
    if index not in (4, 5, 6, 7):
        raise ValueError("Only physical GPUs 4--7 are authorized")
    import os
    sys.path.insert(0, str(ROOT / "AlphaRAD/local_setup"))
    import run_pretrained
    # Physical devices are host-local. Explicit /tmp avoids the model launchers'
    # NFS TMPDIR and shares the already established local UUID lease convention.
    run_pretrained.LOCKS = Path("/tmp") / f"medical-gpu-locks-{os.getuid()}"
    reserve_gpu = run_pretrained.reserve_gpu
    lock, gpu = reserve_gpu(index)
    if lock is None:
        raise RuntimeError("GPU is reserved or in use: " + str(gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    def close_in_child():
        if not lock.closed:
            lock.close()
    os.register_at_fork(after_in_child=close_in_child)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return lock, gpu


def release_gpu(lock):
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def source_hashes(model):
    root = ROOT / "baselines" / model
    files = list(Path(__file__).parent.glob("*.py"))
    files += list((root / "local_setup").glob("*.py"))
    files += [root / "local_setup/manifest.json"]
    return {str(p.resolve()): sha256(p) for p in files if p.is_file()}


def freeze_backend(backend):
    import torch
    seen, modules = set(), []
    def visit(value, depth):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            value.eval().requires_grad_(False)
            modules.append(value)
        elif depth and isinstance(value, (tuple, list, dict)):
            for item in value.values() if isinstance(value, dict) else value:
                visit(item, depth-1)
        elif depth and hasattr(value, "__dict__") and not isinstance(value, type):
            for item in vars(value).values():
                visit(item, depth-1)
    visit(backend, 4)
    return modules


def query_hashes(backend):
    import torch
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in vars(backend).items() if isinstance(value, torch.Tensor)}
