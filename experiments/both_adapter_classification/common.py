"""Protocol constants and independent, identity-initialized residual adapters."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import (
    DATASETS, read_dataset, save, sha256, residual_mlp)

MODELS = ("AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo")
DIMS = {name: 512 if name == "BiomedCLIP" else 128 if name == "BioViL-T" else 768 for name in MODELS}
BASE = ROOT / "experiments/baseline_token_classification/runs/20261119"
DIRECT = ROOT / "experiments/direct_anchor_classification/runs/20261118"
ALPHA = ROOT / "experiments/direct_anchor_classification/runs/20261118_alpha_token_mlp"
PLACEMENT = ROOT / "experiments/adapter_placement_classification/runs/20261119"


def both_mlp(dim, bottleneck=64):
    from torch import nn
    # Sequential draws give independent initial down projections; both up
    # projections start at zero. The pre draw matches the earlier single MLP.
    return nn.ModuleDict({"pre": residual_mlp(dim, bottleneck), "post": residual_mlp(dim, bottleneck)})


def parameter_count(dim, bottleneck=64):
    return 2 * (2 * dim * bottleneck + bottleneck + dim)


def reference_folder(model, dataset, seed, method):
    if method == "post_mlp":
        return PLACEMENT / "results" / model / dataset / f"seed{seed}" / "post_mlp"
    if model in ("AlphaRAD", "RadZero"):
        root = ALPHA if model == "AlphaRAD" and method == "pre_mlp" else DIRECT
        old_method = "token_mlp" if model == "AlphaRAD" else "existing"
        return root / "results" / model.lower() / dataset / f"seed{seed}" / (old_method if method == "pre_mlp" else "baseline")
    return BASE / "results" / model / dataset / f"seed{seed}" / ("token_mlp" if method == "pre_mlp" else "baseline" if method == "original" else method)
