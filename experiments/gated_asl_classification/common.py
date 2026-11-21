"""CPU-light constants for the four-group, freshly fitted Pre adapter study."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.both_adapter_classification.common import (
    MODELS, DIMS, BASE, DIRECT, ALPHA, PLACEMENT, DATASETS,
    read_dataset, save, sha256, reference_folder)

METHODS = ("pre_bce", "gated_bce", "pre_asl", "gated_asl")
METHOD_NAMES = {
    "pre_bce": "Pre Token MLP + weighted BCE",
    "gated_bce": "Dynamic-gated Pre Token MLP + weighted BCE",
    "pre_asl": "Pre Token MLP + ASL",
    "gated_asl": "Dynamic-gated Pre Token MLP + ASL",
}
SEEDS = (42, 43, 44)
ASL_CONFIG = dict(gamma_neg=4.0, gamma_pos=0.0, clip=0.05,
                  eps=1e-8, reduction="mean", detach_focal=False)
GATE_CONFIG = dict(input="nonaffine_layer_norm", layer_norm_eps=1e-5,
                   output="per_token_scalar_sigmoid", shared_across_tokens=True,
                   weight_init="zeros", initial_probability=0.1)


def parameter_count(dim, bottleneck=64, method="pre_bce"):
    if method not in METHODS:
        raise ValueError(method)
    return 2 * dim * bottleneck + bottleneck + dim + (dim + 1 if method.startswith("gated_") else 0)
