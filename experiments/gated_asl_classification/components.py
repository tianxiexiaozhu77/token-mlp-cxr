"""Independent Pre-MLP and loss factors for the four-group campaign.

ASL primary implementation inspected on 2026-11-19:
https://github.com/Alibaba-MIIL/ASL/blob/8c9e0bd8d5d450cf19093363fc08aa7244ad4408/src/loss_functions/losses.py
Raw source SHA256: ce743a0cd3156a4eae10bbf673ce97fed161e20621d0550d13bd4a48b1ea9c28
Reference paper: https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html

The probability clipping and asymmetric focal factors follow that source.
This campaign explicitly uses gamma_pos=0, gamma_neg=4, clip=0.05,
eps=1e-8, full focal-weight differentiation (detach_focal=False), and a
mean over all batch/class entries. The reference class defaults to
gamma_pos=1 and a sum; its optimized version also supports full focal
autograd. The mean makes effective-batch accumulation match the BCE
reduction. No positive-class weights are applied to ASL. This changes
the loss factor only; ASL groups use the same adapter initialization.

Imports stay CPU-light so protocol preparation does not require PyTorch.
"""
import math

from experiments.gated_asl_classification.common import (
    METHODS, METHOD_NAMES, ASL_CONFIG, GATE_CONFIG, parameter_count)
from experiments.baseline_token_classification.common import residual_mlp

METHOD_SPECS = {
    method: {
        "name": METHOD_NAMES[method],
        "gated": method.startswith("gated_"),
        "loss": "asl" if method.endswith("_asl") else "weighted_bce",
        "fresh_fit": True,
        "placement": "pre",
    }
    for method in METHODS
}

ASL_REFERENCE = {
    "repository": "https://github.com/Alibaba-MIIL/ASL",
    "commit": "8c9e0bd8d5d450cf19093363fc08aa7244ad4408",
    "source_url": "https://raw.githubusercontent.com/Alibaba-MIIL/ASL/8c9e0bd8d5d450cf19093363fc08aa7244ad4408/src/loss_functions/losses.py",
    "source_sha256": "ce743a0cd3156a4eae10bbf673ce97fed161e20621d0550d13bd4a48b1ea9c28",
    "inspected_on": "2026-11-19",
}


def _check_method(method):
    if method not in METHODS:
        raise ValueError(f"Unknown four-group method: {method}")


def make_adapter(dim, bottleneck=64, method="pre_bce"):
    """Make identity-initialized Pre adapter with unchanged legacy MLP keys.

    The exact legacy constructor runs before any gate construction, so a
    shared seed produces bit-identical down/up initialization in all four
    methods. Only the gate input is normalized; the MLP sees raw tokens.
    """
    _check_method(method)
    if not isinstance(dim, int) or dim <= 0 or not isinstance(bottleneck, int) or bottleneck <= 0:
        raise ValueError("dim and bottleneck must be positive integers")
    base = residual_mlp(dim, bottleneck)
    if not METHOD_SPECS[method]["gated"]:
        return base

    import torch
    from torch import nn
    from torch.nn import functional as F

    class GatedResidualTokenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = base.down
            self.activation = base.activation
            self.up = base.up
            self.gate = nn.Linear(dim, 1)
            nn.init.zeros_(self.gate.weight)
            probability = GATE_CONFIG["initial_probability"]
            nn.init.constant_(self.gate.bias, math.log(probability / (1.0 - probability)))

        def gate_values(self, tokens):
            normalized = F.layer_norm(
                tokens, (dim,), weight=None, bias=None,
                eps=GATE_CONFIG["layer_norm_eps"])
            return torch.sigmoid(self.gate(normalized))

        def forward(self, tokens):
            residual = self.up(self.activation(self.down(tokens)))
            return tokens + self.gate_values(tokens) * residual

    return GatedResidualTokenMLP()


def make_loss(method, pos_weight=None, asl_config=None):
    """Return a scalar mean loss; ASL deliberately ignores pos_weight.

    ``pos_weight`` is required by BCE and is retained on its current device.
    The ASL configuration can be overridden for unit validation; the frozen
    experiment protocol supplies the fixed ASL_CONFIG verbatim.
    """
    _check_method(method)
    import torch
    from torch import nn

    if METHOD_SPECS[method]["loss"] == "weighted_bce":
        if pos_weight is None:
            raise ValueError("Weighted BCE requires TRAIN-derived pos_weight")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    config = dict(ASL_CONFIG)
    if asl_config is not None:
        if set(asl_config) - set(config):
            raise ValueError("Unknown ASL configuration keys")
        config.update(asl_config)
    for key in ("gamma_neg", "gamma_pos", "clip", "eps"):
        config[key] = float(config[key])
        if not math.isfinite(config[key]):
            raise ValueError(f"Non-finite ASL {key}")
    if config["gamma_neg"] < 0 or config["gamma_pos"] < 0:
        raise ValueError("ASL focusing powers must be nonnegative")
    if not 0 <= config["clip"] < 1 or not 0 < config["eps"] < 1:
        raise ValueError("ASL clip/eps outside permitted bounds")
    if config["reduction"] != "mean" or config["detach_focal"] is not False:
        raise ValueError("This campaign requires mean ASL and full focal autograd")

    class AsymmetricMeanLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = dict(config)

        def forward(self, logits, targets):
            if logits.shape != targets.shape:
                raise ValueError("ASL logits/targets must have identical shape")
            probabilities = torch.sigmoid(logits)
            negative_probabilities = (1.0 - probabilities + config["clip"]).clamp(max=1.0)
            positive_term = torch.log(probabilities.clamp(min=config["eps"]))
            negative_term = torch.log(negative_probabilities.clamp(min=config["eps"]))
            # Separate scalar powers retain full autograd, including the
            # probability shift; gamma_pos=0 gives an exact unit factor.
            positive_term = positive_term * (1.0 - probabilities).pow(config["gamma_pos"])
            negative_term = negative_term * (1.0 - negative_probabilities).pow(config["gamma_neg"])
            targets = targets.to(dtype=logits.dtype)
            return -(targets * positive_term + (1.0 - targets) * negative_term).mean()

    return AsymmetricMeanLoss()
