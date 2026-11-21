"""RadZero Direct-anchor: inject into VL-CABS context before its L2 norm."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from direct_anchor_classification.model import DirectAnchor, FrozenClassificationHead


def export_head(base, text_features):
    loss = base.loss_fns["KeyPhraseAlignmentLoss"]
    ln = loss.layer_norm
    return {
        "model": "radzero", "queries": text_features.detach().cpu().tolist(),
        "include_cls": loss.use_vision_cls_token, "sim_op": loss.sim_op,
        "attention_temperature": float((loss.attn_temperature if loss.attn_temperature is not None else loss.loss_temperature).exp()),
        "loss_temperature": float(loss.loss_temperature.exp()),
        "layer_norm_weight": ln.weight.detach().cpu().tolist() if ln is not None else None,
        "layer_norm_bias": ln.bias.detach().cpu().tolist() if ln is not None else None,
        "layer_norm_eps": ln.eps if ln is not None else None,
    }
