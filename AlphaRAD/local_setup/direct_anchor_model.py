"""AlphaRAD Direct-anchor: adapt concatenated FLaS contexts before head norms."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from direct_anchor_classification.model import DirectAnchor, FrozenClassificationHead


def export_head(model, queries):
    return {
        "model": "alpharad", "queries": queries.tolist(),
        "include_cls": model.radzero_head_enable_cross_attend_cls_token,
        "heads": model.num_radzero_heads,
        "scale": float(model.radzero_logit_scale.exp()),
        "head_index": model.head_index,
        "head_weights": model.radzero_head_weights.softmax(0).detach().cpu().tolist() if model.use_learnable_head_weights else None,
    }
