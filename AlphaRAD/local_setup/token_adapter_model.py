"""Shared residual token MLP before AlphaRAD's native FLaS aggregation.

The MLP is exactly the existing RadZero implementation. Inputs are projected
raw CLS/patch tokens, before per-head normalization. All original encoders,
projections, disease queries, scales and head weights stay frozen for fitting.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RadZero/scripts"))
from adapter_model import ResidualAdapter as ResidualTokenAdapter

sys.path.insert(0, str(ROOT / "experiments"))
from direct_anchor_classification.model import FrozenClassificationHead


def classify_with_token_adapter(model, image_embeddings, label_embeddings, adapter, **kwargs):
    """Call the native classifier with adapted projected visual tokens.

    Unlike a no-grad evaluation wrapper this preserves adapter gradients.
    The caller freezes/evals the original model. Image/text embeddings must
    come from the matching frozen checkpoint and original preprocessing.
    """
    cls = image_embeddings["image_cls_raw_features"]
    patches = image_embeddings["image_patch_raw_features"]
    if len(cls) != 1 or len(patches) != 1:
        raise ValueError("The current FLaS branch supports one projected layer")
    tokens = adapter(torch.cat([cls[0][:, None], patches[0]], dim=1))
    adapted = dict(image_embeddings)
    adapted["image_cls_raw_features"] = [tokens[:, 0]]
    adapted["image_patch_raw_features"] = [tokens[:, 1:]]
    return model.multihead_radzero_classification_branch(adapted, label_embeddings, **kwargs)


def load_token_adapter(checkpoint_path, device="cpu"):
    """Return frozen scoring head and fitted MLP for native projected tokens."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["method"] != "token_mlp" or payload["head_config"]["model"] != "alpharad":
        raise ValueError("Expected an AlphaRAD token_mlp checkpoint")
    head = FrozenClassificationHead(payload["head_config"]).to(device).eval()
    adapter = ResidualTokenAdapter(payload["dim"], payload["bottleneck"]).to(device).eval()
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return head, adapter, payload["classes"]
