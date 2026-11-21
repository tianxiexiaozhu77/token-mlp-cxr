"""Independently check the pre-normalization anchor against original source."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from experiment_utils import ROOT, save
from model import FrozenClassificationHead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / "AlphaRAD"))
    from cxrclip.model.cxrclip_plus_dqn import CXRClipPlusWithDQN
    results = []
    for path in sorted((args.run_dir / "cache").glob("*/*/cache_*.json")):
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            continue
        config = json.loads(Path(record["head_path"]).read_text())
        if config["model"] != "alpharad":
            continue
        head = FrozenClassificationHead(config).eval()
        source = record["token_shards"][0]
        tokens = torch.from_numpy(np.array(np.load(source["path"], mmap_mode="r")[:3], copy=True))
        native_self = SimpleNamespace(radzero_logit_scale=torch.tensor(config["scale"]).log(),
            radzero_head_enable_cross_attend_cls_token=config["include_cls"], num_radzero_heads=config["heads"],
            use_learnable_head_weights=config["head_weights"] is not None, head_index=config["head_index"],
            radzero_head_enable_bce_loss=True, radzero_head_enable_softmax_loss=False)
        if config["head_weights"] is not None:
            native_self.radzero_head_weights = torch.tensor(config["head_weights"]).log()
        output = CXRClipPlusWithDQN.multihead_radzero_classification_branch(native_self,
            {"image_cls_raw_features": [tokens[:,0]], "image_patch_raw_features": [tokens[:,1:]]},
            {"text_cls_raw_features": head.queries})
        raw = torch.einsum("bqkh,bkhd->bqhd", output["radzero_attention_weights_h"], output["radzero_image_kv_h"]).flatten(-2)
        actual = head.context(tokens)
        torch.testing.assert_close(actual, raw, atol=2e-6, rtol=2e-5)
        cached = torch.from_numpy(np.array(np.load(record["context_path"], mmap_mode="r")[:3], copy=True))
        torch.testing.assert_close(cached, raw, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(head.score(actual), output["logits_label_to_image"].squeeze(-1), atol=2e-5, rtol=2e-5)
        results.append(dict(cache=str(path), raw_anchor_max_error=float((actual-raw).abs().max()),
                            cached_anchor_max_error=float((cached-raw).abs().max())))
    save(args.run_dir / "native_anchor_validation.json", dict(status="passed", checked_shards=results))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
