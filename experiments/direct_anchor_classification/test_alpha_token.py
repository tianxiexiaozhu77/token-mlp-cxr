"""Verify the new token boundary against the actual native AlphaRAD branch."""
import json
import sys
from types import SimpleNamespace, MethodType
import unittest

import torch

from experiment_utils import ROOT
sys.path.insert(0, str(ROOT / "AlphaRAD/local_setup"))
sys.path.insert(0, str(ROOT / "AlphaRAD"))
from token_adapter_model import ResidualTokenAdapter, classify_with_token_adapter
from cxrclip.model.cxrclip_plus_dqn import CXRClipPlusWithDQN
from model import FrozenClassificationHead


class TokenTests(unittest.TestCase):
    def test_native_zero_nonzero_and_gradients(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        path = ROOT / "experiments/direct_anchor_classification/runs/20261118/cache/alpharad/chexpert/head_00.json"
        cfg = json.loads(path.read_text())
        head = FrozenClassificationHead(cfg).eval()
        native = SimpleNamespace(radzero_logit_scale=torch.tensor(cfg["scale"]).log(),
            radzero_head_enable_cross_attend_cls_token=cfg["include_cls"], num_radzero_heads=cfg["heads"],
            use_learnable_head_weights=cfg["head_weights"] is not None, head_index=cfg["head_index"],
            radzero_head_enable_bce_loss=True, radzero_head_enable_softmax_loss=False)
        if cfg["head_weights"] is not None:
            native.radzero_head_weights = torch.tensor(cfg["head_weights"]).log()
        native.multihead_radzero_classification_branch = MethodType(
            CXRClipPlusWithDQN.multihead_radzero_classification_branch, native)
        tokens = torch.randn(2, 17, head.dim)
        visual = {"image_cls_raw_features": [tokens[:,0]], "image_patch_raw_features": [tokens[:,1:]]}
        text = {"text_cls_raw_features": head.queries}
        original = native.multihead_radzero_classification_branch(visual, text)
        adapter = ResidualTokenAdapter(head.dim, 64)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 99136)
        self.assertFalse(any(p.requires_grad for p in head.parameters()))
        torch.testing.assert_close(adapter(tokens), tokens, atol=0, rtol=0)
        torch.testing.assert_close(head(tokens, token_adapter=adapter), original["logits_label_to_image"].squeeze(-1))
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.01)
        target = torch.randn_like(original["logits_label_to_image"].squeeze(-1))
        ((head(tokens, token_adapter=adapter)-target)**2).mean().backward()
        self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0)
        self.assertEqual(float(adapter.down.weight.grad.abs().sum()), 0)
        optimizer.step(); optimizer.zero_grad()
        modified = classify_with_token_adapter(native, visual, text, adapter)
        logits = head(tokens, token_adapter=adapter)
        torch.testing.assert_close(logits, modified["logits_label_to_image"].squeeze(-1), atol=2e-5, rtol=2e-5)
        self.assertFalse(torch.equal(original["radzero_attention_weights_h"], modified["radzero_attention_weights_h"]))
        ((logits-target)**2).mean().backward()
        self.assertGreater(float(adapter.down.weight.grad.abs().sum()), 0)
        for p in adapter.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
        torch.testing.assert_close(visual["image_cls_raw_features"][0], tokens[:,0], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
