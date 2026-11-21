"""Numerical and gradient tests for the actual adaptation boundary."""
import importlib.util
import unittest

import torch
from torch.nn import functional as F

from experiment_utils import ROOT
from model import DirectAnchor, FrozenClassificationHead


class AnchorTests(unittest.TestCase):
    def test_zero_initialization_and_gradient_flow(self):
        torch.manual_seed(42)
        h = torch.randn(5, 3, 16)
        adapter = DirectAnchor(3, 16, 4)
        torch.testing.assert_close(adapter(h), h, atol=0, rtol=0)
        target = torch.randn_like(h)
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.1)
        ((adapter(h)-target)**2).mean().backward()
        self.assertGreater(float(adapter.up.grad.abs().sum()), 0)
        self.assertEqual(float(adapter.down.grad.abs().sum()), 0)
        optimizer.step(); optimizer.zero_grad()
        ((adapter(h)-target)**2).mean().backward()
        self.assertGreater(float(adapter.down.grad.abs().sum()), 0)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 3*16+16*4+3*4*16)

    def test_class_specific_parameters(self):
        h = torch.randn(2, 3, 16)
        adapter = DirectAnchor(3, 16, 4)
        with torch.no_grad():
            adapter.bias[1].fill_(.2)
            adapter.up[1].fill_(.1)
        torch.testing.assert_close(adapter(h)[:, [0, 2]], h[:, [0, 2]], atol=0, rtol=0)
        self.assertFalse(torch.equal(adapter(h)[:, 1], h[:, 1]))

    def test_against_published_radzero_including_singletons(self):
        spec = importlib.util.spec_from_file_location("native_losses", ROOT / "RadZero/checkpoints/RadZero/losses.py")
        native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
        for batch, classes in ((3, 4), (1, 4), (3, 1), (1, 1)):
            q, tokens = torch.randn(classes, 16), torch.randn(batch, 7, 16)
            cfg = dict(model="radzero", queries=q.tolist(), sim_op="cos", include_cls=False,
                       attention_temperature=.17, loss_temperature=.13)
            head = FrozenClassificationHead(cfg)
            logits, _ = native.SimilarityLogit("cos")(q, tokens[:, 1:], temperature=.17)
            expected = logits.T.reshape(batch, classes)/.13
            torch.testing.assert_close(head(tokens), expected)
            adapter = DirectAnchor(classes, 16, 2)
            torch.testing.assert_close(head(tokens, adapter), expected)

    def test_bias_control_preserves_native_alpha_head_aggregation(self):
        for head_index, weights in ((None, None), (1, None), (None, [.2, .8])):
            q, tokens = torch.randn(3, 16), torch.randn(2, 9, 16)
            cfg = dict(model="alpharad", queries=q.tolist(), heads=2, scale=9., include_cls=True,
                       head_index=head_index, head_weights=weights)
            head = FrozenClassificationHead(cfg)
            h = head.context(tokens)
            adapter = DirectAnchor(3, 16, 0)
            torch.testing.assert_close(head.score(h, adapter), head(tokens), atol=0, rtol=0)
            with torch.no_grad(): adapter.bias.fill_(.05)
            actual = head.score(h, adapter)
            expected = (F.normalize((h+.05).reshape(2,3,2,8), dim=-1) * F.normalize(q.reshape(3,2,8),dim=-1)).sum(-1)*9
            expected = expected[...,head_index] if head_index is not None else (expected.mean(-1) if weights is None else (expected*torch.tensor(weights)).sum(-1))
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
