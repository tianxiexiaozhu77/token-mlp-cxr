"""CPU invariants for a controlled pre/post FLaS residual-MLP comparison."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alpharad_backend import AlphaRADPlacementBackend, ResidualAdapter
from experiments.baseline_token_classification.common import residual_mlp


class AlphaRADPlacementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.cfg = dict(model="alpharad", queries=torch.randn(4, 768).tolist(),
                        heads=8, include_cls=True, scale=15.07514,
                        head_index=None, head_weights=None)
        self.backend = AlphaRADPlacementBackend(self.cfg)
        self.tokens = torch.randn(3, 17, 768)

    def native_equations(self):
        values = F.normalize(self.tokens.reshape(3, 17, 8, 96), dim=-1)
        query = F.normalize(torch.tensor(self.cfg["queries"]).reshape(4, 8, 96), dim=-1)
        similarity = torch.einsum("bthd,chd->bcth", values, query) * self.cfg["scale"]
        contexts = torch.einsum("bcth,bthd->bchd", similarity.softmax(2), values)
        logits = torch.einsum("bchd,chd->bch", F.normalize(contexts, dim=-1), query)
        return contexts.flatten(-2), (logits * self.cfg["scale"]).mean(-1)

    def test_identity_shape_and_parameter_match(self):
        context, reference = self.native_equations()
        pre, post = ResidualAdapter(768, 64), ResidualAdapter(768, 64)
        self.assertEqual(sum(p.numel() for p in pre.parameters()), 99136)
        self.assertEqual(sum(p.numel() for p in pre.parameters()),
                         sum(p.numel() for p in post.parameters()))
        self.assertEqual(tuple(context.shape), (3, 4, 768))
        torch.testing.assert_close(self.backend.context(self.tokens), context)
        for placement, adapter in (("none", None), ("pre", pre), ("post", post)):
            torch.testing.assert_close(self.backend(self.tokens, adapter, placement), reference)
        torch.testing.assert_close(self.backend.forward_context(context, post), reference)
        self.assertFalse(any(p.requires_grad for p in self.backend.parameters()))

    def test_common_mlp_has_exact_same_initial_state(self):
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            native = ResidualAdapter(768, 64)
            torch.manual_seed(seed)
            common = residual_mlp(768, 64)
            self.assertEqual(list(native.state_dict()), list(common.state_dict()))
            for name, value in native.state_dict().items():
                torch.testing.assert_close(value, common.state_dict()[name], atol=0, rtol=0)

    def test_both_placements_receive_gradients_and_context_cache_matches(self):
        target = torch.randn(3, 4)
        frozen_context = self.backend.context(self.tokens).detach()
        for placement in ("pre", "post"):
            adapter = ResidualAdapter(768, 64)
            optimizer = torch.optim.SGD(adapter.parameters(), lr=.01)
            initial = self.backend(self.tokens).detach()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                logits = self.backend(self.tokens, adapter, placement)
                if placement == "post":
                    torch.testing.assert_close(logits, self.backend.forward_context(frozen_context, adapter))
                F.mse_loss(logits, target).backward()
                self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0.)
                if step:
                    self.assertGreater(float(adapter.down.weight.grad.abs().sum()), 0.)
                else:
                    self.assertEqual(float(adapter.down.weight.grad.abs().sum()), 0.)
                for parameter in adapter.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                optimizer.step()
            self.assertFalse(torch.equal(initial, self.backend(self.tokens, adapter, placement)))
            self.assertFalse(any(p.grad is not None for p in self.backend.parameters()))

    def test_post_adapter_is_shared_over_diseases_and_pre_final_normalization(self):
        adapter = ResidualAdapter(768, 64)
        torch.nn.init.normal_(adapter.up.weight, std=.01)
        context = self.backend.context(self.tokens)
        adapted = adapter(context)
        manual = torch.stack([adapter(context[:, c]) for c in range(4)], 1)
        torch.testing.assert_close(adapted, manual)
        h = F.normalize(adapted.reshape(3, 4, 8, 96), dim=-1)
        q = F.normalize(self.backend.head.queries.reshape(4, 8, 96), dim=-1)
        reference = (torch.einsum("bchd,chd->bch", h, q) * self.cfg["scale"]).mean(-1)
        torch.testing.assert_close(self.backend(self.tokens, adapter, "post"), reference)

    def test_invalid_model_shapes_and_placement_rejected(self):
        with self.assertRaises(ValueError):
            AlphaRADPlacementBackend(dict(self.cfg, model="radzero"))
        with self.assertRaises(ValueError):
            self.backend.context(self.tokens[..., :96])
        with self.assertRaises(ValueError):
            self.backend.forward_context(torch.randn(3, 5, 768))
        with self.assertRaises(ValueError):
            self.backend(self.tokens, placement="after_logits")
        with self.assertRaises(ValueError):
            self.backend(self.tokens, ResidualAdapter(), "none")


if __name__ == "__main__":
    unittest.main()
