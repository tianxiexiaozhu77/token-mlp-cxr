"""CPU checks of loss mathematics, matched initialization, and dynamic gating.

Run with AlphaRAD/local_setup/python_local.sh; no datasets or GPUs needed.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from experiments.baseline_token_classification.common import residual_mlp
from experiments.gated_asl_classification.components import (
    METHODS, ASL_CONFIG, ASL_REFERENCE, GATE_CONFIG,
    make_adapter, make_loss, parameter_count)


def explicit_scalar_asl(value, target, config):
    """Independent Python-scalar formula and its analytic derivative."""
    p = 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))
    dp = p * (1.0 - p)
    if target == 1:
        q, dq, gamma = p, dp, config["gamma_pos"]
    else:
        unbounded = 1.0 - p + config["clip"]
        q, dq, gamma = min(unbounded, 1.0), -dp if unbounded < 1 else 0.0, config["gamma_neg"]
    w = (1.0 - q) ** gamma
    dw = -gamma * (1.0 - q) ** (gamma - 1.0) if gamma != 0 else 0.0
    logq = math.log(max(q, config["eps"]))
    dlogq = 1.0 / q if q > config["eps"] else 0.0
    return -w * logq, -(dw * logq + w * dlogq) * dq


class ComponentsTests(unittest.TestCase):
    def test_plain_exact_legacy_and_shared_initialization(self):
        for dim in (128, 512, 768):
            torch.manual_seed(42)
            old = residual_mlp(dim, 64)
            old_state = copy.deepcopy(old.state_dict())
            tokens = torch.randn(2, 3, dim)
            for method in METHODS:
                torch.manual_seed(42)
                adapter = make_adapter(dim, 64, method)
                state = adapter.state_dict()
                self.assertEqual(set(state), set(old_state) | ({"gate.weight", "gate.bias"} if method.startswith("gated_") else set()))
                for name, value in old_state.items():
                    self.assertTrue(torch.equal(value, state[name]), (dim, method, name))
                self.assertTrue(torch.equal(adapter(tokens), tokens))
                self.assertEqual(sum(p.numel() for p in adapter.parameters()), parameter_count(dim, 64, method))
                if hasattr(adapter, "gate"):
                    self.assertTrue(torch.equal(adapter.gate.weight, torch.zeros_like(adapter.gate.weight)))
                    torch.testing.assert_close(adapter.gate_values(tokens), torch.full((2, 3, 1), 0.1), atol=1e-8, rtol=1e-7)
                else:
                    # Nonzero legacy and new adapters must remain identical,
                    # not merely both return identity at zero initialization.
                    changed = {k: v + 0.01 for k, v in old_state.items()}
                    old.load_state_dict(changed)
                    adapter.load_state_dict(changed)
                    self.assertTrue(torch.equal(adapter(tokens), old(tokens)))
                    old.load_state_dict(old_state)

    def test_gate_normalizes_only_its_input_and_is_shared(self):
        torch.manual_seed(9)
        adapter = make_adapter(11, 4, "gated_bce").double()
        with torch.no_grad():
            adapter.gate.weight.copy_(torch.randn_like(adapter.gate.weight))
            adapter.up.weight.copy_(torch.randn_like(adapter.up.weight))
            adapter.up.bias.copy_(torch.randn_like(adapter.up.bias))
        tokens = torch.randn(2, 5, 11, dtype=torch.float64)
        expected_gate = torch.sigmoid(F.linear(F.layer_norm(tokens, (11,), eps=1e-5), adapter.gate.weight, adapter.gate.bias))
        expected = tokens + expected_gate * F.linear(F.gelu(F.linear(tokens, adapter.down.weight, adapter.down.bias)), adapter.up.weight, adapter.up.bias)
        torch.testing.assert_close(adapter(tokens), expected, atol=0, rtol=0)
        self.assertEqual(tuple(adapter.gate_values(tokens).shape), (2, 5, 1))
        torch.testing.assert_close(adapter(tokens[:, :1].expand(-1, 3, -1)), adapter(tokens[:, :1]).expand(-1, 3, -1), atol=1e-14, rtol=1e-14)

    def test_identity_then_up_down_gate_gradients(self):
        evidence = {}
        for method in METHODS:
            torch.manual_seed(8)
            adapter = make_adapter(12, 5, method).double()
            optimizer = torch.optim.SGD(adapter.parameters(), lr=0.3)
            tokens = torch.randn(6, 7, 12, dtype=torch.float64)
            target = torch.randn_like(tokens)
            self.assertTrue(torch.equal(adapter(tokens), tokens))
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                F.mse_loss(adapter(tokens), target).backward()
                norms = {name: float(p.grad.norm()) for name, p in adapter.named_parameters()}
                self.assertGreater(norms["up.weight"], 0)
                if step == 0:
                    self.assertEqual(norms["down.weight"], 0)
                    if hasattr(adapter, "gate"):
                        self.assertEqual(norms["gate.weight"], 0)
                else:
                    self.assertGreater(norms["down.weight"], 0)
                    if hasattr(adapter, "gate"):
                        self.assertGreater(norms["gate.weight"], 0)
                        self.assertGreater(norms["gate.bias"], 0)
                optimizer.step()
            if hasattr(adapter, "gate"):
                values = adapter.gate_values(tokens).detach()
                self.assertGreater(float(values.std()), 1e-9)
                self.assertTrue(bool(((values > 0) & (values < 1)).all()))
            evidence[method] = norms
        self.__class__.gradient_evidence = evidence

    def test_asl_independent_values_and_gradients_extremes(self):
        values = [-1000, -100, -21, -12, -3.4, -2.1, -0.5, 0, 0.7, 2.4, 9, 21, 100, 1000]
        for override in ({}, {"gamma_pos": 2.0}, {"gamma_neg": 0.0, "gamma_pos": 0.0, "clip": 0.0}):
            config = dict(ASL_CONFIG, **override)
            logits = torch.tensor([values, values], dtype=torch.float64, requires_grad=True)
            targets = torch.tensor([[0.0] * len(values), [1.0] * len(values)], dtype=torch.float64)
            expected = [explicit_scalar_asl(v, y, config) for y in (0, 1) for v in values]
            loss = make_loss("pre_asl", None, config)(logits, targets)
            gradients, = torch.autograd.grad(loss, logits)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(bool(torch.isfinite(gradients).all()))
            self.assertAlmostEqual(float(loss.detach()), sum(v[0] for v in expected) / logits.numel(), places=13)
            torch.testing.assert_close(gradients.flatten(), torch.tensor([v[1] / logits.numel() for v in expected], dtype=torch.float64), atol=1e-14, rtol=1e-12)

    def test_asl_float32_extremes_and_real_loss_gradient_flow(self):
        logits = torch.tensor([[-1000., -100., -20., 0., 20., 100., 1000.]] * 2, requires_grad=True)
        labels = torch.tensor([[0.] * 7, [1.] * 7])
        loss = make_loss("pre_asl", None, ASL_CONFIG)(logits, labels)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        for method in METHODS:
            torch.manual_seed(31)
            adapter = make_adapter(12, 5, method)
            tokens = torch.randn(6, 7, 12)
            fixed_classifier = torch.randn(12, 3, requires_grad=False)
            labels = torch.randint(0, 2, (6, 3)).float()
            loss_fn = make_loss(method, torch.tensor([2., 3., 4.]), ASL_CONFIG)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                logits = adapter(tokens).mean(dim=1) @ fixed_classifier
                loss_fn(logits, labels).backward()
                self.assertGreater(float(adapter.up.weight.grad.norm()), 0)
                if step:
                    self.assertGreater(float(adapter.down.weight.grad.norm()), 0)
                    if hasattr(adapter, "gate"):
                        self.assertGreater(float(adapter.gate.weight.grad.norm()), 0)
                optimizer.step()
            self.assertIsNone(fixed_classifier.grad)

    def test_asl_zero_focusing_and_clip_reduces_to_bce(self):
        logits = torch.tensor([[-3.0, 1.3, 6.0], [0.1, -0.4, 0.0]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]], dtype=torch.float64)
        asl = make_loss("gated_asl", torch.ones(3) * 999, dict(gamma_neg=0, gamma_pos=0, clip=0))(logits, labels)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        torch.testing.assert_close(asl, bce, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(torch.autograd.grad(asl, logits, retain_graph=True)[0], torch.autograd.grad(bce, logits)[0], atol=1e-14, rtol=1e-14)

    def test_full_focal_gradient_and_no_positive_weight(self):
        logits = torch.tensor([[0.2, 1.2, -1.3]], dtype=torch.float64, requires_grad=True)
        labels = torch.zeros_like(logits)
        loss = make_loss("pre_asl", torch.tensor([2., 7., 20.]), ASL_CONFIG)(logits, labels)
        unweighted = make_loss("pre_asl", None, ASL_CONFIG)(logits, labels)
        self.assertTrue(torch.equal(loss, unweighted))
        grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        negp = (1 - torch.sigmoid(logits) + 0.05).clamp(max=1)
        detached_loss = -((1 - negp).pow(4).detach() * negp.log()).mean()
        detached_grad = torch.autograd.grad(detached_loss, logits)[0]
        self.assertGreater(float((grad - detached_grad).abs().max()), 0.01)

    def test_mean_loss_and_gradient_microbatch_invariance(self):
        torch.manual_seed(4)
        logits = torch.randn(11, 6, dtype=torch.float64)
        labels = torch.randint(0, 2, (11, 6), dtype=torch.float64)
        weight = torch.tensor([1., 2., 7., 3., 20., 1.], dtype=torch.float64)
        for method in METHODS:
            loss_fn = make_loss(method, weight, ASL_CONFIG)
            whole = logits.clone().requires_grad_()
            whole_loss = loss_fn(whole, labels)
            whole_loss.backward()
            split = logits.clone().requires_grad_()
            total = 0
            for lo, hi in ((0, 4), (4, 8), (8, 11)):
                local_loss = loss_fn(split[lo:hi], labels[lo:hi]) * ((hi - lo) / len(logits))
                total += float(local_loss.detach())
                local_loss.backward()
            self.assertAlmostEqual(float(whole_loss.detach()), total, places=14)
            torch.testing.assert_close(whole.grad, split.grad, atol=1e-15, rtol=1e-14)

    def test_bce_is_exact_existing_weighted_mean_loss(self):
        logits = torch.tensor([[-4.0, 0.3], [2.0, -1.2]], requires_grad=True)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = torch.tensor([5.0, 19.0])
        expected = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights)
        for method in ("pre_bce", "gated_bce"):
            actual = make_loss(method, weights, ASL_CONFIG)(logits, labels)
            self.assertTrue(torch.equal(actual, expected))

    def test_rejects_protocol_drift_and_invalid_config(self):
        with self.assertRaises(ValueError):
            make_loss("pre_bce")
        for config in ({"detach_focal": True}, {"reduction": "sum"}, {"gamma_neg": -1}, {"eps": 0}, {"clip": 1}, {"gamma_pos": float("nan")}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                make_loss("pre_asl", None, config)
        with self.assertRaises(ValueError):
            make_adapter(12, 4, "both")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ComponentsTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "torch_version": torch.__version__, "device": "cpu",
        "asl_config": ASL_CONFIG, "gate_config": GATE_CONFIG,
        "asl_reference": ASL_REFERENCE,
        "gradient_evidence": getattr(ComponentsTests, "gradient_evidence", {}),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence, indent=2, allow_nan=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
