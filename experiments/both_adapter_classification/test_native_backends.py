"""CPU identity, native-readout parity and joint-gradient checks for Both.

Use CARZero's local Python launcher with ``--real-carzero`` to additionally
load the released frozen CARZero model and compare to a classifier input
hook on its unchanged native forward path. No GPU is used by these checks.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.both_adapter_classification.native_backends import BothAlphaRAD, BothRadZero, BothCARZero
from experiments.baseline_token_classification.common import residual_mlp
from RadZero.scripts.adapter_model import ResidualAdapter

REPORT = {"device": "cpu", "checks": []}


def joint_gradients(backend, tokens, classes):
    torch.manual_seed(42)
    pre, post = residual_mlp(768, 64), residual_mlp(768, 64)
    assert sum(p.numel() for p in list(pre.parameters()) + list(post.parameters())) == 198272
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(pre.parameters(), post.parameters()))
    optimizer = torch.optim.AdamW(list(pre.parameters()) + list(post.parameters()), lr=1e-4)
    target = torch.arange(len(tokens) * classes).reshape(len(tokens), classes).remainder(2).float()
    backend.train()
    original = backend.score_tokens(tokens).detach()
    torch.testing.assert_close(backend.score_tokens(tokens, pre, post), original, atol=0, rtol=0)
    norms = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = backend.score_tokens(tokens, pre, post)
        F.binary_cross_entropy_with_logits(logits, target).backward()
        step_norms = {}
        for name, module in (("pre", pre), ("post", post)):
            for param_name, parameter in module.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                step_norms[f"{name}.{param_name}"] = float(parameter.grad.norm())
            assert float(module.up.weight.grad.norm()) > 0
            if step:
                assert float(module.down.weight.grad.norm()) > 0
            else:
                assert float(module.down.weight.grad.norm()) == 0
        assert all(not p.requires_grad and p.grad is None for p in backend.parameters())
        optimizer.step()
        norms.append(step_norms)
    assert not torch.equal(backend.score_tokens(tokens, pre, post), original)
    return norms


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.q = torch.randn(3, 768)
        self.tokens = torch.randn(2, 11, 768)

    def test_original_mlp_exact_state_keys_and_initialization(self):
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            original = ResidualAdapter(768, 64)
            torch.manual_seed(seed)
            common = residual_mlp(768, 64)
            self.assertEqual(set(original.state_dict()), set(common.state_dict()))
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, common.state_dict()[name], atol=0, rtol=0)

    def test_alpha_joint_gradients_and_independent_native_equations(self):
        cfg = dict(model="alpharad", queries=self.q.tolist(), heads=8,
                   include_cls=True, scale=15., head_index=None, head_weights=None)
        backend = BothAlphaRAD(cfg)
        value = F.normalize(self.tokens.reshape(2, 11, 8, 96), dim=-1)
        query = F.normalize(self.q.reshape(3, 8, 96), dim=-1)
        attention = (torch.einsum("bthd,chd->bcth", value, query) * 15.).softmax(2)
        context = torch.einsum("bcth,bthd->bchd", attention, value)
        reference = (torch.einsum("bchd,chd->bch", F.normalize(context, dim=-1), query) * 15.).mean(-1)
        torch.testing.assert_close(backend.score_tokens(self.tokens), reference)
        norms = joint_gradients(backend, self.tokens, 3)
        REPORT["checks"].append({"model": "AlphaRAD", "check": "synthetic native equations and joint gradients", "gradient_norms": norms})

    def test_rad_joint_gradients_and_independent_native_equations(self):
        cfg = dict(model="radzero", queries=self.q.tolist(), sim_op="cos", include_cls=False,
                   layer_norm_weight=torch.ones(768).tolist(), layer_norm_bias=torch.zeros(768).tolist(),
                   layer_norm_eps=1e-5, attention_temperature=.17, loss_temperature=.13)
        backend = BothRadZero(cfg)
        value = F.normalize(F.layer_norm(self.tokens, (768,), torch.ones(768), torch.zeros(768), 1e-5)[:, 1:], dim=-1)
        query = F.normalize(self.q, dim=-1)
        attention = (torch.einsum("btd,cd->bct", value, query) / .17).softmax(-1)
        context = torch.einsum("bct,btd->bcd", attention, value)
        reference = torch.einsum("bcd,cd->bc", F.normalize(context, dim=-1), query) / .13
        torch.testing.assert_close(backend.score_tokens(self.tokens), reference)
        norms = joint_gradients(backend, self.tokens, 3)
        REPORT["checks"].append({"model": "RadZero", "check": "synthetic native equations and joint gradients", "gradient_norms": norms})

    def test_all_six_real_head_and_token_cache_identities(self):
        cache = ROOT / "experiments/direct_anchor_classification/runs/20261118/cache"
        for model, cls in (("alpharad", BothAlphaRAD), ("radzero", BothRadZero)):
            for dataset in ("openi", "chexpert", "chestxray14"):
                record = json.loads((cache / model / dataset / "cache_00.json").read_text())
                cfg = json.loads(Path(record["head_path"]).read_text())
                backend = cls(cfg)
                spec = record["token_shards"][0]
                selected = [0, len(spec["indices"]) - 1]
                tokens = torch.tensor(np.array(np.load(spec["path"], mmap_mode="r")[selected], copy=True))
                ids = [spec["indices"][i] for i in selected]
                locations = {sample: i for i, sample in enumerate(record["indices"])}
                reference = torch.tensor(np.array(np.load(record["logits_path"], mmap_mode="r")[[locations[i] for i in ids]], copy=True))
                pre, post = residual_mlp(768, 64), residual_mlp(768, 64)
                actual = backend.score_tokens(tokens, pre, post).detach()
                torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
                # Pre-only is the existing frozen head's token-adapter path.
                nn.init.normal_(pre.up.weight, std=.001)
                torch.testing.assert_close(backend.score_tokens(tokens, pre), backend.head(tokens, token_adapter=pre), atol=0, rtol=0)
                REPORT["checks"].append({"model": model, "dataset": dataset,
                    "check": "real cached native scores; nonzero original-pre parity",
                    "selected_rows": ids, "native_max_abs_error": float((actual-reference).abs().max())})


class RealCARZeroTests(unittest.TestCase):
    def test_native_bidirectional_hook_parity_all_datasets_and_joint_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        path = ROOT / "baselines/CARZero/local_setup/token_mlp_adapter.py"
        spec = importlib.util.spec_from_file_location("both_test_carzero_token", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        token_backend = module.TokenBackend(device="cpu")
        cache = ROOT / "experiments/baseline_token_classification/runs/20261119/cache/CARZero"
        for dataset in ("openi", "chexpert", "chestxray14"):
            record = json.loads((cache / dataset / "cache.json").read_text())
            token_backend.prepare_classes(record["class_names"])
            backend = BothCARZero(token_backend)
            tokens = torch.tensor(np.array(np.load(record["files"]["tokens.npy"]["path"], mmap_mode="r")[:2], copy=True))
            cached = torch.tensor(np.array(np.load(record["files"]["aggregation_logits.npy"]["path"], mmap_mode="r")[:2], copy=True))
            pre, post = residual_mlp(768, 64), residual_mlp(768, 64)
            native = token_backend.score_tokens(tokens).detach()
            initial = backend.score_tokens(tokens, pre, post).detach()
            torch.testing.assert_close(initial, native, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(initial, cached, atol=3e-4, rtol=2e-5)
            nn.init.normal_(pre.up.weight, std=.001)
            nn.init.normal_(post.up.weight, std=.001)
            torch.testing.assert_close(backend.score_tokens(tokens, pre), token_backend.score_tokens(tokens, pre), atol=1e-5, rtol=1e-5)
            both = backend.score_tokens(tokens, pre, post)
            # Independent native forward: inject the post MLP at the original
            # classifier input. The same hook runs once in each direction.
            calls = []
            def adapt_classifier_input(module, args):
                calls.append(tuple(args[0].shape))
                return (post(args[0]),)
            handle = token_backend.model.fusion_module.mlp_head.register_forward_pre_hook(adapt_classifier_input)
            try:
                hooked = token_backend.score_tokens(tokens, pre)
            finally:
                handle.remove()
            self.assertEqual(len(calls), 2)
            torch.testing.assert_close(both, hooked, atol=1e-5, rtol=1e-5)
            row = {"model": "CARZero", "dataset": dataset, "check": "real bidirectional native classifier hook parity",
                   "native_max_abs_error": float((initial-native).abs().max()),
                   "cache_max_abs_error": float((initial-cached).abs().max()),
                   "nonzero_both_native_hook_max_abs_error": float((both-hooked).abs().max()),
                   "post_hook_shapes": calls}
            if dataset == "chexpert":
                row["gradient_norms"] = joint_gradients(backend, tokens, len(record["class_names"]))
                self.assertFalse(backend.fusion.training)
            REPORT["checks"].append(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-carzero", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SimilarityTests)
    if args.real_carzero:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RealCARZeroTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(status="passed" if result.wasSuccessful() else "failed", tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(REPORT, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
