"""CPU integration checks on real cached tokens for all four new methods.

Run each model with its local Python launcher. The test preserves original
backends/checkpoints and does not create experiment training results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.gated_asl_classification.common import (
    MODELS, METHODS, ASL_CONFIG, DIRECT, BASE, save, sha256)
from experiments.gated_asl_classification.components import make_adapter, make_loss, parameter_count
from experiments.baseline_token_classification.common import load_backend, freeze_backend
from experiments.both_adapter_classification.native_backends import BothAlphaRAD, BothRadZero, BothCARZero
from experiments.both_adapter_classification.pooling_backends import wrap_backend


def validate(model):
    torch.set_num_threads(2)
    old = None
    if model not in ("AlphaRAD", "RadZero"):
        old = load_backend(model, "cpu")
        freeze_backend(old)
    evidence = []
    for dataset in ("openi", "chexpert", "chestxray14"):
        path = (DIRECT / "cache" / model.lower() / dataset / "cache_00.json"
                if old is None else BASE / "cache" / model / dataset / "cache.json")
        cache = json.loads(path.read_text())
        if old is None:
            head = json.loads(Path(cache["head_path"]).read_text())
            backend = (BothAlphaRAD if model == "AlphaRAD" else BothRadZero)(head)
            shard = cache["token_shards"][0]
            x = torch.from_numpy(np.array(np.load(shard["path"], mmap_mode="r")[:2], copy=True))
            location = {row: i for i, row in enumerate(cache["indices"])}
            expected = torch.from_numpy(np.array(np.load(cache["logits_path"], mmap_mode="r")[[location[i] for i in shard["indices"][:2]]], copy=True))
            native_score = lambda adapter: backend.head(x, token_adapter=adapter)
        else:
            old.prepare_classes(cache["class_names"])
            freeze_backend(old)
            backend = BothCARZero(old) if model == "CARZero" else wrap_backend(model, old)
            x = torch.from_numpy(np.array(np.load(cache["files"]["tokens.npy"]["path"], mmap_mode="r")[:2], copy=True))
            expected = torch.from_numpy(np.array(np.load(cache["files"]["aggregation_logits.npy"]["path"], mmap_mode="r")[:2], copy=True))
            native_score = lambda adapter: old.score_tokens(x, adapter)
        frozen = freeze_backend(backend)
        baseline = backend.score_tokens(x).detach()
        torch.testing.assert_close(baseline, expected, atol=3e-4, rtol=3e-4)
        target = torch.arange(baseline.numel()).reshape_as(baseline).remainder(2).float()
        initializer = None
        for method in METHODS:
            torch.manual_seed(42)
            adapter = make_adapter(backend.dim, 64, method)
            assert sum(p.numel() for p in adapter.parameters()) == parameter_count(backend.dim, 64, method)
            initial = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in adapter.state_dict().items() if not k.startswith("gate.")}
            if initializer is None:
                initializer = initial
            assert initial == initializer
            torch.testing.assert_close(backend.score_tokens(x, pre_adapter=adapter), baseline, atol=0, rtol=0)
            loss_fn = make_loss(method, torch.full((target.shape[1],), 3.), ASL_CONFIG)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
            gradients = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                output = backend.score_tokens(x, pre_adapter=adapter)
                loss = loss_fn(output, target)
                assert torch.isfinite(loss)
                loss.backward()
                norms = {}
                for name, parameter in adapter.named_parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (model, dataset, method, name)
                    norms[name] = float(parameter.grad.norm())
                assert norms["up.weight"] > 0
                if step:
                    assert norms["down.weight"] > 0
                    if method.startswith("gated_"):
                        assert norms["gate.weight"] > 0
                assert all(not p.requires_grad and p.grad is None for module in frozen for p in module.parameters())
                optimizer.step()
                gradients.append(norms)
            output = backend.score_tokens(x, pre_adapter=adapter).detach()
            original = native_score(adapter).detach()
            torch.testing.assert_close(output, original, atol=3e-4, rtol=3e-5)
            assert not torch.equal(output, baseline)
            evidence.append(dict(model=model, dataset=dataset, method=method,
                native_max_abs_error=float((output-original).abs().max()),
                identity_cache_max_abs_error=float((baseline-expected).abs().max()),
                initializer_sha256=initial, gradients=gradients, frozen_no_grad=True))
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    checks = validate(a.model)
    save(a.output, dict(status="passed", device="cpu", model=a.model,
        datasets=3, methods=4, checks=checks, test_source_sha256=sha256(__file__)))
    print(a.model, "CPU integration passed: 3 datasets x 4 methods")


if __name__ == "__main__":
    main()
