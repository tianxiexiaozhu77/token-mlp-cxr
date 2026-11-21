"""CPU audits of Both boundaries, using original backends as independent oracles.

Run with a model's python_local.sh and --model MODEL --real-cache. Without
--real-cache, BiomedCLIP/BioViL-T/MaCo use small synthetic cached-token fixtures;
KAD always uses its real frozen decoder. No training experiment is launched.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import (
    freeze_backend, load_backend, query_hashes, residual_mlp, sha256)
from experiments.both_adapter_classification.pooling_backends import wrap_backend


def independent_scores(model, backend, tokens, pre, post, pooling="text_guided"):
    """Original KAD model + head hook, or per-query matmul pooling oracle."""
    tokens = pre(tokens)
    if model == "KAD":
        handle = backend.model.mlp_head.register_forward_pre_hook(
            lambda module, inputs: (post(inputs[0]),))
        try:
            scores = backend.model(tokens, backend.query)
        finally:
            handle.remove()
        return scores[..., 1] - scores[..., 0]
    if pooling == "native_mean":
        pooled = post(tokens.mean(1))
        return pooled @ backend.positive.T - pooled @ backend.negative.T
    scores = []
    for i, query in enumerate(backend.queries):
        attention_query = backend.attention_queries[i] if model == "MaCo" else query
        weights = (F.normalize(tokens, dim=-1) @ attention_query).softmax(-1)
        context = post((tokens * weights.unsqueeze(-1)).sum(1))
        if model != "MaCo":
            context = F.normalize(context, dim=-1)
        scores.append(context @ query)
    scores = torch.stack(scores, dim=1)
    count = len(backend.class_names)
    return scores[:, :count] - scores[:, count:] if model == "MaCo" else scores[:, count:] - scores[:, :count]


def validate(model, backend, tokens, expected=None, pooling="text_guided"):
    wrapper = wrap_backend(model, backend, pooling=pooling)
    frozen = freeze_backend(backend)
    query_before = query_hashes(backend)
    torch.manual_seed(17)
    pre, post = residual_mlp(wrapper.dim, 64), residual_mlp(wrapper.dim, 64)
    assert not {id(p) for p in pre.parameters()}.intersection(id(p) for p in post.parameters())
    expected_parameters = {768: 198272, 512: 132224, 128: 33152}[wrapper.dim]
    assert sum(p.numel() for adapter in (pre, post) for p in adapter.parameters()) == expected_parameters
    native = backend.score_tokens(tokens, pooling=pooling) if model == "MaCo" else backend.score_tokens(tokens)
    initial = wrapper.score_tokens(tokens, pre, post)
    torch.testing.assert_close(initial, native, rtol=0, atol=0)
    cached_error = None
    if expected is not None:
        torch.testing.assert_close(initial, expected, rtol=3e-4, atol=3e-4)
        cached_error = float((initial - expected).detach().abs().max())
    parameters = list(pre.parameters()) + list(post.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    labels = (torch.arange(initial.numel()).reshape_as(initial) % 2).float()
    F.binary_cross_entropy_with_logits(initial, labels).backward()
    first_gradients = {}
    for name, adapter in (("pre", pre), ("post", post)):
        assert adapter.up.weight.grad is not None and adapter.up.weight.grad.norm() > 0
        assert adapter.up.bias.grad is not None and adapter.up.bias.grad.norm() > 0
        assert adapter.down.weight.grad.norm() == 0
        assert adapter.down.bias.grad.norm() == 0
        first_gradients[name] = {key: float(p.grad.norm()) for key, p in adapter.named_parameters()}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated = wrapper.score_tokens(tokens, pre, post)
    oracle = independent_scores(model, backend, tokens, pre, post, pooling)
    torch.testing.assert_close(updated, oracle, rtol=3e-5, atol=3e-4)
    # Removing only the post adapter exactly recovers the prior token MLP path.
    legacy_pre = (backend.score_tokens(tokens, pre, pooling=pooling) if model == "MaCo"
                  else backend.score_tokens(tokens, pre))
    new_pre = wrapper.score_tokens(tokens, pre_adapter=pre)
    torch.testing.assert_close(new_pre, legacy_pre, rtol=0, atol=0)
    if model in ("KAD", "MaCo"):
        legacy_post = (backend.score_tokens(tokens, post, adapter_position="post", pooling=pooling)
                       if model == "MaCo" else backend.score_tokens(tokens, post, adapter_position="post"))
        torch.testing.assert_close(wrapper.score_tokens(tokens, post_adapter=post), legacy_post, rtol=0, atol=0)
    F.binary_cross_entropy_with_logits(updated, labels).backward()
    second_gradients = {}
    for name, adapter in (("pre", pre), ("post", post)):
        second_gradients[name] = {}
        for key, p in adapter.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0, (name, key)
            second_gradients[name][key] = float(p.grad.norm())
    assert all(not module.training for module in frozen)
    assert all(not p.requires_grad and p.grad is None for module in frozen for p in module.parameters())
    assert query_hashes(backend) == query_before
    with torch.no_grad():
        contexts = wrapper.pooled_features(tokens)
    expected_context_shape = (len(tokens), wrapper.dim) if pooling == "native_mean" else (
        len(tokens), len(backend.class_names) * (1 if model == "KAD" else 2), wrapper.dim)
    assert tuple(contexts.shape) == expected_context_shape
    report = dict(status="passed", model=model, pooling=pooling, dim=wrapper.dim,
                  tokens_shape=list(tokens.shape), context_shape=list(contexts.shape),
                  independent_adapters=True, trainable_parameters=expected_parameters,
                  identity_max_abs_error=float((initial - native).detach().abs().max()),
                  cached_aggregation_max_abs_error=cached_error,
                  native_oracle_max_abs_error=float((updated - oracle).detach().abs().max()),
                  legacy_pre_max_abs_error=float((new_pre - legacy_pre).detach().abs().max()),
                  first_step_gradients=first_gradients, second_step_gradients=second_gradients,
                  frozen_parameters_no_grad=True, query_hashes_unchanged=True,
                  metadata=wrapper.metadata)
    return report, wrapper, pre, post


def synthetic_backend(model):
    path = ROOT / "baselines" / model / "local_setup/token_mlp_adapter.py"
    spec = importlib.util.spec_from_file_location("both_fixture_" + model.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.TokenBackend.__new__(module.TokenBackend)
    backend.class_names = ("a", "b", "c")
    backend.metadata = {}
    backend.device = torch.device("cpu")
    if model == "MaCo":
        backend.positive, backend.negative = torch.randn(2, 3, backend.dim)
        backend.queries = torch.cat((backend.positive, backend.negative))
        backend.attention_queries = F.normalize(backend.queries, dim=-1)
    else:
        backend.queries = F.normalize(torch.randn(6, backend.dim), dim=-1)
    return backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("KAD", "BiomedCLIP", "BioViL-T", "MaCo"))
    parser.add_argument("--real-cache", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "experiments/baseline_token_classification/runs/20261119/cache")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    reports = []
    if args.real_cache:
        import numpy as np
        backend = load_backend(args.model, "cpu")
        freeze_backend(backend)
        for dataset in ("openi", "chexpert", "chestxray14"):
            record_path = args.cache_root / args.model / dataset / "cache.json"
            record = json.loads(record_path.read_text())
            backend.prepare_classes(record["class_names"])
            freeze_backend(backend)
            # Historical caches were encoded on CUDA; CPU text matmuls need
            # not be bitwise identical. Record both hashes and validate the
            # resulting cached logits numerically below. Production CUDA fits
            # still require exact query hashes in the central trainer.
            cpu_queries = query_hashes(backend)
            array = np.load(record["files"]["tokens.npy"]["path"], mmap_mode="r")
            indices = [0, len(array) // 2, len(array) - 1]
            tokens = torch.from_numpy(np.array(array[indices], copy=True))
            expected = torch.from_numpy(np.array(np.load(
                record["files"]["aggregation_logits.npy"]["path"], mmap_mode="r")[indices], copy=True))
            report, wrapper, pre, post = validate(args.model, backend, tokens, expected=expected)
            report.update(dataset=dataset, cache_record_sha256=sha256(record_path), indices=indices,
                          cpu_query_sha256=cpu_queries, cached_cuda_query_sha256=record["query_sha256"],
                          cpu_cuda_query_bitwise_equal=cpu_queries == record["query_sha256"])
            reports.append(report)
            print(f"{args.model}/{dataset}: identity, legacy pre, independent Both oracle, two-step gradients passed", flush=True)
        # Releasing encoders must not release the live aggregation/readout.
        before = wrapper.score_tokens(tokens, pre, post).detach()
        backend.release_image_encoder()
        torch.testing.assert_close(wrapper.score_tokens(tokens, pre, post), before, rtol=0, atol=0)
        reports[-1]["encoder_release_score_parity"] = True
    elif args.model == "KAD":
        backend = load_backend(args.model, "cpu")
        backend.prepare_classes(["Atelectasis", "Pneumothorax", "Pneumonia"])
        tokens = torch.randn(2, 13, backend.dim)
        reports.append(validate(args.model, backend, tokens)[0])
    else:
        backend = synthetic_backend(args.model)
        tokens = torch.randn(2, 13, backend.dim)
        reports.append(validate(args.model, backend, tokens)[0])
        if args.model == "MaCo":
            reports.append(validate(args.model, backend, tokens, pooling="native_mean")[0])
    output = dict(status="passed", model=args.model, real_cached_tokens=args.real_cache,
                  reports=reports, sources_sha256={str(p): sha256(p) for p in (
                      Path(__file__), Path(__file__).with_name("pooling_backends.py"),
                      ROOT / "baselines" / args.model / "local_setup/token_mlp_adapter.py")})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "passed", "model": args.model,
                      "conditions": len(reports), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
