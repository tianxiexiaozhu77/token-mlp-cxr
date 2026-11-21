"""Real-checkpoint CPU/GPU validation for four added-aggregation backends.

Run with each model's own local_setup/python_local.sh. This checks native readout
parity, an independent per-prompt aggregation calculation, zero-MLP equivalence,
nonzero-MLP gradients and frozen encoder parameters. No training dataset is fit.
"""
from pathlib import Path
import argparse
import importlib.util
import json

import torch
import torch.nn.functional as F

BASELINES = Path(__file__).resolve().parents[2]
MEDICAL = BASELINES.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_score(backend, tokens, model):
    """Deliberately use loops and matmul rather than production einsum axes."""
    queries = backend.queries
    count = len(backend.class_names)
    member_tokens = tokens.reshape(tokens.shape[0], 10, 50, 512) if model == "CheXzero" else tokens[:, None]
    member_queries = queries if model == "CheXzero" else queries[None]
    outputs = []
    for i in range(member_tokens.shape[1]):
        features = member_tokens[:, i]
        scores = []
        for q in member_queries[i]:
            similarity = F.normalize(features, dim=-1) @ F.normalize(q, dim=-1)
            pooled = (similarity.softmax(-1).unsqueeze(-1) * features).sum(1)
            if model != "MaCo":
                pooled = F.normalize(pooled, dim=-1)
            scores.append(pooled @ q)
        scores = torch.stack(scores, dim=1)
        if model in ("CheXzero", "MaCo"):
            outputs.append(scores[:, :count] - scores[:, count:])
        else:
            outputs.append(scores[:, count:] - scores[:, :count])
    if model == "CheXzero":
        return torch.logit(torch.stack(outputs).sigmoid().mean(0), eps=1e-7)
    return outputs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["BiomedCLIP", "BioViL-T", "CheXzero", "MaCo"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image", action="append")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    images = args.image or [str(MEDICAL / "AlphaRAD/local_assets/data_prepared/author_default512/RSNA/397fe03a-f56e-4903-8132-8494d0e9f82c.png")]
    module = load(BASELINES / args.model / "local_setup/token_mlp_adapter.py", "tested_token_backend")
    backend = module.TokenBackend(args.device)
    backend.prepare_classes(["Pneumonia", "Pneumothorax", "Pleural_Thickening"])
    tokens = backend.encode_images(images)
    probabilities = backend.native_predictions(images)
    native = backend.native_scores_from_tokens(tokens)
    torch.testing.assert_close(native.sigmoid(), probabilities, rtol=1e-5, atol=2e-6)
    direct_native = backend.native_scores(images)
    torch.testing.assert_close(direct_native.sigmoid(), probabilities, rtol=1e-5, atol=2e-6)
    aggregation = backend.score_tokens(tokens)
    torch.testing.assert_close(aggregation, independent_score(backend, tokens, args.model), rtol=1e-5, atol=2e-5)
    adapter_module = load(MEDICAL / "RadZero/scripts/adapter_model.py", "tested_residual_adapter")
    adapter = adapter_module.ResidualAdapter(dim=backend.dim, bottleneck=64).to(backend.device)
    torch.testing.assert_close(aggregation, backend.score_tokens(tokens, adapter), rtol=0, atol=0)
    with torch.no_grad():
        adapter.up.weight.normal_(std=0.01)
        adapter.up.bias.normal_(std=0.01)
    adapted = backend.score_tokens(tokens, adapter)
    torch.testing.assert_close(adapted, independent_score(backend, adapter(tokens), args.model), rtol=1e-5, atol=2e-5)
    loss = F.binary_cross_entropy_with_logits(adapted, torch.tensor([[1., 0., 1.]], device=backend.device).expand(len(images), -1))
    loss.backward()
    gradient_norms = {}
    for name, parameter in adapter.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        norm = float(parameter.grad.norm())
        assert norm > 0, name
        gradient_norms[name] = norm
    models = backend.native.models if args.model == "CheXzero" else [backend.native.model]
    if args.model == "BioViL-T":
        models += [backend.native.text_model]
    for model in models:
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    report = {
        "status": "passed", "model": args.model, "device": args.device,
        "images": images, "token_shape": list(tokens.shape),
        "native_probability_max_abs_error": float((native.sigmoid() - probabilities).abs().max()),
        "native_logit_max_abs_error": float((native - direct_native).abs().max()),
        "aggregation_logits": aggregation.detach().cpu().tolist(),
        "native_logits": native.detach().cpu().tolist(),
        "zero_adapter_equals_aggregation_only": True,
        "nonzero_adapter_max_change": float((adapted - aggregation).abs().max()),
        "gradient_norms": gradient_norms, "backbone_frozen": True,
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "metadata": backend.metadata,
    }
    before_release = adapted.detach().clone()
    backend.release_image_encoder()
    torch.testing.assert_close(backend.score_tokens(tokens, adapter), before_release, rtol=0, atol=0)
    report["encoder_release_score_parity"] = True
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
