"""CPU native-probability, equivalent-logit and adapter-gradient audit."""
from pathlib import Path
import json

import torch
from torch import nn
from token_mlp_adapter import TokenBackend


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    medical = Path(__file__).resolve().parents[3]
    paths = [str(medical / "data/alpharad/ChestXDet10/test_data" / name)
             for name in ("36199.png", "36302.png")]
    backend = TokenBackend("cpu")
    backend.prepare_classes(["Atelectasis", "Pneumothorax"])
    tokens = backend.encode_images(paths)
    native = backend.native_scores(paths)
    probabilities = backend.native_predictions(paths)
    direct = backend.score_tokens(tokens)
    torch.testing.assert_close(direct, native, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(direct.sigmoid(), probabilities, rtol=1e-5, atol=1e-7)

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(backend.dim, 64)
            self.up = nn.Linear(64, backend.dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        def forward(self, x):
            return x + self.up(torch.nn.functional.gelu(self.down(x)))

    adapter = Residual()
    zero = backend.score_tokens(tokens, adapter)
    torch.testing.assert_close(zero, native, rtol=1e-6, atol=1e-6)
    nn.init.normal_(adapter.up.weight, std=.001)
    nn.init.normal_(adapter.up.bias, std=.001)
    predicted = backend.score_tokens(tokens, adapter)
    adapted = adapter(tokens).detach()
    # Inject adapted tokens into the original prediction method, leaving its
    # native softmax and query path intact for an independent score check.
    encoder = backend.native.image_encoder
    old_forward = encoder.forward
    encoder.forward = lambda pixels: (adapted, adapted.mean(1))
    try:
        manual_probabilities = backend.native_predictions(paths)
    finally:
        del encoder.forward
    torch.testing.assert_close(predicted.sigmoid(), manual_probabilities, rtol=1e-5, atol=1e-7)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, torch.eye(2))
    loss.backward()
    norms = {name: float(param.grad.norm()) for name, param in adapter.named_parameters()}
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in adapter.parameters())
    assert norms["down.weight"] > 0 and norms["up.weight"] > 0
    for model in (backend.native.image_encoder, backend.native.text_encoder, backend.model):
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert not model.training
    report = {"status": "passed", "device": "cpu", "token_shape": list(tokens.shape),
              "native_logit_max_abs_error": float((direct-native).abs().max()),
              "native_probability_max_abs_error": float((direct.sigmoid()-probabilities).abs().max()),
              "zero_init_max_abs_error": float((zero-native).abs().max()),
              "nonzero_native_probability_max_abs_error": float((predicted.sigmoid()-manual_probabilities).abs().max()),
              "nonzero_score_change": float((predicted-direct).abs().max()),
              "gradient_norms": norms, "frozen_backbone": True, "metadata": backend.metadata}
    Path(__file__).with_name("token_mlp_cpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
