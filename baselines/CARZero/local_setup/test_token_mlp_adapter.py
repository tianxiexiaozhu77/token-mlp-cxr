"""CPU audit against the released single-image bidirectional scoring path."""
from pathlib import Path
import json

import torch
from torch import nn
from token_mlp_adapter import TokenBackend


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    reference = json.loads(Path(__file__).with_name("validation.json").read_text())
    backend = TokenBackend("cpu")
    backend.prepare_classes(reference["classes"])
    tokens = backend.encode_images(reference["images"])
    native = backend.native_scores(reference["images"])
    direct = backend.score_tokens(tokens)
    torch.testing.assert_close(direct, native, rtol=1e-5, atol=1e-5)

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
    torch.testing.assert_close(zero, native, rtol=1e-5, atol=1e-5)
    nn.init.normal_(adapter.up.weight, std=.001)
    nn.init.normal_(adapter.up.bias, std=.001)
    adapted = adapter(tokens)
    predicted = backend.score_tokens(tokens, adapter)
    # Independent released single-image scoring path at a nonzero adapter.
    manual = []
    for i in range(tokens.shape[0]):
        i2t, _ = backend.model.fusion_module(adapted[i:i+1], backend.query, return_atten=True)
        t2i, _ = backend.model.fusion_module(backend.text_memory, adapted[i:i+1, 0], return_atten=True)
        manual.append((i2t.squeeze(-1) + t2i.squeeze(-1).transpose(0, 1)) / 2)
    manual = torch.cat(manual)
    torch.testing.assert_close(predicted, manual, rtol=1e-5, atol=1e-5)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, torch.eye(2))
    loss.backward()
    norms = {name: float(param.grad.norm()) for name, param in adapter.named_parameters()}
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in adapter.parameters())
    assert norms["down.weight"] > 0 and norms["up.weight"] > 0
    assert all(not p.requires_grad and p.grad is None for p in backend.model.parameters())
    assert not backend.model.training
    report = {"status": "passed", "device": "cpu", "token_shape": list(tokens.shape),
              "native_max_abs_error": float((direct-native).abs().max()),
              "zero_init_max_abs_error": float((zero-native).abs().max()),
              "nonzero_native_max_abs_error": float((predicted-manual).abs().max()),
              "nonzero_score_change": float((predicted-direct).abs().max()),
              "gradient_norms": norms, "frozen_backbone": True, "metadata": backend.metadata}
    Path(__file__).with_name("token_mlp_cpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
