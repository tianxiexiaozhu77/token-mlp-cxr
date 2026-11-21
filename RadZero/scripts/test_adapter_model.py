"""CPU checks against the released RadZero loss implementation and weights.

Run: bash scripts/python_local.sh scripts/test_adapter_model.py
The image encoder is intentionally omitted; these tests exercise the original
matching code, frozen-parameter gradients, and the live/cached adapter paths.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from adapter_model import ResidualAdapter, attach_adapter, encode_prompts, match_tokens


CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints/RadZero"
SPEC = importlib.util.spec_from_file_location(
    "released_radzero_losses", CHECKPOINT / "losses.py"
)
RELEASED_LOSSES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASED_LOSSES)
LOSS_CONFIG = json.loads((CHECKPOINT / "config.json").read_text())["kwargs"]["loss"][
    "KeyPhraseAlignmentLoss"
]
with safe_open(CHECKPOINT / "model.safetensors", framework="pt", device="cpu") as stream:
    LOSS_PREFIX = "loss_fns.KeyPhraseAlignmentLoss."
    LOSS_STATE = {
        key.removeprefix(LOSS_PREFIX): stream.get_tensor(key)
        for key in stream.keys()
        if key.startswith(LOSS_PREFIX)
    }


class SyntheticBase(nn.Module):
    """Tiny encoders around the checkpoint's unmodified matching module."""

    def __init__(self, classes=4, include_cls=True, projected_text=False):
        super().__init__()
        self.align_transformer = nn.Linear(768, 768)
        width = 1536 if projected_text else 768
        self.text_model = nn.Embedding(classes, width)
        loss = RELEASED_LOSSES.KeyPhraseAlignmentLoss(**LOSS_CONFIG)
        loss.load_state_dict(LOSS_STATE, strict=True)
        loss.use_vision_cls_token = include_cls
        self.loss_fns = nn.ModuleDict({"KeyPhraseAlignmentLoss": loss})

    def forward_text_model(self, encoded):
        features = self.text_model(encoded["input_ids"][:, 0])
        return {
            "text_features_wo_l2_norm": features,
            "text_features": F.normalize(features, dim=-1),
        }


def tokenize(prompts, **kwargs):
    return {
        "input_ids": torch.arange(len(prompts)).reshape(-1, 1),
        "attention_mask": torch.ones(len(prompts), 1, dtype=torch.long),
    }


def released_match(base, tokens, encoded):
    """Execute the released loss.forward, mirroring model.compute_logits."""
    phrases = [
        {key: value[i:i + 1] for key, value in encoded.items()}
        for i in range(encoded["input_ids"].shape[0])
    ]
    loss = base.loss_fns["KeyPhraseAlignmentLoss"]
    output = loss(
        phrases, tokens, base.forward_text_model,
        ddp_gather=False, need_attn_weights=True, compute_loss=False,
    )
    logits = output["t2i_logits"].T / loss.loss_temperature.exp()
    similarity = torch.stack(output["t2i_attn_weights"]).mean(dim=0)
    if loss.use_vision_cls_token:
        similarity = similarity[:, :, 1:]
    return {
        "logits": logits.reshape(tokens.shape[0], len(phrases)),
        "similarity_scores": similarity,
    }


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def test_zero_initialization_preserves_tokens_and_parameter_count(self):
        adapter = ResidualAdapter()
        tokens = torch.randn(2, 13, 768)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 99136)
        self.assertTrue(torch.equal(adapter(tokens), tokens))
        self.assertGreater(adapter.down.weight.abs().sum().item(), 0)
        self.assertEqual(adapter.up.weight.count_nonzero().item(), 0)
        self.assertEqual(adapter.up.bias.count_nonzero().item(), 0)

    def test_cached_matching_equals_released_loss_with_singleton_axes(self):
        for batch in (1, 3):
            for classes in (1, 4):
                for include_cls in (False, True):
                    with self.subTest(batch=batch, classes=classes, cls=include_cls):
                        base = SyntheticBase(classes, include_cls).requires_grad_(False).eval()
                        tokens = torch.randn(batch, 9, 768)
                        text, encoded = encode_prompts(
                            base, tokenize, [f"prompt {i}" for i in range(classes)]
                        )
                        expected = released_match(base, tokens, encoded)
                        actual = match_tokens(
                            base, tokens, text, ResidualAdapter(), need_maps=True
                        )
                        self.assertEqual(actual["logits"].shape, (batch, classes))
                        self.assertEqual(
                            actual["similarity_scores"].shape, (batch, classes, 8)
                        )
                        for key in expected:
                            torch.testing.assert_close(
                                actual[key], expected[key], rtol=0, atol=0
                            )

    def test_text_projection_selection_matches_released_loss(self):
        base = SyntheticBase(projected_text=True).requires_grad_(False).eval()
        text, encoded = encode_prompts(base, tokenize, ["a", "b", "c", "d"])
        self.assertEqual(text.shape, (4, 768))
        self.assertFalse(text.requires_grad)
        self.assertFalse(torch.is_inference(text))
        tokens = torch.randn(2, 9, 768)
        expected = released_match(base, tokens, encoded)
        actual = match_tokens(base, tokens, text, need_maps=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_live_hook_equals_cached_path_before_and_after_training(self):
        base = SyntheticBase().eval()
        inputs = torch.randn(2, 9, 768)
        with torch.no_grad():
            cached = base.align_transformer(inputs).clone()
        original_parameters = {
            name: parameter.detach().clone() for name, parameter in base.named_parameters()
        }
        adapter = attach_adapter(base)
        self.assertFalse(base.training)
        for name, parameter in base.named_parameters():
            self.assertEqual(
                parameter.requires_grad, name.startswith("residual_adapter."),
                msg=name,
            )
        text, encoded = encode_prompts(base, tokenize, ["a", "b", "c", "d"])
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        target = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
        first_up_grad = None
        second_down_grad = None
        for step in range(2):
            live_tokens = base.align_transformer(inputs)
            actual = match_tokens(base, cached, text, adapter, need_maps=True)
            live = released_match(base, live_tokens, encoded)
            for key in live:
                torch.testing.assert_close(actual[key], live[key], rtol=0, atol=0)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(actual["logits"], target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(adapter.up.weight.grad).all())
            if step == 0:
                first_up_grad = adapter.up.weight.grad.abs().sum().item()
                self.assertEqual(adapter.down.weight.grad.count_nonzero().item(), 0)
            else:
                second_down_grad = adapter.down.weight.grad.abs().sum().item()
            optimizer.step()
        self.assertGreater(first_up_grad, 0)
        self.assertGreater(second_down_grad, 0)
        self.assertGreater(adapter.up.weight.abs().sum().item(), 0)
        self.assertFalse(torch.equal(adapter(cached), cached))
        for name, before in original_parameters.items():
            parameter = dict(base.named_parameters())[name]
            self.assertIsNone(parameter.grad, msg=name)
            self.assertTrue(torch.equal(parameter, before), msg=name)

    def test_adapter_checkpoint_roundtrip_and_double_attach_rejected(self):
        base = SyntheticBase()
        initial_base_state = {key: value.clone() for key, value in base.state_dict().items()}
        adapter = attach_adapter(base)
        with torch.no_grad():
            adapter.up.weight.normal_(std=0.01)
            adapter.up.bias.normal_(std=0.01)
        payload = {
            "adapter_config": {"dim": 768, "bottleneck": 64},
            "adapter_state_dict": adapter.state_dict(),
        }
        inputs = torch.randn(2, 5, 768)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(payload, path)
            restored_base = SyntheticBase()
            restored_base.load_state_dict(initial_base_state)
            restored = attach_adapter(restored_base, path)
            self.assertIsInstance(restored, ResidualAdapter)
            torch.testing.assert_close(
                restored_base.align_transformer(inputs),
                base.align_transformer(inputs), rtol=0, atol=0,
            )
        with self.assertRaisesRegex(ValueError, "already attached"):
            attach_adapter(base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
