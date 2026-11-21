"""Small residual adaptation of the published RadZero token stream.

All published weights, layer norms, and temperatures remain frozen. The adapter
is applied after the original visual aligner and before the original VL-CABS.
"""
from pathlib import Path

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck=64):
        super().__init__()
        self.dim, self.bottleneck = dim, bottleneck
        self.down = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        return tokens + self.up(self.activation(self.down(tokens)))


def unwrap(model):
    return model.inner if hasattr(model, "inner") else model


def attach_adapter(model, checkpoint_path=None):
    base = unwrap(model)
    if hasattr(base, "residual_adapter"):
        raise ValueError("An adapter is already attached")
    base.requires_grad_(False)
    base.eval()
    payload = None
    if checkpoint_path is not None:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    config = payload["adapter_config"] if payload else {"dim": 768, "bottleneck": 64}
    adapter = ResidualAdapter(**config).to(next(base.parameters()).device)
    if payload:
        adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    base.add_module("residual_adapter", adapter)
    # The hook preserves the original graph and all original output branches.
    base._residual_adapter_hook = base.align_transformer.register_forward_hook(
        lambda module, inputs, output: adapter(output)
    )
    return adapter


@torch.no_grad()
def encode_prompts(base, tokenizer, prompts):
    """Use exactly the checkpoint's text projection and normalization."""
    base = unwrap(base)
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    tokens = {key: value.to(next(base.parameters()).device) for key, value in tokens.items()}
    phrases = [{key: value[i:i + 1] for key, value in tokens.items()} for i in range(len(prompts))]
    loss = base.loss_fns["KeyPhraseAlignmentLoss"]
    features, _ = loss.compute_text_features(phrases, base.forward_text_model, ddp_gather=False)
    return features.detach(), tokens


def match_tokens(base, tokens, text_features, adapter=None, need_maps=False):
    """Differentiable original VL-CABS, accepting cached aligned image tokens."""
    base = unwrap(base)
    loss = base.loss_fns["KeyPhraseAlignmentLoss"]
    batch, classes = tokens.shape[0], text_features.shape[0]
    if adapter is not None:
        tokens = adapter(tokens)
    if loss.layer_norm is not None:
        tokens = loss.layer_norm(tokens)
    if not loss.use_vision_cls_token:
        tokens = tokens[:, 1:]
    transposed_logits, scores = loss.compute_t2i_logits(text_features, tokens, need_maps)
    logits = transposed_logits.T.reshape(batch, classes) / loss.loss_temperature.exp()
    result = {"logits": logits}
    if need_maps:
        similarity = torch.stack(scores).mean(dim=0)
        if loss.use_vision_cls_token:
            similarity = similarity[:, :, 1:]
        result["similarity_scores"] = similarity
    return result
