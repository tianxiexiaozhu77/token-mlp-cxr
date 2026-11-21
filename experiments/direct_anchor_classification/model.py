"""Direct-anchor and faithful, differentiable frozen classification readouts."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DirectAnchor(nn.Module):
    """Shared down / disease-specific up, without an intermediate activation.

    The bias is a feature vector, not a scalar added to classification logits.
    rank=0 is the paper's static disease-bias control.
    """
    def __init__(self, classes, dim, rank=8):
        super().__init__()
        self.classes, self.dim, self.rank = classes, dim, rank
        self.bias = nn.Parameter(torch.zeros(classes, dim))
        if rank:
            self.down = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
            self.up = nn.Parameter(torch.zeros(classes, rank, dim))

    def forward(self, context):
        if context.shape[-2:] != (self.classes, self.dim):
            raise ValueError("Anchor axes must be [..., class, feature]")
        residual = self.bias
        if self.rank:
            normalized = context.float() * torch.rsqrt(context.float().square().mean(-1, keepdim=True) + 1e-6)
            residual = residual + torch.einsum("...ck,ckd->...cd", normalized @ self.down, self.up)
        return context + residual


class FrozenClassificationHead(nn.Module):
    """Original scoring operations, with an explicit pre-normalization anchor.

    AlphaRAD concatenates its head contexts only for the adapter, then restores
    the original head axes, normalization, scale and aggregation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kind = config["model"]
        self.register_buffer("queries", torch.tensor(config["queries"], dtype=torch.float32))
        self.dim = self.queries.shape[-1]
        self.heads = config.get("heads", 1)
        if config.get("layer_norm_weight") is not None:
            self.register_buffer("ln_weight", torch.tensor(config["layer_norm_weight"]))
            self.register_buffer("ln_bias", torch.tensor(config["layer_norm_bias"]))
        else:
            self.ln_weight = self.ln_bias = None
        if config.get("head_weights") is not None:
            self.register_buffer("head_weights", torch.tensor(config["head_weights"]))
        else:
            self.head_weights = None

    def context(self, tokens, queries=None):
        q = self.queries if queries is None else queries
        if self.kind == "radzero":
            if self.ln_weight is not None:
                tokens = F.layer_norm(tokens, (self.dim,), self.ln_weight, self.ln_bias,
                                      self.config["layer_norm_eps"])
            if not self.config["include_cls"]:
                tokens = tokens[:, 1:]
            q = q.unsqueeze(0).expand(len(tokens), -1, -1)
            if self.config["sim_op"] == "cos":
                q = F.normalize(q, dim=-1)
                tokens = F.normalize(tokens, dim=-1)
                denominator = self.config["attention_temperature"]
            else:
                denominator = math.sqrt(self.dim)
            scores = torch.bmm(q, tokens.transpose(1, 2)) / denominator
            return torch.matmul(scores.softmax(-1), tokens)
        if not self.config["include_cls"]:
            tokens = tokens[:, 1:]
        values = F.normalize(tokens.reshape(*tokens.shape[:2], self.heads, -1), dim=-1)
        q = F.normalize(q.reshape(len(q), self.heads, -1), dim=-1)
        scores = torch.einsum("bkhd,qhd->bqkh", values, q) * self.config["scale"]
        context = torch.einsum("bqkh,bkhd->bqhd", scores.softmax(2), values)
        return context.flatten(-2)

    def score(self, context, adapter=None, queries=None):
        q = self.queries if queries is None else queries
        if adapter is not None:
            context = adapter(context)
        if self.kind == "radzero":
            q = F.normalize(q, dim=-1).unsqueeze(0).expand(len(context), -1, -1)
            h = F.normalize(context, dim=-1)
            return torch.matmul(q.unsqueeze(2), h.unsqueeze(-1)).squeeze(-1).squeeze(-1) / self.config["loss_temperature"]
        h = F.normalize(context.reshape(*context.shape[:2], self.heads, -1), dim=-1)
        q = F.normalize(q.reshape(len(q), self.heads, -1), dim=-1)
        scores = torch.einsum("bqhd,qhd->bqh", h, q) * self.config["scale"]
        index = self.config.get("head_index")
        if index is not None and index >= 0:
            return scores[..., index]
        if self.head_weights is not None:
            return (scores * self.head_weights).sum(-1)
        return scores.mean(-1)

    def forward(self, tokens, adapter=None, queries=None, token_adapter=None):
        if token_adapter is not None:
            tokens = token_adapter(tokens)
        return self.score(self.context(tokens, queries), adapter, queries)
