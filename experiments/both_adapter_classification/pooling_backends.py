"""Differentiable pre + post MLP boundaries for four frozen token backends.

This module does not alter existing backends or their recorded experiments.
Each adapter is one shared residual MLP, with independent pre/post parameters.
In particular, ``KAD.encode_contexts`` cannot be reused: its no_grad decorator
would prevent the classification loss from training the pre adapter.
"""
from pathlib import Path

import torch
import torch.nn.functional as F


SUPPORTED_MODELS = ("KAD", "BiomedCLIP", "BioViL-T", "MaCo")


class BothPoolingBackend:
    """Wrap an already prepared/frozen backend, including after encoder release.

    The final L2 readout remains *after* the post adapter in BiomedCLIP and
    BioViL-T. MaCo retains its raw contexts and raw-dot margin. KAD retains its
    final native decoder LayerNorm/dropout before the post adapter and its
    original two-class linear head after it. No frozen contexts are cached in
    a Both fit: contexts depend on the currently trainable pre adapter.
    """

    def __init__(self, model, backend, pooling="text_guided"):
        if model not in SUPPORTED_MODELS:
            raise ValueError("Unsupported Both pooling backend: " + str(model))
        if pooling not in ("text_guided", "native_mean"):
            raise ValueError("Unknown pooling: " + str(pooling))
        if pooling == "native_mean" and model != "MaCo":
            raise ValueError("native_mean is supported only for MaCo")
        self.name = model
        self.backend = backend
        self.dim = backend.dim
        self.pooling = pooling
        if backend.class_names is None:
            raise RuntimeError("prepare_classes must precede wrapping the backend")
        self.class_names = tuple(backend.class_names)
        self.metadata = dict(backend.metadata)
        if model == "KAD":
            pre_boundary = "raw 768D projected spatial tokens before native TQN decoder_norm"
            post_boundary = "768D disease contexts after native decoder final LayerNorm and eval dropout, before original two-class head"
            aggregation = "unchanged frozen native TQN decoder; differentiable with respect to adapted image tokens"
        elif model in ("BiomedCLIP", "BioViL-T"):
            pre_boundary = "raw projected CLS/patch tokens before existing fixed text-guided pooling" if model == "BiomedCLIP" else "raw projected patch tokens before existing fixed text-guided pooling"
            post_boundary = "raw weighted contexts before original final L2 normalization and cosine positive-minus-negative readout"
            aggregation = "unchanged fixed temperature-1 cosine softmax; negative then positive query contexts"
        else:
            pre_boundary = "raw img_mlp-projected patch tokens before selected pooling"
            post_boundary = "raw pooled contexts before unchanged raw-dot positive-minus-negative readout; no final L2"
            aggregation = ("unchanged fixed temperature-1 cosine softmax; positive then negative query contexts"
                           if pooling == "text_guided" else "native arithmetic mean of projected patches")
        self.metadata.update(
            adapter_position="both", adapter_positions=["pre", "post"],
            pre_adapter_boundary=pre_boundary, post_adapter_boundary=post_boundary,
            adapter_sharing="two independent shared residual MLPs; each shared over all tokens/contexts/diseases/prompt polarities",
            adapter_optimization="joint classification loss; frozen aggregation parameters retain input gradients",
            both_aggregation=aggregation, pooling=pooling,
            post_dim=self.dim, frozen_context_cache_allowed=False,
            both_source_path=str(Path(__file__).resolve()),
        )

    def _check_tokens(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"Expected B x N x {self.dim} projected tokens, got {tuple(tokens.shape)}")
        if tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("Expected nonempty projected tokens")

    def pooled_features(self, tokens):
        """Keep autograd connected through frozen aggregation to the pre MLP."""
        self._check_tokens(tokens)
        backend = self.backend
        if self.name == "KAD":
            if backend.model.training:
                raise RuntimeError("Frozen KAD decoder must remain in eval mode")
            memory = backend.model.decoder_norm(tokens.transpose(0, 1))
            queries = backend.model.decoder_norm(
                backend.query.unsqueeze(1).repeat(1, tokens.shape[0], 1))
            features, _ = backend.model.decoder(
                queries, memory, memory_key_padding_mask=None, pos=None, query_pos=None)
            return backend.model.dropout_feas(features).transpose(0, 1)
        if self.name == "MaCo":
            return backend.pooled_features(tokens, pooling=self.pooling)
        # Deliberately preserve the existing prepared query tensor, including
        # its normalization, rather than re-normalizing it or changing scale.
        weights = torch.einsum("bnd,qd->bqn", F.normalize(tokens, dim=-1), backend.queries).softmax(-1)
        return torch.einsum("bqn,bnd->bqd", weights, tokens)

    def score_features(self, features, post_adapter=None):
        backend = self.backend
        if post_adapter is not None:
            features = post_adapter(features)
        if self.name == "KAD":
            pair_logits = backend.model.mlp_head(features)
            return pair_logits[..., 1] - pair_logits[..., 0]
        if self.name == "MaCo":
            return backend.score_features(features, pooling=self.pooling)
        scores = torch.einsum("bqd,qd->bq", F.normalize(features, dim=-1), backend.queries)
        count = len(self.class_names)
        return scores[:, count:] - scores[:, :count]

    def score_tokens(self, tokens, pre_adapter=None, post_adapter=None):
        self._check_tokens(tokens)
        if pre_adapter is not None:
            tokens = pre_adapter(tokens)
        return self.score_features(self.pooled_features(tokens), post_adapter)


def wrap_backend(model, old_backend, pooling="text_guided"):
    """Wrap an existing prepared backend; no checkpoint is loaded or changed."""
    return BothPoolingBackend(model, old_backend, pooling=pooling)


def create_backend(model, device="cuda", class_names=None, pooling="text_guided"):
    """Convenience constructor; training can instead wrap its verified backend."""
    from experiments.baseline_token_classification.common import load_backend, freeze_backend
    backend = load_backend(model, device)
    freeze_backend(backend)
    if class_names is None:
        raise ValueError("class_names are required; otherwise use load_backend then wrap_backend")
    backend.prepare_classes(class_names)
    freeze_backend(backend)
    return wrap_backend(model, backend, pooling=pooling)
