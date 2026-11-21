"""Joint pre/post MLP boundaries for three native disease-guided backbones.

These wrappers contain only frozen operations. Their ``score_tokens`` methods
deliberately have no ``no_grad`` decorator: the frozen aggregation must pass
the classification gradient back to the caller's pre-aggregation adapter.
"""
from pathlib import Path
import sys

from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.direct_anchor_classification.model import FrozenClassificationHead


class _BothSimilarityHead(nn.Module):
    dim = 768
    kind = None

    def __init__(self, head_config):
        super().__init__()
        if head_config.get("model") != self.kind:
            raise ValueError(f"Expected {self.kind} frozen head configuration")
        self.head = FrozenClassificationHead(head_config).requires_grad_(False).eval()
        if self.head.dim != self.dim:
            raise ValueError(f"Expected {self.dim}-dimensional projected tokens")
        self.class_names = tuple(head_config["class_names"]) if "class_names" in head_config else None
        self.metadata = dict(model=self.kind, dim=self.dim, adapter_method="both",
            adapter_insertion="independent pre- and post-aggregation shared residual MLPs jointly optimized",
            trainable="only the caller's independent pre/post residual MLPs",
            **BOUNDARIES[self.kind])

    def train(self, mode=True):
        super().train(mode)
        self.head.eval()
        return self

    def score_tokens(self, tokens, pre_adapter=None, post_adapter=None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim or tokens.shape[1] < 2:
            raise ValueError(f"Expected B x (CLS+patches) x {self.dim} projected tokens")
        if pre_adapter is not None:
            tokens = pre_adapter(tokens)
        context = self.head.context(tokens)
        return self.head.score(context, adapter=post_adapter)

    forward = score_tokens


class BothAlphaRAD(_BothSimilarityHead):
    kind = "alpharad"

    def __init__(self, head_config):
        super().__init__(head_config)
        if self.head.heads != 8:
            raise ValueError("AlphaRAD requires eight frozen 96D FLaS heads")


class BothRadZero(_BothSimilarityHead):
    kind = "radzero"


class BothCARZero(nn.Module):
    """Both native DQN branches, with one shared 768D post-aggregation MLP.

    The pre MLP changes visual memory in i2t and the visual global query in
    t2i, exactly as the existing token adapter. The independent post MLP is
    applied to each branch's 768D decoder feature after its final LayerNorm
    and eval dropout, before the original shared nonlinear classifier.
    That one post MLP shares parameters across branches as well as diseases.
    The original arithmetic mean of the two classifier logits is retained.
    No 1536D concatenation/mixing or replacement readout is introduced.

    ``token_backend`` is an already prepared CARZero TokenBackend. Keeping
    creation/query preparation external lets the runner audit the existing
    source/checkpoint/query hashes before constructing this wrapper.
    """
    dim = 768

    def __init__(self, token_backend):
        super().__init__()
        if token_backend.dim != self.dim or token_backend.class_names is None:
            raise ValueError("Expected a prepared 768D CARZero TokenBackend")
        self.class_names = tuple(token_backend.class_names)
        self.fusion = token_backend.model.fusion_module.requires_grad_(False).eval()
        self.register_buffer("query", token_backend.query.detach().clone())
        self.register_buffer("text_memory", token_backend.text_memory.detach().clone())
        if self.query.shape != (len(self.class_names), self.dim):
            raise ValueError("CARZero disease query shape does not match class order")
        if self.text_memory.ndim != 3 or self.text_memory.shape[0] != len(self.class_names) or self.text_memory.shape[-1] != self.dim:
            raise ValueError("CARZero frozen text memory shape is invalid")
        self.metadata = dict(token_backend.metadata)
        self.metadata.update(BOUNDARIES["CARZero"])
        self.metadata.update(adapter_method="both",
            adapter_insertion="independent pre-token and post-decoder shared residual MLPs jointly optimized",
            trainable="only the caller's independent pre/post residual MLPs")

    def train(self, mode=True):
        super().train(mode)
        self.fusion.eval()
        return self

    def _features(self, memory, queries):
        batch = memory.shape[0]
        memory = self.fusion.decoder_norm(memory.transpose(0, 1))
        queries = self.fusion.decoder_norm(queries.unsqueeze(1).repeat(1, batch, 1))
        features, _ = self.fusion.decoder(queries, memory,
            memory_key_padding_mask=None, pos=None, query_pos=None)
        return self.fusion.dropout_feas(features).transpose(0, 1)

    def score_tokens(self, tokens, pre_adapter=None, post_adapter=None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim or tokens.shape[1] < 2:
            raise ValueError("Expected B x (global+patches) x 768 projected tokens")
        if self.fusion.training:
            raise RuntimeError("Frozen CARZero fusion and classifier must remain in eval mode")
        adapted = tokens if pre_adapter is None else pre_adapter(tokens)
        i2t = self._features(adapted, self.query)
        # Native reverse direction treats text classes as its batch. The
        # decoder has no query self-attention, so image queries are independent.
        t2i = self._features(self.text_memory, adapted[:, 0]).transpose(0, 1)
        if post_adapter is not None:
            i2t, t2i = post_adapter(i2t), post_adapter(t2i)
        i2t = self.fusion.mlp_head(i2t).squeeze(-1)
        t2i = self.fusion.mlp_head(t2i).squeeze(-1)
        return (i2t + t2i) / 2

    forward = score_tokens


BOUNDARIES = {
    "alpharad": {
        "pre_boundary": "raw projected CLS/patch tokens before FLaS head split and token L2",
        "post_boundary": "concatenated eight FLaS contexts (768D) before native per-head context L2 and cosine readout",
        "post_sharing": "one shared 768D MLP across diseases",
        "aggregation_origin": "native FLaS",
    },
    "radzero": {
        "pre_boundary": "aligned CLS/patch tokens before native LayerNorm, CLS selection and VL-CABS",
        "post_boundary": "768D VL-CABS context before native final L2, cosine similarity and loss temperature",
        "post_sharing": "one shared 768D MLP across diseases",
        "aggregation_origin": "native VL-CABS",
    },
    "CARZero": {
        "pre_boundary": "projected global/patch tokens before native bidirectional DQN; adapts i2t visual memory and t2i global query",
        "post_boundary": "each 768D i2t/t2i decoder feature after final LayerNorm and eval dropout, before original frozen nonlinear MLP classifier",
        "post_sharing": "one shared 768D MLP across both directions and all diseases; no 1536D branch concatenation",
        "aggregation_origin": "native bidirectional DQN",
        "score_transform": "unchanged arithmetic mean of frozen per-branch classifier logits",
    },
}
