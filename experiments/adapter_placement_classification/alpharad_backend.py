"""AlphaRAD adapter-position comparison, using its audited frozen FLaS head.

Public API: AlphaRADPlacementBackend(head_config), ResidualAdapter(dim, bottleneck).
All scoring methods remain differentiable; only the residual MLP is trainable.
"""
from pathlib import Path
import sys

from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AlphaRAD.local_setup.token_adapter_model import ResidualTokenAdapter
from experiments.direct_anchor_classification.model import FrozenClassificationHead


class AlphaRADPlacementBackend(nn.Module):
    """Frozen FLaS with parameter-matched pre/post aggregation MLP boundaries.

    ``pre`` adapts the projected raw CLS/patch tokens, before the eight-head
    split and token normalization. ``post`` first runs unchanged FLaS and
    adapts its concatenated eight head contexts, before the final per-head
    L2 normalization. Both boundaries have width 768. The same residual MLP
    definition is shared across tokens (pre) or diseases (post); no disease-
    specific parameters or additional learnable normalization are introduced.

    Cached ``contexts_*.npy`` from direct_anchor_classification/cache.py are
    exactly the input to ``forward_context``. They must be paired with their
    matching head configuration. No no-grad decorator is used: gradients
    through the frozen FLaS computations must reach a pre-aggregation MLP.
    """

    def __init__(self, head_config):
        super().__init__()
        if head_config.get("model") != "alpharad":
            raise ValueError("Expected an AlphaRAD frozen head configuration")
        self.head = FrozenClassificationHead(head_config).requires_grad_(False).eval()
        self.dim, self.heads = self.head.dim, self.head.heads
        if self.dim != 768 or self.heads != 8:
            raise ValueError("This AlphaRAD comparison requires eight 96D heads (768D total)")

    def train(self, mode=True):
        super().train(mode)
        self.head.eval()
        return self

    def context(self, tokens):
        """Return unchanged pre-final-L2 FLaS contexts, shape [B, C, 768]."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError("Projected visual tokens must have shape [B, T, 768]")
        return self.head.context(tokens)

    def score(self, context, adapter=None):
        """Apply optional shared MLP, then native per-head L2/cosine/readout."""
        if context.ndim != 3 or context.shape[1:] != self.head.queries.shape:
            raise ValueError("FLaS contexts must have shape [B, C, 768] in frozen query order")
        return self.head.score(context, adapter=adapter)

    def forward_context(self, context, adapter=None):
        """Post-aggregation scoring directly from audited frozen contexts."""
        return self.score(context, adapter)

    def forward(self, tokens, adapter=None, placement="pre"):
        if placement not in ("pre", "post", "none"):
            raise ValueError(f"Unknown adapter placement: {placement}")
        if placement == "none" and adapter is not None:
            raise ValueError("placement='none' cannot receive an adapter")
        if placement == "pre" and adapter is not None:
            tokens = adapter(tokens)
        context = self.context(tokens)
        return self.score(context, adapter if placement == "post" else None)


ResidualAdapter = ResidualTokenAdapter

PLACEMENT_DESCRIPTION = {
    "pre": "Projected CLS/patch tokens, before FLaS head split and token L2",
    "post": "Concatenated frozen FLaS contexts, before final per-head context L2",
    "none": "Unchanged frozen FLaS aggregation and readout",
}
