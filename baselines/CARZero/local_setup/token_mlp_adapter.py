"""Differentiable token-adapter boundary for frozen, native CARZero scoring.

The supplied adapter acts on every projected global/patch image token before
the original bidirectional DQN fusion. It changes visual memory in the i2t
branch and the visual global query in the t2i branch. No FLaS layer is added.
"""
from pathlib import Path
import importlib.util

import torch


def _load_native():
    path = Path(__file__).with_name("adapter.py")
    spec = importlib.util.spec_from_file_location("carzero_token_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CARZeroAdapter


class TokenBackend:
    dim = 768

    def __init__(self, device="cuda"):
        self.native = _load_native()(device=device)
        self.device = self.native.device
        self.model = self.native.model
        self.class_names = None
        self.metadata = dict(self.native.metadata)
        self.metadata.update({
            "adapter_insertion": "projected global+patch tokens before native bidirectional DQN fusion and its LayerNorm",
            "token_boundary": "image_encoder_forward projected global/local output; before fusion_module.decoder_norm",
            "aggregation_origin": "native",
            "dim": self.dim,
            "token_layout": "[image global token, spatial patch tokens], B x N x 768",
            "aggregation": "native bidirectional DQN cross-attention; not AlphaRAD FLaS",
            "adapter_sharing": "one shared residual MLP for all image tokens and diseases",
            "adapted_branches": "i2t visual global+local memory and t2i visual global query",
            "score_kind": "native arithmetic mean of i2t and t2i logits",
            "training_score_kind": "logit",
            "native_score_kind": "bidirectional fusion logit arithmetic mean",
            "score_transform": "identity",
            "frozen": "complete author image/text encoders, projection, DQN and classifier; eval mode",
            "python_launcher": str(Path(__file__).with_name("python_local.sh")),
            "source_files": [str(Path(__file__)), str(Path(__file__).with_name("adapter.py")),
                             str(Path(__file__).parents[1] / "CARZero/models/CARZero_model_dqn_wo_self_atten_gl_mlp.py"),
                             str(Path(__file__).parents[1] / "CARZero/models/dqn_wo_self_atten_mlp.py")],
            "cache_batch": 16,
            "training_batch": 16,
            "cache_batch_size": 16,
            "train_microbatch_size": 8,
        })

    @torch.no_grad()
    def prepare_classes(self, class_names):
        if not class_names or any(not isinstance(x, str) or not x.strip() for x in class_names):
            raise ValueError("Expected nonempty disease names")
        local, global_, _ = self.native._text(class_names)
        self.class_names = tuple(class_names)
        self.query = global_.detach().clone()
        local = local.reshape(local.shape[0], local.shape[1], -1).transpose(1, 2)
        self.text_memory = torch.cat([self.query.unsqueeze(1), local], dim=1).detach().clone()
        self.metadata["class_names"] = list(class_names)
        self.metadata["last_prompts"] = self.native.metadata["last_prompts"]
        self.metadata["query_texts"] = [self.native.metadata["last_prompts"][name][0] for name in class_names]

    @torch.no_grad()
    def encode_images(self, image_paths):
        if not image_paths:
            raise ValueError("Expected at least one image")
        pixels = self.model.process_img(image_paths, self.device)
        local, global_ = self.model.image_encoder_forward(pixels)
        local = local.reshape(local.shape[0], local.shape[1], -1).transpose(1, 2)
        return torch.cat([global_.unsqueeze(1), local], dim=1).detach()

    def score_tokens(self, tokens, token_adapter=None):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before scoring")
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim or tokens.shape[1] < 2:
            raise ValueError(f"Expected B x (1+patches) x {self.dim} tokens, got {tuple(tokens.shape)}")
        adapted = tokens if token_adapter is None else token_adapter(tokens)
        # The native decoder has no query self-attention. Batched queries in
        # the reverse direction are independent and give the same scoring as
        # the released single-image loop (validated against that loop).
        i2t = self.model.fusion_module(adapted, self.query).squeeze(-1)
        t2i = self.model.fusion_module(self.text_memory, adapted[:, 0]).squeeze(-1).transpose(0, 1)
        return (i2t + t2i) / 2

    @torch.no_grad()
    def native_scores(self, image_paths):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before scoring")
        result = self.native.predict(image_paths, self.class_names)
        return torch.as_tensor(result, device=self.device)

    native_predictions = native_scores

    @torch.no_grad()
    def native_scores_from_tokens(self, tokens):
        return self.score_tokens(tokens)

    def release_image_encoder(self):
        """Move encoders to CPU after caching; queries and fusion stay put."""
        self.model.img_encoder.cpu()
        self.model.text_encoder.cpu()


def create_backend(device="cuda"):
    return TokenBackend(device=device)
