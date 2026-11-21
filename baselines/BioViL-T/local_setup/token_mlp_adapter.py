"""Added text-guided aggregation on BioViL-T's native projected patch space.

Native classification uses the mean of projected patches. The new parameter-free
cosine attention is not FLaS and has its own aggregation-only control.
"""
from pathlib import Path
import importlib.util

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATH = ROOT.parent / "protocols/alpharad_binary/BioViL-T/local_setup/adapter.py"


class TokenBackend:
    dim = 128
    tokens_per_image = 196

    def __init__(self, device="cuda"):
        spec = importlib.util.spec_from_file_location("_biovil_token_native", NATIVE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.native = module.create_adapter(device)
        self.device = self.native.device
        self.class_names = None
        self.metadata = {
            **self.native.metadata, "aggregation_origin": "added",
            "dim": self.dim, "tokens_per_image": self.tokens_per_image,
            "token_boundary": "native unnormalized 128D projected 14x14 patch embeddings, before global averaging",
            "aggregation": "separate positive/negative cosine softmax over patches, fixed temperature 1; raw projected values; L2-normalized aggregate cosine readout",
            "attention_temperature": 1.0, "native_score_kind": self.native.metadata["score_kind"],
            "score_transform": "positive-minus-negative cosine margin (logit of native binary probability)",
            "zero_adapter_reference": "aggregation_only", "native_aggregation": "mean of all projected patches",
            "recommended_cache_batch_size": 16, "recommended_train_batch_size": 32,
            "cache_batch_size": 16, "train_microbatch_size": 32,
            "source_path": str(Path(__file__).resolve()), "native_adapter_path": str(NATIVE_PATH),
            "source_files": [str(NATIVE_PATH), str(ROOT / "local_setup/adapter.py"),
                             str(ROOT / "official/hi-ml-multimodal/src/health_multimodal/image/model/model.py")],
            "python_launcher": str(ROOT / "local_setup/python_local.sh"),
        }

    @torch.no_grad()
    def prepare_classes(self, class_names):
        self.class_names = tuple(class_names)
        if not self.class_names or any(not isinstance(x, str) or not x.strip() for x in self.class_names):
            raise ValueError("Nonempty ordered class names required")
        phrases = [name.replace("_", " ") for name in self.class_names]
        prompts = [f"There is no {name}." for name in phrases] + [f"There is {name}." for name in phrases]
        self.queries = self.native.text_engine.get_embeddings_from_prompt(prompts, verbose=False).detach().clone()
        self.metadata.update(class_names=list(self.class_names), prompts=prompts)

    @torch.no_grad()
    def encode_images(self, paths):
        images, _ = self.native._images(paths)
        tokens = self.native.model.get_patchwise_projected_embeddings(images, normalize=False)
        tokens = tokens.reshape(len(paths), -1, self.dim).float()
        if tokens.shape[1] != self.tokens_per_image:
            raise RuntimeError(f"Unexpected projected token shape: {tuple(tokens.shape)}")
        return tokens.detach()

    def score_tokens(self, tokens, token_adapter=None):
        if token_adapter is not None:
            tokens = token_adapter(tokens)
        weights = torch.einsum("bnd,qd->bqn", F.normalize(tokens, dim=-1), self.queries).softmax(-1)
        contexts = F.normalize(torch.einsum("bqn,bnd->bqd", weights, tokens), dim=-1)
        scores = torch.einsum("bqd,qd->bq", contexts, self.queries)
        count = len(self.class_names)
        return scores[:, count:] - scores[:, :count]

    def native_scores_from_tokens(self, tokens):
        scores = F.normalize(tokens.mean(1), dim=-1) @ self.queries.T
        count = len(self.class_names)
        return scores[:, count:] - scores[:, :count]

    @torch.no_grad()
    def native_predictions(self, paths):
        return torch.as_tensor(self.native.predict(paths, self.class_names), device=self.device)

    @torch.no_grad()
    def native_scores(self, paths):
        return torch.logit(self.native_predictions(paths), eps=1e-7)

    def release_image_encoder(self):
        if self.class_names is None:
            raise RuntimeError("prepare_classes must precede encoder release")
        self.native = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
