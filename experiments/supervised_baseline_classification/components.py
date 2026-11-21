"""Explicit global-feature baselines; no trainable text or native readout.

CLIP-style adapter follows the released visual adapter's two bias-free linear
layers with ReLU after each, reduction=4, and fixed mixing ratio=0.2. Native
biomedical readouts and multilabel weighted BCE replace CLIP's cosine softmax
and multiclass CE. It must be reported as a transplant, not an exact CLIP run.
"""
import torch
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, mean, std, classes):
        super().__init__()
        self.register_buffer('feature_mean', mean.detach().clone())
        self.register_buffer('feature_std', std.detach().clone().clamp_min(1e-6))
        self.classifier = nn.Linear(len(mean), classes)

    def forward(self, features):
        return self.classifier((features - self.feature_mean) / self.feature_std)


class ClipGlobalAdapter(nn.Module):
    def __init__(self, dim=768, reduction=4, ratio=0.2):
        super().__init__()
        self.ratio = ratio
        self.fc = nn.Sequential(nn.Linear(dim, dim // reduction, bias=False), nn.ReLU(),
                                nn.Linear(dim // reduction, dim, bias=False), nn.ReLU())

    def forward(self, features):
        return (1 - self.ratio) * features + self.ratio * self.fc(features)


def make_module(method, features, train_ids, classes):
    if method == 'linear_probe':
        train = features[train_ids]
        return LinearProbe(train.mean(0), train.std(0, unbiased=False), classes)
    if method == 'clip_global':
        return ClipGlobalAdapter(features.shape[-1])
    raise ValueError(method)


def global_scores(model, backend, features):
    if model == 'AlphaRAD':
        return backend.score(features[:, None, :].expand(-1, len(backend.queries), -1))
    if model == 'KAD':
        # Global mean is a single visual-memory token, still using all frozen
        # disease queries and native query interactions in their original order.
        return backend.score_tokens(features[:, None, :])
    if model == 'MaCo':
        return backend.score_features(features, pooling='native_mean')
    raise ValueError(model)
