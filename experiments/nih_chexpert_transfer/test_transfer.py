"""Regression tests for source-only normalization, semantic mapping and inference."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nih_chexpert_transfer.common import *
from experiments.large_patient_multilabel.features import forward
from experiments.direct_anchor_classification.model import FrozenClassificationHead
from experiments.classification_uncertainty.bootstrap import weighted_binary

torch.set_num_threads(2)


class TransferTests(unittest.TestCase):
    def test_semantic_projection_with_reordered_axis(self):
        classes = ['unshared'] + [s for s, _ in MAPPING][::-1]
        scores = np.arange(16).reshape(2, 8)
        np.testing.assert_array_equal(project(scores, classes), scores[:, 7:0:-1])
        with self.assertRaises(ValueError):
            project(scores, ['duplicate'] * 8)
        with self.assertRaises(ValueError):
            project(scores[:, :2], classes[:2])

    def test_all_selected_modules_against_explicit_source_operations(self):
        torch.manual_seed(42)
        source_classes = read(SOURCE / 'features/AlphaRAD/head.json')['class_names']
        # Deliberately shifted inputs catch accidentally fitting target normalization.
        tokens = torch.randn(2, 5, 768) * 1.7 + 3.
        for model, method, seed in jobs():
            if method == 'original':
                continue
            payload = torch.load(source_folder(model, method, seed) / 'adapter.pt',
                                 map_location='cpu', weights_only=True)
            s = payload['module_state_dict']
            head = FrozenClassificationHead(payload['head_config']).eval()
            module = build_module(method, s, 14)
            features = SimpleNamespace(device='cpu', config=payload['head_config'], head=head,
                                       contexts=head.context(tokens), global_features=tokens[:, 0])
            x = tokens[:, 0]
            if method == 'linear_probe':
                expected = F.linear((x - s['feature_mean']) / s['feature_std'],
                                    s['classifier.weight'], s['classifier.bias'])
                self.assertTrue(torch.equal(module.feature_mean, s['feature_mean']))
                self.assertTrue(torch.equal(module.feature_std, s['feature_std']))
            elif method == 'clip_global':
                x = .8*x + .2*F.relu(F.linear(F.relu(F.linear(x, s['fc.0.weight'])), s['fc.2.weight']))
                if head.kind == 'radzero' and head.ln_weight is not None:
                    x = F.layer_norm(x, (768,), head.ln_weight, head.ln_bias, head.config['layer_norm_eps'])
                expected = head.score(x[:, None].expand(-1, 14, -1))
            else:
                x = tokens if method == 'pre_mlp' else features.contexts
                x = x + F.linear(F.gelu(F.linear(x, s['down.weight'], s['down.bias'])),
                                 s['up.weight'], s['up.bias'])
                expected = head(x) if method == 'pre_mlp' else head.score(x)
            actual = forward(features, np.arange(2), module, method, tokens)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(project(actual.detach().numpy(), source_classes),
                actual.detach().numpy()[:, [0, 1, 2, 6, 7, 8, 9]])
            self.assertFalse(any(p.requires_grad for p in module.parameters()))

    def test_reject_wrong_source_class_axis_and_protocol(self):
        payload = torch.load(source_folder('AlphaRAD', 'pre_mlp', 42) / 'adapter.pt',
                             map_location='cpu', weights_only=True)
        protocol = dict(source_class_names=payload['classes'])
        selection = dict(best_epoch=payload['best_epoch'])
        validate_payload(payload, 'AlphaRAD', 'pre_mlp', 42, protocol, selection)
        for key, value in [('classes', payload['classes'][::-1]), ('protocol_sha256', 'wrong'),
                           ('model', 'RadZero'), ('seed', 99)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_payload(dict(payload, **{key: value}), 'AlphaRAD', 'pre_mlp', 42, protocol, selection)

    def test_weighted_ties_match_expanded_patient_resampling(self):
        from sklearn.metrics import roc_auc_score, average_precision_score
        y = np.asarray([0, 1, 1, 0, 1, 0])
        scores = np.asarray([.1, .5, .5, .5, .9, .9])
        weights = np.asarray([[1, 1, 1, 1, 1, 1], [2, 2, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1]])
        actual = weighted_binary(y, scores, weights)
        for i in range(2):
            idx = np.repeat(np.arange(len(y)), weights[i])
            expected = [roc_auc_score(y[idx], scores[idx]), average_precision_score(y[idx], scores[idx])]
            np.testing.assert_allclose(actual[i], expected, atol=1e-12, rtol=0)
        self.assertTrue(np.isnan(actual[2]).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
