"""Check native normalization, placement, gradient flow and singleton axes."""
import importlib.util
import unittest
import torch
from torch.nn import functional as F
from experiments.radzero_supervised_classification.common import ROOT, residual_mlp
from experiments.radzero_supervised_classification.features import global_scores
from experiments.direct_anchor_classification.model import FrozenClassificationHead
from experiments.supervised_baseline_classification.components import ClipGlobalAdapter


class RadZeroTests(unittest.TestCase):
    def test_native_global_and_post(self):
        spec = importlib.util.spec_from_file_location('rad_native_losses', ROOT/'RadZero/checkpoints/RadZero/losses.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        for batch, classes in ((3,4), (1,4), (3,1), (1,1)):
            torch.manual_seed(42)
            q, tokens = torch.randn(classes,768), torch.randn(batch,7,768)
            weight, bias = torch.rand(768)+.5, torch.randn(768)*.1
            cfg = dict(model='radzero', queries=q.tolist(), sim_op='cos', include_cls=True,
                attention_temperature=.17, loss_temperature=.13, layer_norm_weight=weight.tolist(),
                layer_norm_bias=bias.tolist(), layer_norm_eps=1e-5)
            head = FrozenClassificationHead(cfg).eval().requires_grad_(False)
            normalized = F.layer_norm(tokens,(768,),weight,bias,1e-5)
            expected, _ = native.SimilarityLogit('cos')(q,normalized,temperature=.17)
            torch.testing.assert_close(head(tokens), expected.T.reshape(batch,classes)/.13)
            single, _ = native.SimilarityLogit('cos')(q,normalized[:,:1],temperature=.17)
            torch.testing.assert_close(global_scores('RadZero',head,tokens[:,0]),single.T.reshape(batch,classes)/.13)
            post = residual_mlp(768,64)
            torch.testing.assert_close(head.score(head.context(tokens),post),head(tokens))
            self.assertEqual(sum(p.numel() for p in post.parameters()),99136)
            (head.score(head.context(tokens),post).square().mean()).backward()
            self.assertGreater(float(post.up.weight.grad.abs().sum()),0.)
            adapter = ClipGlobalAdapter()
            global_scores('RadZero',head,adapter(tokens[:,0])).square().mean().backward()
            self.assertGreater(float(adapter.fc[0].weight.grad.abs().sum()),0.)
            self.assertTrue(all(p.grad is None for p in head.parameters()))


if __name__ == '__main__':
    unittest.main()
