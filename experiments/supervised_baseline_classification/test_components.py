import unittest
import torch
from experiments.supervised_baseline_classification.components import LinearProbe,ClipGlobalAdapter,make_module,global_scores


class BaselineTests(unittest.TestCase):
    def test_clip_formula_and_parameters(self):
        module=ClipGlobalAdapter();x=torch.randn(4,768,requires_grad=True)
        expected=.8*x+.2*torch.relu(module.fc[2](torch.relu(module.fc[0](x))))
        torch.testing.assert_close(module(x),expected)
        self.assertEqual(sum(p.numel() for p in module.parameters()),294912)
        module(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_linear_standardization_train_only(self):
        x=torch.tensor([[1.,2.],[3.,2.],[1000.,-4000.]])
        probe=make_module('linear_probe',x,[0,1],3)
        torch.testing.assert_close(probe.feature_mean,torch.tensor([2.,2.]))
        self.assertEqual(sum(p.numel() for p in probe.parameters()),9)
        self.assertTrue(torch.isfinite(probe(x)).all())

    def test_kad_global_memory_keeps_gradient(self):
        class Backend:
            def score_tokens(self,tokens):
                self.shape=tokens.shape
                return tokens.sum(-1)
        backend=Backend();x=torch.randn(5,768,requires_grad=True)
        global_scores('KAD',backend,x).sum().backward()
        self.assertEqual(backend.shape,(5,1,768))
        torch.testing.assert_close(x.grad,torch.ones_like(x))


if __name__=='__main__':unittest.main()
