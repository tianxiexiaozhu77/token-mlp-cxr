"""Checks for leakage, fixed labels, checkpoint selection, and resume ordering."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.large_patient_multilabel.prepare import split_groups,UnionFind
from experiments.large_patient_multilabel.common import check_disjoint,selected_epoch,metrics
from experiments.large_patient_multilabel.train import training_targets,make_loader,predict
from experiments.large_patient_multilabel.features import make_module,forward
from experiments.direct_anchor_classification.model import FrozenClassificationHead


class DataAndSelection(unittest.TestCase):
    def test_split_determinism_capacity_and_rare_labels(self):
        groups={f'p{i:03}':np.array([i<30, i%3==0, i in (3,17,25,36,58,81,91,95,96,97)]) for i in range(100)}
        a,sizes=split_groups(groups,20261121)
        b,_=split_groups(dict(reversed(list(groups.items()))),20261121)
        self.assertEqual(a,b);self.assertEqual(sizes,dict(train=60,val=10,test=30))
        for split,size in sizes.items():
            selected=[groups[k] for k in groups if a[k]==split]
            self.assertEqual(len(selected),size)
            counts=np.asarray(selected).sum(0)
            self.assertTrue(((counts>0)&(counts<size)).all())

    def test_patient_and_duplicate_leakage_rejected(self):
        rows=[dict(id=str(i),patient_id=f'p{i}',group_id=f'g{i}',file_sha256=f'f{i}',pixel_sha256=f'x{i}',split=s)
              for i,s in enumerate(('train','val','test'))]
        check_disjoint(rows)
        for field in ('patient_id','group_id','file_sha256','pixel_sha256'):
            bad=copy.deepcopy(rows);bad[2][field]=bad[0][field]
            with self.assertRaises(ValueError):check_disjoint(bad)
        u=UnionFind(['a','b','c']);u.union('a','b');u.union('b','c')
        self.assertEqual(u.find('a'),u.find('c'))

    def test_test_labels_do_not_change_training_weights(self):
        rows=[dict(labels=[0,1]),dict(labels=[1,0]),dict(labels=[0,1]),dict(labels=[1,1])]
        ids={'train':np.array([0,1]),'val':np.array([2]),'test':np.array([3])}
        before=training_targets(rows,ids)
        rows[3]['labels']=[999,float('nan')]
        after=training_targets(rows,ids)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_epoch_zero_and_earliest_tie(self):
        history=[dict(epoch=1,val={'macro_auroc':.7}),dict(epoch=2,val={'macro_auroc':.8}),dict(epoch=3,val={'macro_auroc':.8})]
        self.assertEqual(selected_epoch(.8,history),0)
        self.assertEqual(selected_epoch(.75,history),2)

    def test_no_silent_class_dropping(self):
        with self.assertRaises(ValueError):metrics(np.array([[0,0],[1,0]]),np.array([[.1,.1],[.9,.2]]),['a','b'])

    def test_test_predictions_require_lock(self):
        with self.assertRaises(ValueError):predict(None,'test','pre_mlp',None)

    def test_sampler_resume_is_worker_seed_independent(self):
        features=SimpleNamespace(ids={'train':np.arange(31)},rows=[])
        generator=torch.Generator().manual_seed(42)
        loader=make_loader(features,'train','post_mlp',batch_size=7,shuffle=True,generator=generator)
        list(loader);saved=generator.get_state()
        expected=torch.cat([idx for _,idx in loader])
        restarted=torch.Generator();restarted.set_state(saved)
        new=make_loader(features,'train','post_mlp',batch_size=7,shuffle=True,generator=restarted)
        torch.testing.assert_close(torch.cat([idx for _,idx in new]),expected)


class Scoring(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)

    def features(self,model):
        config=dict(model=model,queries=torch.randn(3,768).tolist(),include_cls=True,
            heads=8 if model=='alpharad' else 1,scale=15.,head_index=-1,head_weights=None,
            layer_norm_weight=torch.ones(768).tolist() if model=='radzero' else None,
            layer_norm_bias=torch.zeros(768).tolist() if model=='radzero' else None,
            layer_norm_eps=1e-5,sim_op='cos',attention_temperature=.07,loss_temperature=.07)
        head=FrozenClassificationHead(config).eval().requires_grad_(False)
        tokens=torch.randn(8,7,768)
        return SimpleNamespace(device='cpu',config=config,head=head,contexts=head.context(tokens),
            global_features=tokens[:,0],ids={'train':np.arange(4),'val':np.arange(4,6),'test':np.arange(6,8)},
            meta={'class_names':['a','b','c']}),tokens

    def test_identity_pre_post_and_frozen_gradients(self):
        for model in ('alpharad','radzero'):
            f,tokens=self.features(model);ids=torch.arange(3)
            original=forward(f,ids,None,'original')
            for method in ('pre_mlp','post_mlp'):
                module=make_module(method,f)
                actual=forward(f,ids,module,method,tokens[ids])
                torch.testing.assert_close(actual,original)
                actual.square().mean().backward()
                self.assertGreater(float(module.up.weight.grad.abs().sum()),0.)
                self.assertEqual(sum(p.numel() for p in module.parameters()),99136)
                self.assertTrue(all(p.grad is None for p in f.head.parameters()))

    def test_linear_standardization_uses_train_only(self):
        f,_=self.features('alpharad')
        a=make_module('linear_probe',f)
        f.global_features[4:]=1e9
        b=make_module('linear_probe',f)
        torch.testing.assert_close(a.feature_mean,b.feature_mean)
        torch.testing.assert_close(a.feature_std,b.feature_std)

    def test_radzero_global_keeps_native_layernorm(self):
        from torch.nn import functional as F
        f,_=self.features('radzero');ids=torch.arange(3)
        x=F.layer_norm(f.global_features[ids],(768,),f.head.ln_weight,f.head.ln_bias,1e-5)
        expected=f.head.score(x[:,None].expand(-1,3,-1))
        torch.testing.assert_close(forward(f,ids,None,'global_frozen'),expected)


if __name__=='__main__':unittest.main()
