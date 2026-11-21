import unittest
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.classification_uncertainty.bootstrap import *


class BootstrapTests(unittest.TestCase):
    def test_weighted_ties_match_expanded_sklearn(self):
        y=np.array([0,1,0,1,1,0]);s=np.array([.8,.8,.2,.6,.2,.1])
        weights=np.array([[2,1,0,3,1,0],[0,1,4,2,0,3],[1,1,1,1,1,1]])
        actual=weighted_binary(y,s,weights)
        for j,w in enumerate(weights):
            ids=np.repeat(np.arange(len(y)),w)
            np.testing.assert_allclose(actual[j],[roc_auc_score(y[ids],s[ids]),average_precision_score(y[ids],s[ids])],atol=1e-14)

    def test_cluster_pairing_seed_average_and_missing_class(self):
        groups=np.array(['a','a','b','c','d','e'])
        labels=np.array([[1,0],[1,0],[0,1],[0,0],[0,0],[0,0]])
        scores=np.stack([np.array([[.1,.2],[.4,.3],[.6,.8],[.2,.1],[.5,.1],[.3,.2]]),
                         np.array([[.7,.2],[.6,.1],[.1,.4],[.3,.5],[.4,.6],[.2,.3]])])
        plan=make_patient_draw_plan(groups,n_resamples=300,seed=17)
        p,r=evaluate(labels,scores,groups,plan)
        p0,r0=evaluate(labels,scores[:1],groups,plan);p1,r1=evaluate(labels,scores[1:],groups,plan)
        np.testing.assert_allclose(p,(p0+p1)/2);np.testing.assert_allclose(r,(r0+r1)/2,equal_nan=True)
        _,inv=np.unique(groups,return_inverse=True)
        for j in range(10):
            ids=np.repeat(np.arange(len(groups)),plan.counts[j,inv])
            for c in range(2):
                if len(np.unique(labels[ids,c]))<2:self.assertTrue(np.isnan(r[j,c]).all());continue
                expected=np.mean([[roc_auc_score(labels[ids,c],v[ids,c]),average_precision_score(labels[ids,c],v[ids,c])] for v in scores],axis=0)
                np.testing.assert_allclose(r[j,c],expected)
        result=summarize_difference(p,r,p,r,['one','two'])
        self.assertGreater(result['macro']['auroc']['invalid_fraction'],0)
        self.assertEqual(result['macro']['auroc']['point'],0)
        valid=np.isfinite(r.mean(1)[:,0]);self.assertTrue(np.all((r-r).mean(1)[valid]==0))
        self.assertEqual(result['macro']['auroc']['valid_replicates'],int(valid.sum()))

    def test_reject_nonfinite_or_misaligned(self):
        with self.assertRaises(ValueError):weighted_binary([0,1],[1,np.nan],[[1,1]])
        with self.assertRaises(ValueError):weighted_binary([0,1],[1,0],[[1]])


if __name__=='__main__':unittest.main()
