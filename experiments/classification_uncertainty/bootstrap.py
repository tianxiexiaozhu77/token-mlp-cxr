"""Image-level AUROC/AP with shared cluster multiplicities and score ties.

The full class set is fixed: a draw missing either label for any class has an
undefined macro score. Percentiles over valid draws are explicitly conditional.
Training seeds are averaged AFTER computing each checkpoint's metric.
"""
import numpy as np
from experiments.chexchonet_stability_20261024.cluster_bootstrap import (
    make_patient_draw_plan, save_patient_draw_plan, load_patient_draw_plan)


def weighted_binary(labels, scores, weights):
    """Return [draw, (AUROC, AP)]; weights are nonnegative cluster counts."""
    y=np.asarray(labels);s=np.asarray(scores);w=np.asarray(weights,dtype=np.float64)
    if y.ndim!=1 or s.shape!=y.shape or w.ndim!=2 or w.shape[1]!=len(y) or not len(y):
        raise ValueError('Misaligned labels/scores/draws')
    if not np.isin(y,[0,1]).all() or not np.isfinite(s).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Invalid labels/scores/weights')
    order=np.argsort(-s,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(s[order])!=0)+1]
    total=np.add.reduceat(w[:,order],starts,axis=1)
    pos=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
    tp=np.cumsum(pos,axis=1);fp=np.cumsum(total-pos,axis=1)
    positives=tp[:,-1];negatives=fp[:,-1];valid=(positives>0)&(negatives>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        # Integrating tied-score blocks includes half of ties, exactly as ROC.
        auc=((total-pos)*(tp-.5*pos)).sum(1)/(positives*negatives)
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        ap=(pos*precision).sum(1)/positives
    result=np.stack([auc,ap],axis=1);result[~valid]=np.nan
    return result


def evaluate(labels,scores,groups,plan,batch_size=64):
    """Scores [seed,image,class]; returns point[class,metric], draws[B,C,2]."""
    labels=np.asarray(labels);scores=np.asarray(scores)
    if scores.ndim!=3 or scores.shape[1:]!=labels.shape or not np.isfinite(scores).all():
        raise ValueError('Need aligned [seed,image,class] finite scores')
    unique,inverse=np.unique(groups,return_inverse=True);plan.validate()
    if not np.array_equal(unique,plan.patient_ids):raise ValueError('Cluster universe mismatch')
    point=np.zeros((labels.shape[1],2),dtype=np.float64)
    draws=np.zeros((len(plan.counts),labels.shape[1],2),dtype=np.float64)
    for values in scores:
        for c in range(labels.shape[1]):
            point[c]+=weighted_binary(labels[:,c],values[:,c],np.ones((1,len(labels))))[0]/len(scores)
            for start in range(0,len(draws),batch_size):
                w=plan.counts[start:start+batch_size,inverse]
                draws[start:start+batch_size,c]+=weighted_binary(labels[:,c],values[:,c],w)/len(scores)
    return point,draws


def interval(point,draws,min_valid=200):
    values=np.asarray(draws);valid=values[np.isfinite(values)];n=len(valid)
    ci=np.quantile(valid,[.025,.975]).tolist() if n>=min_valid else None
    return dict(point=float(point) if np.isfinite(point) else None,ci95=ci,
        valid_replicates=n,total_replicates=len(values),invalid_fraction=1-n/len(values),
        conditional_on_all_classes_present=n<len(values),
        interval_status='estimated' if ci is not None else 'insufficient_valid_draws')


def summarize_difference(point_a,draws_a,point_b,draws_b,classes):
    delta=point_a-point_b;draws=draws_a-draws_b
    # Ordinary mean deliberately propagates NaN; never average a class subset.
    macro=draws.mean(axis=1);p=delta.mean(axis=0)
    return dict(macro={name:interval(p[k],macro[:,k]) for k,name in enumerate(('auroc','ap'))},
        per_class={c:{name:interval(delta[j,k],draws[:,j,k]) for k,name in enumerate(('auroc','ap'))}
            for j,c in enumerate(classes)})
