"""Paired patient-group bootstrap of all prespecified comparisons."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *
from experiments.classification_uncertainty.bootstrap import (
    make_patient_draw_plan,save_patient_draw_plan,evaluate,summarize_difference)


def read_predictions(run,model,method,seed,rows,ids,classes):
    out=folder(run,model,method,seed)
    result=json.loads((out/'result.json').read_text())
    if result['status']!='complete' or sha256(out/'test_predictions.npz')!=result['predictions_sha256']:
        raise ValueError('Prediction artifact differs')
    with np.load(out/'test_predictions.npz') as a:
        np.testing.assert_array_equal(a['indices'],ids)
        np.testing.assert_array_equal(a['ids'],[rows[i]['id'] for i in ids])
        np.testing.assert_array_equal(a['groups'],[rows[i]['group_id'] for i in ids])
        np.testing.assert_array_equal(a['patients'],[rows[i]['patient_id'] for i in ids])
        np.testing.assert_array_equal(a['labels'],[rows[i]['labels'] for i in ids])
        np.testing.assert_array_equal(a['class_names'],classes)
        actual=metrics(a['labels'],a['logits'],classes)
        for name in ('macro_auroc','macro_auprc'):
            if abs(actual[name]-result['test'][name])>1e-12:raise ValueError('Metric replay mismatch')
        return a['logits'].copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    args=parser.parse_args();run=args.run_dir.resolve();p=load_protocol(run)
    meta,rows=load_data();ids=np.asarray([i for i,r in enumerate(rows) if r['split']=='test'])
    labels=np.asarray([rows[i]['labels'] for i in ids]);groups=np.asarray([rows[i]['group_id'] for i in ids])
    out=run/'uncertainty';out.mkdir(parents=True,exist_ok=True)
    plan=make_patient_draw_plan(groups,n_resamples=p['uncertainty']['resamples'],seed=p['uncertainty']['seed'])
    if not (out/'draw_plan.npz').exists():save_patient_draw_plan(plan,out/'draw_plan.npz')
    comparisons=[];started=time.time()
    for model in MODELS:
        evaluated={}
        for method in ('original',*METHODS):
            target=out/(model+'_'+method+'.npz')
            seeds=(42,) if method=='original' else SEEDS
            values=np.stack([read_predictions(run,model,method,s,rows,ids,meta['class_names']) for s in seeds])
            fingerprint={str(folder(run,model,method,s)/'test_predictions.npz'):
                sha256(folder(run,model,method,s)/'test_predictions.npz') for s in seeds}
            if target.exists():
                with np.load(target) as a:
                    if str(a['plan_sha256'])!=plan.sha256 or json.loads(str(a['sources']))!=fingerprint:
                        raise ValueError('Bootstrap cache differs')
                    point,draws=a['point'].copy(),a['draws'].copy()
            else:
                point,draws=evaluate(labels,values,groups,plan)
                np.savez_compressed(target,point=point,draws=draws,plan_sha256=plan.sha256,sources=json.dumps(fingerprint))
            evaluated[method]=(point,draws)
            print('Bootstrap',model,method,'elapsed',round(time.time()-started),flush=True)
        for baseline in ('original','post_mlp','linear_probe','clip_global'):
            comparisons.append(dict(model=model,method_a='pre_mlp',method_b=baseline,
                **summarize_difference(*evaluated['pre_mlp'],*evaluated[baseline],meta['class_names'])))
    save(out/'summary.json',dict(status='complete',protocol_sha256=sha256(run/'PROTOCOL.json'),
        selection_lock_sha256=sha256(run/'SELECTION_LOCK.json'),draw_plan_sha256=plan.sha256,
        images=len(ids),patient_groups=len(set(groups)),comparisons=comparisons,
        interpretation='Paired group resampling of fixed checkpoints; average per-seed metrics, not predictions. '
        'No training-cohort uncertainty or multiplicity adjustment. A missing class invalidates the entire macro draw; '
        'intervals conditional on full class presence when invalid draws occur.',seconds=time.time()-started))
    csv_rows=[]
    for c in comparisons:
        for metric,v in c['macro'].items():
            csv_rows.append(dict(model=c['model'],comparison='pre_mlp minus '+c['method_b'],metric=metric,
                delta_pp=100*v['point'],ci95_low_pp=100*v['ci95'][0] if v['ci95'] else '',
                ci95_high_pp=100*v['ci95'][1] if v['ci95'] else '',valid_replicates=v['valid_replicates'],
                total_replicates=v['total_replicates'],conditional=v['conditional_on_all_classes_present']))
    with (out/'macro_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)


if __name__=='__main__':main()
