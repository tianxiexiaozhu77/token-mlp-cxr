"""Audit saved predictions and run paired grouped bootstrap without retraining."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.baseline_token_classification.common import DATASETS,read_dataset,sha256,save,metrics
from experiments.both_adapter_classification.common import MODELS,reference_folder
from experiments.supervised_baseline_classification.common import DEFAULT_RUN
from experiments.classification_uncertainty.bootstrap import *


def load_predictions(folder,rows,indices,classes):
    result=json.loads((folder/'result.json').read_text());assert result['status']=='complete'
    with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
        np.testing.assert_array_equal(p['indices'],indices)
        np.testing.assert_array_equal(p['ids'],[rows[i]['id'] for i in indices])
        np.testing.assert_array_equal(p['groups'],[rows[i]['group_id'] for i in indices])
        np.testing.assert_array_equal(p['labels'],[rows[i]['labels'] for i in indices])
        if 'class_names' in p:np.testing.assert_array_equal(p['class_names'],classes)
        values=p['logits'].copy();recomputed=metrics(p['labels'],values,classes)
    for name in ('macro_auroc','macro_auprc'):
        assert abs(recomputed[name]-result['test'][name])<1e-12,(folder,name)
    return values,dict(folder=str(folder),result_sha256=sha256(folder/'result.json'),
        predictions_sha256=sha256(folder/'test_predictions.npz'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--scope',choices=['main','supervised'],default='main')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'experiments/classification_uncertainty/runs/20261120')
    args=parser.parse_args();out=args.run_dir/args.scope/args.dataset;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(out/'summary.json')
    meta,rows=read_dataset(DATASETS[args.dataset]);indices=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    labels=np.asarray([rows[i]['labels'] for i in indices]);classes=meta['class_names']
    groups=np.asarray([rows[i]['group_id'] for i in indices]);plan=make_patient_draw_plan(groups,n_resamples=2000,seed=20261120)
    save_patient_draw_plan(plan,out/'draw_plan.npz')
    models=MODELS if args.scope=='main' else ('AlphaRAD','KAD','MaCo')
    pairs=[('pre_mlp','original')] if args.scope=='main' else [('pre_mlp',m) for m in ('post_mlp','linear_probe','clip_global')]
    methods=list(dict.fromkeys(m for pair in pairs for m in pair));sources=[];comparisons=[];started=time.time()
    protocol=dict(dataset=args.dataset,scope=args.scope,classes=classes,images=len(indices),groups=len(set(groups)),
        grouping='report/exact-duplicate groups; not verified patients' if args.dataset=='openi' else 'patient/exact-file-duplicate groups',
        dataset_sha256=sha256(DATASETS[args.dataset]),samples_sha256=meta['samples_sha256'],
        resamples=2000,bootstrap_seed=20261120,draw_plan_sha256=plan.sha256,confidence=.95,
        estimand='Mean of per-checkpoint image-level metrics over the three fixed training seeds; paired cluster bootstrap of test groups.',
        uncertainty='Conditional on fitted checkpoints and fixed split; does not cover training-seed or training-cohort uncertainty.',
        missing_class='Reject full macro replicate if any class lacks either label. Do not change class set or redraw. Conditional percentiles with at least200 valid draws.',
        metric_names=['AUROC','average precision (AP), NOT trapezoidal PR AUC'],multiple_comparisons='Descriptive unadjusted95% intervals; no family-wise significance claim.',
        comparisons=pairs,source_sha256={str(p):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        per_class_counts={c:dict(positive_images=int(labels[:,j].sum()),negative_images=int((1-labels[:,j]).sum()),
            positive_groups=len(set(groups[labels[:,j]==1])),negative_groups=len(set(groups[labels[:,j]==0]))) for j,c in enumerate(classes)})
    save(out/'PROTOCOL.json',protocol)
    for model in models:
        evaluated={}
        for method in methods:
            values=[]
            for seed in ((42,) if method=='original' else (42,43,44)):
                folder=DEFAULT_RUN/'results'/model/args.dataset/f'seed{seed}'/method if method in ('linear_probe','clip_global') else reference_folder(model,args.dataset,seed,method)
                scores,source=load_predictions(folder,rows,indices,classes);sources.append(source);values.append(scores)
            point,draws=evaluate(labels,np.stack(values),groups,plan);evaluated[method]=(point,draws)
            np.savez_compressed(out/f'{model}_{method}.npz',point=point,replicates=draws,classes=classes,metrics=['auroc','ap'],plan_sha256=plan.sha256)
            print(args.dataset,model,method,'seconds',round(time.time()-started),flush=True)
        for a,b in pairs:
            point_a,draws_a=evaluated[a];point_b,draws_b=evaluated[b]
            comparisons.append(dict(model=model,method_a=a,method_b=b,**summarize_difference(point_a,draws_a,point_b,draws_b,classes)))
    save(out/'summary.json',dict(status='complete',protocol=protocol,sources=sources,comparisons=comparisons,seconds=time.time()-started))
    csv_rows=[]
    for comp in comparisons:
        for name,v in comp['macro'].items():
            csv_rows.append(dict(dataset=args.dataset,model=comp['model'],difference=comp['method_a']+' minus '+comp['method_b'],metric=name,
                delta_pp=100*v['point'],ci95_lower_pp=100*v['ci95'][0] if v['ci95'] else None,ci95_upper_pp=100*v['ci95'][1] if v['ci95'] else None,
                valid_replicates=v['valid_replicates'],total_replicates=v['total_replicates'],conditional=v['conditional_on_all_classes_present']))
    with (out/'macro_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)


if __name__=='__main__':main()
