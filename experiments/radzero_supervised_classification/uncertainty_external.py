"""Patient-paired bootstrap of the prelocked RSNA transfer evaluations."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.radzero_supervised_classification.common import *
from experiments.classification_uncertainty.bootstrap import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--output',type=Path,default=DEFAULT_RUN/'uncertainty/rsna_transfer')
    args=parser.parse_args();run=args.run_dir;root=run/'external';protocol=json.loads((root/'PROTOCOL.json').read_text())
    rows=json.loads(Path(protocol['samples_file']).read_text());assert sha256(protocol['samples_file'])==protocol['samples_sha256']
    labels=np.asarray([r['labels'] for r in rows]);groups=np.asarray([r['group_id'] for r in rows]);classes=['Pneumonia']
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(out/'summary.json')
    plan=make_patient_draw_plan(groups,n_resamples=2000,seed=20261120);save_patient_draw_plan(plan,out/'draw_plan.npz')
    pp=dict(dataset='rsna_transfer',images=len(rows),patients=len(set(groups)),positives=int(labels.sum()),
        positive_patients=len(set(groups[labels[:,0]==1])),negative_patients=len(set(groups[labels[:,0]==0])),
        source_protocol_sha256=sha256(root/'PROTOCOL.json'),grouping='NIH patients verified with official RSNA mapping',
        resamples=2000,seed=20261120,draw_plan_sha256=plan.sha256,metric_names=['AUROC','AP'],
        uncertainty='Test-patient uncertainty conditional on the fixed CheXpert-fitted checkpoints; mean of per-seed metrics, no score ensembling.',
        contrasts='Pre MLP minus each of Original, Post MLP, Linear probe, CLIP-style global adapter.',
        multiple_comparisons='Unadjusted descriptive95% intervals.',code_sha256=sha256(__file__))
    save(out/'PROTOCOL.json',pp);sources=[];comparisons=[];performance=[]
    for model in MODELS:
        assert json.loads((root/'results'/model/'summary.json').read_text())['status']=='complete'
        evaluated={}
        for method in protocol['methods']:
            scores=[]
            for seed in ((42,) if method in ('original','global_frozen') else SEEDS):
                folder=root/'results'/model/f'seed{seed}'/method;r=json.loads((folder/'result.json').read_text())
                assert r['status']=='complete' and sha256(folder/'test_predictions.npz')==r['predictions_sha256']
                with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['labels'],labels);np.testing.assert_array_equal(p['groups'],groups)
                    np.testing.assert_array_equal(p['ids'],[r['id'] for r in rows]);np.testing.assert_array_equal(p['indices'],np.arange(len(rows)))
                    np.testing.assert_array_equal(p['class_names'],classes)
                    np.testing.assert_array_equal(p['logits'],p['source_logits'][:,[protocol['source_column']]])
                    rec=metrics(labels,p['logits'],classes)
                    for k in ('macro_auroc','macro_auprc'):assert abs(rec[k]-r['test'][k])<1e-12
                    scores.append(p['logits'].copy())
                sources.append(dict(model=model,method=method,seed=seed,folder=str(folder),result_sha256=sha256(folder/'result.json'),predictions_sha256=r['predictions_sha256']))
            point,draws=evaluate(labels,np.stack(scores),groups,plan);evaluated[method]=(point,draws)
            np.savez_compressed(out/f'{model}_{method}.npz',point=point,replicates=draws,classes=classes,metrics=['auroc','ap'],plan_sha256=plan.sha256)
            performance.append(dict(model=model,method=method,metrics={name:interval(point[0,k],draws[:,0,k]) for k,name in enumerate(('auroc','ap'))}))
            print(model,method,flush=True)
        for other in ('original','post_mlp','linear_probe','clip_global'):
            pa,da=evaluated['pre_mlp'];pb,db=evaluated[other]
            comparisons.append(dict(model=model,method_a='pre_mlp',method_b=other,**summarize_difference(pa,da,pb,db,classes)))
    save(out/'summary.json',dict(status='complete',protocol=pp,sources=sources,performance=performance,comparisons=comparisons))
    csv_rows=[]
    for comp in comparisons:
        for metric,v in comp['macro'].items():
            csv_rows.append(dict(dataset='rsna_transfer',model=comp['model'],difference='pre_mlp minus '+comp['method_b'],metric=metric,
                delta_pp=100*v['point'],ci95_lower_pp=100*v['ci95'][0],ci95_upper_pp=100*v['ci95'][1],valid_replicates=v['valid_replicates']))
    with (out/'macro_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)


if __name__=='__main__':main()
