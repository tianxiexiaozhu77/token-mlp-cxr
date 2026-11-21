"""Audit every required fit/prediction and export complete, unfiltered results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *
from experiments.large_patient_multilabel.uncertainty import read_predictions


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    args=parser.parse_args();run=args.run_dir.resolve();p=load_protocol(run);ph=sha256(run/'PROTOCOL.json')
    meta,rows=load_data();check_disjoint(rows)
    audit=json.loads((run/'SPLIT_AUDIT.json').read_text())
    if audit['status']!='passed' or any(audit['cross_split_overlap'].values()):raise ValueError('Split audit failed')
    lock=json.loads((run/'SELECTION_LOCK.json').read_text());lock_sha=sha256(run/'SELECTION_LOCK.json')
    registered=set(expected_jobs())
    if len(lock['selections'])!=24 or {(r['model'],r['method'],r['seed']) for r in lock['selections']}!=registered:
        raise ValueError('Selection lock coverage differs')
    for record in lock['selections']:
        out=folder(run,record['model'],record['method'],record['seed'])
        for key,name in [('checkpoint_sha256','adapter.pt'),('fit_sha256','fit.json'),('history_sha256','training.json')]:
            if sha256(out/name)!=record[key]:raise ValueError('Selected artifact changed after lock')
        history=json.loads((out/'training.json').read_text());fit=json.loads((out/'fit.json').read_text())
        if fit['test_accessed'] or not fit['frozen_head_verified']:raise ValueError('Fit invariant failed')
        if selected_epoch(history['initial_val']['macro_auroc'],history['history'])!=fit['best_epoch']:
            raise ValueError('Incorrect VAL selection')
    ids=np.asarray([i for i,r in enumerate(rows) if r['split']=='test']);classes=meta['class_names']
    per_seed=[];per_class=[];summary=[];predictions=[]
    methods=('original','global_frozen','linear_probe','clip_global','post_mlp','pre_mlp')
    for model in MODELS:
        parity=json.loads((run/'evaluation'/f'{model}_cache_parity.json').read_text())
        if parity['status']!='passed':raise ValueError('Frozen cache parity failed')
        for method in methods:
            seeds=(42,) if method in ('original','global_frozen') else SEEDS
            group=[]
            for seed in seeds:
                out=folder(run,model,method,seed);result=json.loads((out/'result.json').read_text())
                if result['protocol_sha256']!=ph or result['selection_lock_sha256']!=lock_sha:
                    raise ValueError('TEST result is outside the locked protocol')
                if result['test_used_for_selection'] or result['full_test_checkpoint_replay_max_error']>1e-6:
                    raise ValueError('TEST replay/selection check failed')
                read_predictions(run,model,method,seed,rows,ids,classes)
                item=dict(model=model,method=method,seed=seed,best_epoch=result['best_epoch'],
                    trainable_parameters=result['trainable_parameters'],
                    auroc_pct=100*result['test']['macro_auroc'],ap_pct=100*result['test']['macro_auprc'])
                per_seed.append(item);group.append(item)
                predictions.append(dict(path=str(out/'test_predictions.npz'),sha256=sha256(out/'test_predictions.npz')))
                for name,v in result['test']['per_class'].items():
                    per_class.append(dict(model=model,method=method,seed=seed,class_name=name,
                        positives=v['positives'],negatives=v['negatives'],auroc_pct=100*v['auroc'],ap_pct=100*v['auprc']))
            summary.append(dict(model=model,method=method,seeds=len(seeds),
                trainable_parameters=group[0]['trainable_parameters'],
                auroc_mean_pct=statistics.mean(r['auroc_pct'] for r in group),
                auroc_sd_pct=statistics.stdev(r['auroc_pct'] for r in group) if len(group)>1 else 0.,
                ap_mean_pct=statistics.mean(r['ap_pct'] for r in group),
                ap_sd_pct=statistics.stdev(r['ap_pct'] for r in group) if len(group)>1 else 0.,
                selected_epochs=','.join(str(r['best_epoch']) for r in group)))
    if len(per_seed)!=28 or len(per_class)!=28*14 or len(summary)!=12:
        raise ValueError('Incomplete evaluation matrix')
    uncertainty=json.loads((run/'uncertainty/summary.json').read_text())
    expected_pairs={(m,'pre_mlp',b) for m in MODELS for b in ('original','post_mlp','linear_probe','clip_global')}
    actual_pairs={(r['model'],r['method_a'],r['method_b']) for r in uncertainty['comparisons']}
    if uncertainty['status']!='complete' or actual_pairs!=expected_pairs or len(uncertainty['comparisons'])!=8:
        raise ValueError('Incomplete uncertainty comparisons')
    if uncertainty['protocol_sha256']!=ph or uncertainty['selection_lock_sha256']!=lock_sha:
        raise ValueError('Uncertainty protocol differs')
    for comparison in uncertainty['comparisons']:
        a=next(r for r in summary if r['model']==comparison['model'] and r['method']=='pre_mlp')
        b=next(r for r in summary if r['model']==comparison['model'] and r['method']==comparison['method_b'])
        for metric in ('auroc','ap'):
            record=comparison['macro'][metric]
            if abs(record['point']*100-(a[metric+'_mean_pct']-b[metric+'_mean_pct']))>1e-8:
                raise ValueError('Bootstrap point estimate differs')
            if record['total_replicates']!=2000:raise ValueError('Incomplete bootstrap draws')
    write_csv(run/'per_seed_results.csv',per_seed);write_csv(run/'per_class_results.csv',per_class)
    write_csv(run/'summary.csv',summary)
    lines=['ChestXray14 large patient-disjoint test replication',
        'AUROC/AP in percent; fitted methods average per-seed metrics, without prediction ensembling.',
        '24 fresh fits, 28 TEST evaluations, 8 paired comparisons (16 macro metric differences).',
        'All 24 validation-selected checkpoint hashes locked before TEST evaluation.',
        'Patient counts and image counts:']
    for split,spec in meta['split_stats'].items():
        lines.append(f"  {split}: {spec['images']} images, {spec['patients']} patients")
    lines+=['','Model | method | AUROC mean +/- SD | AP mean +/- SD | selected epochs']
    for r in summary:
        lines.append(f"{r['model']} | {r['method']} | {r['auroc_mean_pct']:.2f} +/- {r['auroc_sd_pct']:.2f} | "
                     f"{r['ap_mean_pct']:.2f} +/- {r['ap_sd_pct']:.2f} | {r['selected_epochs']}")
    lines+=['','Paired patient-group bootstrap (percentage points):']
    for c in uncertainty['comparisons']:
        for metric,v in c['macro'].items():
            interval='unavailable' if v['ci95'] is None else f"[{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}]"
            lines.append(f"{c['model']} Pre - {c['method_b']} {metric}: {100*v['point']:+.2f} {interval}; "
                         f"valid {v['valid_replicates']}/{v['total_replicates']}")
    lines+=['','Scope and limitations:',*p['limitations'],uncertainty['interpretation'],
        'The global CLS route replaces native disease aggregation; its difference from Pre is not a pure insertion-point effect.',
        'All methods used the fixed prior protocol; no claim that each baseline is optimally tuned.',
        'Hernia has 11 positive TEST images from 9 patients; per-class support and bootstrap validity are retained.']
    (run/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    tex=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llcc}',r'\toprule',
         r'Model & Method & AUROC (\%) & AP (\%) \\',r'\midrule']
    labels={'original':'Original','global_frozen':'Frozen global','linear_probe':'Linear probe','clip_global':'Global adapter',
            'post_mlp':'Post MLP','pre_mlp':'Pre MLP'}
    for r in summary:
        cells=[]
        for metric in ('auroc','ap'):
            value=f"{r[metric+'_mean_pct']:.2f}"
            if r['seeds']>1:
                value+=f" \\pm {r[metric+'_sd_pct']:.2f}"
            cells.append('$'+value+'$')
        tex.append(f"{r['model']} & {labels[r['method']]} & "+' & '.join(cells)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',
        r'\caption{Patient-disjoint local ChestXray14 replication on 6,727 test images from 1,856 patients, retaining all 14 labels. '
        r'Training/validation use 13,406/2,300 images from 3,712/619 patients. Fitted methods report three-seed mean $\pm$ sample SD; '
        r'unfitted methods are evaluated once. This is a repartition of a previously used local cohort, not an official benchmark split or a new blind external cohort.}',
        r'\label{tab:large_patient_multilabel}',r'\end{table*}']
    (run/'paper_table.tex').write_text('\n'.join(tex)+'\n')
    save(run/'AUDIT.json',dict(status='passed',checked_fits=24,checked_test_evaluations=28,checked_classes_per_evaluation=14,
        checked_paired_comparisons=8,all_validation_selections_verified=True,all_prediction_metrics_recomputed=True,
        all_test_checkpoint_replays_passed=True,all_predictions_aligned_to_locked_test=True,
        split_overlap=audit['cross_split_overlap'],prediction_files=predictions,
        source_sha256={str(path):sha256(path) for path in HERE.glob('*.py')}))
    outputs=['REPORT.txt','summary.csv','per_seed_results.csv','per_class_results.csv','paper_table.tex',
        'AUDIT.json','SELECTION_LOCK.json','PROTOCOL.json','uncertainty/summary.json','uncertainty/macro_differences.csv']
    save(run/'COMPLETION.json',dict(status='complete',finished=time.time(),fits=24,test_evaluations=28,paired_comparisons=8,
        output_sha256={name:sha256(run/name) for name in outputs}))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
