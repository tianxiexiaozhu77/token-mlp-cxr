"""Recompute, audit and summarize the complete supervised baseline matrix."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *
from experiments.classification_uncertainty.run import load_predictions


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    import torch
    torch.set_num_threads(4)
    run=DEFAULT_RUN;p=json.loads((run/'PROTOCOL.json').read_text());ph=sha256(run/'PROTOCOL.json')
    assert json.loads((run/'training_matrix.json').read_text())['status']=='complete'
    for file,digest in p['source_sha256'].items():assert sha256(file)==digest
    for ref in p['references']:
        for file,digest in ref['files_sha256'].items():assert sha256(Path(ref['source'])/file)==digest
    records=[];summary=[];sources=[];new_fit_count=0;normalization_checks=[];seed_orders={}
    for dataset,path in DATASETS.items():
        meta,rows=read_dataset(path);classes=meta['class_names'];assert sha256(path)==p['datasets'][dataset]['sha256']
        indices=np.asarray([i for i,r in enumerate(rows) if r['split']=='test']);train_ids=[i for i,r in enumerate(rows) if r['split']=='train']
        for model in MODELS:
            for method in ('original','global_frozen','linear_probe','clip_global','post_mlp','pre_mlp'):
                seeds=(42,) if method in ('original','global_frozen') else SEEDS;method_rows=[]
                for seed in seeds:
                    folder=reference_folder(model,dataset,seed,method) if method in ('original','pre_mlp','post_mlp') else run/'results'/model/dataset/f'seed{seed}'/method
                    values,source=load_predictions(folder,rows,indices,classes);sources.append(source);r=json.loads((folder/'result.json').read_text())
                    assert len(classes)==values.shape[1]
                    count=r['trainable_parameters'];expected=0 if method in ('original','global_frozen') else 769*len(classes) if method=='linear_probe' else 294912 if method=='clip_global' else 99136
                    assert count==expected
                    if method in METHODS:
                        new_fit_count+=1;assert r['protocol_sha256']==ph and r['dataset_sha256']==sha256(path)
                        assert r['frozen_backbone_verified'] and r['full_checkpoint_replay_max_error']<=1e-6 and not r['test_used_for_selection']
                        t=json.loads((folder/'training.json').read_text());h=t['history']
                        epoch_metrics=[t['initial_val']['macro_auroc']]+[e['val']['macro_auroc'] for e in h]
                        assert int(np.argmax(epoch_metrics))==r['best_epoch']
                        assert abs(max(epoch_metrics)-r['best_val']['macro_auroc'])<1e-12
                        assert 8<=len(h)<=20 and r['config']==p['train']
                        if len(h)<20:assert len(h)-r['best_epoch']>=5
                        if len(h)>8:assert all(e<8 or e-max(range(e+1),key=lambda j:epoch_metrics[j])<5 for e in range(1,len(h)))
                        saved=torch.load(folder/'adapter.pt',map_location='cpu',weights_only=True)
                        assert saved['best_epoch']==r['best_epoch'] and saved['protocol_sha256']==ph
                        assert sum(v.numel() for k,v in saved['module_state_dict'].items() if not k.startswith('feature_'))==expected
                        if method=='linear_probe':
                            features=np.load(run/'features'/model/dataset/'global.npy');train=torch.from_numpy(features[train_ids])
                            # Training statistics were reduced on GPU; bound CPU/GPU summation error.
                            st=saved['module_state_dict'];mu=train.mean(0);sd=train.std(0,unbiased=False).clamp_min(1e-6)
                            torch.testing.assert_close(st['feature_mean'],mu,atol=1e-5,rtol=1e-5)
                            torch.testing.assert_close(st['feature_std'],sd,atol=1e-5,rtol=1e-5)
                            normalization_checks.append(dict(model=model,dataset=dataset,seed=seed,train_only=True,
                                mean_max_abs_error=float((st['feature_mean']-mu).abs().max()),std_max_abs_error=float((st['feature_std']-sd).abs().max())))
                        for e in h:
                            key=(dataset,seed,e['epoch']);digest=e['train_order_sha256']
                            if key in seed_orders:assert seed_orders[key]==digest,'New methods saw different training order'
                            seed_orders[key]=digest
                    row=dict(dataset=dataset,model=model,method=method,seed=seed,parameters=count,best_epoch=r['best_epoch'],
                        auroc_percent=100*r['test']['macro_auroc'],ap_percent=100*r['test']['macro_auprc'])
                    records.append(row);method_rows.append(row)
                summary.append(dict(dataset=dataset,model=model,method=method,parameters=count,seeds=len(seeds),
                    auroc_mean=np.mean([r['auroc_percent'] for r in method_rows]),auroc_std=np.std([r['auroc_percent'] for r in method_rows],ddof=1) if len(seeds)>1 else None,
                    ap_mean=np.mean([r['ap_percent'] for r in method_rows]),ap_std=np.std([r['ap_percent'] for r in method_rows],ddof=1) if len(seeds)>1 else None,
                    epoch_zero_selections=sum(r['best_epoch']==0 for r in method_rows) if len(seeds)>1 else None))
    assert new_fit_count==54 and len(records)==126 and len(summary)==54
    write_csv(run/'all_runs.csv',records);write_csv(run/'supervised_comparison.csv',summary)
    for dataset in DATASETS:
        ci=json.loads((ROOT/'experiments/classification_uncertainty/runs/20261120/supervised'/dataset/'summary.json').read_text())
        assert ci['status']=='complete'
        for c in ci['comparisons']:
            a=next(r for r in summary if r['dataset']==dataset and r['model']==c['model'] and r['method']==c['method_a'])
            b=next(r for r in summary if r['dataset']==dataset and r['model']==c['model'] and r['method']==c['method_b'])
            for key,column in [('auroc','auroc_mean'),('ap','ap_mean')]:assert abs(100*c['macro'][key]['point']-(a[column]-b[column]))<1e-10
    # MaCo's unadapted global path must reproduce its original native mean readout.
    parity=[]
    for dataset in DATASETS:
        orig=np.load(reference_folder('MaCo',dataset,42,'original')/'test_predictions.npz')['logits']
        glob=np.load(run/'results/MaCo'/dataset/'seed42/global_frozen/test_predictions.npz')['logits']
        np.testing.assert_allclose(orig,glob,atol=3e-4,rtol=3e-4);parity.append(dict(dataset=dataset,max_abs_error=float(np.abs(orig-glob).max())))
    save(run/'audit.json',dict(status='passed',new_fits=54,historical_evaluations=63,global_frozen_controls=9,
        all_prediction_membership_and_metrics_verified=True,validation_selection_verified=True,parameter_counts_verified=True,
        checkpoint_replays_verified=True,shared_training_orders_verified_for_new_methods=True,train_only_normalization_checks=normalization_checks,
        maco_native_global_parity=parity,sources=sources,protocol_sha256=ph,report_code_sha256=sha256(__file__)))
    text=['已完成54次新增训练：3模型×3数据集×2监督基线×3seed；另纳入63次原有评估及9次不训练全局对照。',
        '下表数值为三seed均值，AUROC/AP均以百分数计；完整std、参数量和epoch0选择见supervised_comparison.csv。',
        'Linear probe参数量=769×类别数；Pre/Post均99136；CLIP-style global=294912，后者不是同参数量消融。',
        'CLIP-style适配器移植官方两层无偏置ReLU、缩减比4、固定混合0.2，保留各模型原生冻结读出，使用多标签weighted BCE；不称为原版CLIP-Adapter复现。',
        'global_frozen单独量化改用全局表示的影响。AlphaRAD/KAD的全局路线改变了原有聚合输入，不能把Pre与CLIP-style的全部差异归因于MLP结构或插入位置；严格位置比较仍是同参数量Pre/Post。',
        '所有方法共享划分、标签、固定单配置及验证集选点预算；该预算不保证每种监督基线已调到最优。',
        'AlphaRAD全局CLS、KAD空间均值、MaCo原生patch均值。线性探针标准化只用TRAIN。','']
    for dataset in DATASETS:
        text.append(dataset)
        for model in MODELS:
            cells=[r for r in summary if r['dataset']==dataset and r['model']==model]
            text.append(model+' | '+' | '.join(r['method']+f" {r['auroc_mean']:.2f}/{r['ap_mean']:.2f}" for r in cells))
        text.append('')
    text+=['患者/报告组paired bootstrap：experiments/classification_uncertainty/runs/20261120/{main,supervised}/<dataset>/macro_differences.csv',
        '每次对相同组重采样，保留组内全部图像，分别计算各seed指标再取均值；区间衡量固定模型的测试样本不确定性，不是训练seed不确定性。',
        '任意类别缺正/负则整次macro无效；不删类别、不重抽直到有效。给出的百分位区间是在完整类别仍可计算条件下的区间。',
        'CheXpert仅312/2000次完整macro有效，OpenI970/2000，ChestXray14 1891/2000。尤其CheXpert区间不能解释为可靠的无条件95%覆盖；只有1阳性的类别问题仍在。',
        'OpenI按已验证报告/精确重复组重采样，不能称患者级区间；全部区间未做多重比较校正。',
        '运行：bash AlphaRAD/local_setup/python_local.sh -m unittest experiments.supervised_baseline_classification.test_components experiments.classification_uncertainty.test_bootstrap -v',
        '重建汇总：bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/report.py',
        '从新目录复跑：prepare.py --run-dir <new_directory>，随后run.py --run-dir <new_directory>。已锁定目录禁止覆盖。',
        '外部迁移结果及局限见external/REPORT.txt；实验代码冻结快照和哈希见PROTOCOL.json及source/。']
    (run/'REPORT.txt').write_text('\n'.join(text)+'\n');print('AUDITED',len(records),'evaluations;54newfits')


if __name__=='__main__':main()
