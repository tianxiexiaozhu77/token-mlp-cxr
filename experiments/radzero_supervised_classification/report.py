"""Audit RadZero fits, paired differences and external transfer; join four models."""
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.radzero_supervised_classification.common import *
from experiments.classification_uncertainty.run import load_predictions


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(dataset, method, records):
    return dict(dataset=dataset, model='RadZero', method=method, parameters=records[0]['parameters'], seeds=len(records),
        auroc_mean=float(np.mean([r['auroc_percent'] for r in records])),
        auroc_std=float(np.std([r['auroc_percent'] for r in records],ddof=1)) if len(records)>1 else None,
        ap_mean=float(np.mean([r['ap_percent'] for r in records])),
        ap_std=float(np.std([r['ap_percent'] for r in records],ddof=1)) if len(records)>1 else None,
        epoch_zero_selections=sum(r.get('best_epoch')==0 for r in records) if len(records)>1 else None)


def main():
    torch.set_num_threads(4)
    run=DEFAULT_RUN; protocol=json.loads((run/'PROTOCOL.json').read_text());verify_sources(protocol)
    ph=sha256(run/'PROTOCOL.json')
    assert json.loads((run/'training_matrix.json').read_text())['status']=='complete'
    for ref in protocol['references']:
        for f,h in ref['files_sha256'].items():assert sha256(Path(ref['source'])/f)==h
    records=[];summary=[];sources=[];normalizations=[];orders={};new_fits=0
    for dataset,spec in protocol['datasets'].items():
        meta,rows=read_dataset(spec['path']);assert sha256(spec['path'])==spec['sha256']
        classes=meta['class_names'];indices=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
        train_ids=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
        for method in ('original','global_frozen','linear_probe','clip_global','post_mlp','pre_mlp'):
            method_rows=[]
            for seed in ((42,) if method in ('original','global_frozen') else SEEDS):
                folder=result_folder(run,dataset,method,seed)
                values,source=load_predictions(folder,rows,indices,classes);sources.append(source)
                result=json.loads((folder/'result.json').read_text())
                expected=0 if method in ('original','global_frozen') else 769*len(classes) if method=='linear_probe' else 294912 if method=='clip_global' else 99136
                assert result['trainable_parameters']==expected
                if method in METHODS:
                    new_fits+=1
                    assert result['protocol_sha256']==ph and result['dataset_sha256']==spec['sha256']
                    assert result['frozen_backbone_verified'] and not result['test_used_for_selection']
                    assert result['full_checkpoint_replay_max_error']<=1e-6
                    training=json.loads((folder/'training.json').read_text());history=training['history']
                    val=[training['initial_val']['macro_auroc']]+[e['val']['macro_auroc'] for e in history]
                    assert int(np.argmax(val))==result['best_epoch']
                    assert abs(max(val)-result['best_val']['macro_auroc'])<1e-12
                    assert 8<=len(history)<=20 and result['config']==protocol['train']
                    if len(history)<20:assert len(history)-result['best_epoch']>=5
                    assert all(e<8 or e-int(np.argmax(val[:e+1]))<5 for e in range(1,len(history)))
                    for e in history:
                        key=(dataset,seed,e['epoch']);digest=e['train_order_sha256']
                        assert orders.setdefault(key,digest)==digest
                    saved=torch.load(folder/'adapter.pt',map_location='cpu',weights_only=True)
                    assert saved['best_epoch']==result['best_epoch'] and saved['protocol_sha256']==ph
                    st=saved['module_state_dict']
                    assert sum(v.numel() for k,v in st.items() if not k.startswith('feature_'))==expected
                    train_y=np.array([rows[i]['labels'] for i in train_ids],dtype=np.float32)
                    weights=np.minimum((len(train_y)-train_y.sum(0))/np.maximum(train_y.sum(0),1),20.)
                    np.testing.assert_array_equal(training['pos_weight'],weights)
                    if method=='linear_probe':
                        features=np.load(run/'features/RadZero'/dataset/'global.npy')
                        train=torch.from_numpy(features[train_ids]);mu=train.mean(0);sd=train.std(0,unbiased=False).clamp_min(1e-6)
                        torch.testing.assert_close(st['feature_mean'],mu,atol=1e-5,rtol=1e-5)
                        torch.testing.assert_close(st['feature_std'],sd,atol=1e-5,rtol=1e-5)
                        normalizations.append(dict(dataset=dataset,seed=seed,train_only=True))
                row=dict(dataset=dataset,model='RadZero',method=method,seed=seed,parameters=expected,
                    best_epoch=result['best_epoch'],auroc_percent=100*result['test']['macro_auroc'],ap_percent=100*result['test']['macro_auprc'])
                records.append(row);method_rows.append(row)
            summary.append(summarize(dataset,method,method_rows))
    assert new_fits==27 and len(records)==42 and len(summary)==18
    write_csv(run/'all_runs.csv',records);write_csv(run/'supervised_comparison.csv',summary)
    with (PARENT_RUN/'supervised_comparison.csv').open() as f:parent_rows=list(csv.DictReader(f))
    write_csv(run/'four_model_supervised_comparison.csv',parent_rows+summary)
    # External evaluation identity, metrics, serialized-source replay and cohort.
    ep=json.loads((run/'external/PROTOCOL.json').read_text())
    external_summary=json.loads((run/'external/results/RadZero/summary.json').read_text())
    assert external_summary['status']=='complete' and len(external_summary['results'])==14
    assert external_summary['frozen_verified'] and external_summary['native_readout_max_abs_error']<3e-4
    assert max(external_summary['source_reencoding_checks']['max_abs_error'].values())<3e-4
    target=json.loads(Path(ep['samples_file']).read_text());external_records=[];external_table=[]
    for method in ep['methods']:
        method_rows=[]
        for seed in ((42,) if method in ('original','global_frozen') else SEEDS):
            folder=run/'external/results/RadZero'/f'seed{seed}'/method
            values,source=load_predictions(folder,target,np.arange(len(target)),['Pneumonia']);sources.append(source)
            result=json.loads((folder/'result.json').read_text())
            assert result['predictions_sha256']==sha256(folder/'test_predictions.npz')
            saved=np.load(folder/'test_predictions.npz')
            np.testing.assert_array_equal(saved['logits'],saved['source_logits'][:,[ep['source_column']]])
            source_folder=result_folder(run,'chexpert',method,seed)
            assert sha256(source_folder/'result.json')==result['source']['result_sha256']
            if method not in ('original','global_frozen'):
                assert sha256(source_folder/'adapter.pt')==result['source']['checkpoint_sha256']
            row=dict(dataset='rsna_transfer',model='RadZero',method=method,seed=seed,parameters=result['trainable_parameters'],
                best_epoch=result['source']['best_epoch'],auroc_percent=100*result['test']['macro_auroc'],ap_percent=100*result['test']['macro_auprc'])
            external_records.append(row);method_rows.append(row)
        external_table.append(summarize('rsna_transfer',method,method_rows))
    write_csv(run/'external/all_runs.csv',external_records)
    write_csv(run/'external/transfer_comparison.csv',external_table)
    with (PARENT_RUN/'external/transfer_comparison.csv').open() as f:
        old_external=list(csv.DictReader(f))
    joint_fields=['model','method','parameters','seeds','auroc_mean','auroc_std','ap_mean','ap_std']
    write_csv(run/'four_model_transfer_comparison.csv',
              [{k:r[k] for k in joint_fields} for r in old_external+external_table])
    ci_rows=[];ci_text=[]
    for dataset in (*DATASETS,'rsna_transfer'):
        path=run/'uncertainty'/('rsna_transfer' if dataset=='rsna_transfer' else 'supervised/'+dataset)/'summary.json'
        ci=json.loads(path.read_text());assert ci['status']=='complete' and len(ci['comparisons'])==4
        table=external_table if dataset=='rsna_transfer' else [r for r in summary if r['dataset']==dataset]
        for comp in ci['comparisons']:
            a=next(r for r in table if r['method']==comp['method_a']);b=next(r for r in table if r['method']==comp['method_b'])
            for metric,column in [('auroc','auroc_mean'),('ap','ap_mean')]:
                v=comp['macro'][metric];assert abs(100*v['point']-(a[column]-b[column]))<1e-10
                ci_rows.append(dict(dataset=dataset,model='RadZero',difference='pre_mlp minus '+comp['method_b'],metric=metric,
                    delta_pp=100*v['point'],ci95_lower_pp=100*v['ci95'][0],ci95_upper_pp=100*v['ci95'][1],
                    valid_replicates=v['valid_replicates'],total_replicates=v['total_replicates'],conditional=v['conditional_on_all_classes_present']))
                ci_text.append(f"{dataset} Pre - {comp['method_b']} | {metric} {100*v['point']:+.2f} [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}] | valid={v['valid_replicates']}/2000")
    assert len(ci_rows)==32
    write_csv(run/'all_macro_differences.csv',ci_rows)
    text=['RadZero补充实验完成：27次新增训练（3数据集×Linear/CLIP-style global/Post×3seed），12次已有Original/Pre评估，3次global_frozen对照，14次RSNA迁移评估。',
        '以下AUROC/AP均为百分数；训练方法先计算每个seed的指标再平均，不集成预测。CSV包含样本标准差与参数量。',
        'Pre/Post均99136参数；CLIP-style global294912；Linear769×类别数。保持原有标签、划分、优化及验证选点预算；不代表各基线已调到最优。',
        '全局基线使用768维对齐后CLS，适配器之后保留原生LayerNorm、冻结文本查询和温度；改变了VL-CABS聚合路径，不能将全部差距归因于MLP结构。',
        'CLIP-style是官方结构的多标签移植，不称原版CLIP-Adapter复现。',
        'Post在VL-CABS聚合后、最终余弦归一化前；Pre使用原有同协议3seed结果。','']
    for dataset in (*DATASETS,'rsna_transfer'):
        table=external_table if dataset=='rsna_transfer' else [r for r in summary if r['dataset']==dataset]
        text.append(dataset+' | '+' | '.join(f"{r['method']} {r['auroc_mean']:.2f}/{r['ap_mean']:.2f}" for r in table))
    text+=['','成对差值及95%患者/报告组bootstrap区间（百分点；2000次，同一组抽样用于所有方法与seed）：']+ci_text
    text+=['','OpenI按报告/精确重复组，不能称患者级；完整macro有效970/2000。CheXpert仅312/2000、ChestXray14 1891/2000，缺失类别不删不重抽，区间是条件区间，尤其不能据CheXpert声称可靠无条件95%覆盖。',
        'RSNA：5297张图、3713位NIH患者、1201阳性。仅使用CheXpert训练并由其VAL选择的checkpoint；保持全部12个源查询，仅评估Pneumonia列。目标训练/调参/选点均为零。',
        'RSNA为先前研究用过的本地固定队列，不是新盲测或官方测试协议；单一肺炎/阴影终点不能替代完整12类验证；未排除预训练重叠。RadZero扩展在已知父实验结果后登记，不称完全盲测。',
        '区间只反映固定模型的测试组抽样不确定性，未覆盖训练集/seed总体变化，未做多重比较校正。',
        '新增结果单独保存，旧实验和论文文件未覆盖。完整四模型同域表：four_model_supervised_comparison.csv。']
    (run/'REPORT.txt').write_text('\n'.join(text)+'\n')
    external_line='RSNA AUROC/AP (%) | '+' | '.join(f"{r['method']} {r['auroc_mean']:.2f}/{r['ap_mean']:.2f}" for r in external_table)
    (run/'external/REPORT.txt').write_text(external_line+'\n\n'+'\n'.join(text[-5:])+'\n\n'+'\n'.join(ci_text[-8:])+'\n')
    audit=dict(status='passed',new_fits=27,historical_evaluations=12,global_frozen_controls=3,external_evaluations=14,
        comparisons=16,metric_differences=32,all_prediction_membership_and_metrics_verified=True,
        validation_selection_verified=True,parameter_counts_verified=True,checkpoint_replays_verified=True,
        shared_new_method_training_orders_verified=True,train_only_normalizations=normalizations,
        external_source_reencoding_checks=external_summary['source_reencoding_checks'],sources=sources,
        protocol_sha256=ph,report_code_sha256=sha256(__file__))
    save(run/'audit.json',audit)
    # Snapshot code added after training registration as well as registered code.
    additional={}
    for path in Path(__file__).parent.glob('*.py'):
        destination=run/'source_final'/path.name;destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path,destination);additional[str(path)]=sha256(path)
    save(run/'final_source_sha256.json',additional)
    paths=[run/p for p in ['PROTOCOL.json','REPORT.txt','all_runs.csv','supervised_comparison.csv',
        'four_model_supervised_comparison.csv','four_model_transfer_comparison.csv','all_macro_differences.csv','audit.json','final_source_sha256.json','environment.json',
        'external/PROTOCOL.json','external/REPORT.txt','external/all_runs.csv','external/transfer_comparison.csv']]
    save(run/'COMPLETION.json',dict(status='complete',new_fits=27,external_evaluations=14,paired_comparisons=16,
        artifact_sha256={str(p.relative_to(run)):sha256(p) for p in paths}))
    print('AUDITED RadZero:27 fits,14 external evaluations,16 paired comparisons')


if __name__=='__main__':main()
