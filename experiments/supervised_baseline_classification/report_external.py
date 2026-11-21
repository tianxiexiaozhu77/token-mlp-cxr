"""Audit and summarize cross-dataset transfer with its patient-level intervals."""
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def main():
    run=DEFAULT_RUN;out=run/'external';p=json.loads((out/'PROTOCOL.json').read_text())
    ci_path=ROOT/'experiments/classification_uncertainty/runs/20261120/rsna_transfer/summary.json'
    ci=json.loads(ci_path.read_text());assert ci['status']=='complete'
    audit=json.loads((out/'cohort_audit.json').read_text());assert audit['status']=='passed'
    records=[];summaries=[];source_checks={};plans={}
    for model in MODELS:
        root=out/'results'/model;summary=json.loads((root/'summary.json').read_text());assert summary['status']=='complete'
        assert summary['images']==5297 and summary['patients']==3713 and summary['frozen_verified']
        assert summary['native_readout_max_abs_error']<=3e-4
        source_checks[model]=summary['source_reencoding_checks'];plan=json.loads((root/'evaluation_plan.json').read_text())
        assert sha256(root/'evaluation_plan.json')==summary['evaluation_plan_sha256'];plans[model]=sha256(root/'evaluation_plan.json')
        for file,digest in plan['code_sha256'].items():assert sha256(file)==digest
        for source in plan['sources'].values():
            folder=Path(source['folder']);assert sha256(folder/'result.json')==source['result_sha256']
            if source['checkpoint_sha256']:assert sha256(folder/'adapter.pt')==source['checkpoint_sha256']
        for method in p['methods']:
            values=[r for r in summary['results'] if r['method']==method]
            assert len(values)==(1 if method in ('original','global_frozen') else 3)
            for r in values:
                path=root/f"seed{r['seed']}"/method/'test_predictions.npz';assert sha256(path)==r['predictions_sha256']
                records.append(dict(model=model,method=method,seed=r['seed'],parameters=r['trainable_parameters'],
                    auroc_percent=100*r['test']['macro_auroc'],ap_percent=100*r['test']['macro_auprc']))
            mean=np.mean([[v['test']['macro_auroc'],v['test']['macro_auprc']] for v in values],axis=0)*100
            sd=np.std([[v['test']['macro_auroc'],v['test']['macro_auprc']] for v in values],axis=0,ddof=1)*100 if len(values)>1 else [None,None]
            intervals=next(r['metrics'] for r in ci['performance'] if r['model']==model and r['method']==method)
            for i,k in enumerate(('auroc','ap')):
                assert intervals[k]['valid_replicates']==2000 and abs(intervals[k]['point']*100-mean[i])<1e-10
            summaries.append(dict(model=model,method=method,parameters=values[0]['trainable_parameters'],seeds=len(values),
                auroc_mean=mean[0],auroc_std=sd[0],auroc_ci_lower=100*intervals['auroc']['ci95'][0],auroc_ci_upper=100*intervals['auroc']['ci95'][1],
                ap_mean=mean[1],ap_std=sd[1],ap_ci_lower=100*intervals['ap']['ci95'][0],ap_ci_upper=100*intervals['ap']['ci95'][1]))
    assert len(records)==42 and len(summaries)==18
    for path,rows in [(out/'all_runs.csv',records),(out/'transfer_comparison.csv',summaries)]:
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['RSNA外部迁移：5297张图，3713位NIH患者，1201张阳性，4096张阴性。',
        '仅使用CheXpert训练/验证所确定的模型，不在RSNA上训练、调参或选择checkpoint；保留源12类查询及其顺序，只读取Pneumonia列。',
        '目标样本、标签及患者映射已在新增训练前锁定，映射逐项核验；与全部668张CheXpert图像未发现文件或解码RGB像素完全一致的样本。',
        '这是此前本地其他研究使用过的固定RSNA队列，不是全新盲测队列，也不是官方RSNA测试协议。NIH患者与Stanford CheXpert适配队列分属不同数据源；无法进行跨机构真实身份链接，也未排除预训练集重叠。',
        'RSNA源自NIH，因此没有使用ChestXray14训练的适配器作独立迁移验证。该单一肺炎/阴影终点不能代替更大规模的12标签CheXpert外部验证。',
        '所有3模型×14条件=42次评估已完成；重编码源TEST的首/中/末3张图，逐条件核对原始预测后才继续推理。冻结状态及目标首尾原生读出一致性均通过。',
        '以下均为AUROC/AP百分数，训练seed先单独算指标再取均值；完整std和患者bootstrap区间见transfer_comparison.csv。','']
    for model in MODELS:
        text.append(model+' | '+' | '.join(r['method']+f" {r['auroc_mean']:.2f}/{r['ap_mean']:.2f}" for r in summaries if r['model']==model))
    text+=['','成对差值（百分点；Pre减去对照；括号为2000次患者分组bootstrap的95%百分位区间）：']
    for c in ci['comparisons']:
        cells=[]
        for k,v in c['macro'].items():
            assert v['valid_replicates']==2000
            cells.append(f"{k} {100*v['point']:.2f} [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}]")
        text.append(c['model']+' Pre - '+c['method_b']+' | '+' | '.join(cells))
    text+=['','区间只度量固定源模型的目标患者抽样不确定性，未覆盖训练集/训练seed变化，也未做多重比较校正。',
        '若出现跨域下降，必须如实报告；本实验不预设Token MLP、全局适配或线性探针始终更优。',
        '重算区间：bash AlphaRAD/local_setup/python_local.sh experiments/classification_uncertainty/external.py --output <new_output_directory>（禁止覆盖现有结果）',
        '复核报告：bash AlphaRAD/local_setup/python_local.sh experiments/supervised_baseline_classification/report_external.py']
    (out/'REPORT.txt').write_text('\n'.join(text)+'\n')
    save(out/'audit.json',dict(status='passed',evaluations=42,models=3,images=5297,patients=3713,
        target_predictions_hashes_verified=True,source_checkpoint_hashes_verified=True,source_reencoding_checks=source_checks,
        all_bootstrap_replicates_valid=True,cohort_audit_sha256=sha256(out/'cohort_audit.json'),ci_sha256=sha256(ci_path),evaluation_plans_sha256=plans,
        code_sha256=sha256(__file__)))
    additional={}
    for source in list(Path(__file__).parent.glob('*.py'))+list((ROOT/'experiments/classification_uncertainty').glob('*.py')):
        dest=run/'source_additional'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        additional[str(source)]=sha256(source)
    save(run/'additional_source_sha256.json',additional)
    print('AUDITED42transfer evaluations;2000/2000valid patient draws')


if __name__=='__main__':main()
