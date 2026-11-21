"""Collect paired macro differences without hiding invalid bootstrap draws."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parent/'runs/20261120');a=p.parse_args()
    reports=[]
    for scope in ('main','supervised'):
        for dataset in ('openi','chexpert','chestxray14'):
            path=a.run_dir/scope/dataset/'summary.json';d=json.loads(path.read_text());assert d['status']=='complete'
            reports.append((scope,dataset,d))
    path=a.run_dir/'rsna_transfer/summary.json'
    if path.exists():reports.append(('external','rsna_transfer',json.loads(path.read_text())))
    output=[];text=['成对分组bootstrap：差值及区间均为百分点（方法A减方法B）。',
        '每个数据集共享2000次组抽样；所有seed与方法使用相同的抽样次数向量。每个checkpoint先计算指标，再平均seed指标，不集成预测。',
        'AUROC与AP均为图像层面的排序指标，组抽样保留组内全部图像。AP采用average_precision，不是梯形PR积分。',
        '任何类别缺正/负均使该次完整macro无效；不重抽、不临时减少类别。标注valid<2000的区间仅是条件区间；尤其不能据此声称稀有病结果稳定。',
        'OpenI是报告/精确重复组，未验证纵向患者身份。CheXpert/ChestXray14/RSNA使用已验证患者分组。',
        '区间仅反映固定模型的测试组抽样不确定性；未覆盖训练队列/seed变化，未做多重比较校正。','']
    for scope,dataset,d in reports:
        text.append(scope+' / '+dataset)
        if 'per_class_counts' in d['protocol']:
            rare=[c+':'+str(v['positive_groups']) for c,v in d['protocol']['per_class_counts'].items() if v['positive_groups']<5]
            if rare:text.append('少于5个阳性组的类别：'+', '.join(rare))
        for c in d['comparisons']:
            cells=[]
            for metric,v in c['macro'].items():
                ci=v['ci95'];lo=100*ci[0] if ci else None;hi=100*ci[1] if ci else None
                output.append(dict(scope=scope,dataset=dataset,model=c['model'],method_a=c['method_a'],method_b=c['method_b'],metric=metric,
                    delta_pp=100*v['point'],ci95_lower_pp=lo,ci95_upper_pp=hi,valid_replicates=v['valid_replicates'],total_replicates=v['total_replicates'],
                    conditional=v['conditional_on_all_classes_present']))
                cells.append(f"{metric}: {100*v['point']:.2f} [{lo:.2f}, {hi:.2f}]" if ci else f"{metric}: interval undefined")
            text.append(c['model']+' '+c['method_a']+' - '+c['method_b']+' | '+' | '.join(cells)+f" | valid={v['valid_replicates']}/{v['total_replicates']}")
        text.append('')
    with (a.run_dir/'all_macro_differences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    (a.run_dir/'REPORT.txt').write_text('\n'.join(text)+'\n');print('Reported',len(output),'paired metric contrasts')


if __name__=='__main__':main()
