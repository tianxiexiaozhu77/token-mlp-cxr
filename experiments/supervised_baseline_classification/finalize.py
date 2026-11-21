"""Wait for all registered jobs, then generate and verify the final deliverables."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def main():
    run=DEFAULT_RUN
    while True:
        training=json.loads((run/'training_matrix.json').read_text())
        additional=json.loads((run/'additional_matrix.json').read_text())
        if additional['status']=='failed':raise RuntimeError('Additional job failed; inspect before finalization')
        if training['status']==additional['status']=='complete':break
        time.sleep(5)
    command=['bash',str(launcher('AlphaRAD'))]
    subprocess.run(command+[str(Path(__file__).with_name('report.py'))],cwd=ROOT,check=True)
    subprocess.run(command+[str(ROOT/'experiments/classification_uncertainty/report.py')],cwd=ROOT,check=True)
    assert json.loads((run/'audit.json').read_text())['status']=='passed'
    assert json.loads((run/'external/audit.json').read_text())['status']=='passed'
    sources={}
    for path in list(Path(__file__).parent.glob('*.py'))+list((ROOT/'experiments/classification_uncertainty').glob('*.py')):
        dest=run/'source_additional'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        sources[str(path)]=sha256(path)
    save(run/'additional_source_sha256.json',sources)
    artifacts=[run/name for name in ('PROTOCOL.json','audit.json','all_runs.csv','supervised_comparison.csv','REPORT.txt','seed_parallelization.json','additional_source_sha256.json')]
    artifacts += [run/'external'/name for name in ('PROTOCOL.json','audit.json','cohort_audit.json','transfer_comparison.csv','REPORT.txt')]
    ci=ROOT/'experiments/classification_uncertainty/runs/20261120'
    artifacts += [ci/name for name in ('all_macro_differences.csv','REPORT.txt','rsna_transfer/pre_vs_original.pdf','rsna_transfer/pre_vs_original.png')]
    save(run/'COMPLETION.json',dict(status='complete',supervised_fits=54,historical_evaluations=63,
        global_frozen_controls=9,transfer_evaluations=42,paired_bootstrap_comparisons=60,
        artifacts_sha256={str(path):sha256(path) for path in artifacts},finalizer_sha256=sha256(__file__),
        limitations=['Fixed single-configuration budget, not optimized benchmarking of every baseline.',
            'CLIP-style native-readout transplant is not exact original CLIP-Adapter reproduction.',
            'CheXpert macro intervals have312/2000valid replicates and are conditional, not robust nominal95% guarantees.',
            'OpenI groups are reports/duplicates, not verified longitudinal patients.',
            'RSNA transfer is one endpoint on a previously used local cohort, not official larger12-label CheXpert testing.']))
    print('COMPLETE:54supervised fits,42transfer evaluations,60paired bootstrap comparisons',flush=True)


if __name__=='__main__':main()
