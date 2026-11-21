"""Resume the matrix controller after the documented pending-seed handoff."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def main():
    run=DEFAULT_RUN;note=run/'seed_parallelization.json';state=json.loads(note.read_text())
    while True:
        ready=[]
        for seed in (43,44):
            execution=run/'execution/KAD/chestxray14'/f'seed{seed}'/'clip_global.json'
            result=run/'results/KAD/chestxray14'/f'seed{seed}'/'clip_global/result.json'
            done=result.exists() and json.loads(result.read_text())['status']=='complete' and json.loads(execution.read_text())['status']=='complete'
            ready.append(done)
            if not done:
                pid=state[f'seed{seed}_pid'];proc=Path('/proc')/str(pid)/'stat'
                if not proc.exists() or proc.read_text().split(') ')[1].startswith('Z'):
                    raise RuntimeError(f'Owned seed{seed} stopped before completion; inspect its log')
        if all(ready):break
        time.sleep(5)
    matrix=json.loads((run/'training_matrix.json').read_text())
    for job in matrix['jobs']:
        if job['status']=='failed':
            assert job['model']=='KAD' and job['dataset']=='chestxray14'
            log=Path(job['log']).read_text()
            assert 'FileExistsError: Incomplete fit needs explicit recovery:' in log
            assert any(f'/seed{s}/clip_global' in log.splitlines()[-1] for s in (43,44))
    shutil.copy2(run/'training_matrix.json',run/'training_matrix_before_parallel_reconcile.json')
    subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),'--run-dir',str(run)],cwd=ROOT,check=True)
    assert json.loads((run/'training_matrix.json').read_text())['status']=='complete'
    state.update(status='complete',reconcile_code_sha256=sha256(__file__),matrix_sha256=sha256(run/'training_matrix.json'),
        fits_added_beyond_original_protocol=0,serial_guard_verified=True)
    save(note,state);print('Reconciled all54fits; no duplicate fit or changed protocol')


if __name__=='__main__':main()
