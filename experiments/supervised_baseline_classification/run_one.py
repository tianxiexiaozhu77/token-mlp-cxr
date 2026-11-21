"""Execute a single already-registered fit using the unchanged frozen trainer.

The old serial worker's incomplete-directory guard prevents a second writer
when it later reaches a seed owned here. Resume its matrix controller after
all owned fits finish; no optimization or checkpoint-selection rule changes.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args();run=a.run_dir.resolve()
    protocol=json.loads((run/'PROTOCOL.json').read_text());assert a.model in protocol['models'] and a.seed in protocol['seeds'] and a.method in protocol['methods']
    execution=run/'execution'/a.model/a.dataset/f'seed{a.seed}';execution.mkdir(parents=True,exist_ok=True)
    fitlock=(execution/(a.method+'.lock')).open('a+');fcntl.flock(fitlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fit=run/'results'/a.model/a.dataset/f'seed{a.seed}'/a.method
    if fit.exists():raise FileExistsError('Existing fit is owned by another worker: '+str(fit))
    state=dict(status='running',model=a.model,dataset=a.dataset,method=a.method,seed=a.seed,gpu_index=a.gpu,
        protocol_sha256=sha256(run/'PROTOCOL.json'),runner_sha256=sha256(__file__),
        training_entry_sha256=sha256(Path(__file__).with_name('train.py')),
        scope='Subset of existing frozen54-fit matrix; no extra seed/configuration and no trainer-source change.')
    save(execution/(a.method+'.json'),state)
    lock,gpu=setup_gpu(a.gpu)
    try:
        import experiments.supervised_baseline_classification.train as trainer
        trainer.SEEDS=(a.seed,);trainer.METHODS=(a.method,)
        trainer.worker(a,gpu)
        state.update(status='complete',result_sha256=sha256(fit/'result.json'))
        save(execution/(a.method+'.json'),state)
    finally:release_gpu(lock)


if __name__=='__main__':main()
