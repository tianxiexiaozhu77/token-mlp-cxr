"""Run fixed NIH-to-CheXpert inference and paired uncertainty; no retraining."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nih_chexpert_transfer.common import *


def execute(model, script, args, log_name):
    path = RUN / 'logs' / log_name
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ['bash', str(launcher(model)), '-u', str(ROOT / script), *map(str, args)]
    print('START', log_name, flush=True)
    with path.open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('COMPLETE', log_name, flush=True)
    return dict(command=command, log=str(path), log_sha256=sha256(path), status='passed')


def model_job(model, gpu):
    state_path = RUN / 'cache' / model / 'cache_00.json'
    load_protocol()
    if not state_path.exists():
        execute(model, 'experiments/direct_anchor_classification/cache.py',
            ['--model', model.lower(), '--dataset', RUN / 'encoder_dataset.json',
             '--output', state_path.parent, '--gpu', gpu, '--batch-size', 12], model + '_encode.log')
    elif read(state_path)['status'] != 'complete':
        raise ValueError('Incomplete cache requires investigation; not overwriting ' + str(state_path))
    execute(model, 'experiments/nih_chexpert_transfer/evaluate.py',
            ['--model', model, '--gpu', gpu], model + '_evaluate.log')


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'COMPLETION.json').exists():
            load_protocol(); check_source()
            for name, digest in read(RUN / 'COMPLETION.json')['output_sha256'].items():
                if sha256(RUN / name) != digest:
                    raise ValueError('Completed output changed: ' + name)
            print('Completed transfer verified; no jobs repeated.', flush=True)
            return
        save(RUN / 'STATUS.json', dict(status='running', started=time.time(), new_training_runs=0))
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                tests = list(pool.map(lambda m: execute(m, 'experiments/nih_chexpert_transfer/test_transfer.py',
                     [], m + '_tests.log'), MODELS))
            save(RUN / 'TEST_CHECKS.json', dict(status='passed', environments=tests))
            execute('AlphaRAD', 'experiments/nih_chexpert_transfer/prepare.py', [], 'prepare.log')
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(model_job, model, gpu) for model, gpu in zip(MODELS, (4, 5))]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(repr(error))
                if errors:
                    raise RuntimeError(errors)
            execute('AlphaRAD', 'experiments/nih_chexpert_transfer/report.py', [], 'report.log')
            save(RUN / 'STATUS.json', dict(status='complete', finished=time.time(), new_training_runs=0,
                 completion_sha256=sha256(RUN / 'COMPLETION.json')))
            print('TRANSFER COMPLETE: 26 evaluations, 8 paired comparisons; zero target fitting.', flush=True)
        except BaseException as error:
            save(RUN / 'STATUS.json', dict(status='failed', error=repr(error), updated=time.time()))
            raise


if __name__ == '__main__':
    main()
