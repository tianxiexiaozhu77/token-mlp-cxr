"""Freeze RadZero extension before fitting; preserve the completed parent run."""
import argparse
import json
import shutil
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.radzero_supervised_classification.common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    run = parser.parse_args().run_dir.resolve()
    if run.exists():
        raise FileExistsError(run)
    parent = json.loads((PARENT_RUN / 'PROTOCOL.json').read_text())
    datasets = parent['datasets']
    references, caches = [], {}
    for dataset, spec in datasets.items():
        meta, rows = read_dataset(spec['path'])
        assert sha256(spec['path']) == spec['sha256']
        for method in ('original', 'pre_mlp'):
            for seed in ((42,) if method == 'original' else SEEDS):
                folder = result_folder(run, dataset, method, seed)
                result = json.loads((folder / 'result.json').read_text())
                assert result['status'] == 'complete'
                assert result['dataset_sha256'] == spec['sha256']
                assert result['samples_sha256'] == spec['samples_sha256']
                files = ['result.json', 'test_predictions.npz']
                if method == 'pre_mlp':
                    assert result['trainable_parameters'] == 99136
                    for k in ('lr', 'weight_decay', 'epochs', 'min_epochs', 'patience'):
                        assert result['config'][k] == parent['train'][k]
                    assert result['config']['batch_size'] == parent['train']['effective_batch_size']
                    files += ['adapter.pt', 'training.json']
                references.append(dict(model='RadZero', dataset=dataset, method=method, seed=seed,
                    source=str(folder), files_sha256={f:sha256(folder/f) for f in files}))
        caches[dataset] = []
        for path in sorted((DIRECT/'cache/radzero'/dataset).glob('cache_*.json')):
            record = json.loads(path.read_text())
            assert record['dataset_sha256'] == spec['sha256']
            files = {}
            for file in [record['head_path'], record['context_path'], record['logits_path']]:
                stat = Path(file).stat()
                files[file] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(file))
            for entry in record['token_shards']:
                stat = Path(entry['path']).stat()
                files[entry['path']] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            assert sha256(ROOT/'RadZero/checkpoints/RadZero/model.safetensors') == record['checkpoint_sha256']
            caches[dataset].append(dict(path=str(path), sha256=sha256(path), files=files))
        assert caches[dataset]
    ep = json.loads((PARENT_RUN/'external/PROTOCOL.json').read_text())
    ep['models'] = ['RadZero']
    ep['extension_parent_protocol_sha256'] = sha256(PARENT_RUN/'external/PROTOCOL.json')
    ep['registration_note'] = 'Same fixed cohort as parent; RadZero extension requested after parent results were known. No RadZero target tuning.'
    save(run/'external/PROTOCOL.json', ep)
    # Reuse the completed cohort identity/duplicate audit after verifying its
    # hashes and every image's size/mtime. No images or groups are reselected.
    audit = json.loads((PARENT_RUN/'external/cohort_audit.json').read_text())
    assert audit['status'] == 'passed'
    assert audit['protocol_sha256'] == sha256(PARENT_RUN/'external/PROTOCOL.json')
    assert audit['image_hashes_sha256'] == sha256(PARENT_RUN/'external/image_hashes.json')
    image_hashes = json.loads((PARENT_RUN/'external/image_hashes.json').read_text())
    for group in ('source', 'target'):
        for entry in image_hashes[group]:
            stat = Path(entry['image']).stat()
            assert stat.st_size == entry['bytes'] and stat.st_mtime_ns == entry['mtime_ns']
    shutil.copy2(PARENT_RUN/'external/image_hashes.json', run/'external/image_hashes.json')
    audit['parent_audit_sha256'] = sha256(PARENT_RUN/'external/cohort_audit.json')
    audit['protocol_sha256'] = sha256(run/'external/PROTOCOL.json')
    save(run/'external/cohort_audit.json', audit)
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/f for f in [
        'experiments/supervised_baseline_classification/components.py',
        'experiments/baseline_token_classification/common.py',
        'experiments/both_adapter_classification/common.py',
        'experiments/direct_anchor_classification/model.py',
        'RadZero/scripts/python_local.sh', 'RadZero/scripts/eval_public.py',
        'RadZero/scripts/adapter_model.py', 'RadZero/scripts/direct_anchor_model.py',
        'RadZero/exp/cxr_pt/inference/dataset.py']]
    paths += list((ROOT/'RadZero/checkpoints/RadZero').glob('*.py'))
    paths += list((ROOT/'RadZero/checkpoints/RadZero').glob('*.json'))
    source_hashes = {str(p):sha256(p) for p in paths}
    protocol = dict(status='frozen', models=list(MODELS), methods=list(METHODS), seeds=list(SEEDS),
        parent_protocol_sha256=sha256(PARENT_RUN/'PROTOCOL.json'),
        parent_completion_sha256=sha256(PARENT_RUN/'COMPLETION.json'),
        datasets=datasets, train=parent['train'], allowed_physical_gpus=[4,5,6,7],
        references=references, caches=caches, source_sha256=source_hashes,
        external_protocol_sha256=sha256(run/'external/PROTOCOL.json'),
        global_features={'RadZero':'768-D aligned raw CLS; TRAIN-only standardization for Linear probe; CLIP-style adapter then frozen native visual LayerNorm, cosine with all frozen source queries, and native loss temperature.'},
        post_mlp='Shared 768->64->768 GELU residual, zero up initialization, after frozen VL-CABS attention and before final cosine normalization; 99136 parameters.',
        clip_global='Released CLIP-style visual adapter: bias-free 768->192->768 with two ReLUs, mixing 0.2; weighted multilabel BCE and native RadZero readout. 294912 parameters; not an exact original CLIP-Adapter reproduction.',
        scope='27 new fits + 12 historical evaluations + 3 frozen-global controls + 14 RSNA evaluations.',
        caveat='Global CLS replaces disease-conditioned aggregation. Comparisons change representation as well as adapter; matched Pre/Post isolates placement. Fixed budget does not guarantee optimal baseline tuning.')
    save(run/'PROTOCOL.json', protocol)
    for path in source_hashes:
        destination = run/'source'/Path(path).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print('LOCKED', run, '27 new fits; 14 RSNA evaluations')


if __name__ == '__main__':
    main()
