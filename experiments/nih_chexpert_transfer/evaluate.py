"""Apply locked NIH checkpoints to freshly encoded CheXpert images; no fitting."""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nih_chexpert_transfer.common import *


def predict(features, tokens, method, module, batch_size):
    import torch
    from experiments.large_patient_multilabel.features import forward
    values = []
    before = {} if module is None else {k: v.clone() for k, v in module.state_dict().items()}
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            ids = np.arange(start, min(start + batch_size, len(tokens)))
            batch = (torch.from_numpy(np.array(tokens[ids], copy=True)).to(features.device)
                     if method == 'pre_mlp' else None)
            values.append(forward(features, ids, module, method, batch).cpu().numpy())
    if module is not None:
        for k, v in module.state_dict().items():
            if not torch.equal(v, before[k]):
                raise ValueError('Checkpoint state changed during inference: ' + k)
    values = np.concatenate(values)
    if values.shape != (len(tokens), 14) or not np.isfinite(values).all():
        raise ValueError('Invalid full-source-axis scores')
    return values


def evaluate_model(model, gpu):
    import torch
    from experiments.direct_anchor_classification.model import FrozenClassificationHead
    p = load_protocol()
    check_source()
    ph = sha256(RUN / 'PROTOCOL.json')
    rows = read(RUN / 'samples.json')
    cache_dir = RUN / 'cache' / model
    cache_path = cache_dir / 'cache_00.json'
    cache = read(cache_path)
    n = p['images'] + len(p['source_validation_check_indices'])
    if (cache['status'] != 'complete' or cache['processed'] != n or
            cache['indices'] != list(range(n)) or cache['shards'] != 1 or
            cache['model'] != model.lower()):
        raise ValueError('Fresh encoder coverage/model mismatch')
    for key, path in (('dataset_sha256', RUN / 'encoder_dataset.json'),
                      ('samples_sha256', RUN / 'encoder_samples.json'),
                      ('head_sha256', Path(cache['head_path']))):
        if cache[key] != sha256(path):
            raise ValueError('Fresh encoder provenance mismatch: ' + key)
    source_cache = read(SOURCE / 'features' / model / 'cache.json')
    config = read(cache['head_path'])
    if (config != read(SOURCE / 'features' / model / 'head.json') or
            cache['checkpoint_sha256'] != source_cache['checkpoint_sha256']):
        raise ValueError('Source encoder or full NIH scoring head changed')
    if len(cache['token_shards']) != 1:
        raise ValueError('Expected one complete fresh token shard')
    spec = cache['token_shards'][0]
    tokens = np.load(spec['path'], mmap_mode='r')
    if list(tokens.shape) != spec['shape'] or spec['indices'] != list(range(n)) or tokens.dtype != np.float32:
        raise ValueError('Invalid token array')
    files = [cache_path, Path(cache['head_path']), Path(cache['context_path']),
             Path(cache['logits_path']), Path(spec['path'])]
    integrity = {str(path): dict(sha256=sha256(path), bytes=path.stat().st_size,
                               mtime_ns=path.stat().st_mtime_ns) for path in files}
    save(RUN / 'evaluation' / (model + '_cache_integrity.json'), integrity)
    features = SimpleNamespace(device='cuda', config=config,
        head=FrozenClassificationHead(config).cuda().eval().requires_grad_(False),
        contexts=torch.from_numpy(np.load(cache['context_path'])).cuda(),
        global_features=torch.from_numpy(np.array(tokens[:, 0, :], copy=True)).cuda())
    native = np.load(cache['logits_path'])
    frozen_state = {k: v.clone() for k, v in features.head.state_dict().items()}
    context_error = 0.
    with torch.inference_mode():
        # Recompute every context, not merely the first/last encoded batches.
        for start in range(0, n, p['scoring_batch_size']):
            end = min(n, start + p['scoring_batch_size'])
            batch = torch.from_numpy(np.array(tokens[start:end], copy=True)).cuda()
            replay = features.head.context(batch)
            torch.testing.assert_close(replay, features.contexts[start:end], atol=3e-5, rtol=3e-5)
            context_error = max(context_error, float((replay - features.contexts[start:end]).abs().max()))
    original = predict(features, tokens, 'original', None, p['scoring_batch_size'])
    np.testing.assert_allclose(original, native, atol=3e-5, rtol=3e-5)
    save(RUN / 'evaluation' / (model + '_cache_parity.json'), dict(status='passed',
        all_contexts_max_error=context_error, all_native_logits_max_error=float(np.abs(original-native).max()),
        fresh_encoder_native_max_error=cache['native_max_abs_error'],
        source_head_exactly_equal=True, source_pretrained_checkpoint_equal=True))
    source_native = np.load(SOURCE / 'features' / model / 'native.npy', mmap_mode='r')
    k = len(p['source_validation_check_indices'])
    tolerance = p['numerical_tolerances']
    labels = np.asarray([r['labels'] for r in rows], dtype=np.float32)
    records = []
    for _, method, seed in jobs(model):
        started = time.time()
        out = folder(model, method, seed)
        if (out / 'result.json').exists():
            result = read(out / 'result.json')
            if result['protocol_sha256'] != ph or result['predictions_sha256'] != sha256(out / 'predictions.npz'):
                raise ValueError('Existing result changed')
            records.append(result)
            continue
        module, selection = None, None
        if method != 'original':
            module, selection = selected_module(p, model, method, seed, 'cuda')
        values = original if method == 'original' else predict(features, tokens, method, module, p['scoring_batch_size'])
        replay_error = 0.
        if module is not None:
            restored, _ = selected_module(p, model, method, seed, 'cuda')
            replay = predict(features, tokens, method, restored, p['scoring_batch_size'])
            np.testing.assert_allclose(replay, values, atol=tolerance['checkpoint_replay_atol'],
                                       rtol=tolerance['checkpoint_replay_rtol'])
            replay_error = float(np.abs(replay - values).max())
            del restored
            with np.load(source_folder(model, method, seed) / 'val_predictions.npz') as a:
                locs = p['source_validation_check_locations']
                np.testing.assert_array_equal(a['indices'][locs], p['source_validation_check_indices'])
                np.testing.assert_array_equal(a['class_names'], p['source_class_names'])
                source_reference = a['logits'][locs]
        else:
            source_reference = source_native[p['source_validation_check_indices']]
        np.testing.assert_allclose(values[:k], source_reference, atol=tolerance['source_reencoding_atol'],
                                   rtol=tolerance['source_reencoding_rtol'])
        target_scores = project(values[k:], p['source_class_names'])
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / 'predictions.npz', logits=target_scores, source_logits=values[k:], labels=labels,
            source_replay_logits=values[:k], source_replay_reference=source_reference,
            ids=np.asarray([r['id'] for r in rows]), patients=np.asarray([r['patient_id'] for r in rows]),
            groups=np.asarray([r['group_id'] for r in rows]), class_names=p['class_names'],
            source_class_names=p['source_class_names'])
        result = dict(status='complete', model=model, method=method, seed=seed, protocol_sha256=ph,
            source_selection=selection, source_head_sha256=sha256(SOURCE / 'features' / model / 'head.json'),
            source_checkpoint_sha256=cache['checkpoint_sha256'],
            target_used_for_training=False, target_used_for_selection=False,
            target_normalization_fitted=False, source_state_unchanged=True,
            test=metrics(labels, target_scores, p['class_names']),
            full_checkpoint_replay_max_error=replay_error,
            source_validation_reencoding_max_error=float(np.abs(values[:k] - source_reference).max()),
            predictions_sha256=sha256(out / 'predictions.npz'), seconds=time.time()-started, gpu=gpu)
        save(out / 'result.json', result)
        records.append(result)
        print('EVALUATED', model, method, seed, flush=True)
        del module
    for key, value in features.head.state_dict().items():
        if not torch.equal(value, frozen_state[key]):
            raise ValueError('Frozen NIH head changed')
    load_protocol(); check_source()
    for path, rec in integrity.items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (rec['bytes'], rec['mtime_ns']):
            raise ValueError('Fresh features changed during evaluation')
    save(RUN / 'evaluation' / (model + '_complete.json'), dict(status='complete', evaluations=len(records),
        protocol_sha256=ph, frozen_source_state_verified=True,
        results_sha256={str(folder(model, r['method'], r['seed']) / 'result.json'):
            sha256(folder(model, r['method'], r['seed']) / 'result.json') for r in records}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    lease, gpu = setup_gpu(args.gpu)
    try:
        evaluate_model(args.model, gpu)
    finally:
        release_gpu(lease)


if __name__ == '__main__':
    main()
