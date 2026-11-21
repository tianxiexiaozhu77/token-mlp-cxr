"""RadZero CLS controls and native, disease-conditioned post-pooling contexts."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.radzero_supervised_classification.common import *
from experiments.direct_anchor_classification.model import FrozenClassificationHead
from experiments.supervised_baseline_classification.components import (
    LinearProbe, ClipGlobalAdapter, make_module as make_global_module)


def global_scores(model, backend, features):
    assert model == 'RadZero'
    # Global CLS replaces the query-conditioned context. Apply the published
    # frozen visual LayerNorm before the native cosine / temperature readout.
    if backend.ln_weight is not None:
        features = F.layer_norm(features, (backend.dim,), backend.ln_weight,
                                backend.ln_bias, backend.config['layer_norm_eps'])
    return backend.score(features[:, None, :].expand(-1, len(backend.queries), -1))


def make_module(method, features, train_ids, classes):
    return residual_mlp(768, 64) if method == 'post_mlp' else make_global_module(method, features, train_ids, classes)


def load_global(model, dataset, run):
    assert model == 'RadZero'
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    spec = protocol['datasets'][dataset]
    meta, rows = read_dataset(spec['path'])
    assert sha256(spec['path']) == spec['sha256']
    assert meta['samples_sha256'] == spec['samples_sha256']
    sources = protocol['caches'][dataset]
    records = []
    for source in sources:
        assert sha256(source['path']) == source['sha256']
        record = json.loads(Path(source['path']).read_text())
        assert record['status'] == 'complete' and record['processed'] == len(record['indices'])
        assert record['dataset_sha256'] == spec['sha256']
        assert record['samples_sha256'] == spec['samples_sha256']
        for f, info in source['files'].items():
            stat = Path(f).stat()
            assert stat.st_size == info['bytes'] and stat.st_mtime_ns == info['mtime_ns'], f
            if info.get('sha256'):
                assert sha256(f) == info['sha256'], f
        records.append(record)
    assert len(records) == records[0]['shards']
    config = json.loads(Path(records[0]['head_path']).read_text())
    assert config['model'] == 'radzero' and config['class_names'] == meta['class_names']
    backend = FrozenClassificationHead(config).cuda().eval().requires_grad_(False)
    contexts = np.empty((len(rows), len(meta['class_names']), 768), dtype=np.float32)
    native = np.empty((len(rows), len(meta['class_names'])), dtype=np.float32)
    features = np.empty((len(rows), 768), dtype=np.float32)
    seen, token_seen = set(), set()
    for record in records:
        assert record['head_sha256'] == records[0]['head_sha256']
        assert record['checkpoint_sha256'] == records[0]['checkpoint_sha256']
        ids = record['indices']
        assert not seen.intersection(ids)
        seen.update(ids)
        contexts[ids] = np.load(record['context_path'], mmap_mode='r')
        native[ids] = np.load(record['logits_path'], mmap_mode='r')
        for entry in record['token_shards']:
            a = np.load(entry['path'], mmap_mode='r')
            assert list(a.shape) == entry['shape'] and a.dtype == np.float32
            assert not token_seen.intersection(entry['indices'])
            token_seen.update(entry['indices'])
            features[entry['indices']] = a[:, 0, :]
            # Independently recompute contexts from cached full tokens, including
            # singleton batches, before relying on the compact context cache.
            sample = sorted(set([0, len(a)//2, len(a)-1]))
            with torch.no_grad():
                t = torch.from_numpy(np.array(a[sample], copy=True)).cuda()
                c = backend.context(t).cpu().numpy()
            np.testing.assert_allclose(c, contexts[np.asarray(entry['indices'])[sample]], atol=3e-5, rtol=3e-5)
    assert seen == token_seen == set(range(len(rows)))
    with torch.no_grad():
        replay = np.concatenate([backend.score(torch.from_numpy(contexts[s:s+64]).cuda()).cpu().numpy()
                                 for s in range(0, len(rows), 64)])
    np.testing.assert_allclose(replay, native, atol=3e-5, rtol=3e-5)
    backend.cached_contexts = torch.from_numpy(contexts).cuda()
    backend.cached_native = native
    out = run / 'features/RadZero' / dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'cache.json').exists():
        saved = json.loads((out / 'cache.json').read_text())
        assert saved['sources'] == sources
        assert sha256(out / 'global.npy') == saved['global_sha256']
        np.testing.assert_array_equal(np.load(out / 'global.npy'), features)
    else:
        np.save(out / 'global.npy', features)
        save(out / 'cache.json', dict(status='complete', sources=sources,
            definition=protocol['global_features']['RadZero'], shape=list(features.shape),
            global_sha256=sha256(out / 'global.npy'), native_replay_max_error=float(np.abs(replay-native).max())))
    assert np.isfinite(features).all() and np.isfinite(contexts).all()
    return backend, features, meta, rows
