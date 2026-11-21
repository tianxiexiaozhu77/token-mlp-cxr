"""Derive compact global features from existing immutable projected-token caches."""
import json
from pathlib import Path
import numpy as np
import torch

from experiments.supervised_baseline_classification.common import *


def load_global(model, dataset, run):
    protocol=json.loads((run/'PROTOCOL.json').read_text());spec=protocol['datasets'][dataset]
    meta,rows=read_dataset(spec['path'])
    assert sha256(spec['path'])==spec['sha256'] and meta['samples_sha256']==spec['samples_sha256']
    out=run/'features'/model/dataset;out.mkdir(parents=True,exist_ok=True)
    provenance=[]
    if model=='AlphaRAD':
        records=[json.loads(p.read_text()) for p in sorted((DIRECT/'cache/alpharad'/dataset).glob('cache_*.json'))]
        assert records and len(records)==records[0]['shards']
        head_path=Path(records[0]['head_path'])
        assert sha256(head_path)==records[0]['head_sha256']
        from experiments.direct_anchor_classification.model import FrozenClassificationHead
        backend=FrozenClassificationHead(json.loads(head_path.read_text())).cuda().eval().requires_grad_(False)
        token_specs=[]
        for record in records:
            assert record['status']=='complete' and record['dataset_sha256']==spec['sha256']
            assert record['samples_sha256']==spec['samples_sha256']
            assert record['head_sha256']==records[0]['head_sha256']
            token_specs.extend(record['token_shards'])
        provenance=[dict(head_path=str(head_path),head_sha256=sha256(head_path),
            checkpoint_sha256=records[0]['checkpoint_sha256'],tokens=token_specs)]
    else:
        cache_path=BASE/'cache'/model/dataset/'cache.json';cache=json.loads(cache_path.read_text())
        assert cache['status']=='complete' and cache['dataset_sha256']==spec['sha256']
        assert cache['samples_sha256']==spec['samples_sha256'] and cache['class_names']==meta['class_names']
        for entry in cache['files'].values():
            stat=Path(entry['path']).stat()
            assert stat.st_size==entry['bytes'] and stat.st_mtime_ns==entry['mtime_ns'],entry['path']
        backend=load_backend(model,'cuda');freeze_backend(backend);backend.prepare_classes(meta['class_names']);freeze_backend(backend)
        assert query_hashes(backend)==cache['query_sha256']
        for path,digest in cache['checkpoint_sha256'].items():
            assert sha256(path)==digest
        backend.release_image_encoder();torch.cuda.empty_cache()
        token_specs=[dict(path=cache['files']['tokens.npy']['path'],indices=list(range(len(rows))))]
        provenance=[dict(cache_path=str(cache_path),cache_sha256=sha256(cache_path),files=cache['files'],
            query_sha256=cache['query_sha256'],checkpoint_sha256=cache['checkpoint_sha256'])]
    if (out/'cache.json').exists():
        saved=json.loads((out/'cache.json').read_text())
        assert saved['dataset_sha256']==spec['sha256'] and saved['sources']==provenance
        assert sha256(out/'global.npy')==saved['global_sha256']
        features=np.load(out/'global.npy')
    else:
        features=np.empty((len(rows),768),dtype=np.float32);seen=set()
        for entry in token_specs:
            a=np.load(entry['path'],mmap_mode='r');ids=entry['indices']
            assert a.shape[0]==len(ids) and a.shape[-1]==768
            assert not seen.intersection(ids);seen.update(ids)
            for start in range(0,len(ids),128):
                end=min(start+128,len(ids))
                # CLS selection does not read the other 1369 AlphaRAD tokens.
                values=np.asarray(a[start:end,0]) if model=='AlphaRAD' else a[start:end].mean(axis=1)
                features[ids[start:end]]=values
        assert seen==set(range(len(rows))) and np.isfinite(features).all()
        np.save(out/'global.npy',features)
        save(out/'cache.json',dict(status='complete',dataset_sha256=spec['sha256'],samples_sha256=spec['samples_sha256'],
            definition=protocol['global_features'][model],shape=list(features.shape),sources=provenance,global_sha256=sha256(out/'global.npy')))
    assert features.shape==(len(rows),768) and np.isfinite(features).all()
    return backend,features,meta,rows
