"""Reuse only verified frozen features, never previously fitted adapters."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.large_patient_multilabel.common import *


def hash_with_progress(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    total, reported = 0, 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
            total += len(block)
            if total - reported >= 2 * 1024**3:
                print(path.parent.parent.name, path.name, round(total/1024**3,1), '/',
                      round(before.st_size/1024**3,1), 'GiB hashed', flush=True)
                reported = total
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed while hashing: ' + str(path))
    return dict(path=str(path),bytes=after.st_size,mtime_ns=after.st_mtime_ns,sha256=h.hexdigest())


def prepare_features():
    meta, rows = load_data()
    old_meta = json.loads(SOURCE_DATA.read_text())
    old_rows = json.loads(Path(old_meta['samples_file']).read_text())
    for a, b in zip(rows, old_rows):
        for field in ('id','image','patient_id','labels','file_sha256'):
            if a[field] != b[field]:
                raise ValueError('Frozen cache row identity mismatch: ' + field)
    if len(rows) != len(old_rows):
        raise ValueError('Different source cohort')
    manifests, files = {}, set()
    for model in MODELS:
        records = []
        for path in sorted((SOURCE_CACHE/model.lower()/'chestxray14').glob('cache_*.json')):
            r = json.loads(path.read_text())
            if r['status'] != 'complete' or r['processed'] != len(r['indices']):
                raise ValueError('Incomplete cache')
            if r['dataset_sha256'] != sha256(SOURCE_DATA) or r['samples_sha256'] != old_meta['samples_sha256']:
                raise ValueError('Cache-source mismatch')
            if sha256(r['head_path']) != r['head_sha256']:
                raise ValueError('Head changed')
            files.update([path,Path(r['head_path']),Path(r['context_path']),Path(r['logits_path'])])
            files.update(Path(s['path']) for s in r['token_shards'])
            records.append(dict(path=str(path),sha256=sha256(path),record=r))
        if not records or len(records) != records[0]['record']['shards']:
            raise ValueError('Missing cache shards')
        manifests[model] = records
    checkpoints = {
        'AlphaRAD': ROOT/'AlphaRAD/local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar',
        'RadZero': ROOT/'RadZero/checkpoints/RadZero/model.safetensors'}
    files.update(checkpoints.values())
    integrity_path = RUN/'source_integrity.json'
    if integrity_path.exists():
        integrity = json.loads(integrity_path.read_text())
        if set(integrity) != {str(p) for p in files}:
            raise ValueError('Source integrity manifest coverage changed')
        for path, rec in integrity.items():
            stat=Path(path).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(rec['bytes'],rec['mtime_ns']):
                raise ValueError('Previously hashed source changed')
    else:
        ordered = sorted(files, key=lambda p:p.stat().st_size, reverse=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            values=list(pool.map(hash_with_progress, ordered))
        integrity={r['path']:r for r in values}
        save(integrity_path, integrity)
    for model, records in manifests.items():
        out = RUN/'features'/model
        out.mkdir(parents=True,exist_ok=True)
        if (out/'cache.json').exists():
            cache=json.loads((out/'cache.json').read_text())
            for path, digest in cache['compact_sha256'].items():
                if sha256(path)!=digest:
                    raise ValueError('Compact cache changed')
            continue
        context=np.empty((len(rows),len(meta['class_names']),768),dtype=np.float32)
        native=np.empty((len(rows),len(meta['class_names'])),dtype=np.float32)
        global_features=np.empty((len(rows),768),dtype=np.float32)
        seen, token_seen, specs=set(),set(),[]
        first=records[0]['record']
        if integrity[str(checkpoints[model])]['sha256']!=first['checkpoint_sha256']:
            raise ValueError('Base checkpoint mismatch')
        config=json.loads(Path(first['head_path']).read_text())
        if config['class_names']!=meta['class_names'] or config['model']!=model.lower():
            raise ValueError('Frozen class order/model mismatch')
        for source in records:
            r=source['record'];ids=r['indices']
            if seen.intersection(ids) or r['head_sha256']!=first['head_sha256'] or r['checkpoint_sha256']!=first['checkpoint_sha256']:
                raise ValueError('Mixed/overlapping cache')
            seen.update(ids)
            context[ids]=np.load(r['context_path'],mmap_mode='r')
            native[ids]=np.load(r['logits_path'],mmap_mode='r')
            for spec in r['token_shards']:
                if token_seen.intersection(spec['indices']):
                    raise ValueError('Duplicate tokens')
                token_seen.update(spec['indices'])
                a=np.load(spec['path'],mmap_mode='r')
                if list(a.shape)!=spec['shape'] or a.dtype!=np.float32:
                    raise ValueError('Invalid token array')
                for start in range(0,len(a),128):
                    global_features[spec['indices'][start:start+128]]=a[start:start+128,0,:]
                specs.append(spec)
        if seen!=token_seen or seen!=set(range(len(rows))):
            raise ValueError('Cache coverage differs')
        for name, value in [('contexts',context),('native',native),('global',global_features)]:
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite features')
            np.save(out/(name+'.npy'),value)
        save(out/'head.json',config)
        save(out/'cache.json',dict(status='complete',model=model,dataset_sha256=sha256(DATA/'dataset.json'),
            samples_sha256=meta['samples_sha256'],source_dataset_sha256=sha256(SOURCE_DATA),
            source_integrity_sha256=sha256(integrity_path),source_records=records,token_shards=specs,
            checkpoint_path=str(checkpoints[model]),checkpoint_sha256=first['checkpoint_sha256'],
            compact_sha256={str(p):sha256(p) for p in out.iterdir() if p.name!='cache.json'},
            reuse_scope='Frozen encoder outputs/native scoring only; all fitted modules retrained on NEW TRAIN'))
        print(model,'frozen features verified and indexed for new split',flush=True)


class FeatureSet:
    def __init__(self, run, model, device='cuda'):
        import torch
        from experiments.direct_anchor_classification.model import FrozenClassificationHead
        self.meta,self.rows=load_data()
        self.device=device
        self.ids={s:np.asarray([i for i,r in enumerate(self.rows) if r['split']==s],dtype=np.int64) for s in SPLITS}
        out=run/'features'/model
        cache=json.loads((out/'cache.json').read_text())
        if cache['dataset_sha256']!=sha256(DATA/'dataset.json'):
            raise ValueError('Feature split mismatch')
        for path,digest in cache['compact_sha256'].items():
            if sha256(path)!=digest:raise ValueError('Compact features changed')
        self.config=json.loads((out/'head.json').read_text())
        self.head=FrozenClassificationHead(self.config).to(device).eval().requires_grad_(False)
        self.contexts=torch.from_numpy(np.load(out/'contexts.npy')).to(device)
        self.global_features=torch.from_numpy(np.load(out/'global.npy')).to(device)
        self.native=np.load(out/'native.npy')
        self.arrays=[];self.locations={}
        integrity=json.loads((run/'source_integrity.json').read_text())
        for spec in cache['token_shards']:
            stat=Path(spec['path']).stat();rec=integrity[spec['path']]
            if (stat.st_size,stat.st_mtime_ns)!=(rec['bytes'],rec['mtime_ns']):
                raise ValueError('Frozen tokens changed')
            a=np.load(spec['path'],mmap_mode='r')
            for local,index in enumerate(spec['indices']):self.locations[index]=(len(self.arrays),local)
            self.arrays.append(a)
        if set(self.locations)!=set(range(len(self.rows))):raise ValueError('Incomplete token mapping')

    def tokens(self, index):
        a,j=self.locations[int(index)]
        return np.array(self.arrays[a][j],copy=True)


def make_module(method, features):
    import torch
    from experiments.supervised_baseline_classification.components import LinearProbe,ClipGlobalAdapter
    if method in ('pre_mlp','post_mlp'):
        return residual_mlp(768,64).to(features.device)
    if method=='clip_global':return ClipGlobalAdapter().to(features.device)
    if method=='linear_probe':
        x=features.global_features[features.ids['train']]
        return LinearProbe(x.mean(0),x.std(0,unbiased=False),len(features.meta['class_names'])).to(features.device)
    raise ValueError(method)


def forward(features, indices, module, method, tokens=None):
    import torch
    from torch.nn import functional as F
    if method=='original':return features.head.score(features.contexts[indices])
    if method=='pre_mlp':return features.head(tokens,token_adapter=module)
    if method=='post_mlp':return features.head.score(features.contexts[indices],module)
    x=features.global_features[indices]
    if method=='linear_probe':return module(x)
    if module is not None:x=module(x)
    if features.head.kind=='radzero' and features.head.ln_weight is not None:
        x=F.layer_norm(x,(768,),features.head.ln_weight,features.head.ln_bias,features.config['layer_norm_eps'])
    return features.head.score(x[:,None,:].expand(-1,len(features.head.queries),-1))


if __name__=='__main__':
    prepare_features()
