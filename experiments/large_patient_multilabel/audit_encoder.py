"""Compare fresh image encodings against the reused token/context/logit cache."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *


def main():
    root=RUN/'encoder_audit/matched_batch'
    meta=json.loads((root/'dataset.json').read_text())
    checks=[]
    for model in MODELS:
        fresh_path=root/model.lower()/'cache_00.json'
        fresh=json.loads(fresh_path.read_text())
        if fresh['status']!='complete' or fresh['processed']!=len(meta['original_indices']):
            raise ValueError('Fresh encoder audit incomplete')
        tokens=np.load(fresh['token_shards'][0]['path'],mmap_mode='r')
        contexts=np.load(fresh['context_path'],mmap_mode='r')
        logits=np.load(fresh['logits_path'],mmap_mode='r')
        locations={}
        for path in sorted((SOURCE_CACHE/model.lower()/'chestxray14').glob('cache_*.json')):
            r=json.loads(path.read_text())
            if r['checkpoint_sha256']!=fresh['checkpoint_sha256'] or r['head_sha256']!=fresh['head_sha256']:
                raise ValueError('Different native encoder or query/scoring configuration')
            t=np.load(r['token_shards'][0]['path'],mmap_mode='r')
            c=np.load(r['context_path'],mmap_mode='r')
            l=np.load(r['logits_path'],mmap_mode='r')
            for local,index in enumerate(r['indices']):
                if index in meta['original_indices']:locations[index]=(t[local],c[local],l[local])
        errors=[0.,0.,0.]
        for i,index in enumerate(meta['original_indices']):
            for j,(a,b) in enumerate(zip((tokens[i],contexts[i],logits[i]),locations[index])):
                np.testing.assert_allclose(a,b,atol=3e-5,rtol=3e-5)
                errors[j]=max(errors[j],float(np.max(np.abs(a-b))))
        checks.append(dict(model=model,images=len(meta['original_indices']),indices=meta['original_indices'],
            primary_train_val_indices=meta['primary_comparison_indices'],
            token_max_abs_error=errors[0],context_max_abs_error=errors[1],logit_max_abs_error=errors[2],
            fresh_cache_sha256=sha256(fresh_path),checkpoint_sha256=fresh['checkpoint_sha256']))
    save(RUN/'ENCODER_AUDIT.json',dict(status='passed',checks=checks,
        scope=meta['scope'],batch_size=meta['batch_size'],
        prior_small_batch_diagnostic_sha256=sha256(RUN/'encoder_audit/numerical_diagnostics.json'),
        source_sha256={str(HERE/'audit_encoder.py'):sha256(HERE/'audit_encoder.py'),
            str(ROOT/'experiments/direct_anchor_classification/cache.py'):sha256(ROOT/'experiments/direct_anchor_classification/cache.py')}))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
