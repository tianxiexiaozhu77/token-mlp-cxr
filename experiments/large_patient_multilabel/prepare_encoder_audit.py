"""Reconstruct the original batches containing fixed TRAIN/VAL audit cases."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *


def main():
    meta,rows=load_data()
    targets=[]
    for split in ('train','val'):
        ids=[i for i,r in enumerate(rows) if r['split']==split]
        targets.extend([ids[0],ids[len(ids)//2],ids[-1]])
    chunks={};sources={}
    for path in sorted((SOURCE_CACHE/'alpharad/chestxray14').glob('cache_*.json')):
        record=json.loads(path.read_text());ids=record['indices'];local={i:j for j,i in enumerate(ids)}
        sources[str(path)]=sha256(path)
        for i in targets:
            if i in local:
                start=(local[i]//12)*12
                chunks[(record['shard'],start)]=ids[start:start+12]
    ordered=sorted(chunks.items(),key=lambda item:(len(item[1])!=12,item[0]))
    indices=[i for _,ids in ordered for i in ids]
    if len(indices)!=len(set(indices)) or not set(targets)<=set(indices):
        raise ValueError('Incomplete or overlapping encoder batches')
    out=RUN/'encoder_audit/matched_batch';out.mkdir(parents=True,exist_ok=True)
    samples=[rows[i] for i in indices]
    path=out/'dataset.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['original_indices']!=indices or old['primary_comparison_indices']!=targets:
            raise ValueError('Existing audit cases differ')
        if json.loads(Path(old['samples_file']).read_text())!=samples or sha256(old['samples_file'])!=old['samples_sha256']:
            raise ValueError('Existing audit images differ')
    else:
        save(out/'samples.json',samples)
        save(path,dict(dataset='chestxray14',class_names=meta['class_names'],prompts=meta['prompts'],
            samples_file=str(out/'samples.json'),samples_sha256=sha256(out/'samples.json'),
            original_indices=indices,primary_comparison_indices=targets,batch_size=12,
            scope='Original batches containing six fixed TRAIN/VAL audit images; label-free encoder parity only, no TEST metrics.'))
    save(out/'creation_audit.json',dict(status='passed',primary_train_val_indices=targets,
        batch_lengths=[len(ids) for _,ids in ordered],dataset_sha256=sha256(DATA/'dataset.json'),
        source_cache_sha256=sources,script_sha256=sha256(Path(__file__))))
    print('Verified fixed original-batch audit manifest:',len(indices),'images.')


if __name__=='__main__':main()
