"""Audit available images and lock a 60/10/30 patient-level split once."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.large_patient_multilabel.common import *


class UnionFind:
    def __init__(self, keys):
        self.parents = {key: key for key in keys}

    def find(self, key):
        if self.parents[key] != key:
            self.parents[key] = self.find(self.parents[key])
        return self.parents[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def split_groups(groups, seed, fractions=(.6, .1, .3)):
    """Iterative multilabel stratification, each whole patient/duplicate group once."""
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    n = len(keys)
    capacity = np.array([0, round(n * fractions[1]), round(n * fractions[2])])
    capacity[0] = n - capacity[1:].sum()
    sizes = capacity.copy()
    labels = np.asarray([groups[k] for k in keys], dtype=int)
    targets = labels.sum(0)[None, :] * np.asarray(fractions)[:, None]
    for col, positives in enumerate(labels.sum(0)):
        if positives >= 3:
            targets[1:, col] = np.maximum(1., targets[1:, col])
            targets[0, col] = positives - targets[1:, col].sum()
    remaining, assignments = set(range(n)), {}
    rank = np.empty(n, dtype=int)
    rank[rng.permutation(n)] = np.arange(n)
    while remaining:
        positives = labels[sorted(remaining)].sum(0)
        available = np.flatnonzero(positives)
        if not len(available):
            for i in sorted(remaining, key=lambda j: rank[j]):
                options = np.flatnonzero(capacity == capacity.max())
                chosen = int(rng.choice(options))
                assignments[keys[i]] = SPLITS[chosen]
                capacity[chosen] -= 1
            break
        rarest = int(available[np.argmin(positives[available])])
        for i in sorted((i for i in remaining if labels[i, rarest]), key=lambda j: rank[j]):
            options = np.flatnonzero(capacity > 0)
            demand = targets[options, rarest]
            options = options[np.isclose(demand, demand.max())]
            options = options[capacity[options] == capacity[options].max()]
            chosen = int(rng.choice(options))
            assignments[keys[i]] = SPLITS[chosen]
            capacity[chosen] -= 1
            targets[chosen] -= labels[i]
            remaining.remove(i)
    if capacity.any():
        raise ValueError('Unfilled groups')
    return assignments, dict(zip(SPLITS, sizes.tolist()))


def image_record(row):
    path = Path(row['image'])
    if not re.fullmatch(r'\d{8}', row['patient_id']) or path.stem.split('_')[0] != row['patient_id']:
        raise ValueError('NIH patient ID / image ID mismatch')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row['file_sha256']:
        raise ValueError('Image bytes changed: ' + str(path))
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        rgb = im.convert('RGB')
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        return dict(file_sha256=digest, pixel_sha256=pixels, width=rgb.width, height=rgb.height,
                    bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_DATA.read_text())
    if sha256(source['samples_file']) != source['samples_sha256']:
        raise ValueError('Source cohort changed')
    rows = json.loads(Path(source['samples_file']).read_text())
    plan = dict(dataset='ChestXray14', source_dataset=str(SOURCE_DATA),
        source_dataset_sha256=sha256(SOURCE_DATA), source_samples_sha256=source['samples_sha256'],
        class_names=source['class_names'], fraction_by_patient_group=[.6, .1, .3], split_seed=20261121,
        grouping='NIH patient ID union exact file/decoded RGB duplicates; no group crosses a split',
        algorithm='One deterministic iterative multilabel split; no seed search or score-based selection',
        models=list(MODELS), methods=['original','global_frozen',*METHODS], training_seeds=list(SEEDS),
        training=dict(optimizer='AdamW',lr=1e-4,weight_decay=1e-4,batch_size=32,epochs=20,
                      min_epochs=8,patience=5,grad_clip=1.,bottleneck=64,pos_weight_cap=20.,
                      selection='VAL macro AUROC, strict improvement; earliest tie; epoch zero eligible'),
        test_access='All 24 model/method/seed fits and selected checkpoint hashes must be locked before TEST evaluation',
        initialization='Fresh adapters from the original frozen published backbone; NEVER reuse fitted adapters from previous splits',
        cache_reuse='Only immutable frozen-encoder tokens/native contexts; exact sample mapping and image hashes verified',
        inference_dtype='float32; TF32 disabled; encoder and downstream native parameters frozen',
        uncertainty=dict(resamples=2000,seed=20261121,unit='patient/duplicate connected component',
                         pairs=['Pre-Original','Pre-Post','Pre-Linear','Pre-Global adapter']),
        limitations=['Repartition of a previously used local NIH cohort, not a new blind cohort or official benchmark split.',
                     'Patient independence applies to the NEW adaptation splits, not prior studies or backbone pretraining.',
                     'Single-site 14-label evaluation; does not expand the small CheXpert cohort or establish cross-institution generalization.',
                     'Exact pixel duplicates checked; semantic/near-duplicate equivalence is not exhaustively audited.'])
    plan_path = RUN / 'SPLIT_PLAN.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing plan differs; no overwrite')
    else:
        save(plan_path, plan)
    if (DATA / 'dataset.json').exists():
        load_data()
        print('Existing locked split verified; not overwritten.', flush=True)
        return
    if (RUN / 'image_audit.json').exists():
        records = json.loads((RUN / 'image_audit.json').read_text())
        if len(records) != len(rows):
            raise ValueError('Incomplete image audit')
    else:
        records = []
        started = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, record in enumerate(pool.map(image_record, rows), 1):
                records.append(record)
                if i % 1000 == 0:
                    print('Decoded and hashed', i, '/', len(rows), 'seconds', round(time.time()-started), flush=True)
        save(RUN / 'image_audit.json', records)
    union = UnionFind({r['patient_id'] for r in rows})
    duplicate_patients = []
    for field in ('file_sha256','pixel_sha256'):
        owners = {}
        for row, record in zip(rows, records):
            previous = owners.setdefault(record[field], row['patient_id'])
            if previous != row['patient_id']:
                union.union(previous, row['patient_id'])
                duplicate_patients.append([field, previous, row['patient_id']])
    groups = {}
    old_splits = {}
    for row, record in zip(rows, records):
        old_splits[row['id']] = row['split']
        row.update(record)
        row['group_id'] = 'nih_' + union.find(row['patient_id'])
        groups.setdefault(row['group_id'], np.zeros(len(source['class_names']), dtype=int))
        groups[row['group_id']] = np.maximum(groups[row['group_id']], row['labels'])
    assignments, targets = split_groups(groups, plan['split_seed'])
    for row in rows:
        row['split'] = assignments[row['group_id']]
    check_disjoint(rows)
    stats = {}
    for split in SPLITS:
        selected = [r for r in rows if r['split'] == split]
        y = np.asarray([r['labels'] for r in selected])
        positive = y.sum(0)
        if (positive == 0).any() or (positive == len(y)).any():
            raise ValueError('Undefined fixed-class metrics; stop without trying another seed')
        stats[split] = dict(images=len(y),patients=len({r['patient_id'] for r in selected}),
            groups=len({r['group_id'] for r in selected}),positive_images=positive.tolist(),
            positive_patients=[len({r['patient_id'] for r in selected if r['labels'][c]}) for c in range(len(positive))])
    DATA.mkdir(parents=True, exist_ok=True)
    save(DATA / 'samples.json', rows)
    for split in SPLITS:
        save(DATA / (split + '.json'), [r for r in rows if r['split'] == split])
    metadata = dict(dataset='chestxray14', experiment='large_patient_multilabel_20261121',
        class_names=source['class_names'],prompts=source['prompts'],samples_file=str(DATA/'samples.json'),
        samples_sha256=sha256(DATA/'samples.json'),split_plan_sha256=sha256(plan_path),split_stats=stats,
        source_dataset=str(SOURCE_DATA),source_dataset_sha256=sha256(SOURCE_DATA),
        source_samples_sha256=source['samples_sha256'],group_targets=targets,
        patient_independence={'verified':True,'scope':'NEW adaptation train/val/test splits only'},
        limitations=plan['limitations'])
    save(DATA / 'dataset.json', metadata)
    save(RUN / 'SPLIT_AUDIT.json', dict(status='passed',images=len(rows),patients=len({r['patient_id'] for r in rows}),
        cross_split_overlap={k:0 for k in ('patient_id','group_id','file_sha256','pixel_sha256')},
        connected_duplicate_patients=duplicate_patients,split_stats=stats,
        prior_split_overlap={s:dict(Counter(old_splits[r['id']] for r in rows if r['split']==s)) for s in SPLITS},
        hashes={str(p):sha256(p) for p in DATA.glob('*.json')},image_audit_sha256=sha256(RUN/'image_audit.json')))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
