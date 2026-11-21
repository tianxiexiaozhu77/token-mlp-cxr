"""Audit all target images, preserve all source selections, and lock transfer."""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nih_chexpert_transfer.common import *


def audit_image(row):
    from PIL import Image
    path = TARGET / row['Path']
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    patient = re.search(r'patient\d+', row['Path']).group()
    labels = [float(row[target]) for _, target in MAPPING]
    if any(x not in (0., 1.) for x in labels):
        raise ValueError('Non-binary shared target label')
    stat = path.stat()
    return dict(id=row['Path'], image=str(path), patient_id=patient, labels=labels,
        split='test', file_sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=pixel,
        bytes=len(raw), mtime_ns=stat.st_mtime_ns,
        view='frontal' if 'frontal' in row['Path'] else 'lateral' if 'lateral' in row['Path'] else 'unknown')


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'PROTOCOL.json').exists():
        load_protocol(); check_source()
        print('Existing locked transfer verified; not overwritten.', flush=True)
        return
    source_protocol, selection = check_source()
    meta = read(source_protocol['dataset'])
    source_rows = read(meta['samples_file'])
    with (TARGET / 'test_labels.csv').open() as stream:
        target_csv = list(csv.DictReader(stream))
    # Fixed before metrics: complete CSV, both views, all seven semantic matches.
    if len(target_csv) != 668:
        raise ValueError('Target release size differs')
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(audit_image, target_csv))
    if len({r['id'] for r in rows}) != 668 or len({r['patient_id'] for r in rows}) != 500:
        raise ValueError('Target identity coverage differs')
    from experiments.large_patient_multilabel.prepare import UnionFind
    union = UnionFind({r['patient_id'] for r in rows})
    overlap = {}
    for field in ('file_sha256', 'pixel_sha256'):
        overlap[field] = len({r[field] for r in rows} & {r[field] for r in source_rows})
        if overlap[field]:
            raise ValueError('Source-target exact image overlap: ' + field)
        owners = {}
        for row in rows:
            union.union(row['patient_id'], owners.setdefault(row[field], row['patient_id']))
    for row in rows:
        row['group_id'] = 'chexpert_' + union.find(row['patient_id'])
    support = {target: dict(positive_images=sum(r['labels'][c] for r in rows),
        positive_patients=len({r['patient_id'] for r in rows if r['labels'][c]}))
        for c, (_, target) in enumerate(MAPPING)}
    if any(v['positive_images'] in (0, 668) for v in support.values()):
        raise ValueError('Undefined full seven-class metrics')
    val_ids = [i for i, r in enumerate(source_rows) if r['split'] == 'val']
    locations = [0, len(val_ids) // 2, len(val_ids) - 1]
    checks = [val_ids[i] for i in locations]
    # Only three NIH validation images are used for inference replay, not selection.
    for i in checks:
        if sha256(source_rows[i]['image']) != source_rows[i]['file_sha256']:
            raise ValueError('Source replay image changed')
    save(RUN / 'samples.json', rows)
    encoder_rows = [dict(id='nih_check_' + str(i), image=source_rows[i]['image']) for i in checks]
    encoder_rows += [dict(id=r['id'], image=r['image']) for r in rows]
    save(RUN / 'encoder_samples.json', encoder_rows)
    # The cache utility's dataset key selects NIH prompts; image origin is explicit.
    save(RUN / 'encoder_dataset.json', dict(dataset='chestxray14',
        image_cohort='Three NIH validation replay cases followed by all 668 CheXpert images',
        class_names=meta['class_names'], prompts=meta['prompts'],
        samples_file=str(RUN / 'encoder_samples.json'), samples_sha256=sha256(RUN / 'encoder_samples.json')))
    old_meta = read(ROOT / 'data/direct_anchor_classification_20261118/chexpert/dataset.json')
    old_rows = read(old_meta['samples_file'])
    from collections import Counter
    audit = dict(status='passed', images=668, patients=500, groups=len({r['group_id'] for r in rows}),
        views=dict(Counter(r['view'] for r in rows)), shared_class_support=support,
        source_target_exact_image_overlap=overlap,
        historical_chexpert_usage=dict(Counter(r['split'] for r in old_rows)),
        history_note='This same CheXpert cohort was used in prior separate fitting/evaluation; not a new blind cohort.')
    save(RUN / 'COHORT_AUDIT.json', audit)
    artifacts = [RUN / n for n in ('samples.json', 'encoder_samples.json', 'encoder_dataset.json', 'COHORT_AUDIT.json')]
    artifacts += [TARGET / 'test_labels.csv', SOURCE / 'PROTOCOL.json', SOURCE / 'SELECTION_LOCK.json',
                  SOURCE / 'COMPLETION.json', Path(source_protocol['dataset']), Path(meta['samples_file'])]
    for model in MODELS:
        artifacts += [SOURCE / 'features' / model / name for name in ('head.json', 'cache.json', 'native.npy')]
    selections = []
    for rec in selection['selections']:
        out = source_folder(rec['model'], rec['method'], rec['seed'])
        artifacts += [out / name for name in ('adapter.pt', 'fit.json', 'training.json', 'val_predictions.npz')]
        selections.append(rec)
    files = list(HERE.glob('*.py')) + [ROOT / name for name in (
        'experiments/direct_anchor_classification/cache.py', 'experiments/direct_anchor_classification/model.py',
        'experiments/direct_anchor_classification/experiment_utils.py',
        'experiments/large_patient_multilabel/features.py', 'experiments/large_patient_multilabel/common.py',
        'experiments/large_patient_multilabel/train.py', 'experiments/supervised_baseline_classification/components.py',
        'experiments/baseline_token_classification/common.py', 'experiments/classification_uncertainty/bootstrap.py',
        'experiments/chexchonet_stability_20261024/cluster_bootstrap.py')]
    for file in files:
        target = RUN / 'source' / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    save(RUN / 'PROTOCOL.json', dict(status='locked', created=time.time(),
        source_dataset='ChestXray14', target_dataset='CheXpert', source_run=str(SOURCE),
        source_class_names=meta['class_names'], class_names=[t for _, t in MAPPING],
        label_mapping=[dict(source=s, target=t, source_column=meta['class_names'].index(s)) for s, t in MAPPING],
        models=MODELS, methods=METHODS, seeds=SEEDS, evaluations=26, additional_training_runs=0,
        images=668, patients=500, groups=audit['groups'], views=audit['views'],
        input_policy='All 668 images in CSV order; frontal and lateral retained; no target fitting, recalibration, selection or prompt changes.',
        scoring_policy='Keep all 14 NIH queries and classifier outputs; select the seven registered columns only after scoring.',
        source_validation_check_indices=checks, source_validation_check_locations=locations,
        source_selection=selections, encoder_batch_size=12, scoring_batch_size=32,
        numerical_tolerances=dict(source_reencoding_atol=3e-4, source_reencoding_rtol=3e-4,
                                  checkpoint_replay_atol=1e-6, checkpoint_replay_rtol=1e-6),
        bootstrap=dict(resamples=2000, seed=20261121, unit='patient plus exact duplicate connected group',
            pairs=['pre_mlp minus '+m for m in ('original', 'post_mlp', 'linear_probe', 'clip_global')]),
        limitations=[audit['history_note'],
            'External relative to NIH adapter fitting, not proof of unseen data or institutions in backbone pretraining.',
            'Seven shared labels, not the earlier 12-class CheXpert result or an official five-task benchmark.',
            'Image-level metrics; multiple views share patient/study labels. Bootstrap groups their images together.',
            'Intervals condition on fixed checkpoints and do not include training-cohort uncertainty or multiplicity correction.',
            'Exact file/pixel overlap is checked; exhaustive near-duplicate or pretraining overlap is not established.'],
        artifact_sha256={str(f):sha256(f) for f in artifacts},
        implementation_sha256={str(f):sha256(f) for f in files}))
    print('LOCKED: 668 images, 500 patients, seven classes, 24 existing NIH checkpoints; no target fitting.', flush=True)


if __name__ == '__main__':
    main()
