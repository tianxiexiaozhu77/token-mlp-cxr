"""Lock supervision, optimization budget, references and external evaluation."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.supervised_baseline_classification.common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / 'PROTOCOL.json').exists():
        raise FileExistsError(run / 'PROTOCOL.json')
    datasets = {}
    for name, path in DATASETS.items():
        meta, rows = read_dataset(path)
        datasets[name] = dict(path=str(path), sha256=sha256(path), samples_sha256=meta['samples_sha256'],
            classes=meta['class_names'], counts={s: sum(r['split'] == s for r in rows) for s in ('train','val','test')},
            grouping='report/exact-duplicate groups' if name == 'openi' else 'patient/exact-file-duplicate groups')
    references = []
    for model in MODELS:
        for dataset in DATASETS:
            for method in ('original', 'pre_mlp', 'post_mlp'):
                for seed in ((42,) if method == 'original' else SEEDS):
                    folder = reference_folder(model, dataset, seed, method)
                    result = json.loads((folder / 'result.json').read_text())
                    assert result['status'] == 'complete'
                    assert result['dataset_sha256'] == datasets[dataset]['sha256']
                    assert result['samples_sha256'] == datasets[dataset]['samples_sha256']
                    files = ['result.json','test_predictions.npz']
                    if method != 'original':
                        assert result['trainable_parameters'] == 99136
                        for k,v in dict(lr=1e-4,weight_decay=1e-4,epochs=20,min_epochs=8,patience=5).items():
                            assert result['config'][k] == v
                        files += ['adapter.pt', 'training.json']
                    references.append(dict(model=model,dataset=dataset,method=method,seed=seed,source=str(folder),
                        files_sha256={f:sha256(folder/f) for f in files}))
    # Lock an additional cross-dataset evaluation before computing its scores.
    # Only CheXpert-fitted models are eligible: RSNA is derived from NIH, and
    # ChestXray14-trained adapters would not provide a patient-disjoint transfer.
    external_source = ROOT / 'experiments/rsna_fixed_patient_cleanup_20261110/corrected_test_manifest.json'
    external = json.loads(external_source.read_text())
    assert external['count'] == len(external['rows']) == 5297
    assert len({r['patient_id'] for r in external['rows']}) == 3713
    assert all(r['patient_id'].startswith('NIH:') for r in external['rows'])
    ext_rows = [dict(id=r['exam_id'],image=r['image'],patient_id=r['patient_id'],
        group_id=r['patient_id'],nih_image_id=r['nih_image_id'],labels=r['labels'],split='test') for r in external['rows']]
    save(run/'external/rsna_samples.json',ext_rows)
    external_plan = dict(dataset='rsna_pneumonia_transfer', samples_file=str(run/'external/rsna_samples.json'),
        samples_sha256=sha256(run/'external/rsna_samples.json'), source_manifest=str(external_source),
        source_manifest_sha256=sha256(external_source), official_patient_mapping=external['official_mapping'],
        images=5297,patients=3713,positives=sum(r['labels'][0] for r in ext_rows),
        adaptation_source='chexpert',source_query_set='all 12 original CheXpert queries, unchanged order',
        target_class='Pneumonia',source_class='Pneumonia',source_column=datasets['chexpert']['classes'].index('Pneumonia'),
        models=list(MODELS),methods=['original','global_frozen','pre_mlp','post_mlp',*METHODS],
        selection='Only source CheXpert validation selects checkpoints. No RSNA training, tuning or checkpoint selection.',
        independence='RSNA patient IDs are mapped to NIH patients. The fitted source cohort is Stanford CheXpert. Exclude exact duplicate images against all source rows before scoring; no claim of pretraining independence.',
        limitations=['Single pneumonia/opacity endpoint, not a replacement for a larger 12-label CheXpert evaluation.',
            'This RSNA cohort was used in earlier different local studies; it is not a newly collected blind cohort.',
            'Do not evaluate ChestXray14-fitted adapters here as independent transfer: RSNA originates in NIH.'])
    save(run/'external/PROTOCOL.json',external_plan)
    protocol = dict(status='frozen',models=list(MODELS),methods=list(METHODS),seeds=list(SEEDS),datasets=datasets,
        new_training_runs=54,allowed_physical_gpus=[4,5,6,7],
        train=dict(optimizer='AdamW',lr=1e-4,weight_decay=1e-4,effective_batch_size=32,epochs=20,min_epochs=8,patience=5,grad_clip=1.,
            pos_weight='TRAIN negative/positive ratio capped20',selection='VAL macro AUROC, strict improvement, earliest tie, epoch zero eligible'),
        tuning_budget='One preregistered fixed configuration for every method, including reused Pre/Post. No hyperparameter search or TEST selection.',
        global_features={'AlphaRAD':'projected raw CLS token','KAD':'mean of projected spatial tokens','MaCo':'native mean of projected patches'},
        linear_probe='TRAIN-only coordinate mean/std normalization (std floor1e-6), then one d-to-C affine layer. Random initialization; epoch0 is not a pretrained classifier.',
        clip_global=dict(dim=768,bottleneck=192,reduction=4,ratio=.2,activation='ReLU after both bias-free linear layers',parameters=294912,
            source='https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py',
            adaptation='CLIP-Adapter-style visual module; native biomedical frozen readout and multilabel weighted BCE. Not exact original CLIP softmax evaluation.',
            readout={'AlphaRAD':'global CLS replicated across queries, native head-wise cosine/scale readout',
                'KAD':'mean feature as singleton visual memory for unchanged disease-query decoder/readout',
                'MaCo':'native global raw positive-minus-negative text dot products'},
            initialization='Default PyTorch Linear, no zero init. Epoch0 includes random residual; global_frozen is a separate no-fit reference.'),
        references=references,source_sha256=current_sources(),external_protocol_sha256=sha256(run/'external/PROTOCOL.json'))
    save(run/'PROTOCOL.json',protocol)
    for path in protocol['source_sha256']:
        source=Path(path); destination=run/'source'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    print(run, 'locked: 54 new fits, 63 historical evaluations, RSNA transfer 5297 images/3713 patients')


if __name__ == '__main__': main()
