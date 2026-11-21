#!/usr/bin/env python3
"""Copy an explicit Token MLP source allowlist; never copy data or run outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

EXPERIMENTS = {
    'direct_anchor_classification': 'AlphaRAD/RadZero primary fits and frozen heads',
    'baseline_token_classification': 'Five other paper backbones; residual MLP and metrics',
    'adapter_placement_classification': 'Pre/Post and MaCo pooling controls',
    'both_adapter_classification': 'Joint Pre+Post adaptation',
    'gated_asl_classification': 'Gating x weighted BCE/ASL',
    'supervised_baseline_classification': 'AlphaRAD supervised baselines',
    'radzero_supervised_classification': 'RadZero supervised baselines',
    'large_patient_multilabel': 'Larger patient-disjoint NIH evaluation',
    'nih_chexpert_transfer': 'NIH-to-CheXpert external inference',
    'classification_uncertainty': 'Paired patient bootstrap and shared statistics',
}
BASELINES = ('CARZero', 'KAD', 'BiomedCLIP', 'BioViL-T', 'MaCo')
SOURCE_REPOS = {
    'AlphaRAD': 'AlphaRAD', 'RadZero': 'RadZero', 'CARZero': 'baselines/CARZero',
    'KAD': 'baselines/KAD', 'BiomedCLIP': 'baselines/BiomedCLIP/official',
    'BioViL-T': 'baselines/BioViL-T/official', 'MaCo': 'baselines/MaCo',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if workspace == output or not (workspace / 'latex_ours/main_token_mlp.tex').is_file():
        raise ValueError('Provide the original medical workspace and a separate output directory')
    selected = {}

    def add(relative, category, required=True):
        path = workspace / relative
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return
        if path.is_symlink() or path.stat().st_size > 5 * 1024**2:
            raise ValueError('Unexpected source file: ' + str(path))
        selected[relative] = category

    for name, role in EXPERIMENTS.items():
        for path in sorted((workspace / 'experiments' / name).glob('*.py')):
            add(str(path.relative_to(workspace)), role)
    add('experiments/__init__.py', 'Package marker', required=False)
    add('experiments/chexchonet_stability_20261024/cluster_bootstrap.py', 'Imported patient-bootstrap dependency')
    add('experiments/chexchonet_stability_20261024/__init__.py', 'Package marker', required=False)

    alpha_files = ('token_adapter_model.py', 'direct_anchor_model.py', 'query_refinement.py',
                   'train_direct_anchor.py', 'run_pretrained.py', 'python_local.sh', 'activate.sh',
                   'runtime-requirements.txt', 'runtime-constraints.txt', 'runtime.lock.txt',
                   'runtime-resolved.txt', 'runtime-upstream-fixes.patch')
    for name in alpha_files:
        add('AlphaRAD/local_setup/' + name, 'AlphaRAD integration/environment', required=False)
    for name in ('adapter_model.py', 'direct_anchor_model.py', 'eval_public.py',
                 'prepare_openi_adapter.py', 'train_direct_anchor.py', 'python_local.sh',
                 'test_adapter_model.py', 'test_gpu_lease.py'):
        add('RadZero/scripts/' + name, 'RadZero integration/environment')
    for model in BASELINES:
        folder = workspace / 'baselines' / model / 'local_setup'
        for name in ('token_mlp_adapter.py', 'adapter.py', 'python_local.sh', 'test_token_mlp_adapter.py',
                     'test_token_mlp_backends.py', 'requirements_eval.txt', 'requirements.lock.txt',
                     'environment_freeze.txt'):
            add(str((folder / name).relative_to(workspace)), model + ' integration/environment', required=False)
    for model, relative in SOURCE_REPOS.items():
        for name in ('LICENSE', 'LICENSE.txt', 'pyproject.toml', 'requirements.txt', 'environment.yml'):
            add(relative + '/' + name, model + ' upstream metadata', required=False)
    for path in sorted((workspace / 'latex_ours/scripts').glob('*token_mlp*.py')):
        add(str(path.relative_to(workspace)), 'Paper table/figure exporter')
    for pattern in ('tables/token_mlp_*.tex', 'sec/token_mlp_*.tex', 'supplement_data/token_mlp_*.csv',
                    'supplement_data/token_mlp_*provenance.json', 'supplement_data/token_mlp_figure_render.json'):
        for path in sorted((workspace / 'latex_ours').glob(pattern)):
            add(str(path.relative_to(workspace)), 'Paper text or aggregate result (no sample arrays)')
    for name in ('main_token_mlp.tex', 'token_mlp_refs.bib', 'cvpr.sty', 'ieeenat_fullname.bst',
                 'figures/token_mlp_pipeline.tex'):
        add('latex_ours/' + name, 'Paper reference source')

    records = []
    for relative, category in sorted(selected.items()):
        source, target = workspace / relative, output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(source):
            raise FileExistsError('Refusing to overwrite an edited snapshot: ' + str(target))
        shutil.copy2(source, target)
        records.append(dict(path=relative, category=category, size=target.stat().st_size, sha256=digest(target)))

    environments = []
    for name in ('requirements-bootstrap.txt', 'requirements-core.in', 'requirements-freeze.txt', 'requirements-lock.txt'):
        source = workspace / '_downloads/radzero_env' / name
        target = output / 'environments/RadZero' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(dict(path=str(target.relative_to(output)), source_path=str(source.relative_to(workspace)),
                            category='RadZero historical environment', size=target.stat().st_size, sha256=digest(target)))
        environments.append(str(target.relative_to(output)))

    models = []
    for model, relative in SOURCE_REPOS.items():
        repo = workspace / relative
        record = dict(model=model, checkout_path=relative, upstream_source_included=False)
        manifest_path = workspace / 'baselines' / model / 'local_setup/manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if (repo / '.git').exists():
            for key, command in [('url', ['remote', 'get-url', 'origin']), ('revision', ['rev-parse', 'HEAD'])]:
                record[key] = subprocess.check_output(['git', '-C', str(repo), *command], text=True).strip()
            # Preserve tracked implementation changes, excluding local ignore rules.
            patch = subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--', '*.py'])
            if patch:
                target = output / 'upstream_patches' / (model + '.patch')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(patch)
                record['local_patch'] = str(target.relative_to(output))
                record['local_patch_sha256'] = digest(target)
        else:
            record.update(url=manifest.get('code_url'), revision=manifest.get('revision'),
                          source_archive_sha256=manifest.get('source_archive_sha256'))
        for key in ('checkpoint_url', 'checkpoint_revision', 'checkpoints'):
            if key in manifest:
                # Relativize local resource paths in this derived index only.
                record[key] = json.loads(json.dumps(manifest[key]).replace(str(workspace) + '/', ''))
        models.append(record)
    write_json(output / 'MODEL_SOURCES.json', dict(source='Local Git metadata and saved model manifests; no network lookup', models=models))

    findings = []
    patterns = {'absolute_local_path': r'/storage/|/home/',
                'host_gpu_constraint': r'4\s*,\s*5\s*,\s*6\s*,\s*7|physical GPUs|physical GPU|zip\(MODELS, \(4, 5\)\)',
                'fixed_historical_run': r'runs/2026|202611\d\d'}
    for row in records:
        path = output / row['path']
        if path.suffix not in ('.py', '.sh', '.txt', '.toml'):
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            for category, pattern in patterns.items():
                if re.search(pattern, line):
                    findings.append(dict(path=row['path'], line=line_number, category=category))
    write_json(output / 'PORTABILITY_AUDIT.json', dict(scope='Static candidate locations, not an execution or completeness audit', findings=findings))
    write_json(output / 'SOURCE_MANIFEST.json', dict(scope='Exact copies of selected original source and aggregate results',
               manuscript_sha256=digest(workspace / 'latex_ours/main_token_mlp.tex'), files=records,
               excluded=['raw medical images', 'sample manifests', 'checkpoints', 'feature caches',
                         'per-image predictions', 'runs directories', 'virtual environments', 'real-image figure assets']))
    with (output / 'CODE_INDEX.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'category', 'size', 'sha256'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in records)
    print(json.dumps(dict(files=len(records), copied_bytes=sum(r['size'] for r in records),
                          portability_candidates=len(findings), output=str(output))))


if __name__ == '__main__':
    main()
