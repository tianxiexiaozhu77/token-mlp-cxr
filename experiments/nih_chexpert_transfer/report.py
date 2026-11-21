"""Verify every transfer prediction and report paired patient bootstrap intervals."""
import csv
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nih_chexpert_transfer.common import *
from experiments.classification_uncertainty.bootstrap import (
    make_patient_draw_plan, save_patient_draw_plan, load_patient_draw_plan,
    evaluate as bootstrap_evaluate, summarize_difference)

DISPLAY = dict(original='Original', linear_probe='Linear probe', clip_global='Global adapter',
               post_mlp='Post MLP', pre_mlp='Pre MLP')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = load_protocol()
    check_source()
    ph = sha256(RUN / 'PROTOCOL.json')
    rows = read(RUN / 'samples.json')
    labels = np.asarray([r['labels'] for r in rows])
    groups = np.asarray([r['group_id'] for r in rows])
    tests = read(RUN / 'TEST_CHECKS.json')
    if tests['status'] != 'passed' or len(tests['environments']) != 2:
        raise ValueError('Missing both-environment regression checks')
    for rec in tests['environments']:
        if rec['status'] != 'passed' or sha256(rec['log']) != rec['log_sha256']:
            raise ValueError('Regression evidence changed')
    for model in MODELS:
        complete = read(RUN / 'evaluation' / (model + '_complete.json'))
        if complete['status'] != 'complete' or complete['evaluations'] != 13 or complete['protocol_sha256'] != ph:
            raise ValueError('Incomplete model evaluation')
        for path, digest in complete['results_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Evaluation result changed')
        integrity = read(RUN / 'evaluation' / (model + '_cache_integrity.json'))
        for path, rec in integrity.items():
            if sha256(path) != rec['sha256']:
                raise ValueError('Feature cache changed: ' + path)
    per_seed, per_class, scores, records = [], [], {}, []
    for model, method, seed in jobs():
        out = folder(model, method, seed)
        result = read(out / 'result.json')
        if ((result['model'], result['method'], result['seed']) != (model, method, seed) or
                result['status'] != 'complete' or result['protocol_sha256'] != ph or
                result['target_used_for_training'] or result['target_used_for_selection'] or
                result['target_normalization_fitted'] or not result['source_state_unchanged'] or
                result['predictions_sha256'] != sha256(out / 'predictions.npz')):
            raise ValueError('Invalid evaluation provenance')
        selected = [r for r in p['source_selection'] if (r['model'], r['method'], r['seed']) == (model, method, seed)]
        if result['source_selection'] != (selected[0] if selected else None):
            raise ValueError('Not the pre-locked NIH checkpoint')
        with np.load(out / 'predictions.npz', allow_pickle=False) as a:
            for key, value in [('labels', labels), ('groups', groups),
                    ('ids', [r['id'] for r in rows]), ('patients', [r['patient_id'] for r in rows]),
                    ('class_names', p['class_names']), ('source_class_names', p['source_class_names'])]:
                np.testing.assert_array_equal(a[key], value)
            if a['source_logits'].shape != (p['images'], 14):
                raise ValueError('Source score axis was truncated')
            np.testing.assert_array_equal(a['logits'], project(a['source_logits'], p['source_class_names']))
            tol = p['numerical_tolerances']
            np.testing.assert_allclose(a['source_replay_logits'], a['source_replay_reference'],
                atol=tol['source_reencoding_atol'], rtol=tol['source_reencoding_rtol'])
            actual = metrics(labels, a['logits'], p['class_names'])
            if actual != result['test']:
                raise ValueError('Saved metrics do not reproduce')
            scores.setdefault((model, method), []).append(a['logits'].copy())
        per_seed.append(dict(model=model, method=method, seed=seed,
            source_selected_epoch=0 if not selected else selected[0]['best_epoch'],
            macro_auroc=actual['macro_auroc'], macro_ap=actual['macro_auprc']))
        for name, item in actual['per_class'].items():
            per_class.append(dict(model=model, method=method, seed=seed, class_name=name,
                                  positives=item['positives'], negatives=item['negatives'],
                                  auroc=item['auroc'], ap=item['auprc']))
        records.append(dict(model=model, method=method, seed=seed,
            predictions_sha256=result['predictions_sha256'], result_sha256=sha256(out / 'result.json'),
            source_reencoding_max_error=result['source_validation_reencoding_max_error'],
            checkpoint_replay_max_error=result['full_checkpoint_replay_max_error']))
    if len(records) != p['evaluations']:
        raise ValueError('Missing registered results')
    summary = []
    for model in MODELS:
        for method in METHODS:
            items = [r for r in per_seed if (r['model'], r['method']) == (model, method)]
            row = dict(model=model, method=method, n_seeds=len(items))
            for key in ('macro_auroc', 'macro_ap'):
                values = [r[key] for r in items]
                row[key + '_mean'] = float(np.mean(values))
                row[key + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
            summary.append(row)
    uncertainty = RUN / 'uncertainty'
    uncertainty.mkdir(exist_ok=True)
    plan = make_patient_draw_plan(groups, n_resamples=p['bootstrap']['resamples'], seed=p['bootstrap']['seed'])
    plan_path = uncertainty / 'patient_draws.npz'
    if plan_path.exists():
        existing = load_patient_draw_plan(plan_path)
        if existing.sha256 != plan.sha256:
            raise ValueError('Existing bootstrap plan differs')
    else:
        save_patient_draw_plan(plan, plan_path)
    unique, inverse = np.unique(groups, return_inverse=True)
    points, draws, sklearn_checks = {}, {}, 0
    for key, values in scores.items():
        values = np.asarray(values)
        point, draw = bootstrap_evaluate(labels, values, groups, plan)
        # Independent direct index expansion verifies clustered weights and ties.
        for b in (0, len(plan.counts)//2, len(plan.counts)-1):
            idx = np.repeat(np.arange(len(labels)), plan.counts[b, inverse].astype(np.int64))
            for c in range(labels.shape[1]):
                if len(np.unique(labels[idx, c])) != 2:
                    if not np.isnan(draw[b, c]).all():
                        raise ValueError('Unsupported class was not marked undefined')
                    continue
                expected = np.mean([[roc_auc_score(labels[idx, c], v[idx, c]),
                    average_precision_score(labels[idx, c], v[idx, c])] for v in values], axis=0)
                np.testing.assert_allclose(draw[b, c], expected, atol=1e-12, rtol=0)
                sklearn_checks += 1
        item = next(r for r in summary if (r['model'], r['method']) == key)
        np.testing.assert_allclose(point.mean(0), [item['macro_auroc_mean'], item['macro_ap_mean']], atol=1e-12, rtol=0)
        points[key], draws[key] = point, draw
        np.savez_compressed(uncertainty / ('_'.join(key) + '.npz'), point=point, draws=draw,
                            class_names=p['class_names'], plan_sha256=np.asarray(plan.sha256))
        print('BOOTSTRAP', *key, '2000 patient draws', flush=True)
    differences, macro_rows = [], []
    for model in MODELS:
        for comparison in ('original', 'post_mlp', 'linear_probe', 'clip_global'):
            delta = summarize_difference(points[model, 'pre_mlp'], draws[model, 'pre_mlp'],
                        points[model, comparison], draws[model, comparison], p['class_names'])
            differences.append(dict(model=model, comparison='pre_mlp minus ' + comparison, **delta))
            for metric, item in delta['macro'].items():
                ci = item['ci95']
                macro_rows.append(dict(model=model, comparison='pre_mlp minus ' + comparison, metric=metric,
                    delta_pp=100*item['point'], ci95_low_pp=None if ci is None else 100*ci[0],
                    ci95_high_pp=None if ci is None else 100*ci[1], valid_replicates=item['valid_replicates'],
                    total_replicates=item['total_replicates'], invalid_fraction=item['invalid_fraction'],
                    conditional_on_all_classes_present=item['conditional_on_all_classes_present']))
    save(uncertainty / 'summary.json', dict(status='complete', plan_sha256=plan.sha256,
        bootstrap=p['bootstrap'], metric_aggregation='mean of per-seed metrics; no score ensemble',
        full_class_policy='A macro draw is undefined if any of the seven classes lacks either label.',
        pairs=differences))
    write_csv(RUN / 'summary.csv', summary)
    write_csv(RUN / 'per_seed_results.csv', per_seed)
    write_csv(RUN / 'per_class_results.csv', per_class)
    write_csv(uncertainty / 'macro_differences.csv', macro_rows)
    table = [r'\begin{table*}[t]', r'\centering', r'\small',
        r'\caption{ChestXray14 $\rightarrow$ CheXpert transfer on all 668 images (500 patients) and seven shared labels. '
        r'Only NIH training and validation data are used to fit adapters and select checkpoints. '
        r'Entries are macro AUROC / AP (\%), averaged over three seeds for fitted methods; bold marks the best in each metric. '
        r'The original model is deterministic. The global adapter operates on the CLS feature.}',
        r'\label{tab:nih_chexpert_transfer}', r'\begin{tabular}{lccccc}', r'\toprule',
        r'Model & Original & Linear probe & Global adapter & Post MLP & Pre MLP \\', r'\midrule']
    text_table = []
    for model in MODELS:
        items = [next(r for r in summary if (r['model'], r['method']) == (model, m)) for m in METHODS]
        best = {key: max(r[key] for r in items) for key in ('macro_auroc_mean', 'macro_ap_mean')}
        cells = []
        for item in items:
            nums = []
            for key in best:
                value = f'{100*item[key]:.2f}'
                nums.append(r'\textbf{' + value + '}' if abs(item[key]-best[key]) < 1e-12 else value)
            cells.append(' / '.join(nums))
        table.append(model + ' & ' + ' & '.join(cells) + r' \\')
        text_table.append(model + ' | ' + ' | '.join(
            f"{100*r['macro_auroc_mean']:.2f} / {100*r['macro_ap_mean']:.2f}" for r in items))
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (RUN / 'paper_table.tex').write_text('\n'.join(table) + '\n')
    report = ['ChestXray14 -> CheXpert: fixed-source-checkpoint transfer',
        '668 images, 500 patients; 518 frontal / 150 lateral; seven shared labels.',
        '24 existing NIH-selected checkpoints, 26 evaluations, zero new training runs.',
        'Source split: 13,406 train / 2,300 validation; target data never used for fitting or selection.',
        'Preserved NIH prompts, all 14 source outputs, native preprocessing and source LP normalization.',
        'Shared labels: ' + ', '.join(p['class_names']),
        'Metrics are image-level; fitted-method summaries average three seed metrics, not predictions.',
        '', 'Macro AUROC / AP (%):', 'Model | Original | Linear probe | Global adapter | Post MLP | Pre MLP',
        *text_table, '', 'Mean +/- sample SD (%), fitted methods:']
    report += [f"{r['model']} {DISPLAY[r['method']]}: {100*r['macro_auroc_mean']:.2f} +/- {100*r['macro_auroc_sd']:.2f} / "
               f"{100*r['macro_ap_mean']:.2f} +/- {100*r['macro_ap_sd']:.2f}" for r in summary if r['n_seeds'] > 1]
    report += ['', 'Paired patient bootstrap: 2,000 shared draws; 95% percentile intervals; differences in percentage points.']
    report += [f"{r['model']} {r['comparison']} {r['metric']}: {r['delta_pp']:+.2f} "
               f"[{r['ci95_low_pp']:.2f}, {r['ci95_high_pp']:.2f}], valid {r['valid_replicates']}/{r['total_replicates']}"
               for r in macro_rows]
    report += ['', 'Label support (positive images / positive patients):']
    report += [f"{name}: {v['positive_images']:.0f} / {v['positive_patients']}"
        for name, v in read(RUN / 'COHORT_AUDIT.json')['shared_class_support'].items()]
    report += ['', 'Limitations:', *['- ' + value for value in p['limitations']], '',
        'Validation: source selection sealed before target metrics; exact source-target file/pixel overlap = 0;',
        'all fresh contexts/native logits reproduced; all selected checkpoint full-target replays passed;',
        'NIH validation reference predictions reproduced within the predeclared tolerance;',
        'all reported metrics recomputed; both model environments passed the regression suite;',
        f'{sklearn_checks} direct sklearn patient-resample/class comparisons passed;',
        'source protocol, code, checkpoints and completion artifacts remain unchanged.']
    (RUN / 'REPORT.txt').write_text('\n'.join(report) + '\n')
    load_protocol(); check_source()
    save(RUN / 'AUDIT.json', dict(status='passed', protocol_sha256=ph, evaluations=records,
        source_frozen=True, target_fitting=False, target_checkpoint_selection=False,
        all_prediction_axes_and_identities_verified=True, all_metrics_recomputed=True,
        bootstrap_direct_sklearn_checks=sklearn_checks, paired_plan_sha256=plan.sha256,
        checks=['both-environment regression tests', 'all-source selections', 'fresh encoder and NIH head equality',
                'all context/native replay', 'full checkpoint reload replay', 'NIH validation reference replay',
                'exact source-target file/pixel overlap audit', 'full-cohort identities/labels/mapping',
                'feature/prediction/code/checkpoint SHA256', 'independent patient draw expansion']))
    outputs = [RUN / n for n in ('REPORT.txt', 'summary.csv', 'per_seed_results.csv', 'per_class_results.csv',
        'paper_table.tex', 'AUDIT.json', 'PROTOCOL.json', 'COHORT_AUDIT.json', 'TEST_CHECKS.json')]
    outputs += list((RUN / 'evaluation').glob('*.json')) + list(uncertainty.iterdir())
    outputs += [folder(m, method, seed) / name for m, method, seed in jobs() for name in ('predictions.npz', 'result.json')]
    save(RUN / 'COMPLETION.json', dict(status='complete', finished=time.time(),
        existing_checkpoints=24, new_training_runs=0, evaluations=26, paired_comparisons=8,
        output_sha256={str(path.relative_to(RUN)): sha256(path) for path in outputs if path.is_file()}))
    print('\n'.join(text_table), flush=True)


if __name__ == '__main__':
    main()
