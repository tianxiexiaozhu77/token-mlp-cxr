"""Reproduce Token MLP paper tables from fixed, independently audited runs.

Run from medical with:
  bash AlphaRAD/local_setup/python_local.sh latex_ours/scripts/build_token_mlp_tables.py
Add --check to verify existing outputs without rewriting them. No GPU is used.
The merged/source CSV digests are pinned; all 93 saved TEST prediction arrays
are checked against their split identities/labels and their metrics recomputed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import metrics, read_dataset

MODELS = ("AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo")
ADDED_AGGREGATION = ("BiomedCLIP", "BioViL-T", "MaCo")
TOKEN_DIMS = {model: {"BiomedCLIP": 512, "BioViL-T": 128}.get(model, 768) for model in MODELS}
MEAN_MODEL = "seven_model_mean"
DATASETS = ("openi", "chexpert", "chestxray14")
NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
SEEDS = (42, 43, 44)
KEYS = ("macro_auroc", "macro_auprc")
MERGED = ROOT / "experiments/classification_results_summary_20261119/all_methods.csv"
MERGED_SHA256 = "8f53a2553f35201fb55be31444d300e172f0e83643f8987640a3da058153ecd3"
DIRECT = ROOT / "experiments/direct_anchor_classification/runs/20261118"
ALPHA_TOKEN = ROOT / "experiments/direct_anchor_classification/runs/20261118_alpha_token_mlp"
BASELINE_TOKEN = ROOT / "experiments/baseline_token_classification/runs/20261119"
CSV_HASHES = {
    DIRECT / "RESULTS.csv": "bb54dae6d9e619f5cff23bfe625c50bcdbd346f8c33f30654d66a206065d08c1",
    ALPHA_TOKEN / "RESULTS.csv": "346873872c2b355f7807a68bf333bf2e56b5b5b3050b986559e0148b1a9d5e95",
    BASELINE_TOKEN / "RESULTS.csv": "0b7e2143ab5d5bc130dad6ab15b902d8a56a555224c3b12afb7f00e60116ef45",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, where):
    if not condition:
        raise ValueError(where)


def close(actual, expected, where):
    if actual is None or expected is None:
        require(actual is expected, where)
    else:
        require(np.isfinite(actual) and np.isfinite(expected) and abs(actual-expected) < 1e-10, where)


def equal_metrics(actual, expected, where):
    if isinstance(actual, dict):
        require(set(actual) == set(expected), where + ": fields")
        for key in actual:
            equal_metrics(actual[key], expected[key], where + "/" + key)
    else:
        close(actual, expected, where)


def source_folder(model, dataset, method, seed):
    if model not in ("AlphaRAD", "RadZero"):
        return BASELINE_TOKEN / "results" / model / dataset / f"seed{seed}" / method
    original_method = "existing" if model == "RadZero" and method == "token_mlp" else method
    run = ALPHA_TOKEN if model == "AlphaRAD" and method == "token_mlp" else DIRECT
    return run / "results" / model.lower() / dataset / f"seed{seed}" / original_method


def audit_result(model, dataset, method, seed, datasets, files):
    folder = source_folder(model, dataset, method, seed)
    result_path = folder / "result.json"
    result = read(result_path)
    expected_method = "existing" if model == "RadZero" and method == "token_mlp" else method
    require(result["status"] == "complete" and result["method"] == expected_method and result["seed"] == seed,
            "Incomplete/wrong result: " + str(result_path))
    require(result["model"].lower() == model.lower(), "Wrong model")
    dataset_path = Path(result.get("dataset_path", result["dataset"])).resolve()
    require(sha256(dataset_path) == result["dataset_sha256"], "Changed dataset metadata")
    if dataset not in datasets:
        metadata, rows = read_dataset(dataset_path)
        labels = np.asarray([row["labels"] for row in rows], dtype=np.float32)
        indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"], dtype=np.int64)
        require(np.isin(labels, [0, 1]).all(), "Nonbinary classification labels")
        group_splits = {}
        for row in rows:
            previous = group_splits.setdefault(row["group_id"], row["split"])
            require(previous == row["split"], "Group leakage across splits")
        datasets[dataset] = dict(path=str(dataset_path), sha256=result["dataset_sha256"],
            samples_sha256=metadata["samples_sha256"], metadata=metadata, rows=rows, labels=labels,
            test_indices=indices, counts=dict(Counter(row["split"] for row in rows)))
    ds = datasets[dataset]
    require(str(dataset_path) == ds["path"] and result["dataset_sha256"] == ds["sha256"], "Models use different datasets")
    require(result["samples_sha256"] == ds["samples_sha256"] and result["counts"] == ds["counts"], "Different samples/splits")
    require(result["class_names"] == ds["metadata"]["class_names"], "Different class order")
    indices, rows, labels = ds["test_indices"], ds["rows"], ds["labels"]
    prediction_path = folder / "test_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as predicted:
        require(set(predicted.files) == {"logits", "labels", "indices", "ids", "groups"}, "Wrong NPZ fields")
        checks = dict(labels=labels[indices], indices=indices,
                      ids=np.asarray([rows[i]["id"] for i in indices]),
                      groups=np.asarray([rows[i]["group_id"] for i in indices]))
        for name, expected in checks.items():
            np.testing.assert_array_equal(predicted[name], expected, err_msg=str(folder) + "/" + name)
        logits = predicted["logits"].copy()
    require(logits.shape == labels[indices].shape and np.isfinite(logits).all(), "Invalid saved logits")
    computed = metrics(labels[indices], logits, result["class_names"])
    equal_metrics(computed, result["test"], "Recomputed TEST metrics")
    artifacts = [result_path, prediction_path]
    if model not in ("AlphaRAD", "RadZero"):
        cache_path = Path(result["cache_path"])
        require(sha256(cache_path) == result["cache_sha256"], "Changed cache provenance")
        cache = read(cache_path)
        require(cache["status"] == "complete", "Incomplete source cache")
        for key in ("dataset_sha256", "samples_sha256", "class_names", "query_sha256", "checkpoint_sha256"):
            require(cache[key] == result[key], "Cache/result provenance mismatch: " + key)
        expected_origin = "added" if model in ADDED_AGGREGATION else "native"
        require(result["backend_metadata"]["aggregation_origin"] == expected_origin, "Unexpected aggregation origin")
        artifacts.append(cache_path)
    epoch_zero_control = None
    if method == "token_mlp":
        training_path = folder / "training.json"
        training = read(training_path)
        require(training["status"] == "complete", "Incomplete training")
        for key in ("best_epoch", "best_val", "class_names", "dataset_sha256", "samples_sha256", "method", "seed", "config"):
            require(training[key] == result[key], "Training/result mismatch: " + key)
        equal_metrics(training["test"], computed, "Training TEST metrics")
        initial = training.get("initial_val", training.get("baseline_val"))
        best, epoch = initial["macro_auroc"], 0
        require([row["epoch"] for row in training["history"]] == list(range(1, len(training["history"])+1)), "Epoch order")
        for row in training["history"]:
            if row["val"]["macro_auroc"] > best:
                best, epoch = row["val"]["macro_auroc"], row["epoch"]
        require(epoch == result["best_epoch"], "VAL selection including epoch 0")
        close(best, result["best_val"]["macro_auroc"], "Selected VAL value")
        dim, bottleneck = TOKEN_DIMS[model], 64
        parameter_count = 2 * dim * bottleneck + bottleneck + dim
        require(result["trainable_parameters"] == training["trainable_parameters"] == parameter_count,
                "Expected shared residual MLP parameter count")
        checkpoint = torch.load(folder / "adapter.pt", map_location="cpu", weights_only=True)
        require(checkpoint["dim"] == dim and checkpoint["bottleneck"] == bottleneck, "Checkpoint MLP dimensions")
        require(checkpoint["method"] == expected_method and checkpoint["best_epoch"] == epoch, "Checkpoint selection/method")
        require(checkpoint.get("class_names", checkpoint.get("classes")) == result["class_names"], "Checkpoint class order")
        for key, value in checkpoint["provenance"].items():
            require(key in result and result[key] == value, "Checkpoint/result provenance mismatch: " + key)
            require(key in training and training[key] == value, "Checkpoint/training provenance mismatch: " + key)
        state = checkpoint["adapter_state_dict"]
        shapes = {"down.weight": (bottleneck, dim), "down.bias": (bottleneck,),
                  "up.weight": (dim, bottleneck), "up.bias": (dim,)}
        require(set(state) == set(shapes), "Unexpected adapter parameters")
        for key, shape in shapes.items():
            require(tuple(state[key].shape) == shape and bool(torch.isfinite(state[key]).all()), "Invalid adapter tensor: " + key)
        require(sum(tensor.numel() for tensor in state.values()) == parameter_count, "Checkpoint parameter count")
        artifacts += [training_path, folder / "adapter.pt"]
        if model not in ("AlphaRAD", "RadZero"):
            require(result["adapter_architecture"] == dict(dim=dim, bottleneck=bottleneck,
                    activation="GELU", linear_bias=True, zero_initialized_up=True), "Unexpected MLP architecture")
            validation = read(folder / "checkpoint_validation.json")
            require(validation == result["checkpoint_validation"] and validation["status"] == "passed", "Checkpoint validation")
            artifacts.append(folder / "checkpoint_validation.json")
        if epoch == 0:
            require(not bool(state["up.weight"].count_nonzero()) and not bool(state["up.bias"].count_nonzero()),
                    "Epoch-0 checkpoint must be the identity adapter")
            control_method = "aggregation_only" if model in ADDED_AGGREGATION else "baseline"
            control_path = source_folder(model, dataset, control_method, 42) / "test_predictions.npz"
            with np.load(control_path, allow_pickle=False) as control:
                for key, expected in checks.items():
                    np.testing.assert_array_equal(control[key], expected)
                control_logits = control["logits"].copy()
            np.testing.assert_allclose(logits, control_logits, rtol=3e-4, atol=3e-4,
                                       err_msg="Epoch-0 TEST predictions differ from the corresponding no-MLP control")
            equal_metrics(computed, metrics(labels[indices], control_logits, result["class_names"]), "Epoch-0 control metrics")
            epoch_zero_control = dict(method=control_method, predictions_path=str(control_path),
                max_abs_logit_error=float(np.max(np.abs(logits-control_logits))), up_parameters_all_zero=True)
            artifacts.append(control_path)
    else:
        require(result["trainable_parameters"] == 0 and result["best_epoch"] == 0, "Frozen control identity")
    for path in artifacts:
        files[str(path)] = sha256(path)
    files[str(dataset_path)] = ds["sha256"]
    files[ds["metadata"]["samples_file"]] = ds["samples_sha256"]
    return dict(model=model, dataset=dataset, method=method, source_method=expected_method,
                seed=seed, best_epoch=result["best_epoch"], trainable_parameters=result["trainable_parameters"],
                token_dim=TOKEN_DIMS[model], aggregation_origin="added" if model in ADDED_AGGREGATION else "native",
                macro_auroc=computed["macro_auroc"]*100, macro_auprc=computed["macro_auprc"]*100,
                result_path=str(result_path), result_sha256=files[str(result_path)],
                predictions_sha256=files[str(prediction_path)], dataset_sha256=ds["sha256"], samples_sha256=ds["samples_sha256"],
                epoch_zero_control=epoch_zero_control)


def summarize(records):
    values = np.asarray([[r[key] for key in KEYS] for r in records])
    means = values.mean(0)
    deviations = values.std(0, ddof=1) if len(records) > 1 else [None, None]
    return {key + suffix: (None if value is None else float(value))
            for i, key in enumerate(KEYS) for suffix, value in (("_mean", means[i]), ("_std", deviations[i]))}


def csv_text(rows):
    stream = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def pair(row, prefix, sd=False):
    values = []
    for key in KEYS:
        mean = row[prefix + key + "_mean"]
        suffix = r"\pm" + f"{row[prefix + key + '_std']:.2f}" if sd else ""
        values.append(f"{mean:.2f}" + suffix)
    return "$" + r"\,/\,".join(values) + "$"


def deltas(row, prefix="delta_"):
    return "$" + r"\,/\,".join(f"{row[prefix + key + '_pp']:+.2f}" for key in KEYS) + "$"


def main_table(rows):
    lines = [r"% Generated by scripts/build_token_mlp_tables.py; do not hand-edit.",
        r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
        r"Model & Dataset & Original (AUROC / AP) & Token MLP (AUROC / AP) & $\Delta$ (pp) \\", r"\midrule"]
    for i, row in enumerate(rows):
        model = "Seven-model mean" if row["model"] == MEAN_MODEL else row["model"]
        dataset_cell = ""
        if i == 0 or row["dataset"] != rows[i - 1]["dataset"]:
            span = 1
            while i + span < len(rows) and rows[i + span]["dataset"] == row["dataset"]:
                span += 1
            dataset_cell = rf"\multirow{{{span}}}{{*}}{{{NAMES[row['dataset']]}}}"
        lines.append(f"{model} & {dataset_cell} & {pair(row, 'original_')} & {pair(row, 'token_', True)} & {deltas(row)} " + r"\\")
        if row["model"] == MEAN_MODEL and i < len(rows)-1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Frozen local baselines (Original) and supervised Token MLP adaptation on the fixed local classification splits. Cells show macro AUROC / macro AUPRC (AP), in percent; AUPRC is computed as average precision. Token MLP reports mean $\pm$ sample SD over seeds 42/43/44; each original is evaluated once. SD measures training-seed variation, not patient sampling uncertainty. Mean rows first average the seven models within each seed, then summarize those three means; datasets are never pooled. $\Delta$ is Token MLP minus original in percentage points. BiomedCLIP, BioViL-T, and MaCo include added fixed aggregation; Table~\ref{tab:token_mlp_aggregation_control} isolates its contribution. BioViL-T/OpenI selects epoch 0 for all three seeds, retaining the no-MLP aggregation control.}",
        r"\label{tab:token_mlp_results}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def control_table(rows):
    lines = [r"% Generated by scripts/build_token_mlp_tables.py; do not hand-edit.",
        r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"Model & Dataset & Original & + Agg. & + Agg. + Token MLP & $\Delta$ vs. Agg. (pp) \\", r"\midrule"]
    for i, row in enumerate(rows):
        dataset_cell = ""
        if i == 0 or row["dataset"] != rows[i - 1]["dataset"]:
            span = 1
            while i + span < len(rows) and rows[i + span]["dataset"] == row["dataset"]:
                span += 1
            dataset_cell = rf"\multirow{{{span}}}{{*}}{{{NAMES[row['dataset']]}}}"
        lines.append(f"{row['model']} & {dataset_cell} & {pair(row, 'original_')} & {pair(row, 'aggregation_')} & {pair(row, 'token_', True)} & {deltas(row, 'delta_vs_aggregation_')} " + r"\\")
        if (i+1) % len(ADDED_AGGREGATION) == 0 and i < len(rows)-1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Aggregation controls for BiomedCLIP, BioViL-T, and MaCo on exactly the same splits. Cells show macro AUROC / macro AUPRC (AP), in percent. Original denotes the frozen local baseline; added fixed cosine-attention aggregation (Agg.) has no fitted parameters. The adapted condition fits the shared residual Token MLP and reports three-seed mean $\pm$ sample SD. The final column subtracts the aggregation-only control, separating the MLP contribution from the change in scoring path. BioViL-T/OpenI selects epoch 0 in every seed and matches its aggregation-only control.}",
        r"\label{tab:token_mlp_aggregation_control}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def dataset_table(datasets):
    lines = [r"% Generated by scripts/build_token_mlp_tables.py; do not hand-edit.",
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Dataset & Classes & Train & Val & Test \\", r"\midrule"]
    for dataset in DATASETS:
        ds = datasets[dataset]
        count = ds["counts"]
        lines.append(f"{NAMES[dataset]} & {len(ds['metadata']['class_names'])} & {count['train']:,} & {count['val']:,} & {count['test']:,} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
        r"\caption{The exact local subsets and fixed splits used by all seven models; these are not the complete public benchmarks. CheXpert includes 12 classes (including No Finding). CheXpert and ChestXray14 are grouped by patient and exact file duplicates. OpenI groups reports and exact decoded-image duplicates; longitudinal patient independence is unverified.}",
        r"\label{tab:token_mlp_dataset_summary}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def epoch_table(records):
    lines = [r"% Generated by scripts/build_token_mlp_tables.py; do not hand-edit.",
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
        r"Model & Dataset & Seed 42 & Seed 43 & Seed 44 \\", r"\midrule"]
    for i, dataset in enumerate(DATASETS):
        for j, model in enumerate(MODELS):
            epochs = [next(r['best_epoch'] for r in records if r['model']==model and r['dataset']==dataset
                           and r['method']=='token_mlp' and r['seed']==seed) for seed in SEEDS]
            dataset_cell = rf"\multirow{{{len(MODELS)}}}{{*}}{{{NAMES[dataset]}}}" if j == 0 else ""
            lines.append(f"{model} & {dataset_cell} & " + " & ".join(str(v) for v in epochs) + r" \\")
        if i < len(DATASETS)-1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Token MLP checkpoints selected by validation macro AUROC, with exact ties retaining the earliest epoch and epoch 0 eligible. All 63 reported seeds are retained and verified against the full saved validation histories. Only BioViL-T/OpenI selects epoch 0 (all three seeds); its zero-output adapter matches the aggregation-only control.}",
        r"\label{tab:token_mlp_selected_epochs}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    require(sha256(MERGED) == MERGED_SHA256, "Merged audit CSV changed; review before updating the pinned digest")
    files = {str(MERGED): MERGED_SHA256}
    for path, digest in CSV_HASHES.items():
        require(sha256(path) == digest, "Source CSV changed: " + str(path))
        files[str(path)] = digest
    with MERGED.open(encoding="utf-8-sig") as stream:
        merged = list(csv.DictReader(stream))
    wanted = {(model, dataset, method) for model in MODELS for dataset in DATASETS
              for method in (("baseline", "aggregation_only", "token_mlp") if model in ADDED_AGGREGATION else ("baseline", "token_mlp"))}
    selected = [row for row in merged if (row["model"], row["dataset"], row["method"]) in wanted]
    require(len(selected) == len(wanted) == 51 and len({(r['model'], r['dataset'], r['method']) for r in selected}) == 51,
            "Missing/duplicate prespecified table cell")
    datasets, raw, summaries = {}, [], {}
    for row in selected:
        model, dataset, method = (row[key] for key in ("model", "dataset", "method"))
        require(CSV_HASHES[Path(row["source_csv"])] == row["source_csv_sha256"], "Source CSV provenance")
        seeds = SEEDS if method == "token_mlp" else (42,)
        require(int(row["runs"]) == len(seeds), "Incorrect seed count")
        runs = [audit_result(model, dataset, method, seed, datasets, files) for seed in seeds]
        require(all(r["source_method"] == row["source_method"] for r in runs), "Incorrect source-method mapping")
        summary = summarize(runs)
        for key, value in summary.items():
            close(value, float(row[key]) if row[key] else None, "CSV/result numerical mismatch: " + key)
        raw.extend(runs); summaries[model, dataset, method] = summary
    require(len(raw) == 93 and sum(row["method"] == "token_mlp" for row in raw) == 63, "Incomplete evaluation matrix")
    zero_epochs = [row for row in raw if row["method"] == "token_mlp" and row["best_epoch"] == 0]
    require({(r['model'], r['dataset'], r['seed']) for r in zero_epochs} == {("BioViL-T", "openi", seed) for seed in SEEDS},
            "Unexpected epoch-0 selection; review the table caption")
    for dataset, expected in {"openi": (18,5966,755,749), "chexpert": (12,534,67,67), "chestxray14": (14,18140,2097,2196)}.items():
        ds = datasets[dataset]
        require((len(ds["metadata"]["class_names"]), *(ds["counts"][s] for s in ("train","val","test"))) == expected, "Changed dataset scope")
    main_rows, control_rows, averages = [], [], {}
    for dataset in DATASETS:
        for model in MODELS:
            row = dict(model=model, dataset=dataset)
            row.update({"original_"+k:v for k,v in summaries[model,dataset,"baseline"].items()})
            row.update({"token_"+k:v for k,v in summaries[model,dataset,"token_mlp"].items()})
            row.update({"delta_"+key+"_pp":row["token_"+key+"_mean"]-row["original_"+key+"_mean"] for key in KEYS})
            main_rows.append(row)
            if model in ADDED_AGGREGATION:
                control = dict(row)
                control.update({"aggregation_"+k:v for k,v in summaries[model,dataset,"aggregation_only"].items()})
                control.update({"delta_vs_aggregation_"+key+"_pp":row["token_"+key+"_mean"]-control["aggregation_"+key+"_mean"] for key in KEYS})
                control_rows.append(control)
        original = {key:float(np.mean([summaries[model,dataset,"baseline"][key+"_mean"] for model in MODELS])) for key in KEYS}
        per_seed = [{key:float(np.mean([r[key] for r in raw if r['dataset']==dataset and r['method']=='token_mlp' and r['seed']==seed])) for key in KEYS} for seed in SEEDS]
        mean = dict(model=MEAN_MODEL,dataset=dataset)
        mean.update({"original_"+key+suffix:original[key] if suffix=="_mean" else None for key in KEYS for suffix in ("_mean","_std")})
        mean.update({"token_"+key:value for key,value in summarize(per_seed).items()})
        mean.update({"delta_"+key+"_pp":mean["token_"+key+"_mean"]-original[key] for key in KEYS})
        main_rows.append(mean); averages[dataset]=dict(row=mean, per_seed_model_means=dict(zip(SEEDS,per_seed)))
    cell_rows = [row for row in main_rows if row["model"] != MEAN_MODEL]
    ranges = {key:dict(minimum_pp=min(row['delta_'+key+'_pp'] for row in cell_rows),
                       maximum_pp=max(row['delta_'+key+'_pp'] for row in cell_rows),
                       positive_cells=sum(row['delta_'+key+'_pp']>0 for row in cell_rows),
                       negative_cells=sum(row['delta_'+key+'_pp']<0 for row in cell_rows),
                       unchanged_cells=sum(row['delta_'+key+'_pp']==0 for row in cell_rows), total_cells=21) for key in KEYS}
    require(next(row for row in cell_rows if row['model']=='RadZero' and row['dataset']=='openi')['delta_macro_auprc_pp'] < 0,
            "RadZero OpenI AP decrease must remain visible")
    dataset_records = {dataset:{key:ds[key] for key in ('path','sha256','samples_sha256','counts')} | {
        "class_names":ds['metadata']['class_names'], "split_method":ds['metadata']['split_method'],
        "patient_independence":ds['metadata']['patient_independence']} for dataset,ds in datasets.items()}
    provenance = dict(status="verified", models=list(MODELS), datasets=dataset_records, seeds=list(SEEDS),
        evaluation_count=93, original_evaluations=21, aggregation_control_evaluations=9, trained_runs=63,
        main_rows=main_rows, aggregation_control_rows=control_rows,
        per_dataset_model_means=averages, improvement_ranges=ranges, runs=raw,
        negative_cells={key:[dict(model=row['model'],dataset=row['dataset'],delta_pp=row['delta_'+key+'_pp'])
                            for row in cell_rows if row['delta_'+key+'_pp']<0] for key in KEYS},
        epoch_zero_runs=zero_epochs,
        adapter_parameters={model:dict(dim=TOKEN_DIMS[model],bottleneck=64,
            trainable_parameters=2*TOKEN_DIMS[model]*64+64+TOKEN_DIMS[model]) for model in MODELS},
        verified_input_sha256=files, generator_sha256=sha256(Path(__file__)),
        notes=["Metrics recomputed from raw saved logits after checking TEST image IDs, groups, labels and indices.",
               "AP is sklearn average_precision_score; percentages and sample SD (ddof=1).",
               "Original/control: one deterministic evaluation. Token MLP: seeds42/43/44, fixed split.",
               "Seven-model mean is formed within each seed before calculating seed SD, separately per dataset.",
               "No across-dataset mean, test selection, patient uncertainty interval, or significance claim.",
               "RadZero existing is Token MLP; AlphaRAD existing query refinement is excluded from this paper table.",
               "BiomedCLIP, BioViL-T and MaCo Token MLP include fixed cosine-attention aggregation; separate aggregation-only controls are retained.",
               "Original denotes each frozen local classification baseline, not a claim of reproducing author benchmark evaluation protocols.",
               "BioViL-T OpenI selects epoch0 for all three seeds: checkpoint up tensors are zero and TEST logits/metrics match aggregation_only.",
               "Every trained adapter checkpoint is loaded on CPU to verify parameter shapes/counts, selected epoch and result/training provenance."])
    paper = ROOT / "latex_ours"
    outputs = {
        paper / "tables/token_mlp_results.tex": main_table(main_rows),
        paper / "tables/token_mlp_aggregation_control.tex": control_table(control_rows),
        paper / "tables/token_mlp_dataset_summary.tex": dataset_table(datasets),
        paper / "tables/token_mlp_selected_epochs.tex": epoch_table(raw),
        paper / "supplement_data/token_mlp_results.csv": csv_text(main_rows),
        paper / "supplement_data/token_mlp_results_per_seed.csv": csv_text(raw),
        paper / "supplement_data/token_mlp_results_aggregation_control.csv": csv_text(control_rows),
        paper / "supplement_data/token_mlp_results_provenance.json": json.dumps(provenance, indent=2, ensure_ascii=False)+"\n",
    }
    for path, content in outputs.items():
        if args.check:
            require(path.is_file() and path.read_text() == content, "Generated output is stale: " + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    print(json.dumps(dict(status="checked" if args.check else "built", evaluations=93,
                         per_dataset_model_means=averages, improvement_ranges=ranges,
                         negative_cells=provenance['negative_cells'], adapter_parameters=provenance['adapter_parameters'],
                         epoch_zero_runs=[{k:r[k] for k in ('model','dataset','seed','best_epoch','epoch_zero_control')} for r in zero_epochs]), indent=2))


if __name__ == "__main__":
    main()
