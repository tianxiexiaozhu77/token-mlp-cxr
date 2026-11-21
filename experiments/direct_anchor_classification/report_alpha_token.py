"""Audit all fixed-seed AlphaRAD token-MLP runs and their four prior controls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment_utils import read_dataset, save, sha256
from train import metrics


DATASETS = ("openi", "chexpert", "chestxray14")
METHODS = ("baseline", "bias", "direct", "existing", "token_mlp")
METRICS = ("macro_auroc", "macro_auprc", "micro_auprc")
NAMES = {
    "baseline": "Original",
    "bias": "Disease bias",
    "direct": "Direct-anchor",
    "existing": "Query refinement (eta=0.3)",
    "token_mlp": "Token MLP (768->64->768, GELU)",
}


def seeds_for(method):
    return (42,) if method in ("baseline", "existing") else (42, 43, 44)


def assert_metrics_equal(actual, recorded, location):
    """Audit aggregate and every per-class metric, including undefined values."""
    if isinstance(actual, dict):
        if not isinstance(recorded, dict) or set(actual) != set(recorded):
            raise ValueError(f"Metric fields differ: {location}")
        for key, value in actual.items():
            assert_metrics_equal(value, recorded[key], f"{location}/{key}")
    elif actual is None:
        if recorded is not None:
            raise ValueError(f"Expected undefined metric: {location}")
    elif not isinstance(recorded, (int, float)) or not np.isfinite(recorded) or abs(actual-recorded) > 1e-12:
        raise ValueError(f"Metric mismatch: {location}: {actual!r} != {recorded!r}")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(row, key):
    mean, sd = row[key+"_mean"], row[key+"_std"]
    return f"{mean:.2f}" if sd is None else f"{mean:.2f} +/- {sd:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    args = parser.parse_args()
    root, reference = args.run_dir.resolve(), args.reference_run.resolve()
    if root == reference:
        raise ValueError("New-run and reference-run directories must differ")

    # Require every prespecified result before producing a complete report.
    folders = {}
    missing = []
    for dataset in DATASETS:
        for method in METHODS:
            source = root if method == "token_mlp" else reference
            for seed in seeds_for(method):
                folder = source / "results" / "alpharad" / dataset / f"seed{seed}" / method
                folders[dataset, method, seed] = folder
                files = ["result.json", "test_predictions.npz"]
                if method == "token_mlp":
                    files += ["training.json", "adapter.pt"]
                missing.extend(str(folder / name) for name in files if not (folder / name).is_file())
    if missing:
        raise FileNotFoundError("Missing required results; no partial report written:\n" + "\n".join(missing))

    rows, per_class, runs, dataset_audits = [], [], [], []
    for dataset in DATASETS:
        baseline_folder = folders[dataset, "baseline", 42]
        original = json.loads((baseline_folder / "result.json").read_text())
        dataset_path = Path(original["dataset"])
        metadata, samples = read_dataset(dataset_path)
        dataset_hash = sha256(dataset_path)
        if metadata["samples_sha256"] != original["samples_sha256"] or dataset_hash != original["dataset_sha256"]:
            raise ValueError(f"Original dataset provenance changed: {dataset}")
        classes = metadata["class_names"]
        expected_indices = np.array([i for i, sample in enumerate(samples) if sample["split"] == "test"])
        expected_ids = np.array([samples[i]["id"] for i in expected_indices])
        expected_groups = np.array([samples[i]["group_id"] for i in expected_indices])
        expected_labels = np.array([samples[i]["labels"] for i in expected_indices], dtype=np.float32)
        counts = {split: sum(sample["split"] == split for sample in samples) for split in ("train", "val", "test")}
        if expected_labels.shape != (counts["test"], len(classes)) or not np.isin(expected_labels, [0, 1]).all():
            raise ValueError(f"Unexpected test label shape or values: {dataset}")
        shared_fields = ("dataset_sha256", "samples_sha256", "head_sha256", "checkpoint_sha256", "prompts", "config", "pos_weight")
        dataset_rows = []
        for method in METHODS:
            values = []
            for seed in seeds_for(method):
                folder = folders[dataset, method, seed]
                data = json.loads((folder / "result.json").read_text())
                if data["status"] != "complete" or data["model"] != "alpharad" or data["method"] != method or data["seed"] != seed:
                    raise ValueError(f"Incorrect or incomplete run identity: {folder}")
                if data["class_names"] != classes or data["counts"] != counts:
                    raise ValueError(f"Class order or split counts differ: {folder}")
                if sha256(data["dataset"]) != dataset_hash:
                    raise ValueError(f"Dataset metadata differs: {folder}")
                for key in shared_fields:
                    if data[key] != original[key]:
                        raise ValueError(f"Shared provenance/protocol differs ({key}): {folder}")
                if method == "token_mlp":
                    if data["trainable_parameters"] != 99136:
                        raise ValueError(f"Unexpected 768->64->768 adapter parameter count: {folder}")
                    if data["gpu"]["physical_index"] not in (4, 5, 6, 7):
                        raise ValueError(f"New training used a GPU outside requested 4-7: {folder}")
                    training = json.loads((folder / "training.json").read_text())
                    if training.get("status") != "complete" or training.get("best_epoch") != data["best_epoch"]:
                        raise ValueError(f"Training record is incomplete or inconsistent: {folder}")
                    assert_metrics_equal(data["test"], training["test"], f"{folder}/training/test")
                with np.load(folder / "test_predictions.npz", allow_pickle=False) as prediction:
                    for key, expected in (("indices", expected_indices), ("ids", expected_ids),
                                          ("groups", expected_groups), ("labels", expected_labels)):
                        np.testing.assert_array_equal(prediction[key], expected, err_msg=f"{folder}: {key}")
                    logits = prediction["logits"]
                    if logits.shape != expected_labels.shape or not np.isfinite(logits).all():
                        raise ValueError(f"Invalid prediction shape/values: {folder}")
                    audited = metrics(expected_labels, logits, classes)
                assert_metrics_equal(audited, data["test"], f"{folder}/test")
                values.append(data)
                run = dict(model="alpharad", dataset=dataset, method=method, seed=seed,
                           folder=str(folder), best_epoch=data["best_epoch"], test=audited,
                           gpu=data["gpu"], result_sha256=sha256(folder / "result.json"),
                           prediction_sha256=sha256(folder / "test_predictions.npz"))
                if method == "token_mlp":
                    run.update(adapter_sha256=sha256(folder / "adapter.pt"),
                               training_sha256=sha256(folder / "training.json"))
                runs.append(run)
                for disease, result in audited["per_class"].items():
                    per_class.append(dict(model="alpharad", dataset=dataset, method=method,
                                          seed=seed, disease=disease, **result))
            row = dict(model="alpharad", dataset=dataset, method=method, method_label=NAMES[method],
                       runs=len(values), test_images=counts["test"], classes=len(classes),
                       trainable_parameters=values[0]["trainable_parameters"],
                       fitted_query_values=values[0].get("fitted_query_values", 0),
                       best_epochs="/".join(str(value["best_epoch"]) for value in values))
            for key in METRICS:
                scores = [value["test"][key]*100 for value in values]
                row[key+"_mean"] = float(np.mean(scores))
                row[key+"_std"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
            dataset_rows.append(row)
        controls = {row["method"]: row for row in dataset_rows}
        for row in dataset_rows:
            for key in METRICS:
                for control, suffix in (("baseline", "baseline"), ("direct", "direct"), ("existing", "query")):
                    row[f"delta_{key}_vs_{suffix}_pp"] = row[key+"_mean"]-controls[control][key+"_mean"]
            rows.append(row)
        dataset_audits.append(dict(dataset=dataset, dataset_path=str(dataset_path),
            dataset_sha256=dataset_hash, samples_sha256=metadata["samples_sha256"],
            head_sha256=original["head_sha256"], checkpoint_sha256=original["checkpoint_sha256"],
            counts=counts, class_names=classes, test_positive_counts=expected_labels.sum(0).astype(int).tolist()))

    if len(runs) != 33 or sum(run["method"] == "token_mlp" for run in runs) != 9:
        raise ValueError("The prespecified 24 reference + 9 new evaluations were not all audited")
    token_comparisons = []
    for row in rows:
        if row["method"] == "token_mlp":
            token_comparisons.append({key: value for key, value in row.items()
                                      if key in ("dataset", "method") or key.startswith("delta_")})
    summary = dict(status="complete", model="alpharad", reference_run=str(reference),
        run_dir=str(root), evaluations=len(runs), new_evaluations=9, rows=rows, runs=runs,
        dataset_audits=dataset_audits, token_mlp_comparisons=token_comparisons,
        all_test_indices_ids_groups_labels_match_metadata=True,
        all_class_orders_and_shared_provenance_match=True,
        all_metrics_independently_recomputed=True,
        new_training_gpus=sorted({run["gpu"]["physical_index"] for run in runs if run["method"] == "token_mlp"}),
        allowed_physical_gpus=[4, 5, 6, 7],
        seed_policy="All prespecified training seeds 42,43,44 retained on a fixed split. Original and deterministic query refinement evaluated once.",
        metric_units="RESULTS.csv and summary rows: percent; deltas: percentage points; per_class_results.csv and per-run test metrics: fractions.",
        uncertainty="Sample SD across three training seeds, not uncertainty across patients or data splits.")
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "RESULTS.csv", rows)
    write_csv(root / "per_class_results.csv", per_class)

    lines = ["AlphaRAD classification: token MLP and four previous controls", "",
        "All 9 prespecified new runs and 24 reference evaluations passed prediction/provenance/metric audits.",
        f"Reference run: {reference}", f"New run: {root}", "",
        "Summary metrics are percentages; differences are percentage points. +/- is sample SD across training seeds 42/43/44.",
        "Seed SD is not patient-level uncertainty, split uncertainty, a confidence interval, or a significance test.",
        "OpenI: original report/duplicate-group split 5966/755/749, 18 classes; longitudinal patient independence unverified.",
        "CheXpert: local official-test cohort repartitioned by patient, 534/67/67, 12 classes including No Finding.",
        "ChestXray14: local official-test cohort repartitioned by patient, 18140/2097/2196, 14 diseases.",
        "These are supervised adaptation experiments on custom splits, not official zero-shot benchmark results.",
        "CheXpert selected labels are binary; Pleural Other/Support Devices were excluded by the existing manifest.",
        "OpenI automatic report labels use the existing mapping; absence of mention is treated as zero.",
        "CheXpert has only 67 test images, and some classes have 1-3 positives. See per_class_results.csv for every class count.",
        "Original: unchanged AlphaRAD classifier, evaluated once.",
        "Disease bias: learned disease-specific feature vector after visual aggregation and before context normalization.",
        "Direct-anchor: shared down + disease-specific up + disease bias, rank8, RMS normalization, no activation, after aggregation.",
        "Existing/query refinement: deterministic train-label statistics, fixed eta0.3; no gradient training. This is not the token MLP.",
        "Token MLP: x + Linear64to768(GELU(Linear768to64(x))); both Linear layers have bias; 99,136 trainable parameters.",
        "The token MLP is shared across all tokens and diseases, applied to projected raw tokens before native FLaS head splitting/L2/attention.",
        "It updates both attention keys and values and mixes head subspaces; original queries, backbone, projection, scale and head readout remain frozen.",
        "The token MLP up-projection starts at zero, so epoch0 equals the original model. It is trained with labels, not training-free.",
        "New token MLP uses all three seeds on physical GPUs4-7. Previous controls are reused with the exact same data/checkpoint/head provenance.",
        "AdamW lr1e-4 wd1e-4, batch32, max20 epochs, minimum8, patience5, training-only positive weights capped at20, gradient norm clipped at1.",
        "Epoch0 eligible; validation macro-AUROC chooses checkpoints, exact ties keep the earlier epoch. No test selection or seed substitution.",
        "Methods are not parameter matched. Separate adapters are fitted per dataset; this is not a cross-dataset transfer evaluation.",
        "", "dataset,method,macro_AUROC,macro_AUPRC,micro_AUPRC,parameters,best_epochs"]
    for row in rows:
        lines.append(f"{row['dataset']},{row['method']},{format_metric(row, 'macro_auroc')},{format_metric(row, 'macro_auprc')},{format_metric(row, 'micro_auprc')},{row['trainable_parameters']},{row['best_epochs']}")
    lines += ["", "Token-MLP mean deltas (percentage points):", "dataset,metric,vs_original,vs_direct_anchor,vs_query_refinement"]
    for row in rows:
        if row["method"] == "token_mlp":
            for key in METRICS:
                lines.append(f"{row['dataset']},{key},{row[f'delta_{key}_vs_baseline_pp']:+.2f},{row[f'delta_{key}_vs_direct_pp']:+.2f},{row[f'delta_{key}_vs_query_pp']:+.2f}")
    (root / "RESULTS.txt").write_text("\n".join(lines)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=True)
    colors = ["#64748b", "#e9a23b", "#267aaf", "#3f9a70", "#9563b7"]
    labels = ["Original", "Disease\nbias", "Direct-\nanchor", "Query\nrefinement", "Token\nMLP"]
    titles = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
    for axis, dataset in zip(axes, DATASETS):
        selected = [row for row in rows if row["dataset"] == dataset]
        axis.bar(range(5), [row["macro_auroc_mean"] for row in selected], color=colors,
                 yerr=[row["macro_auroc_std"] or 0 for row in selected], capsize=3)
        axis.set_xticks(range(5), labels, fontsize=9)
        axis.set_ylim(0, 100)
        axis.set_title(titles[dataset])
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.2)
        for index, row in enumerate(selected):
            height = row["macro_auroc_mean"]+(row["macro_auroc_std"] or 0)+1.2
            axis.text(index, height, f"{row['macro_auroc_mean']:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Macro AUROC (%)")
    figure.suptitle("AlphaRAD: shared token MLP before disease-conditioned visual aggregation")
    figure.text(.5, .025, "Fixed custom splits; error bars: sample SD across 3 training seeds. Original/query refinement: deterministic once.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .065, 1, .96))
    figure.savefig(root / "comparison.png", dpi=180)
    figure.savefig(root / "comparison.pdf")
    plt.close(figure)
    summary["outputs_sha256"] = {name: sha256(root / name) for name in
        ("RESULTS.csv", "RESULTS.txt", "per_class_results.csv", "comparison.png", "comparison.pdf")}
    # Write the completed summary last, after every report artifact exists.
    save(root / "summary.json", summary)
    print("\n".join(lines[-27:]))


if __name__ == "__main__":
    main()
