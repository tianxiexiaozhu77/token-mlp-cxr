"""Audit and summarize all prespecified runs; no test-based run selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment_utils import read_dataset, save, sha256
from train import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir
    rows, per_class, runs = [], [], []
    datasets = ("openi", "chexpert", "chestxray14")
    methods = ("baseline", "bias", "direct", "existing")
    for dataset in datasets:
        common_ids = common_labels = None
        expected_ids = expected_labels = expected_indices = expected_groups = None
        for model in ("radzero", "alpharad"):
            baseline_metric = None
            for method in methods:
                seeds = [42] if method == "baseline" or (model == "alpharad" and method == "existing") else [42,43,44]
                values = []
                for seed in seeds:
                    folder = root / "results" / model / dataset / f"seed{seed}" / method
                    data = json.loads((folder / "result.json").read_text())
                    if data["status"] != "complete":
                        raise ValueError(f"Incomplete: {folder}")
                    pred = np.load(folder / "test_predictions.npz", allow_pickle=False)
                    if sha256(data["dataset"]) != data["dataset_sha256"]:
                        raise ValueError("Dataset metadata changed during training")
                    if expected_ids is None:
                        metadata, samples = read_dataset(data["dataset"])
                        expected_indices = np.array([i for i,s in enumerate(samples) if s["split"] == "test"])
                        expected_ids = np.array([samples[i]["id"] for i in expected_indices])
                        expected_groups = np.array([samples[i]["group_id"] for i in expected_indices])
                        expected_labels = np.array([samples[i]["labels"] for i in expected_indices])
                    np.testing.assert_array_equal(pred["indices"], expected_indices)
                    np.testing.assert_array_equal(pred["ids"], expected_ids)
                    np.testing.assert_array_equal(pred["groups"], expected_groups)
                    np.testing.assert_array_equal(pred["labels"], expected_labels)
                    if common_ids is None:
                        common_ids, common_labels = pred["ids"], pred["labels"]
                    np.testing.assert_array_equal(pred["ids"], common_ids)
                    np.testing.assert_array_equal(pred["labels"], common_labels)
                    audited = metrics(pred["labels"], pred["logits"], data["class_names"])
                    for key in ("macro_auroc", "macro_auprc", "micro_auprc"):
                        if abs(audited[key]-data["test"][key]) > 1e-12:
                            raise ValueError(f"Metric mismatch: {folder}")
                    values.append(data)
                    runs.append(dict(model=model, dataset=dataset, method=method, seed=seed,
                                     folder=str(folder.resolve()), best_epoch=data["best_epoch"],
                                     test=audited, prediction_sha256=sha256(folder / "test_predictions.npz")))
                    for name, result in audited["per_class"].items():
                        per_class.append(dict(model=model, dataset=dataset, method=method, seed=seed, disease=name, **result))
                row = dict(model=model, dataset=dataset, method=method, runs=len(values),
                           test_images=len(common_ids), classes=values[0]["test"]["total_classes"],
                           trainable_parameters=values[0]["trainable_parameters"],
                           fitted_query_values=values[0].get("fitted_query_values", 0),
                           best_epochs="/".join(str(v["best_epoch"]) for v in values))
                for key in ("macro_auroc", "macro_auprc", "micro_auprc"):
                    scores = [v["test"][key]*100 for v in values]
                    row[key+"_mean"] = float(np.mean(scores))
                    row[key+"_std"] = float(np.std(scores, ddof=1)) if len(scores)>1 else None
                if method == "baseline":
                    baseline_metric = row["macro_auroc_mean"]
                row["delta_auroc_pp"] = row["macro_auroc_mean"]-baseline_metric
                rows.append(row)
    for filename, data in (("RESULTS.csv", rows), ("per_class_results.csv", per_class)):
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    save(root / "summary.json", dict(status="complete", rows=rows, runs=runs,
        all_test_ids_and_labels_match=True, all_metrics_independently_recomputed=True,
        seed_policy="All prespecified seeds 42,43,44 retained. Fixed split. Deterministic baseline/AlphaRAD query refinement evaluated once."))
    lines = ["Direct-anchor classification: completed four-way comparison", "",
        "All metrics below are percentages. +/- is sample SD over training seeds 42/43/44, not uncertainty across patients.",
        "OpenI: existing report/duplicate-group split 5966/755/749; longitudinal patient independence unverified.",
        "CheXpert: local official-test cohort repartitioned by patient, 534/67/67, 12 classes including No Finding.",
        "ChestXray14: local official-test cohort repartitioned by patient, 18140/2097/2196, 14 diseases.",
        "These are supervised adaptation experiments on custom splits, not official zero-shot benchmark results.",
        "CheXpert selected labels are binary (no uncertain/missing values); unused Pleural Other/Support Devices excluded by the existing manifest.",
        "OpenI automatic report labels use the existing author mapping; absence of mention is treated as zero.",
        "Each model retains native prompts, preprocessing, scales and readout. Compare changes within each model.",
        "Bias means a learned disease-specific FEATURE VECTOR before normalization, not a scalar logit bias.",
        "Direct-anchor: shared down + disease-specific up + bias, rank 8, RMS normalization, no activation.",
        "RadZero existing: original 768->64->768 GELU token adapter, retrained using the same split/seed/selection protocol.",
        "AlphaRAD existing: original deterministic query refinement eta=0.3, fitted using all TRAIN rows; zero gradient-trained parameters.",
        "Bias/direct use 3 seeds for both models; RadZero existing uses 3 seeds; deterministic methods use one run.",
        "AdamW lr=1e-4 wd=1e-4, batch32, max20 epochs, minimum8, patience5, train-only positive weights capped at20.",
        "Epoch0 eligible; VAL macro-AUROC chooses checkpoints, ties favor earlier epoch. No TEST selection or seed substitution.",
        "The compared methods are not parameter matched. AlphaRAD Direct-anchor mixes heads through the low-rank mapping then restores native head normalization.",
        "Some diseases have only 1-3 positive test images; per-class counts are supplied. Three seed SDs do not measure this sampling uncertainty.",
        "", "model,dataset,method,macro_AUROC,macro_AUPRC,micro_AUPRC,delta_AUROC_pp,parameters,best_epochs"]
    for r in rows:
        def fmt(key):
            mean, sd = r[key+"_mean"], r[key+"_std"]
            return f"{mean:.2f}" if sd is None else f"{mean:.2f} +/- {sd:.2f}"
        lines.append(f"{r['model']},{r['dataset']},{r['method']},{fmt('macro_auroc')},{fmt('macro_auprc')},{fmt('micro_auprc')},{r['delta_auroc_pp']:+.2f},{r['trainable_parameters']},{r['best_epochs']}")
    (root / "RESULTS.txt").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    colors = ["#64748b", "#e9a23b", "#267aaf", "#3f9a70"]
    for row_index, model in enumerate(("radzero", "alpharad")):
        for col, dataset in enumerate(datasets):
            ax = axes[row_index,col]
            selected = [r for r in rows if r["model"]==model and r["dataset"]==dataset]
            ax.bar(range(4), [r["macro_auroc_mean"] for r in selected], color=colors,
                   yerr=[r["macro_auroc_std"] or 0 for r in selected], capsize=3)
            ax.set_xticks(range(4), ["Original", "Bias", "Direct", "Existing"])
            ax.set_ylim(0, 100); ax.set_title(f"{model} / {dataset}")
            ax.set_ylabel("Macro AUROC (%)"); ax.grid(axis="y", alpha=.2)
            for i,r in enumerate(selected): ax.text(i, r["macro_auroc_mean"]+1.8, f"{r['macro_auroc_mean']:.1f}", ha="center", fontsize=9)
    fig.suptitle("Classification adaptation on fixed custom splits (error bars: training-seed SD)")
    fig.tight_layout(); fig.savefig(root / "comparison.png", dpi=180); fig.savefig(root / "comparison.pdf")
    print("\n".join(lines[-25:]))


if __name__ == "__main__":
    main()
