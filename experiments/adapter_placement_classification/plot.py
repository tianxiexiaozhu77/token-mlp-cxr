"""Plot the nine complete paired pre-minus-post comparisons; never partial rows.

Input: report.py's summary.csv. Error bars are sample SDs of paired seed
differences (n=3), not confidence intervals or statistical-significance tests.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

MODELS = ("AlphaRAD", "KAD", "MaCo")
DATASETS = ("openi", "chexpert", "chestxray14")
DATASET_NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
METRICS = ("macro_auroc", "macro_auprc")
# Okabe-Ito colors; model names also label every row, so color is redundant.
COLORS = {"AlphaRAD": "#0072B2", "KAD": "#D55E00", "MaCo": "#009E73"}


def read_complete_summary(path):
    """Return nine validated rows in fixed model/dataset order, plus CSV hash."""
    payload = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"model", "dataset", "pooling", "pre_method", "post_method",
                "pre_n", "post_n", "paired_n", "complete"}
    for metric in METRICS:
        required.update((f"pre_minus_post_{metric}_mean_pp", f"pre_minus_post_{metric}_sd_pp",
                         f"pre_{metric}_mean_pct", f"post_{metric}_mean_pct"))
    missing_columns = required.difference(reader.fieldnames or ())
    if missing_columns:
        raise ValueError("Summary is missing columns: " + ", ".join(sorted(missing_columns)))
    selected = {}
    for row in reader:
        if row["pooling"] != "text_guided":
            continue
        key = row["model"], row["dataset"]
        if key[0] not in MODELS or key[1] not in DATASETS:
            raise ValueError("Unexpected main-comparison row: " + repr(key))
        if key in selected:
            raise ValueError("Duplicate main-comparison row: " + repr(key))
        selected[key] = row
    expected = [(model, dataset) for model in MODELS for dataset in DATASETS]
    missing = [key for key in expected if key not in selected]
    incomplete = [key for key, row in selected.items()
                  if row["complete"].strip().lower() != "true"
                  or any(row[field].strip() != "3" for field in ("pre_n", "post_n", "paired_n"))]
    if missing or incomplete:
        pending = ", ".join("/".join(key) for key in missing + incomplete)
        raise ValueError("Refusing partial figure: all 9 comparisons need 3 paired seeds. Pending: " + pending)
    ordered = []
    for key in expected:
        row = selected[key]
        if row["pre_method"] != "pre_mlp" or row["post_method"] != "post_mlp":
            raise ValueError("Unexpected placement methods: " + repr(key))
        result = {"model": key[0], "dataset": key[1]}
        for metric in METRICS:
            mean = float(row[f"pre_minus_post_{metric}_mean_pp"])
            sd = float(row[f"pre_minus_post_{metric}_sd_pp"])
            pre = float(row[f"pre_{metric}_mean_pct"])
            post = float(row[f"post_{metric}_mean_pct"])
            if not all(math.isfinite(value) for value in (mean, sd, pre, post)) or sd < 0:
                raise ValueError("Nonfinite metric or negative SD: " + repr(key))
            if not 0 <= pre <= 100 or not 0 <= post <= 100:
                raise ValueError("Metrics must be percentages in [0, 100]: " + repr(key))
            if not math.isclose(mean, pre - post, abs_tol=1e-9, rel_tol=1e-9):
                raise ValueError("Paired mean delta disagrees with pre minus post: " + repr(key))
            result[metric] = dict(mean_pp=mean, paired_seed_sd_pp=sd)
        ordered.append(result)
    return ordered, hashlib.sha256(payload).hexdigest()


def make_figure(rows):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 6.4), sharey=True)
    figure.subplots_adjust(left=.225, right=.98, top=.86, bottom=.22, wspace=.25)
    ys = [model_index * 4 + dataset_index
          for model_index in range(len(MODELS)) for dataset_index in range(len(DATASETS))]
    labels = [f"{row['model']}  /  {DATASET_NAMES[row['dataset']]}" for row in rows]
    for panel, metric in enumerate(METRICS):
        axis = axes[panel]
        for bottom, top in ((-.55, 2.55), (7.45, 10.55)):
            axis.axhspan(bottom, top, facecolor="#F4F5F6", edgecolor="none", zorder=0)
        axis.axvline(0, color="#555555", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        extent = max(abs(row[metric]["mean_pp"]) + row[metric]["paired_seed_sd_pp"] for row in rows)
        span = max(1., extent * 1.15)
        for y, row in zip(ys, rows):
            value = row[metric]
            axis.errorbar(value["mean_pp"], y, xerr=value["paired_seed_sd_pp"],
                          fmt="o", markersize=6.2, color=COLORS[row["model"]],
                          markeredgecolor="white", markeredgewidth=.7,
                          elinewidth=1.5, capsize=3.1, capthick=1.2, zorder=3)
        axis.set_xlim(-span, span)
        axis.set_ylim(10.7, -.7)
        axis.set_yticks(ys)
        axis.set_yticklabels(labels)
        axis.tick_params(axis="both", length=0, pad=7)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.grid(axis="x", color="#E0E2E5", linewidth=.65, zorder=0)
        axis.set_axisbelow(True)
        for name in ("top", "right", "left"):
            axis.spines[name].set_visible(False)
        axis.spines["bottom"].set_color("#B8BCC2")
        axis.set_xlabel("Pre − post (percentage points)", labelpad=9)
        axis.set_title(("(a) Macro AUROC", "(b) Macro AUPRC")[panel], loc="left", pad=14)
    axes[1].tick_params(axis="y", labelleft=False)
    for tick, row in zip(axes[0].get_yticklabels(), rows):
        tick.set_color(COLORS[row["model"]])
    figure.suptitle("Effect of MLP placement across frozen chest X-ray models", fontsize=13.5, y=.965)
    figure.text(.225, .125,
                "Points: mean paired difference over seeds 42, 43, 44.  Bars: ±1 sample SD (n = 3), not confidence intervals.",
                fontsize=9, ha="left", color="#30343B")
    figure.text(.225, .075,
                "Positive: pre-aggregation higher.  Negative: post-aggregation higher.  No statistical significance is implied.",
                fontsize=9, ha="left", color="#30343B")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        rows, summary_hash = read_complete_summary(run / "summary.csv")
        audit_path = run / "AUDIT.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit.get("errors") or audit.get("status") == "failed":
                raise ValueError("Report audit failed; refusing to plot its summary")
        protocol_path = run / "PROTOCOL.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text())["seeds"] != [42, 43, 44]:
            raise ValueError("This figure expects the locked seeds 42, 43, 44")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    figure = make_figure(rows)
    temporaries = []
    try:
        for extension in ("pdf", "svg"):
            destination = run / f"placement_delta.{extension}"
            temporary = destination.with_name(f".{destination.stem}.tmp.{extension}")
            figure.savefig(temporary, format=extension, facecolor="white", bbox_inches="tight", pad_inches=.15)
            temporaries.append((temporary, destination))
        for temporary, destination in temporaries:
            temporary.replace(destination)
    finally:
        from matplotlib import pyplot as plt
        plt.close(figure)
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)
    metadata = dict(summary_csv=str(run / "summary.csv"), summary_sha256=summary_hash,
                    plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    comparisons=9, paired_seeds=[42, 43, 44],
                    delta="pre minus post, percentage points",
                    error_bars="plus/minus one sample SD (ddof=1) of paired seed differences; not a confidence interval",
                    statistical_significance_claimed=False, rows=rows)
    (run / "placement_delta_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print("Wrote", run / "placement_delta.pdf", "and", run / "placement_delta.svg")


if __name__ == "__main__":
    main()
