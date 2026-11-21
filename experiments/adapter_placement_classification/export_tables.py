"""Export three LaTeX tables only after all 72 fits pass report.py's audit.

The fragment requires booktabs and graphicx. It does not modify the manuscript.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

MODELS = ("AlphaRAD", "KAD", "MaCo")
DATASETS = ("openi", "chexpert", "chestxray14")
NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
METRICS = ("macro_auroc", "macro_auprc")


def finite(value, name, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Invalid numeric field: " + name)
    return value


def load_inputs(run):
    audit_bytes = (run / "AUDIT.json").read_bytes()
    audit = json.loads(audit_bytes)
    if (audit.get("status") != "passed" or audit.get("errors") or audit.get("missing_fits")
            or audit.get("completed_fits") != 72 or audit.get("expected_fits") != 72
            or audit.get("completed_new_fits") != 45 or audit.get("completed_reused_fits") != 27):
        raise ValueError("Refusing partial or failed tables: all 72 fits (45 new + 27 reused) must pass the audit")
    protocol_bytes = (run / "PROTOCOL.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    if hashlib.sha256(protocol_bytes).hexdigest() != audit["protocol_sha256"]:
        raise ValueError("Protocol hash differs from completed audit")
    expected_config = dict(lr=1e-4, effective_batch_size=32, epochs=20, min_epochs=8, patience=5)
    if (protocol["seeds"] != [42, 43, 44] or protocol["trainable_parameters"] != 99136
            or any(protocol["train"][key] != value for key, value in expected_config.items())):
        raise ValueError("Protocol does not match the captions' fixed comparison settings")

    summary_bytes = (run / "summary.csv").read_bytes()
    groups = {}
    expected = {(model, dataset, "text_guided") for model in MODELS for dataset in DATASETS}
    expected.update(("MaCo", dataset, "native_mean") for dataset in DATASETS)
    for row in csv.DictReader(io.StringIO(summary_bytes.decode("utf-8-sig"))):
        key = row["model"], row["dataset"], row["pooling"]
        if key not in expected or key in groups:
            raise ValueError("Unexpected or duplicate summary row: " + repr(key))
        if (row["complete"].strip().lower() != "true"
                or any(row[field] != "3" for field in ("pre_n", "post_n", "paired_n"))):
            raise ValueError("Incomplete three-seed summary row: " + repr(key))
        methods = ("native_mean_pre", "native_mean_post") if key[2] == "native_mean" else ("pre_mlp", "post_mlp")
        if (row["pre_method"], row["post_method"]) != methods:
            raise ValueError("Unexpected method names: " + repr(key))
        for metric in METRICS:
            for prefix in ("pre", "post"):
                mean = finite(row[f"{prefix}_{metric}_mean_pct"], prefix + metric)
                if not 0 <= mean <= 100:
                    raise ValueError("Percentage outside [0, 100]: " + repr(key))
                finite(row[f"{prefix}_{metric}_sd_pct"], prefix + metric + " SD", nonnegative=True)
            delta = finite(row[f"pre_minus_post_{metric}_mean_pp"], metric + " delta")
            finite(row[f"pre_minus_post_{metric}_sd_pp"], metric + " paired SD", nonnegative=True)
            difference = float(row[f"pre_{metric}_mean_pct"]) - float(row[f"post_{metric}_mean_pct"])
            if not math.isclose(delta, difference, abs_tol=1e-9, rel_tol=1e-9):
                raise ValueError("Summary delta arithmetic mismatch: " + repr(key))
        groups[key] = row
    if set(groups) != expected:
        raise ValueError("Expected all 12 complete summary rows")

    controls = {}
    for row in audit["maco_pooling_controls"]:
        dataset = row["dataset"]
        if dataset not in DATASETS or dataset in controls or row["model"] != "MaCo" or row["trainable_parameters"] != 0:
            raise ValueError("Unexpected no-MLP control row")
        for metric in METRICS:
            native = finite(row[f"native_mean_{metric}_pct"], metric + " native control")
            guided = finite(row[f"text_guided_{metric}_pct"], metric + " guided control")
            delta = finite(row[f"text_guided_minus_native_{metric}_pp"], metric + " control delta")
            if not (0 <= native <= 100 and 0 <= guided <= 100
                    and math.isclose(delta, guided - native, abs_tol=1e-9, rel_tol=1e-9)):
                raise ValueError("Invalid no-MLP control metrics")
        controls[dataset] = row
    if set(controls) != set(DATASETS):
        raise ValueError("Expected all three audited no-MLP controls")
    hashes = {"summary.csv": hashlib.sha256(summary_bytes).hexdigest(),
              "AUDIT.json": hashlib.sha256(audit_bytes).hexdigest(),
              "PROTOCOL.json": hashlib.sha256(protocol_bytes).hexdigest()}
    return groups, controls, hashes


def metric_pair(row, prefix, delta=False):
    parts = []
    for metric in METRICS:
        unit = "pp" if delta else "pct"
        mean = float(row[f"{prefix}_{metric}_mean_{unit}"])
        sd = float(row[f"{prefix}_{metric}_sd_{unit}"])
        mean_text = f"{mean:+.2f}" if delta else f"{mean:.2f}"
        parts.append(mean_text + r"\!\pm\!" + f"{sd:.2f}")
    return "$" + r"\,/\,".join(parts) + "$"


def frozen_pair(row, prefix, delta=False):
    unit = "pp" if delta else "pct"
    return "$" + r"\,/\,".join(
        f"{float(row[f'{prefix}_{metric}_{unit}']):+.3f}" if delta else f"{float(row[f'{prefix}_{metric}_{unit}']):.2f}"
        for metric in METRICS) + "$"


def table(caption, label, columns, header, rows, resize=True):
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.14}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [r"\begin{tabular}{" + columns + "}", r"\toprule", header,
              r"\midrule", *rows, r"\bottomrule", r"\end{tabular}%"]
    if resize:
        lines.append("}")
    return "\n".join(lines + [r"\end{table*}", ""])


def render_tables(groups, controls, hashes):
    lines = ["% Generated by adapter_placement_classification/export_tables.py; requires booktabs and graphicx.",
             "% Three-seed sample SD; no cross-dataset average or significance claim."]
    lines += [f"% {name} SHA256: {digest}" for name, digest in hashes.items()]
    main_rows = []
    for model_index, model in enumerate(MODELS):
        if model_index:
            main_rows.append(r"\addlinespace[3pt]")
        for dataset_index, dataset in enumerate(DATASETS):
            row = groups[model, dataset, "text_guided"]
            name = r"\textbf{" + model + "}" if dataset_index == 0 else model
            main_rows.append(" & ".join((name, NAMES[dataset], metric_pair(row, "pre"),
                                        metric_pair(row, "post"), metric_pair(row, "pre_minus_post", delta=True))) + r" \\")
    lines.append(table(
        r"\textbf{Matched pre- versus post-aggregation MLP placement.} "
        r"Each cell reports macro AUROC / macro AUPRC (average precision). Pre/post scores are percentages; "
        r"$\Delta$ is pre minus post in percentage points. Values are mean $\pm$ sample SD over seeds 42, 43, and 44; "
        r"the SD of $\Delta$ is computed from paired seed differences and is not a confidence interval. "
        r"All MLPs have 99,136 trainable parameters, learning rate $10^{-4}$, effective batch size 32, and at most "
        r"20 epochs, with the same validation-based early stopping and epoch-zero eligibility. "
        r"AlphaRAD and KAD retain native disease fusion; MaCo uses fixed text-guided pooling. "
        r"The 27 pre fits are reused; the 27 post fits are newly trained. No significance claim is made.",
        "tab:placement_ablation_pre_post", "llccc",
        r"Model & Dataset & Pre MLP & Post MLP & $\Delta$ pre$-$post \\", main_rows))

    factorial_rows = []
    for dataset in DATASETS:
        native, guided = groups["MaCo", dataset, "native_mean"], groups["MaCo", dataset, "text_guided"]
        factorial_rows.append(" & ".join((NAMES[dataset], metric_pair(native, "pre"), metric_pair(native, "post"),
                                          metric_pair(guided, "pre"), metric_pair(guided, "post"))) + r" \\")
    lines.append(table(
        r"\textbf{MaCo pooling $\times$ placement control.} Cells report macro AUROC / macro AUPRC (\%), "
        r"mean $\pm$ sample SD over three fitting seeds. All four conditions use the same 99,136-parameter MLP "
        r"and the learning rate, batch size, and epoch budget in Table~\ref{tab:placement_ablation_pre_post}. "
        r"Native mean pooling and fixed text-guided pooling introduce no trainable pooling parameters. "
        r"The text-guided pre column reuses existing fits; the other three conditions are new fits. "
        r"Datasets are reported separately; seed SD is not a confidence interval.",
        "tab:placement_ablation_maco_factorial", "lcccc",
        "\n".join([r"Dataset & \multicolumn{2}{c}{Native mean pooling} & \multicolumn{2}{c}{Text-guided pooling} \\",
                    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"& Pre MLP & Post MLP & Pre MLP & Post MLP \\"]), factorial_rows))

    control_rows = []
    for dataset in DATASETS:
        row = controls[dataset]
        control_rows.append(" & ".join((NAMES[dataset], frozen_pair(row, "native_mean"),
                                        frozen_pair(row, "text_guided"),
                                        frozen_pair(row, "text_guided_minus_native", delta=True))) + r" \\")
    lines.append(table(
        r"\textbf{MaCo frozen pooling controls without an MLP.} Cells report macro AUROC / macro AUPRC (\%); "
        r"$\Delta$ is text-guided minus native mean pooling, in percentage points (three decimals). "
        r"Both controls have zero trainable parameters and one deterministic result per dataset, so no seed SD "
        r"is attached. Metrics are recomputed from the audited cached scores. These controls isolate the "
        r"pooling change before supervised MLP fitting; no cross-dataset mean or significance claim is provided.",
        "tab:placement_ablation_maco_frozen_controls", "lccc",
        r"Dataset & Native mean, no MLP & Text-guided, no MLP & $\Delta$ guided$-$native \\", control_rows, resize=False))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        inputs = load_inputs(run)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = run / "placement_tables.tex"
    temporary = output.with_suffix(".tex.tmp")
    temporary.write_text(render_tables(*inputs))
    temporary.replace(output)
    print("Wrote", output)


if __name__ == "__main__":
    main()
