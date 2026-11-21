"""Export audited Original/Pre/Post/Both tables, separately for each dataset.

Default export requires all 63 Both fits to pass report.py. --allow-partial
permits clearly marked previews, but never failed audits or inconsistent CSV
statistics. --compile also builds a standalone PDF; no manuscript is edited.
The LaTeX fragments require booktabs and graphicx.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

MODELS = ("AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo")
DATASETS = ("openi", "chexpert", "chestxray14")
NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
METHODS = ("original", "pre_mlp", "post_mlp", "both_mlp")
POST_MODELS = ("AlphaRAD", "KAD", "MaCo")
METRICS = ("macro_auroc", "macro_auprc")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value, field, nonnegative=False):
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError("Invalid numeric field: " + field)
    return number


def atomic_text(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def load_inputs(run, allow_partial=False):
    data = {name: (run / name).read_bytes() for name in ("summary.csv", "PROTOCOL.json", "AUDIT.json")}
    protocol, audit = json.loads(data["PROTOCOL.json"]), json.loads(data["AUDIT.json"])
    hashes = {name: digest(value) for name, value in data.items()}
    if audit.get("errors") or audit.get("status") not in ("passed", "partial"):
        raise ValueError("Refusing tables from a failed or unsupported audit")
    if audit.get("protocol_sha256") != hashes["PROTOCOL.json"]:
        raise ValueError("Protocol hash differs from the audit")
    if (audit.get("expected_new_fits") != 63 or audit.get("expected_reference_evaluations") != 120
            or audit.get("completed_reference_evaluations") != 120):
        raise ValueError("Expected 63 Both fits and all 120 audited historical evaluations")
    completed = audit.get("completed_new_fits")
    if not isinstance(completed, int) or not 0 <= completed <= 63:
        raise ValueError("Invalid completed Both fit count")
    if not allow_partial and (audit["status"] != "passed" or completed != 63 or audit.get("pending")):
        raise ValueError("Final tables require AUDIT passed and all 63 Both fits; use --allow-partial for a marked preview")
    if (audit["status"] == "passed") != (completed == 63 and not audit.get("pending")):
        raise ValueError("Audit completion status is inconsistent")
    if (protocol["models"] != list(MODELS) or set(protocol["datasets"]) != set(DATASETS)
            or protocol["seeds"] != [42, 43, 44] or protocol["new_training_runs"] != 63
            or protocol["bottleneck"] != 64 or not protocol["joint_training"] or not protocol["independent_adapters"]):
        raise ValueError("Protocol differs from the table definitions")

    # Check CSV means/SDs against the audit's independent prediction-derived
    # metrics, so stale summaries cannot be packaged with a newer audit.
    audited, unique = defaultdict(dict), set()
    for entry in audit["runs"]:
        model, dataset, seed, method = entry["key"]
        key = tuple(entry["key"])
        if key in unique:
            raise ValueError("Duplicate audited run: " + repr(key))
        unique.add(key)
        if method in METHODS:
            audited[model, dataset, method][seed] = entry
    groups, total_both = {}, 0
    for row in csv.DictReader(io.StringIO(data["summary.csv"].decode("utf-8-sig"))):
        key = row["model"], row["dataset"]
        model, dataset = key
        if model not in MODELS or dataset not in DATASETS or key in groups:
            raise ValueError("Unexpected or duplicate summary row: " + repr(key))
        spec = protocol["datasets"][dataset]
        if (int(row["classes"]) != len(spec["classes"]) or int(row["test_images"]) != spec["counts"]["test"]
                or int(row["both_trainable_parameters"]) != protocol["model_specs"][model]["trainable_parameters"]):
            raise ValueError("Summary dataset or architecture mismatch: " + repr(key))
        for method in METHODS:
            expected = 1 if method == "original" else 0 if method == "post_mlp" and model not in POST_MODELS else 3
            runs = audited[model, dataset, method]
            count = int(row[method + "_runs"])
            wanted_status = "unavailable" if not expected else "complete" if count == expected else "partial"
            if (int(row[method + "_expected_runs"]) != expected or count != len(runs)
                    or not 0 <= count <= expected or row[method + "_status"] != wanted_status
                    or row[method + "_seeds"] != ",".join(str(seed) for seed in sorted(runs))
                    or not set(runs) <= ({42} if method == "original" else {42, 43, 44})):
                raise ValueError("Summary run identity mismatch: " + repr((key, method)))
            if method != "both_mlp" and count != expected:
                raise ValueError("Historical comparison is incomplete: " + repr((key, method)))
            for metric in METRICS:
                field = method + "_" + metric
                values = [100 * runs[seed]["recomputed_test"][metric] for seed in sorted(runs)]
                if not values:
                    if row[field + "_mean_pct"] or row[field + "_sd_pct"]:
                        raise ValueError("Summary invents an unavailable metric")
                    continue
                mean = finite(row[field + "_mean_pct"], field)
                if not 0 <= mean <= 100 or not math.isclose(mean, statistics.mean(values), rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError("CSV mean differs from audited runs: " + repr((key, field)))
                if len(values) == 1:
                    if row[field + "_sd_pct"]:
                        raise ValueError("A single evaluation cannot have seed SD")
                else:
                    sd = finite(row[field + "_sd_pct"], field + " SD", nonnegative=True)
                    if not math.isclose(sd, statistics.stdev(values), rel_tol=1e-10, abs_tol=1e-9):
                        raise ValueError("CSV sample SD differs from audited runs: " + repr((key, field)))
            if method == "both_mlp":
                total_both += count
        groups[key] = row
    if set(groups) != {(m, d) for m in MODELS for d in DATASETS} or total_both != completed:
        raise ValueError("Expected all 21 dataset/model rows and the audited completed-fit count")
    return groups, protocol, audit, hashes


def metric_pair(row, method):
    if row[method + "_status"] == "unavailable":
        return r"\textemdash"
    count = int(row[method + "_runs"])
    if not count:
        return r"\textit{pending (0/3)}"
    parts = []
    for metric in METRICS:
        prefix = method + "_" + metric
        value = f"{float(row[prefix + '_mean_pct']):.2f}"
        if row[prefix + "_sd_pct"]:
            value += r"\!\pm\!" + f"{float(row[prefix + '_sd_pct']):.2f}"
        parts.append(value)
    pair = "$" + r"\,/\,".join(parts) + "$"
    if row[method + "_status"] != "complete":
        return r"\shortstack{" + pair + r"\\{\scriptsize partial " + str(count) + r"/3}}"
    return pair


def provenance_comments(hashes, partial, completed):
    return "\n".join([
        "% Generated by both_adapter_classification/export_tables.py; requires booktabs and graphicx.",
        "% " + (f"PARTIAL PREVIEW: {completed}/63 Both fits audited." if partial else "FINAL: 63/63 Both fits audited."),
        "% Metrics are percentages. SD is sample SD (ddof=1), not a confidence interval.",
        *[f"% {name} SHA256: {value}" for name, value in hashes.items()], ""])


def render_table(dataset, groups, protocol, audit):
    partial = audit["status"] != "passed"
    preview = (r"\textbf{PARTIAL PREVIEW: " + str(audit["completed_new_fits"]) + r"/63 Both fits complete.} " if partial else "")
    classes = len(protocol["datasets"][dataset]["classes"])
    images = protocol["datasets"][dataset]["counts"]["test"]
    caption = (preview + r"\textbf{" + NAMES[dataset] + r": joint pre- and post-aggregation adaptation.} "
        + f"{classes} labels, {images:,} test images. "
        + r"Each cell is macro AUROC / macro AUPRC (average precision), in percent. "
        r"Fitted methods report mean $\pm$ sample SD over seeds 42, 43, and 44; "
        r"Original is a deterministic evaluation without seed SD. "
        r"Pre and available Post results are reused; Both jointly fits two independent rank-64 residual MLPs "
        r"with twice the single-MLP parameter count. An em dash denotes an unrun Post condition.")
    if partial:
        caption += r" Incomplete Both cells are explicitly marked and must not be reported as final three-seed results."
    lines = [r"\begin{table*}[!ht]", r"\centering", r"\small", r"\caption{" + caption + "}",
        r"\label{tab:both_mlp_" + dataset + "}", r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.16}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccc}",
        r"\toprule", r"Model & Original & Pre Token MLP & Post MLP & Both MLP \\", r"\midrule"]
    for model in MODELS:
        row = groups[model, dataset]
        lines.append(" & ".join([model, *(metric_pair(row, method) for method in METHODS)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\par\vspace{3pt}",
        r"\begin{minipage}{\textwidth}\footnotesize",
        r"Both retains the existing Token MLP aggregation. For BiomedCLIP, BioViL-T, and MaCo, "
        r"this includes added fixed text-guided pooling; identity adapters recover that pooling control, "
        r"which differs from Original. Best checkpoints use validation macro AUROC with epoch zero eligible. "
        r"Seed SD does not establish statistical significance. Datasets are not averaged.",
        r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(lines)


def render_preview(partial, completed):
    title = "Both MLP adaptation: " + (f"partial preview ({completed}/63 fits)" if partial else "complete audited results")
    return "\n".join([
        r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=1.8cm]{geometry}",
        r"\usepackage{booktabs,graphicx}", r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}", r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\setlength{\textfloatsep}{13pt}", r"\setlength{\floatsep}{12pt}",
        r"\newcommand{\BothDatasetBreak}{\clearpage}", r"\begin{document}",
        r"\noindent{\large\bfseries " + title + r"}\\[4pt]",
        r"\noindent Original / Pre / Post / Both; macro AUROC and macro AUPRC. "
        r"Results remain separated by dataset.\par\vspace{3pt}",
        r"\input{both_tables.tex}", r"\clearpage", r"\end{document}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Build both_tables.pdf using pdflatex")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        groups, protocol, audit, hashes = load_inputs(run, args.allow_partial)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    partial, completed = audit["status"] != "passed", audit["completed_new_fits"]
    header = provenance_comments(hashes, partial, completed)
    fragments = [render_table(dataset, groups, protocol, audit) for dataset in DATASETS]
    outputs = {}
    for dataset, fragment in zip(DATASETS, fragments):
        outputs[f"both_{dataset}_table.tex"] = header + fragment
    outputs["both_tables.tex"] = (header + r"\providecommand{\BothDatasetBreak}{}" + "\n"
        + fragments[0] + "\n" + fragments[1] + "\n" + r"\BothDatasetBreak" + "\n" + fragments[2])
    outputs["both_tables_preview.tex"] = header + render_preview(partial, completed)
    for name, text in outputs.items():
        atomic_text(run / name, text)
    build = None
    if args.compile:
        executable = shutil.which("pdflatex")
        if not executable:
            parser.error("pdflatex is unavailable; LaTeX fragments were written")
        directory = run / "tables_build"
        directory.mkdir(exist_ok=True)
        command = [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                   "-output-directory", str(directory), "both_tables_preview.tex"]
        logs = []
        for pass_number in (1, 2):
            process = subprocess.run(command, cwd=run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logs.append(f"PASS {pass_number}\n" + process.stdout)
            atomic_text(directory / "build_stdout.log", "\n".join(logs))
            if process.returncode:
                parser.error("pdflatex failed; inspect " + str(directory / "build_stdout.log"))
        pdf = directory / "both_tables_preview.pdf"
        shutil.copy2(pdf, run / "both_tables.pdf")
        build = dict(command=command, passes=2, returncode=process.returncode, log=str(directory / "build_stdout.log"))
    metadata = dict(status="partial_preview" if partial else "complete", completed_new_fits=completed,
        expected_new_fits=63, datasets_separate=True, metric_order=list(METRICS), unit="percent",
        sd="sample SD (ddof=1) across available seeds; only 3/3 is final", source_sha256=hashes,
        exporter_sha256=digest(Path(__file__).read_bytes()),
        outputs_sha256={name: digest((run / name).read_bytes()) for name in outputs}, pdf_build=build)
    if args.compile:
        metadata["outputs_sha256"]["both_tables.pdf"] = digest((run / "both_tables.pdf").read_bytes())
    atomic_text(run / "both_tables_metadata.json", json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(status=metadata["status"], completed_new_fits=completed,
                         fragment=str(run / "both_tables.tex"), pdf=str(run / "both_tables.pdf") if args.compile else None), indent=2))


if __name__ == "__main__":
    main()
