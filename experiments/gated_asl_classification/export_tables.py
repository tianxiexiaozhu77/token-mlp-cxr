"""Export separate dataset tables for the 252-fit fresh gating/loss campaign.

Final export requires the complete independent audit. --allow-partial permits
explicitly marked previews, never audit errors or inconsistent CSV statistics.
--compile makes a landscape A4 PDF with one dataset per page. No manuscript or
runtime source is changed. Only small reporting files are read.
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
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gated_asl_classification.common import MODELS, METHODS, SEEDS, ASL_CONFIG, GATE_CONFIG, parameter_count

DATASETS = ("openi", "chexpert", "chestxray14")
NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
COLUMNS = ("original", *METHODS)
METRICS = ("macro_auroc", "macro_auprc")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def load_inputs(run, allow_partial=False):
    data = {name: (run / name).read_bytes() for name in ("summary.csv", "PROTOCOL.json", "AUDIT.json")}
    protocol, audit = json.loads(data["PROTOCOL.json"]), json.loads(data["AUDIT.json"])
    hashes = {name: digest(value) for name, value in data.items()}
    require(audit.get("status") in ("passed", "partial") and not audit.get("errors"), "Refusing failed or unsupported audit")
    require(audit.get("protocol_sha256") == hashes["PROTOCOL.json"], "Protocol differs from audit")
    require(audit.get("expected_new_fits") == 252 and audit.get("expected_reference_evaluations") == 120
            and audit.get("completed_reference_evaluations") == 120, "Require 252 planned fresh fits and all 120 historical evaluations")
    count = audit.get("completed_new_fits")
    require(isinstance(count, int) and 0 <= count <= 252, "Invalid completed-fit count")
    complete = count == 252 and not audit.get("pending")
    require((audit["status"] == "passed") == complete, "Inconsistent audit completion status")
    require(allow_partial or complete, "Final tables require AUDIT passed and 252/252 fits; --allow-partial creates a marked preview")
    require(protocol["models"] == list(MODELS) and protocol["methods"] == list(METHODS)
            and protocol["seeds"] == list(SEEDS) and set(protocol["datasets"]) == set(DATASETS)
            and protocol["new_training_runs"] == 252 and protocol["all_methods_fresh"] is True
            and protocol["reused_training_runs"] == 0, "Unexpected experiment matrix or reused fits")
    require(protocol["asl_config"] == ASL_CONFIG and protocol["gate_config"] == GATE_CONFIG, "Loss/gate configuration differs from captions")
    audited, keys = defaultdict(dict), set()
    for entry in [*audit["runs"], *audit["historical_references"]]:
        key = tuple(entry["key"])
        require(key not in keys, "Duplicate audited run: " + repr(key))
        keys.add(key)
        model, dataset, seed, method = key
        if method in COLUMNS:
            audited[model, dataset, method][seed] = entry
    groups, fresh_total = {}, 0
    for row in csv.DictReader(io.StringIO(data["summary.csv"].decode("utf-8-sig"))):
        key = row["model"], row["dataset"]
        model, dataset = key
        require(model in MODELS and dataset in DATASETS and key not in groups, "Unexpected/duplicate summary row: " + repr(key))
        dataset_spec, model_spec = protocol["datasets"][dataset], protocol["model_specs"][model]
        require(int(row["classes"]) == len(dataset_spec["classes"])
                and int(row["test_images"]) == dataset_spec["counts"]["test"], "Dataset summary mismatch")
        for method in COLUMNS:
            expected = 1 if method == "original" else 3
            runs = audited[model, dataset, method]
            number = int(row[method + "_runs"])
            status = "complete" if number == expected else "partial"
            require(number == len(runs) and 0 <= number <= expected
                    and int(row[method + "_expected_runs"]) == expected
                    and row[method + "_status"] == status
                    and row[method + "_seeds"] == ",".join(str(seed) for seed in sorted(runs))
                    and set(runs) <= ({42} if method == "original" else set(SEEDS)), "Summary run identities differ: " + repr((key, method)))
            if method == "original":
                require(number == 1, "Historical Original must be complete")
            else:
                fresh_total += number
                require(model_spec["trainable_parameters_by_method"][method] == parameter_count(model_spec["dim"], 64, method), "Incorrect adapter parameter budget")
            for metric in METRICS:
                prefix = method + "_" + metric
                values = [100 * runs[seed]["recomputed_test"][metric] for seed in sorted(runs)]
                if not values:
                    require(not row[prefix + "_mean_pct"] and not row[prefix + "_sd_pct"], "Invented metric for pending group")
                    continue
                mean = float(row[prefix + "_mean_pct"])
                require(math.isfinite(mean) and 0 <= mean <= 100 and math.isclose(mean, statistics.mean(values), abs_tol=1e-9, rel_tol=1e-10), "CSV mean differs from audited predictions: " + repr((key, prefix)))
                if len(values) == 1:
                    require(not row[prefix + "_sd_pct"], "A single evaluation cannot have a seed SD")
                else:
                    sd = float(row[prefix + "_sd_pct"])
                    require(math.isfinite(sd) and sd >= 0 and math.isclose(sd, statistics.stdev(values), abs_tol=1e-9, rel_tol=1e-10), "CSV sample SD differs from audited predictions: " + repr((key, prefix)))
        groups[key] = row
    require(set(groups) == {(model, dataset) for model in MODELS for dataset in DATASETS}
            and fresh_total == count, "Require all 21 rows and the audited fresh-fit count")
    return groups, protocol, audit, hashes


def metric_pair(row, method):
    count = int(row[method + "_runs"])
    if not count:
        return r"\textit{pending (0/3)}"
    values = []
    for metric in METRICS:
        prefix = method + "_" + metric
        text = f"{float(row[prefix + '_mean_pct']):.2f}"
        if row[prefix + "_sd_pct"]:
            text += r"\!\pm\!" + f"{float(row[prefix + '_sd_pct']):.2f}"
        values.append(text)
    pair = "$" + r"\,/\,".join(values) + "$"
    return (r"\shortstack{" + pair + r"\\{\scriptsize partial " + str(count) + r"/3}}"
            if row[method + "_status"] != "complete" else pair)


def comments(hashes, audit):
    return "\n".join(["% Generated by gated_asl_classification/export_tables.py; requires booktabs and graphicx.",
        "% " + ("FINAL: all 252 fresh fits audited." if audit["status"] == "passed" else f"PARTIAL PREVIEW: {audit['completed_new_fits']}/252 fresh fits audited."),
        "% Percentages; sample SD (ddof=1), not a patient-level confidence interval.",
        *[f"% {name} SHA256: {value}" for name, value in hashes.items()], ""])


def render_table(dataset, groups, protocol, audit):
    prefix = (r"\textbf{PARTIAL PREVIEW: " + str(audit["completed_new_fits"]) + r"/252 fits.} " if audit["status"] != "passed" else "")
    spec = protocol["datasets"][dataset]
    caption = (prefix + r"\textbf{" + NAMES[dataset] + r": pre-token gating and loss ablation.} "
        + f"{len(spec['classes'])} labels and {spec['counts']['test']:,} test images. "
        + r"Cells show macro AUROC / macro AUPRC (average precision), in percent. "
        r"The four adaptation groups are freshly trained; completed groups report mean $\pm$ sample SD over seeds 42, 43, and 44. "
        r"Original is a deterministic frozen-model evaluation without seed SD. "
        r"BCE denotes class-weighted binary cross-entropy using TRAIN-only weights; ASL uses no class weights.")
    if audit["status"] != "passed":
        caption += r" Pending or partial groups are explicitly marked and are not final three-seed results."
    lines = [r"\begin{table*}[!ht]", r"\centering", r"\small", r"\caption{" + caption + "}",
        r"\label{tab:gated_asl_" + dataset + "}", r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.25}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule",
        r"Model & Original & Pre + BCE & Gated + BCE & Pre + ASL & Gated + ASL \\", r"\midrule"]
    for model in MODELS:
        lines.append(" & ".join([model, *(metric_pair(groups[model, dataset], method) for method in COLUMNS)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\par\vspace{5pt}", r"\begin{minipage}{\textwidth}\footnotesize",
        r"All adapters act before aggregation and use a shared residual GELU MLP with bottleneck 64; no post adapter is added. "
        r"Gated variants multiply the residual by a learned scalar sigmoid gate per token, initialized to 0.1, "
        r"and add $d+1$ trainable parameters. ASL uses $\gamma_{-}=4$, $\gamma_{+}=0$, "
        r"negative-probability shift 0.05, and a batch/class mean with full focal-gradient propagation. "
        r"Matched seeds share MLP initialization and training sample order across the four groups. "
        r"Validation macro AUROC selects the checkpoint, including epoch zero. "
        r"BiomedCLIP, BioViL-T, and MaCo retain added fixed text-guided pooling; their identity adapters recover that "
        r"pooling control rather than Original. Datasets are not averaged; seed SD is not a significance test.",
        r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(lines)


def render_wrapper(audit):
    title = ("Complete audited gating/loss ablation" if audit["status"] == "passed"
             else f"Partial gating/loss preview ({audit['completed_new_fits']}/252 fresh fits)")
    return "\n".join([r"\documentclass[10pt,a4paper,landscape]{article}", r"\usepackage[margin=1.6cm]{geometry}",
        r"\usepackage{booktabs,graphicx}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern,caption}",
        r"\captionsetup{font=small,labelfont=bf}", r"\newcommand{\GatedASLDatasetBreak}{\clearpage}",
        r"\begin{document}", r"\noindent{\large\bfseries " + title + r"}\par\vspace{4pt}",
        r"\input{gated_asl_tables.tex}", r"\clearpage", r"\end{document}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        groups, protocol, audit, hashes = load_inputs(run, args.allow_partial)
        header = comments(hashes, audit)
        fragments = [render_table(dataset, groups, protocol, audit) for dataset in DATASETS]
        outputs = {f"gated_asl_{dataset}_table.tex": header + fragment for dataset, fragment in zip(DATASETS, fragments)}
        outputs["gated_asl_tables.tex"] = header + r"\providecommand{\GatedASLDatasetBreak}{}" + "\n" + ("\n" + r"\GatedASLDatasetBreak" + "\n").join(fragments)
        outputs["gated_asl_tables_preview.tex"] = header + render_wrapper(audit)
        for name, value in outputs.items():
            atomic_text(run / name, value)
        build = None
        if args.compile:
            executable = shutil.which("pdflatex")
            require(executable is not None, "pdflatex unavailable; fragments were written")
            directory = run / "tables_build"
            directory.mkdir(exist_ok=True)
            command = [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-output-directory", str(directory), "gated_asl_tables_preview.tex"]
            logs = []
            for number in (1, 2):
                process = subprocess.run(command, cwd=run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                logs.append(f"PASS {number}\n" + process.stdout)
                atomic_text(directory / "build_stdout.log", "\n".join(logs))
                require(process.returncode == 0, "pdflatex failed: " + str(directory / "build_stdout.log"))
            shutil.copy2(directory / "gated_asl_tables_preview.pdf", run / "gated_asl_tables.pdf")
            build = dict(command=command, passes=2, returncode=0, log=str(directory / "build_stdout.log"))
        output_hashes = {name: digest((run / name).read_bytes()) for name in outputs}
        if args.compile:
            output_hashes["gated_asl_tables.pdf"] = digest((run / "gated_asl_tables.pdf").read_bytes())
        metadata = dict(status="complete" if audit["status"] == "passed" else "partial_preview", completed_new_fits=audit["completed_new_fits"],
            expected_new_fits=252, completed_reference_evaluations=120, methods=list(METHODS), datasets_separate=True,
            metric_order=list(METRICS), unit="percent", sd="sample SD (ddof=1); only 3/3 is a final fitted group",
            source_sha256=hashes, exporter_sha256=digest(Path(__file__).read_bytes()), outputs_sha256=output_hashes, pdf_build=build)
        atomic_text(run / "gated_asl_tables_metadata.json", json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(dict(status=metadata["status"], completed_new_fits=audit["completed_new_fits"], fragment=str(run / "gated_asl_tables.tex"),
                             pdf=str(run / "gated_asl_tables.pdf") if args.compile else None), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
