"""Seal final small artifacts after report.py and export_tables.py --compile.

Requires all 63 Both fits and a complete audited PDF. No token arrays or model
checkpoints are read. The original runtime/extra-source archives are untouched;
postprocessing and I/O helper sources get their own supplemental archive.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = ("AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo")
DATASETS = ("openi", "chexpert", "chestxray14")
TABLE_OUTPUTS = ("both_tables.tex", "both_tables_preview.tex", "both_tables.pdf",
                 "both_openi_table.tex", "both_chexpert_table.tex", "both_chestxray14_table.tex")
FILES = ("PROTOCOL.json", "source_manifest.json", "EXTRA_SOURCE_MANIFEST.json", "AUDIT.json",
         "summary.csv", "per_seed.csv", "aggregation_controls.csv", "REPORT.txt", "training_matrix.json",
         "both_tables_metadata.json", *TABLE_OUTPUTS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def inspect_inputs(run):
    protocol, audit, table = (read(run / name) for name in ("PROTOCOL.json", "AUDIT.json", "both_tables_metadata.json"))
    phash = sha256(run / "PROTOCOL.json")
    require(audit["status"] == "passed" and not audit["errors"] and not audit["pending"], "AUDIT must pass without pending fits or errors")
    require(audit["expected_new_fits"] == audit["completed_new_fits"] == 63, "All 63 Both fits must be audited")
    require(audit["expected_reference_evaluations"] == audit["completed_reference_evaluations"] == 120, "All 120 historical evaluations must be audited")
    require(audit["protocol_sha256"] == phash and protocol["new_training_runs"] == 63, "Protocol/audit mismatch")
    require(protocol["models"] == list(MODELS) and set(protocol["datasets"]) == set(DATASETS) and protocol["seeds"] == [42, 43, 44], "Unexpected experiment matrix")
    require(table["status"] == "complete" and table["completed_new_fits"] == table["expected_new_fits"] == 63, "Tables/PDF are partial; rerun final export with --compile")
    require(table["pdf_build"] and table["pdf_build"]["returncode"] == 0, "Final PDF compilation is missing or failed")
    require(set(table["outputs_sha256"]) == set(TABLE_OUTPUTS), "Table metadata must cover all final fragments and the PDF")
    for name in ("summary.csv", "PROTOCOL.json", "AUDIT.json"):
        require(table["source_sha256"][name] == sha256(run / name), "Stale table inputs: " + name)
    require(table["exporter_sha256"] == sha256(HERE / "export_tables.py"), "Exporter differs from compiled table provenance")
    for name, digest in table["outputs_sha256"].items():
        require(sha256(run / name) == digest, "Table artifact differs: " + name)
    require((run / "both_tables.pdf").read_bytes().startswith(b"%PDF-"), "Final PDF header is invalid")
    matrix = read(run / "training_matrix.json")
    require(matrix["status"] == "complete" and matrix["protocol_sha256"] == phash, "Training controller is not complete")
    require(len(matrix["jobs"]) == 21 and all(j["status"].startswith("complete") for j in matrix["jobs"]), "Training jobs are incomplete")
    runtime, extra = read(run / "source_manifest.json"), read(run / "EXTRA_SOURCE_MANIFEST.json")
    require(runtime["protocol_sha256"] == extra["protocol_sha256"] == phash, "Source manifests use a different protocol")
    require(extra["status"] == "complete" and extra["protected_runtime_sources_unchanged"] is True
            and extra["runtime_manifest_sha256"] == sha256(run / "source_manifest.json"), "Extra-source archive provenance differs")
    for source, digest in runtime["sources_sha256"].items():
        require(sha256(source) == sha256(runtime["archives"][source]) == digest, "Protected runtime source changed: " + source)
    with (run / "per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    new = [r for r in rows if r["method"] == "both_mlp"]
    expected = {(m, d, s) for m in MODELS for d in DATASETS for s in (42, 43, 44)}
    require(len(rows) == 183 and len(new) == 63 and {(r["model"], r["dataset"], int(r["seed"])) for r in new} == expected, "Per-seed output does not cover the complete audited matrix")
    for row in new:
        spec = protocol["model_specs"][row["model"]]
        require(int(row["trainable_parameters"]) == spec["trainable_parameters"] and 0 <= int(row["best_epoch"]) <= protocol["train"]["epochs"], "Both parameter/epoch record mismatch")
    parameters = {}
    for model, spec in protocol["model_specs"].items():
        dim, rank = spec["dim"], spec["bottleneck"]
        single = 2 * dim * rank + rank + dim
        require(rank == 64 and spec["trainable_parameters"] == 2 * single, "Unexpected Both parameter count")
        parameters[model] = dict(dim=dim, bottleneck=rank, single_parameters=single, both_parameters=2 * single)
    zeros = [{k: int(r[k]) if k in ("seed", "best_epoch") else r[k] for k in ("model", "dataset", "seed", "method", "best_epoch", "source")}
             for r in rows if r["method"] in ("pre_mlp", "post_mlp", "both_mlp") and int(r["best_epoch"]) == 0]
    return protocol, parameters, zeros, {name: sha256(run / name) for name in FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Check final gates without writing or archiving anything")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        protocol, parameters, zeros, artifacts = inspect_inputs(run)
        if args.check_only:
            print("Final gates passed: 63 Both fits, 120 references, complete matching PDF; no files written")
            return
        require(not (run / "ARTIFACTS.json").exists(), "ARTIFACTS.json already exists; final records are not overwritten")
        paths = [HERE / name for name in ("report.py", "export_tables.py", "final_artifacts.py", "README.txt", "status.py",
                                          "warm_tokens.py", "warm_missing.py", "release_completed_cache.py")]
        paths.append(ROOT / "experiments/adapter_placement_classification/report.py")
        archived = []
        for source in paths:
            destination = run / "postprocessing_source" / source.relative_to(ROOT)
            digest = sha256(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                require(sha256(destination) == digest, "Existing postprocessing archive differs: " + str(destination))
            else:
                shutil.copy2(source, destination)
            require(sha256(destination) == digest, "Postprocessing archive copy failed")
            archived.append(dict(source=str(source), archive=str(destination), sha256=digest))
        timestamp = datetime.now(timezone.utc).isoformat()
        manifest = dict(status="complete", created_utc=timestamp, purpose="Final postprocessing and non-mutating cache I/O helper source archive",
            protocol_sha256=artifacts["PROTOCOL.json"], runtime_manifest_sha256=artifacts["source_manifest.json"],
            protected_runtime_archive_unchanged=True, sources=archived)
        write_json(run / "postprocessing_source_manifest.json", manifest)
        artifacts["postprocessing_source_manifest.json"] = sha256(run / "postprocessing_source_manifest.json")
        require(all(sha256(run / name) == digest for name, digest in artifacts.items()), "An input changed while sealing artifacts")
        write_json(run / "ARTIFACTS.json", dict(status="complete", created_utc=timestamp,
            completed_new_fits=63, expected_new_fits=63, completed_reference_evaluations=120,
            datasets_separate=True, seeds=protocol["seeds"], metric_order=["macro_auroc", "macro_auprc"],
            unit="percent", sd="sample SD (ddof=1); deterministic Original has no seed SD",
            parameters=parameters, both_epoch_zero_count=sum(r["method"] == "both_mlp" for r in zeros),
            supervised_epoch_zero_runs=zeros, final_pdf="both_tables.pdf", artifacts_sha256=artifacts,
            notes=["Both uses twice the single-position MLP parameters; no parameter-matched claim.",
                   "Original runtime and extra-source archives remain unchanged.",
                   "No token cache or model checkpoint was read by this final-artifact seal."]))
        print("Sealed", run / "ARTIFACTS.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
