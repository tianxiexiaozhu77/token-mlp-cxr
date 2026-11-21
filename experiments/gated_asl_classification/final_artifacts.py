"""Seal the completed 252-fit campaign after final audit and PDF export.

No tokens or image files are read and no checkpoints are deserialized.
Migrated fit artifacts, including adapter files, are integrity-hashed.
Small audited fit records independently verify cross-method initialization
and sample-order pairing. Runtime archives are untouched; final reporting,
tests and validation summaries are copied to a postprocessing archive.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.gated_asl_classification.common import MODELS, METHODS, SEEDS, ASL_CONFIG, GATE_CONFIG, parameter_count
from experiments.gated_asl_classification.components import ASL_REFERENCE
from experiments.gated_asl_classification.migrate import verify_migration

DATASETS = ("openi", "chexpert", "chestxray14")
TABLE_OUTPUTS = ("gated_asl_tables.tex", "gated_asl_tables_preview.tex", "gated_asl_tables.pdf",
                "gated_asl_openi_table.tex", "gated_asl_chexpert_table.tex", "gated_asl_chestxray14_table.tex")
FILES = ("PROTOCOL.json", "source_manifest.json", "PREFLIGHT.json", "AUDIT.json", "summary.csv", "per_seed.csv",
         "historical_results.csv", "historical_pre_comparison.csv", "factorial_effects.csv", "factorial_effects_per_seed.csv",
         "REPORT.txt", "training_matrix.json", "gated_asl_tables_metadata.json", *TABLE_OUTPUTS)
IO_HELPER = ROOT / "experiments/both_adapter_classification/warm_missing.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def inspect_preflight(run, protocol_hash, runtime):
    """Bind all 84 CPU integration cases and their sources to the frozen run."""
    path = run / "PREFLIGHT.json"
    preflight = read(path)
    require(preflight["status"] == "passed" and preflight["protocol_sha256"] == protocol_hash
            and preflight["backend_integration_cases"] == 84 and set(preflight["models"]) == set(MODELS),
            "Preflight status/protocol/84-case matrix differs")
    required_sources = {str(HERE / name) for name in ("components.py", "test_components.py", "test_backends.py",
        "common.py", "prepare.py", "run.py", "snapshot.py", "train.py")}
    require(required_sources <= set(preflight["sources"]), "Preflight omits required validation/runtime sources")
    for source, digest in preflight["sources"].items():
        require(sha256(source) == digest, "Preflight source changed: " + source)
        if source in runtime["sources_sha256"]:
            require(runtime["sources_sha256"][source] == digest, "Preflight/runtime source digests disagree: " + source)
    component = preflight["component_tests"]
    require(Path(component["path"]).resolve() == HERE / "components_cpu_validation.json"
            and sha256(component["path"]) == component["sha256"], "Preflight component validation changed")
    cpu = read(component["path"])
    require(cpu["passed"] is True and cpu["failures"] == cpu["errors"] == 0 and cpu["tests_run"] == 10,
            "Preflight component tests are incomplete or failed")
    expected_cases = {(dataset, method) for dataset in DATASETS for method in METHODS}
    for model, record in preflight["models"].items():
        require(record["cases"] == 12 and Path(record["path"]).resolve() == HERE / "validation" / f"{model}_cpu.json"
                and sha256(record["path"]) == record["sha256"], "Preflight integration report changed: " + model)
        validation = read(record["path"])
        require(validation["status"] == "passed" and validation["model"] == model and validation["device"] == "cpu"
                and validation["datasets"] == 3 and validation["methods"] == 4 and len(validation["checks"]) == 12
                and {(item["dataset"], item["method"]) for item in validation["checks"]} == expected_cases,
                "Preflight integration cases are incomplete/duplicated: " + model)
        require(validation["test_source_sha256"] == preflight["sources"][str(HERE / "test_backends.py")],
                "Integration test source differs from preflight: " + model)
    return dict(status="passed", path=str(path), sha256=sha256(path), backend_integration_cases=84,
                component_tests=component, models=preflight["models"], sources_sha256=preflight["sources"])


def inspect_io_logs(run, protocol, protocol_hash):
    """Optionally seal terminal, verified read-only warming logs; never read tokens.

    Stopped/budget-limited warming is a successful I/O integrity record, not a
    failed training run. Running/failed/unknown records remain explicitly
    excluded from the final file manifest because they are not sealable.
    """
    source_hash = sha256(IO_HELPER)
    frozen_files, records = {}, {}
    for cache_group in protocol["caches"].values():
        for record in cache_group:
            records[record["path"]] = record
            frozen_files.update(record["files"])
    included, excluded, hashes = [], [], {}
    for path in sorted((run / "io_warming").glob("*.json")):
        relative = str(path.relative_to(run))
        record = read(path)
        status = record.get("status")
        if status not in ("complete", "stopped", "read_budget_reached"):
            excluded.append(dict(path=relative, observed_status=status,
                reason="Not a verified terminal read-only warming record; mutable/unverified file is not sealed"))
            continue
        require(record.get("read_only") is True and record.get("protocol_sha256") == protocol_hash
                and record.get("files_unchanged") is True, "Warming log lacks matching read-only/integrity evidence: " + relative)
        require(record.get("helper_sha256") == source_hash, "Warming helper differs from recorded source: " + relative)
        require(record.get("sources"), "Warming log omits its validated source files: " + relative)
        for source in record["sources"]:
            expected = frozen_files.get(source["path"])
            cache = records.get(source["cache_record"])
            require(expected is not None and cache is not None
                    and source["cache_record_sha256"] == cache["sha256"]
                    and sha256(cache["path"]) == cache["sha256"], "Warming source/cache is outside the frozen protocol: " + relative)
            stat = Path(source["path"]).stat()
            require(source["bytes"] == expected["bytes"] == stat.st_size
                    and source["mtime_ns"] == expected["mtime_ns"] == stat.st_mtime_ns,
                    "Warmed source file metadata changed: " + source["path"])
        hashes[relative] = sha256(path)
        included.append(dict(path=relative, sha256=hashes[relative], status=status, read_only=True,
            protocol_sha256=protocol_hash, files_unchanged=True, current_source_stats_match=True,
            helper_source=str(IO_HELPER), helper_sha256=source_hash,
            scan_complete=record.get("scan_complete"), read_bytes=record.get("read_bytes"),
            source_files=len(record["sources"])))
    return dict(optional=True, helper_source=str(IO_HELPER), helper_sha256=source_hash,
        included_terminal_logs=included, excluded_unsealed_logs=excluded,
        note="Warming is not required for this campaign. Complete/stopped/read_budget_reached are accepted after integrity checks; source data contents are not read by sealing."), hashes


def inspect_supervision(run, protocol_hash, migration=None):
    """Optionally include only a finished, current successful watcher record."""
    path = run / "supervision.json"
    if not path.exists():
        return dict(optional=True, included=False, reason="No supervisor was used; manual finalization is supported"), {}
    record = read(path)
    require(record.get("migration") == (migration["evidence"] if migration else None),
            "Supervisor migration provenance differs")
    identity = record.get("supervisor_identity", {})
    process_live = False
    try:
        fields = (Path("/proc") / str(int(identity["pid"])) / "stat").read_text().rsplit(")", 1)[1].split()
        process_live = fields[0] not in ("Z", "X", "x") and int(fields[19]) == identity["starttime"]
    except (FileNotFoundError, ProcessLookupError):
        pass
    if record.get("status") != "ready_for_final_review" or process_live:
        return dict(optional=True, included=False, observed_status=record.get("status"),
            supervisor_process_live=process_live,
            reason="Only a terminated ready_for_final_review supervisor is sealable; this mutable/nonfinal record and its logs are excluded"), {}
    require(record["protocol_sha256"] == protocol_hash
            and record["source_manifest_sha256"] == sha256(run / "source_manifest.json")
            and record["observed_new_fits"] == record["audited_new_fits"] == record["expected_new_fits"] == 252
            and record["completed_reference_evaluations"] == record["expected_reference_evaluations"] == 120
            and record["complete_worker_jobs"] == record["expected_worker_jobs"] == 21
            and record["matrix_status"] == "complete" and record["exit_code"] == 0
            and record["ended_utc"] and record["active_command"] is None,
            "Finished supervisor does not confirm the frozen 252/120/21 campaign")
    require(record["audit_sha256"] == sha256(run / "AUDIT.json")
            and record["export_metadata_sha256"] == sha256(run / "gated_asl_tables_metadata.json"),
            "Supervisor references stale audit/export metadata")
    require(len(record["commands"]) >= 2
            and [entry["stage"] for entry in record["commands"][-2:]] == ["strict_report", "strict_export_tables"],
            "Supervisor lacks its final strict report/export sequence")
    source_hashes = record["postprocessing_sources_sha256"]
    require(str(HERE / "supervise.py") in source_hashes, "Supervisor omits its own source hash")
    for source, digest in source_hashes.items():
        require(sha256(source) == digest, "Supervised postprocessing source changed: " + source)
    hashes, logs = {"supervision.json": sha256(path)}, []
    for command in record["commands"]:
        relative = command["log"]
        logfile = (run / relative).resolve()
        require(logfile.parent == (run / "supervision_logs").resolve()
                and logfile.suffix == ".log" and relative not in hashes
                and command["status"] == "passed" and command["returncode"] == 0 and command["ended_utc"],
                "Supervisor log is mutable, duplicated, outside the run, or records a failed command")
        require(sha256(logfile) == command["log_sha256"], "Supervisor command log changed: " + relative)
        hashes[relative] = command["log_sha256"]
        logs.append(dict(path=relative, sha256=hashes[relative], stage=command["stage"], returncode=0))
    return dict(optional=True, included=True, status="ready_for_final_review", supervisor_process_live=False,
        path="supervision.json", sha256=hashes["supervision.json"], audited_new_fits=252,
        completed_reference_evaluations=120, audit_sha256=record["audit_sha256"],
        export_metadata_sha256=record["export_metadata_sha256"], postprocessing_sources_sha256=source_hashes,
        command_logs=logs), hashes


def inspect_inputs(run):
    protocol, audit, table = (read(run / name) for name in ("PROTOCOL.json", "AUDIT.json", "gated_asl_tables_metadata.json"))
    phash = sha256(run / "PROTOCOL.json")
    require(audit["status"] == "passed" and not audit["errors"] and not audit["pending"], "AUDIT must pass without pending fits/errors")
    require(audit["completed_new_fits"] == audit["expected_new_fits"] == 252 and
            audit["completed_reference_evaluations"] == audit["expected_reference_evaluations"] == 120, "Require all 252 fresh fits and 120 historical references")
    require(protocol["models"] == list(MODELS) and protocol["methods"] == list(METHODS)
            and protocol["seeds"] == list(SEEDS) and set(protocol["datasets"]) == set(DATASETS)
            and protocol["new_training_runs"] == 252 and protocol["all_methods_fresh"] is True
            and protocol["reused_training_runs"] == 0 and audit["protocol_sha256"] == phash, "Wrong campaign protocol")
    require(protocol["asl_config"] == ASL_CONFIG and protocol["gate_config"] == GATE_CONFIG, "Loss/gate definitions changed")
    require(table["status"] == "complete" and table["completed_new_fits"] == table["expected_new_fits"] == 252
            and table["completed_reference_evaluations"] == 120, "PDF metadata is not final 252/252")
    require(table["pdf_build"] and table["pdf_build"]["returncode"] == 0, "Final PDF compilation missing/failed")
    require(set(table["outputs_sha256"]) == set(TABLE_OUTPUTS), "Incomplete table/PDF output manifest")
    for name in ("summary.csv", "PROTOCOL.json", "AUDIT.json"):
        require(table["source_sha256"][name] == sha256(run / name), "Stale PDF source: " + name)
    require(table["exporter_sha256"] == sha256(HERE / "export_tables.py"), "Exporter changed after PDF compilation")
    for name, digest in table["outputs_sha256"].items():
        require(sha256(run / name) == digest, "Modified table/PDF: " + name)
    require((run / "gated_asl_tables.pdf").read_bytes().startswith(b"%PDF-"), "Invalid PDF header")
    matrix, runtime = read(run / "training_matrix.json"), read(run / "source_manifest.json")
    migration = verify_migration(run, runtime)
    require(audit.get("migration") == (migration["evidence"] if migration else None),
            "Final audit migration provenance differs")
    archives = migration["archives"] if migration else runtime["archives"]
    require(matrix["status"] == "complete" and matrix["protocol_sha256"] == phash and matrix["new_fits"] == 252
            and len(matrix["jobs"]) == 21 and all(job["status"].startswith("complete") for job in matrix["jobs"]), "Controller is not complete")
    require(runtime["protocol_sha256"] == phash, "Runtime snapshot protocol differs")
    for source, digest in runtime["sources_sha256"].items():
        require(sha256(source) == sha256(archives[source]) == digest, "Runtime source/archive changed: " + source)
    preflight = inspect_preflight(run, phash, runtime)
    for source, digest in audit["reporter_sources_sha256"].items():
        require(sha256(source) == digest, "Reporting source changed after final audit: " + source)
    require(len(audit["matching_groups"]) == 21, "Missing cross-method group audit")
    for group in audit["matching_groups"]:
        require(group["completed_fresh_fits"] == group["expected_fresh_fits"] == 12
                and group["saved_worker_summary"] == "passed"
                and sha256(group["path"]) == group["sha256"], "Worker matching summary missing/changed")
    parameters = {}
    for model in MODELS:
        spec = protocol["model_specs"][model]
        require(spec["bottleneck"] == 64, "Unexpected bottleneck")
        wanted = {method: parameter_count(spec["dim"], 64, method) for method in METHODS}
        require(spec["trainable_parameters_by_method"] == wanted, "Parameter budget differs: " + model)
        parameters[model] = dict(dim=spec["dim"], bottleneck=64, trainable_parameters_by_method=wanted,
            additional_gate_parameters=spec["dim"] + 1)
    with (run / "per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fresh = [row for row in rows if row["method"] in METHODS]
    expected = {(model, dataset, seed, method) for model in MODELS for dataset in DATASETS for seed in SEEDS for method in METHODS}
    require(len(fresh) == 252 and {(r["model"], r["dataset"], int(r["seed"]), r["method"]) for r in fresh} == expected, "Fresh per-seed matrix is incomplete/duplicated")
    audit_by_key = {tuple(entry["key"]): entry for entry in audit["runs"]}
    grouped, zeros = defaultdict(dict), []
    for row in fresh:
        key = row["model"], row["dataset"], int(row["seed"]), row["method"]
        model, dataset, seed, method = key
        folder = run / "results" / model / dataset / f"seed{seed}" / method
        require(Path(row["source"]).resolve() == folder, "Fresh fit points outside current campaign")
        evidence = audit_by_key[key]
        for filename in ("result.json", "training.json"):
            require(sha256(folder / filename) == evidence["files_sha256"][filename], "Fit record changed after audit: " + str(folder / filename))
        result, training = read(folder / "result.json"), read(folder / "training.json")
        for record in (result, training):
            require(record["status"] == "complete" and record["fresh_fit"] is True
                    and record["historical_reference_used_for_training"] is False
                    and record["protocol_sha256"] == phash
                    and record["matched_initialization_verified"] is True
                    and record["matched_overlapping_epoch_orders_verified"] is True, "Fresh/pairing verification missing")
        require(int(row["trainable_parameters"]) == parameters[model]["trainable_parameters_by_method"][method], "Per-seed parameter count differs")
        require(int(row["best_epoch"]) == result["best_epoch"] == training["best_epoch"], "Selected epoch record mismatch")
        if result["best_epoch"] == 0:
            zeros.append(dict(model=model, dataset=dataset, seed=seed, method=method, best_epoch=0, source=str(folder)))
        require(result["shared_initialization_sha256"] == training["shared_initialization_sha256"], "Initialization evidence differs")
        grouped[model, dataset, seed][method] = training
    paired = []
    for (model, dataset, seed), group in sorted(grouped.items()):
        require(set(group) == set(METHODS), "Incomplete four-method pairing group")
        initial = group[METHODS[0]]["shared_initialization_sha256"]
        require(set(initial) == {"down.weight", "down.bias", "up.weight", "up.bias"}, "Wrong shared initialization tensor keys")
        orders, lengths = {}, {}
        for method, record in group.items():
            require(record["shared_initialization_sha256"] == initial, "Cross-method shared initialization differs")
            lengths[method] = len(record["history"])
            for epoch in record["history"]:
                number, digest = epoch["epoch"], epoch["train_order_sha256"]
                require(orders.setdefault(number, digest) == digest, "Cross-method overlapping sample order differs")
        paired.append(dict(model=model, dataset=dataset, seed=seed, shared_initialization_sha256=initial,
            epochs_by_method=lengths, common_epochs_all_four=min(lengths.values()), all_overlapping_epoch_orders_match=True))
    require(len(paired) == 63, "Require 63 paired model/dataset/seed groups")
    cpu = read(HERE / "components_cpu_validation.json")
    require(cpu["passed"] is True and cpu["failures"] == cpu["errors"] == 0
            and cpu["asl_reference"] == ASL_REFERENCE and cpu["asl_config"] == ASL_CONFIG
            and cpu["gate_config"] == GATE_CONFIG, "Component CPU validation differs or failed")
    for model in MODELS:
        validation = read(HERE / "validation" / f"{model}_cpu.json")
        require(validation["status"] == "passed" and validation["model"] == model
                and validation["datasets"] == 3 and validation["methods"] == 4, "Incomplete native CPU integration validation")
    artifacts = {name: sha256(run / name) for name in FILES}
    if migration:
        artifacts.update(migration["files_sha256"])
    for name in ("pairing_audit.json", "EXTRA_SOURCE_MANIFEST.json", "RECOVERY_LAUNCH.json"):
        if (run / name).is_file():
            artifacts[name] = sha256(run / name)
    io_warming, io_hashes = inspect_io_logs(run, protocol, phash)
    artifacts.update(io_hashes)
    supervision, supervision_hashes = inspect_supervision(run, phash, migration)
    artifacts.update(supervision_hashes)
    return protocol, parameters, zeros, paired, artifacts, preflight, io_warming, supervision, migration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        protocol, parameters, zeros, paired, artifacts, preflight, io_warming, supervision, migration = inspect_inputs(run)
        if args.check_only:
            print("Final gates passed: 252 fresh fits, 120 references, 63 paired groups, matching complete PDF; no files written")
            return
        require(not (run / "ARTIFACTS.json").exists(), "ARTIFACTS.json exists; refusing to overwrite final seal")
        paths = [HERE / name for name in ("report.py", "export_tables.py", "final_artifacts.py", "README.txt", "test_components.py", "test_backends.py", "components_cpu_validation.json", "migrate.py", "test_migrate.py")]
        paths += [HERE / "validation" / f"{model}_cpu.json" for model in MODELS]
        paths += [Path(source) for source in read(run / "AUDIT.json")["reporter_sources_sha256"]]
        paths += [Path(source) for source in preflight["sources_sha256"]]
        paths.append(IO_HELPER)
        # Include local postprocessing/I/O helpers only if they exist. Runtime
        # copies remain protected in source/; no original archive is changed.
        paths += [HERE / name for name in ("supervise.py", "status.py", "warm_tokens.py", "warm_missing.py", "release_completed_cache.py") if (HERE / name).is_file()]
        expected_supplement_hashes = dict(preflight["sources_sha256"])
        expected_supplement_hashes[preflight["component_tests"]["path"]] = preflight["component_tests"]["sha256"]
        expected_supplement_hashes.update({record["path"]: record["sha256"] for record in preflight["models"].values()})
        expected_supplement_hashes[str(IO_HELPER)] = io_warming["helper_sha256"]
        if supervision["included"]:
            expected_supplement_hashes.update(supervision["postprocessing_sources_sha256"])
            paths += [Path(source) for source in supervision["postprocessing_sources_sha256"]]
        archived = []
        for source in sorted(set(paths)):
            destination = run / "postprocessing_source" / source.relative_to(ROOT)
            digest = sha256(source)
            if str(source) in expected_supplement_hashes:
                require(digest == expected_supplement_hashes[str(source)], "Validated source/report changed during archive: " + str(source))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                require(sha256(destination) == digest, "Existing supplemental archive differs: " + str(destination))
            else:
                shutil.copy2(source, destination)
            require(sha256(destination) == digest, "Archive copy verification failed")
            archived.append(dict(source=str(source), archive=str(destination), sha256=digest))
        stamp = datetime.now(timezone.utc).isoformat()
        write_json(run / "postprocessing_source_manifest.json", dict(status="complete", created_utc=stamp,
            protocol_sha256=artifacts["PROTOCOL.json"], runtime_manifest_sha256=artifacts["source_manifest.json"],
            protected_runtime_archive_unchanged=True, sources=archived))
        artifacts["postprocessing_source_manifest.json"] = sha256(run / "postprocessing_source_manifest.json")
        require(all(sha256(run / name) == digest for name, digest in artifacts.items()), "Reporting artifact changed during seal")
        write_json(run / "ARTIFACTS.json", dict(status="complete", created_utc=stamp,
            completed_new_fits=252, expected_new_fits=252, completed_reference_evaluations=120,
            all_methods_fresh=True, reused_training_runs=0, methods=list(METHODS), seeds=list(SEEDS), datasets_separate=True,
            metric_order=["macro_auroc", "macro_auprc"], unit="percent", sd="sample SD (ddof=1); deterministic Original has no seed SD",
            parameters=parameters, asl_reference=ASL_REFERENCE, asl_config=protocol["asl_config"], gate_config=protocol["gate_config"],
            epoch_zero_count=len(zeros), epoch_zero_by_method={method: sum(row["method"] == method for row in zeros) for method in METHODS},
            epoch_zero_runs=zeros, paired_initialization_and_order=dict(status="passed", groups=63, evidence=paired),
            preflight=preflight, optional_read_only_io_warming=io_warming, optional_supervision=supervision,
            migration=migration["evidence"] if migration else None,
            final_pdf="gated_asl_tables.pdf", artifacts_sha256=artifacts,
            notes=["All four method groups are fresh supervised fits, including the plain weighted-BCE control.",
                "Gated adapters add dim+1 parameters; no exact parameter-matching claim.",
                "Matching covers shared MLP initialization and every overlapping training epoch; early-stopping lengths may differ.",
                "No token array or image was read and no checkpoint was deserialized; migrated artifacts were integrity-hashed."]))
        print("Sealed", run / "ARTIFACTS.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
