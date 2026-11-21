"""Copy the interrupted 20-fit campaign into an independent, new run directory.

The origin is read-only, including its NFS locks. This is a byte-preserving
copy of completed fits, not checkpoint continuation. No training is started.
The original source manifest remains byte-identical; verify_migration supplies
the relocated archive map without changing that frozen manifest.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat


EXPECTED_FITS = 20
EXPECTED_SOURCES = 65
EXPECTED_REFERENCES = 120
MODELS = ("AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo")
DATASETS = ("openi", "chexpert", "chestxray14")
METHODS = ("pre_bce", "gated_bce", "pre_asl", "gated_asl")
SEEDS = (42, 43, 44)
FIT_FILES = ("result.json", "training.json", "adapter.pt",
             "validation_predictions.npz", "test_predictions.npz", "progress.json")
AUDIT_FILES = set(FIT_FILES) - {"progress.json"}
ORIGIN_FILES = ("PROTOCOL.json", "source_manifest.json", "AUDIT.json", "PREFLIGHT.json")
EXPECTED_KEYS = {
    (model, "chestxray14", seed, method)
    for model in ("AlphaRAD", "RadZero") for seed in SEEDS for method in METHODS[:3]
} | {("AlphaRAD", "chestxray14", seed, "gated_asl") for seed in (42, 43)}
PARTIAL_KEYS = {("AlphaRAD", "chestxray14", 44, "gated_asl"),
                ("RadZero", "chestxray14", 42, "gated_asl")}
VALIDATION_NAMES = {"frozen_path.json", "matched_four_groups.json"} | {
    f"reused_pre_seed{seed}.json" for seed in SEEDS}
VALIDATION_POLICY = ("Working validation/ is an initial snapshot only and may be rewritten by "
                     "the frozen worker. Immutable origin copies retain the previous audit evidence.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    p = Path(value)
    require(".." not in p.parts, "Parent traversal is forbidden: " + str(value))
    return Path(os.path.abspath(p))


def safe_path(value, *, file=False, directory=False, missing=False):
    """Check every existing component before resolving, so links cannot disappear."""
    p = absolute(value)
    for part in reversed((p, *p.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            require(missing, "Missing path: " + str(part))
            continue
        require(not stat.S_ISLNK(info.st_mode), "Symlink is forbidden: " + str(part))
        if part != p:
            require(stat.S_ISDIR(info.st_mode), "Non-directory ancestor: " + str(part))
    if file:
        require(p.is_file(), "Not a regular file: " + str(p))
        require(stat.S_ISREG(p.lstat().st_mode), "Special file is forbidden: " + str(p))
        require(p.stat().st_nlink == 1, "Hard-linked file is forbidden: " + str(p))
    if directory:
        require(p.is_dir(), "Not a directory: " + str(p))
    return p


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value, "Invalid relative path")
    p = Path(value)
    require(p.parts and not p.is_absolute() and ".." not in p.parts and p.as_posix() == value,
            "Noncanonical relative path: " + value)
    return p


def under(root, value, *, file=False):
    p = safe_path(value, file=file)
    require(p != root and p.is_relative_to(root), "Path escapes its root: " + str(p))
    return p


def sha256(path):
    p = safe_path(path, file=True)
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    # Duplicate keys would otherwise hide conflicting provenance entries.
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "Duplicate JSON key: " + str(key))
            output[key] = value
        return output
    with safe_path(path, file=True).open() as stream:
        return json.load(stream, object_pairs_hook=pairs)


def check_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Invalid SHA256 for " + str(path))
    require(sha256(path) == expected, "SHA256 differs: " + str(path))


def tree_files(root):
    root = safe_path(root, directory=True)
    output = {}
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            safe_path(Path(folder) / name, directory=True)
        for name in names:
            p = safe_path(Path(folder) / name, file=True)
            output[p.relative_to(root).as_posix()] = p
    return output


def key(entry):
    value = tuple(entry["key"]) if "key" in entry else tuple(
        entry[name] for name in ("model", "dataset", "seed", "method"))
    require(len(value) == 4 and type(value[2]) is int, "Invalid fit key")
    return value


def fit_relative(k):
    model, dataset, seed, method = k
    require(model in MODELS and dataset in DATASETS and seed in SEEDS and method in METHODS,
            "Not a fresh factorial fit: " + str(k))
    return f"results/{model}/{dataset}/seed{seed}/{method}"


def expected_references():
    result = {(m, d, 42, "original") for m in MODELS for d in DATASETS}
    result |= {(m, d, s, "pre_mlp") for m in MODELS for d in DATASETS for s in SEEDS}
    result |= {(m, d, s, "post_mlp") for m in ("AlphaRAD", "KAD", "MaCo")
               for d in DATASETS for s in SEEDS}
    result |= {(m, d, 42, "aggregation_only") for m in ("BiomedCLIP", "BioViL-T", "MaCo")
               for d in DATASETS}
    return result


def verify_external(protocol, audit):
    """Hash small provenance and historical artifacts; never read token arrays."""
    require(protocol["status"] == "frozen" and protocol["campaign"] == "gated_asl_classification",
            "Not the frozen factorial campaign")
    require(protocol["models"] == list(MODELS) and protocol["methods"] == list(METHODS)
            and protocol["seeds"] == list(SEEDS) and set(protocol["datasets"]) == set(DATASETS),
            "Campaign axes differ")
    require(protocol["new_training_runs"] == 252 and protocol["all_methods_fresh"] is True
            and protocol["reused_training_runs"] == 0, "Fresh-fit accounting differs")
    parent = protocol["parent_protocol"]
    check_hash(parent["path"], parent["sha256"])
    previous = read(parent["path"])
    for name in ("datasets", "references", "caches"):
        require(protocol[name] == previous[name], "Parent provenance differs: " + name)
    for spec in protocol["datasets"].values():
        check_hash(spec["path"], spec["sha256"])
        meta = read(spec["path"])
        require(meta["samples_sha256"] == spec["samples_sha256"] and meta["class_names"] == spec["classes"],
                "Dataset labels/provenance differ")
        check_hash(meta["samples_file"], spec["samples_sha256"])
        with safe_path(meta["samples_file"], file=True).open() as stream:
            first = stream.read(1)
            stream.seek(0)
            rows = json.load(stream) if first == "[" else (json.loads(line) for line in stream if line.strip())
            counts = Counter(row["split"] for row in rows)
        require(dict(counts) == spec["counts"], "Dataset split counts differ")
    refs = protocol["references"]
    refmap = {key(r): r for r in refs}
    historical = {key(r): r for r in audit["historical_references"]}
    require(len(refs) == len(refmap) == len(historical) == EXPECTED_REFERENCES
            and set(refmap) == set(historical) == expected_references(), "Historical reference matrix differs")
    for k, entry in refmap.items():
        folder = safe_path(entry["source"], directory=True)
        old = historical[k]
        require(old["source"] == str(folder) and old.get("new_fit") is False,
                "Historical audit identity differs")
        require(set(old["files_sha256"]) <= set(entry["files_sha256"]), "Historical audit files differ")
        for filename, digest in entry["files_sha256"].items():
            require(relative(filename).name == filename, "Nested historical artifact name")
            check_hash(folder / filename, digest)
        for filename, digest in old["files_sha256"].items():
            require(digest == entry["files_sha256"][filename], "Historical audit hash differs")
    require(set(protocol["caches"]) == {f"{m}/{d}" for m in MODELS for d in DATASETS}, "Cache matrix differs")
    for records in protocol["caches"].values():
        require(isinstance(records, list) and records, "Empty cache specification")
        for record in records:
            check_hash(record["path"], record["sha256"])
            for filename, expected in record["files"].items():
                info = safe_path(filename, file=True).stat()
                require(info.st_size == expected["bytes"] and info.st_mtime_ns == expected["mtime_ns"],
                        "Frozen cache size/mtime differs: " + filename)


def verify_sources(origin, manifest, *, logical_origin=None):
    logical_origin = origin if logical_origin is None else logical_origin
    sources, archives = manifest["sources_sha256"], manifest["archives"]
    require(len(sources) == len(archives) == EXPECTED_SOURCES and set(sources) == set(archives),
            "Expected exactly 65 runtime sources and archives")
    require(manifest["all_methods_fresh"] is True and manifest["new_fits"] == 252, "Wrong source campaign")
    result = {}
    for source, digest in sources.items():
        require(Path(source).is_absolute() and Path(archives[source]).is_absolute(),
                "Runtime source and archive must be absolute")
        check_hash(source, digest)
        recorded = absolute(archives[source])
        require(recorded.is_relative_to(logical_origin / "source") and recorded != logical_origin / "source",
                "Recorded runtime archive escapes origin/source")
        archived = under(origin / "source", origin / recorded.relative_to(logical_origin), file=True)
        check_hash(archived, digest)
        rel = archived.relative_to(origin).as_posix()
        require(rel not in result, "Two runtime sources share one archive")
        result[rel] = (str(archived), digest, source)
    require(set(tree_files(origin / "source")) == {
        str(Path(name).relative_to("source")) for name in result}, "Unexpected source archive files")
    return result


def verify_fits(origin, protocol, manifest, audit, recovery, *, logical_origin=None, allow_additional=False):
    logical_origin = origin if logical_origin is None else logical_origin
    require(audit["status"] == "partial" and audit["errors"] == [] and audit["completed_new_fits"] == EXPECTED_FITS
            and audit["expected_new_fits"] == 252 and audit["completed_reference_evaluations"] == EXPECTED_REFERENCES
            and audit["expected_reference_evaluations"] == EXPECTED_REFERENCES
            and audit["old_fits_counted_as_new"] == 0, "Origin audit is not the clean 20-fit audit")
    audited = {key(r): r for r in audit["runs"]}
    require(len(audit["runs"]) == len(audited) == EXPECTED_FITS and set(audited) == EXPECTED_KEYS,
            "Origin audit completed-fit set differs")
    require(recovery["run_dir"] == str(logical_origin) and recovery["completed_fits_preserved"] == EXPECTED_FITS
            and recovery["runtime_sources_verified"] == EXPECTED_SOURCES,
            "Recovery snapshot belongs to another origin/campaign")
    expected_files = recovery["completed_files_sha256"]
    require(len(expected_files) == EXPECTED_FITS * len(FIT_FILES), "Recovery must bind all 120 fit files")
    if not allow_additional:
        actual_complete = {p.parent.relative_to(origin).as_posix() for p in (origin / "results").glob("*/*/seed*/*/result.json")}
        require(actual_complete == {fit_relative(k) for k in EXPECTED_KEYS}, "Origin completed results changed")
    fits, files = [], {}
    for k in sorted(EXPECTED_KEYS):
        rel = fit_relative(k)
        folder = under(origin, origin / rel)
        require(set(tree_files(folder)) == set(FIT_FILES), "Complete fit must have exactly six files: " + rel)
        entry = audited[k]
        require(entry["source"] == str(logical_origin / rel) and entry["fresh_fit"] is True
                and set(entry["files_sha256"]) == AUDIT_FILES, "Origin audit fit binding differs")
        digests = {}
        for filename in FIT_FILES:
            name = rel + "/" + filename
            digest = expected_files.get(name)
            check_hash(folder / filename, digest)
            if filename in AUDIT_FILES:
                require(entry["files_sha256"][filename] == digest, "Recovery/AUDIT artifact hash differs")
            digests[filename] = digest
            files[name] = digest
        result, training = read(folder / "result.json"), read(folder / "training.json")
        require(result == {n: v for n, v in training.items() if n != "history"}, "Result/training records differ")
        require(key(result) == k and result["status"] == "complete" and result["fresh_fit"] is True
                and result["historical_reference_used_for_training"] is False,
                "An old/partial result cannot count as a fresh completed fit")
        require(result["protocol_sha256"] == audit["protocol_sha256"]
                and result["source_manifest_sha256"] == recovery["source_manifest_sha256"]
                and result["sources_sha256"] == manifest["sources_sha256"]
                and result["config"] == protocol["train"] and result["method_spec"] == protocol["method_specs"][k[3]],
                "Fit protocol/runtime/config differs")
        spec = protocol["datasets"][k[1]]
        for left, right in (("dataset_sha256", "sha256"), ("samples_sha256", "samples_sha256"),
                            ("class_names", "classes"), ("counts", "counts")):
            require(result[left] == spec[right], "Fit dataset identity differs")
        for name in ("frozen_backbone_verified", "matched_initialization_verified", "matched_overlapping_epoch_orders_verified"):
            require(result[name] is True, "Fit validation flag missing: " + name)
        require(result["test_selection"] is False, "TEST was used for selection")
        for name, filename in (("adapter_sha256", "adapter.pt"), ("test_predictions_sha256", "test_predictions.npz"),
                               ("validation_predictions_sha256", "validation_predictions.npz")):
            require(result[name] == digests[filename], "Fit artifact digest differs")
        require(entry["result_current_sha256"] == digests["result.json"], "AUDIT result digest differs")
        for left, right in (("selected_val", "recomputed_val"), ("test", "recomputed_test")):
            for metric in ("macro_auroc", "macro_auprc"):
                value = result[left][metric]
                require(isinstance(value, (int, float)) and math.isfinite(value)
                        and math.isclose(value, entry[right][metric], rel_tol=1e-12, abs_tol=1e-12),
                        "Origin audit metric differs")
        fits.append(dict(key=list(k), relative_dir=rel, files_sha256=digests))
    require(files == expected_files, "Recovery has additional/missing fit artifacts")
    return fits, files


def validation_files(origin, audit, *, logical_origin=None):
    logical_origin = origin if logical_origin is None else logical_origin
    output = {}
    root = origin / "validation"
    if root.exists():
        for rel, path in tree_files(root).items():
            parts = Path(rel).parts
            require(len(parts) == 3 and parts[0] in MODELS and parts[1] in DATASETS
                    and parts[2] in VALIDATION_NAMES, "Unexpected validation file: " + rel)
            output["validation/" + rel] = (str(path), sha256(path))
    for entry in audit["runs"]:
        evidence = entry.get("runtime_validation", {})
        records = ([evidence["frozen_path"]] if "frozen_path" in evidence else []) + evidence.get("reused_pre_replays", [])
        for record in records:
            require(Path(record["path"]).is_absolute(), "Validation evidence path must be absolute")
            recorded = absolute(record["path"])
            require(recorded.is_relative_to(logical_origin / "validation"), "Recorded validation path escapes origin")
            p = under(root, origin / recorded.relative_to(logical_origin), file=True)
            rel = p.relative_to(origin).as_posix()
            require(rel in output and record["sha256"] == output[rel][1], "AUDIT validation snapshot changed")
    return output


def _build_plan(source_run, destination_run, recovery_dir, *, verifying_existing=False):
    origin = safe_path(source_run, directory=True)
    destination = safe_path(destination_run, missing=True)
    if verifying_existing:
        safe_path(destination, directory=True)
    else:
        require(not destination.exists(), "Destination must be a brand-new directory")
    safe_path(destination.parent, directory=True)
    require(not destination.is_relative_to(origin) and not origin.is_relative_to(destination),
            "Origin and destination cannot contain one another")
    recovery_root = under(origin / "recovery", recovery_dir)
    safe_path(recovery_root, directory=True)
    require(not (origin / "migration.json").exists(), "Chained migration is not supported")
    protocol, manifest = read(origin / "PROTOCOL.json"), read(origin / "source_manifest.json")
    audit, preflight = read(origin / "AUDIT.json"), read(origin / "PREFLIGHT.json")
    recovery = read(recovery_root / "recovery.json")
    protocol_hash, manifest_hash = sha256(origin / "PROTOCOL.json"), sha256(origin / "source_manifest.json")
    require(protocol_hash == manifest["protocol_sha256"] == audit["protocol_sha256"]
            == preflight["protocol_sha256"] == recovery["protocol_sha256"], "Protocol identity differs")
    require(manifest_hash == recovery["source_manifest_sha256"], "Frozen manifest identity differs")
    require(preflight["status"] == "passed", "Preflight did not pass")
    sources = verify_sources(origin, manifest)
    verify_external(protocol, audit)
    fits, fit_hashes = verify_fits(origin, protocol, manifest, audit, recovery)
    snapshots = validation_files(origin, audit)
    copied, immutable = {}, {}

    def add(target, source, digest=None, *, mutable=False):
        relative(target)
        require(target not in copied, "Duplicate migration destination: " + target)
        source = safe_path(source, file=True)
        digest = sha256(source) if digest is None else digest
        check_hash(source, digest)
        copied[target] = str(source)
        if not mutable:
            immutable[target] = digest

    for name in ORIGIN_FILES:
        add("migration/origin/" + name, origin / name)
        if name != "AUDIT.json":
            add(name, origin / name)
    relocated = {}
    for rel, (archived, digest, source) in sources.items():
        add(rel, archived, digest)
        relocated[source] = str(destination / rel)
    for rel, digest in fit_hashes.items():
        add(rel, origin / rel, digest)
    for rel, (source, digest) in snapshots.items():
        add(rel, source, digest, mutable=True)
        add("migration/origin/" + rel, source, digest)
    recovery_files = tree_files(recovery_root)
    require("recovery.json" in recovery_files, "Missing recovery evidence")
    for name, digest in recovery["archived_files"].items():
        relative(name)
        require(name in recovery_files, "Recovery archive entry missing: " + name)
        check_hash(recovery_files[name], digest)
    for name in ("PROTOCOL.json", "source_manifest.json", "AUDIT.json"):
        require(sha256(recovery_root / name) == sha256(origin / name), "Recovery/origin evidence differs: " + name)
    for rel, source in recovery_files.items():
        require(not rel.endswith(".lock"), "Locks cannot be included in recovery evidence")
        if Path(rel).parts[0] == "interrupted_fits":
            continue  # The metadata keeps their hashes; no partial adapter is copied.
        require(rel in recovery["archived_files"] or rel in ("recovery.json", "lock_observations.jsonl"),
                "Unlisted recovery evidence: " + rel)
        add("migration/origin/recovery/" + rel, source)
    excluded = []
    for k in sorted(PARTIAL_KEYS):
        folder = origin / fit_relative(k)
        require(folder.is_dir() and not (folder / "result.json").exists(), "Expected interrupted fit changed")
        hashes = {name: sha256(path) for name, path in tree_files(folder).items()}
        prefix = "interrupted_fits/" + fit_relative(k).removeprefix("results/") + "/"
        archived_hashes = {name.removeprefix(prefix): digest for name, digest in recovery["archived_files"].items()
                           if name.startswith(prefix)}
        require(hashes == archived_hashes, "Interrupted fit differs from the recovery snapshot")
        excluded.append(dict(key=list(k), relative_dir=fit_relative(k), files_sha256=hashes))
    return dict(version=1, status="verified_copy_plan", origin_run=str(origin), destination_run=str(destination),
                recovery_dir=str(recovery_root), protocol_sha256=protocol_hash, source_manifest_sha256=manifest_hash,
                source_archives=relocated, copied_fits=fits, immutable_files_sha256=immutable,
                copy_sources=copied, validation_snapshot_sha256={rel: value[1] for rel, value in snapshots.items()},
                validation_policy=VALIDATION_POLICY, excluded_partial_fits=excluded,
                historical_references_verified=EXPECTED_REFERENCES, runtime_sources_verified=EXPECTED_SOURCES,
                copied_complete_fit_files=EXPECTED_FITS * len(FIT_FILES),
                transfer="independent_regular_file_byte_copies", original_locks_accessed=False,
                interrupted_fits_policy="Not copied. Retrain from the original seed; no optimizer/RNG continuation.")


def build_plan(source_run, destination_run, recovery_dir):
    """Validate every source and return a deterministic, read-only copy plan."""
    return _build_plan(source_run, destination_run, recovery_dir)


def copy_file(source, destination, expected):
    safe_path(source, file=True)
    safe_path(destination.parent, directory=True)
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(source_fd, "rb") as src:
        source_info = os.fstat(src.fileno())
        require(stat.S_ISREG(source_info.st_mode) and source_info.st_nlink == 1, "Source changed before copy")
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
    check_hash(destination, expected)
    require(destination.stat().st_nlink == 1 and not os.path.samefile(source, destination),
            "Destination is not an independent copy")


def migrate(source_run, destination_run, recovery_dir, check_only=False):
    plan = build_plan(source_run, destination_run, recovery_dir)
    if check_only:
        return plan
    destination = Path(plan["destination_run"])
    destination.mkdir()  # Exclusively creates the new run; never overwrites an existing directory.
    for rel, source in plan["copy_sources"].items():
        target = destination / relative(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = plan["immutable_files_sha256"].get(rel, plan["validation_snapshot_sha256"].get(rel))
        copy_file(source, target, digest)
    # Revalidate origin after copying: an interrupted or concurrently changing
    # source can never be published as a completed migration.
    for rel, source in plan["copy_sources"].items():
        digest = plan["immutable_files_sha256"].get(rel, plan["validation_snapshot_sha256"].get(rel))
        check_hash(source, digest)
    record = dict(plan, status="complete", created_utc=datetime.now(timezone.utc).isoformat())
    with (destination / "migration.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return verify_migration(destination)


def verify_migration(run, runtime=None):
    """Return verified relocated sources and seal evidence, or None if unmigrated.

    runtime, when supplied, is the already-read source_manifest.json dictionary.
    No files or process state are modified. Active validation/ and later result
    fits are deliberately excluded from the immutable migration file set.
    """
    run = safe_path(run, directory=True)
    migration_path = run / "migration.json"
    if not migration_path.exists():
        require(not migration_path.is_symlink(), "Dangling migration manifest symlink")
        return None
    record = read(migration_path)
    require(record.get("version") == 1 and record.get("status") == "complete", "Incomplete/unknown migration")
    require(record["destination_run"] == str(run), "Migration destination differs")
    # Origin paths are provenance labels only. Neither the old run nor its
    # diagnostics, partial fits, or NFS locks need remain present or unchanged.
    require(Path(record["origin_run"]).is_absolute() and Path(record["recovery_dir"]).is_absolute(),
            "Origin provenance paths must be absolute")
    origin = absolute(record["origin_run"])
    require(not run.is_relative_to(origin) and not origin.is_relative_to(run), "Migration directories overlap")
    recovery_label = absolute(record["recovery_dir"])
    require(recovery_label.is_relative_to(origin / "recovery"), "Recovery provenance escapes origin")
    snapshot = safe_path(run / "migration/origin", directory=True)
    recovery_root = safe_path(snapshot / "recovery", directory=True)
    protocol, original_manifest = read(snapshot / "PROTOCOL.json"), read(snapshot / "source_manifest.json")
    audit, preflight, recovery = read(snapshot / "AUDIT.json"), read(snapshot / "PREFLIGHT.json"), read(recovery_root / "recovery.json")
    manifest = read(run / "source_manifest.json")
    if runtime is not None:
        require(runtime == manifest, "Caller runtime manifest differs")
    protocol_hash, manifest_hash = sha256(snapshot / "PROTOCOL.json"), sha256(snapshot / "source_manifest.json")
    require(protocol_hash == record["protocol_sha256"] == manifest["protocol_sha256"] == audit["protocol_sha256"]
            == preflight["protocol_sha256"] == recovery["protocol_sha256"], "Copied protocol identities differ")
    require(manifest_hash == record["source_manifest_sha256"] == recovery["source_manifest_sha256"],
            "Copied manifest identities differ")
    require(preflight["status"] == "passed" and manifest == original_manifest, "Copied preflight/manifest differs")
    for name in ("PROTOCOL.json", "source_manifest.json", "PREFLIGHT.json"):
        require(sha256(run / name) == sha256(snapshot / name),
                "Original and copied protocol/manifest/preflight bytes differ")
    sources = verify_sources(run, manifest, logical_origin=origin)
    verify_external(protocol, audit)
    fits, fit_hashes = verify_fits(run, protocol, manifest, audit, recovery,
                                  logical_origin=origin, allow_additional=True)
    snapshots = validation_files(snapshot, audit, logical_origin=origin)
    copied, immutable, relocated = {}, {}, {}

    def add(rel, source, digest, *, mutable=False):
        relative(rel)
        require(rel not in copied, "Duplicate migration evidence path")
        copied[rel] = str(source)
        if not mutable:
            immutable[rel] = digest

    for name in ORIGIN_FILES:
        digest = sha256(snapshot / name)
        add("migration/origin/" + name, origin / name, digest)
        if name != "AUDIT.json":
            add(name, origin / name, digest)
    for rel, (_, digest, source) in sources.items():
        add(rel, manifest["archives"][source], digest)
        relocated[source] = str(run / rel)
    for rel, digest in fit_hashes.items():
        add(rel, origin / rel, digest)
    for rel, (_, digest) in snapshots.items():
        add(rel, origin / rel, digest, mutable=True)
        add("migration/origin/" + rel, origin / rel, digest)
    recovery_files = tree_files(recovery_root)
    wanted_recovery = {name for name in recovery["archived_files"]
                       if not name.startswith("interrupted_fits/")} | {"recovery.json"}
    if "lock_observations.jsonl" in recovery_files:
        wanted_recovery.add("lock_observations.jsonl")
    require(set(recovery_files) == wanted_recovery, "Copied recovery evidence file set differs")
    for rel, path in recovery_files.items():
        relative(rel)
        require(not rel.endswith(".lock"), "A lock file was copied")
        if rel in recovery["archived_files"]:
            check_hash(path, recovery["archived_files"][rel])
        add("migration/origin/recovery/" + rel, recovery_label / rel, sha256(path))
    for name in ("PROTOCOL.json", "source_manifest.json", "AUDIT.json"):
        require(sha256(recovery_root / name) == sha256(snapshot / name), "Copied recovery/origin evidence differs")
    excluded = []
    for k in sorted(PARTIAL_KEYS):
        prefix = "interrupted_fits/" + fit_relative(k).removeprefix("results/") + "/"
        hashes = {name.removeprefix(prefix): digest for name, digest in recovery["archived_files"].items()
                  if name.startswith(prefix)}
        require(hashes and set(hashes) <= {"adapter.pt", "training.json", "progress.json"},
                "Unexpected interrupted-fit evidence")
        excluded.append(dict(key=list(k), relative_dir=fit_relative(k), files_sha256=hashes))
    expected = dict(version=1, origin_run=str(origin), destination_run=str(run), recovery_dir=str(recovery_label),
                    protocol_sha256=protocol_hash, source_manifest_sha256=manifest_hash, source_archives=relocated,
                    copied_fits=fits, immutable_files_sha256=immutable, copy_sources=copied,
                    validation_snapshot_sha256={rel: value[1] for rel, value in snapshots.items()},
                    validation_policy=VALIDATION_POLICY, excluded_partial_fits=excluded,
                    historical_references_verified=EXPECTED_REFERENCES, runtime_sources_verified=EXPECTED_SOURCES,
                    copied_complete_fit_files=EXPECTED_FITS * len(FIT_FILES), transfer="independent_regular_file_byte_copies",
                    original_locks_accessed=False,
                    interrupted_fits_policy="Not copied. Retrain from the original seed; no optimizer/RNG continuation.")
    comparable = {name: value for name, value in record.items() if name not in ("status", "created_utc")}
    require(comparable == expected, "Migration manifest differs from its copied provenance and fixed scope")
    require(isinstance(record.get("created_utc"), str), "Missing migration creation timestamp")
    datetime.fromisoformat(record["created_utc"])
    expected_origin = {str(Path(rel).relative_to("migration/origin")) for rel in record["immutable_files_sha256"]
                       if rel.startswith("migration/origin/")}
    require(set(tree_files(run / "migration/origin")) == expected_origin, "Migration origin evidence file set differs")
    require(set(tree_files(run / "migration")) == {"origin/" + name for name in expected_origin},
            "Unexpected file in migration evidence directory")
    for rel, digest in record["immutable_files_sha256"].items():
        path = run / relative(rel)
        check_hash(path, digest)
        require(path.stat().st_nlink == 1, "Linked migration file: " + rel)
    for fit in record["copied_fits"]:
        require(set(tree_files(run / fit["relative_dir"])) == set(FIT_FILES), "Copied complete fit file set changed")
    require(migration_path.stat().st_nlink == 1, "Hard-linked migration manifest")
    digest = sha256(migration_path)
    files = dict(record["immutable_files_sha256"], **{"migration.json": digest})
    evidence = dict(manifest_sha256=digest, origin_run=str(origin), destination_run=str(run),
                    copied_fits=EXPECTED_FITS, copied_files=EXPECTED_FITS * len(FIT_FILES),
                    runtime_sources_verified=EXPECTED_SOURCES, historical_references_verified=EXPECTED_REFERENCES,
                    protocol_sha256=record["protocol_sha256"], source_manifest_sha256=record["source_manifest_sha256"],
                    validation_policy=VALIDATION_POLICY, transfer=record["transfer"],
                    origin_evidence="migration/origin", immutable_files=len(files))
    return dict(archives=record["source_archives"], evidence=evidence, files_sha256=files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination-run", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Read-only validation and copy plan; create nothing")
    args = parser.parse_args(argv)
    result = migrate(args.source_run, args.destination_run, args.recovery_dir, args.check_only)
    if args.check_only:
        print(json.dumps(dict(status=result["status"], source=result["origin_run"], destination=result["destination_run"],
              copied_fits=len(result["copied_fits"]), complete_fit_files=result["copied_complete_fit_files"],
              runtime_sources=result["runtime_sources_verified"], historical_references=result["historical_references_verified"],
              immutable_files=len(result["immutable_files_sha256"]), validation_files=len(result["validation_snapshot_sha256"]),
              excluded_partial_fits=result["excluded_partial_fits"], writes_performed=False), indent=2))
    else:
        print(json.dumps(result["evidence"], indent=2))


if __name__ == "__main__":
    main()
