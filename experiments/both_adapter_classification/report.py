"""Independently audit Both predictions and report seven models by dataset.

No GPU or model loading is needed. Use a model's python_local.sh for NumPy,
PyTorch and scikit-learn. --allow-partial permits pending new runs, never failed
arithmetic, altered references, unsupported identities, or invented summaries.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import metrics, residual_mlp, sha256
from experiments.adapter_placement_classification.report import (
    atomic_text, expected_order_hashes, load_datasets, read, require,
    selection_audit, write_csv,
)

MODELS = ("AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo")
DATASETS = ("openi", "chexpert", "chestxray14")
NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
METRICS = ("macro_auroc", "macro_auprc")
METHODS = ("original", "pre_mlp", "post_mlp", "both_mlp")
POST_MODELS = ("AlphaRAD", "KAD", "MaCo")
ADDED_MODELS = ("BiomedCLIP", "BioViL-T", "MaCo")


def same(actual, recorded, context):
    """Verify aggregate and per-class metrics, including coverage and nulls."""
    if isinstance(actual, dict):
        require(isinstance(recorded, dict) and set(actual) == set(recorded), context + ": fields differ")
        for key, value in actual.items():
            same(value, recorded[key], context + "/" + key)
    elif actual is None:
        require(recorded is None, context + ": undefined value differs")
    else:
        require(isinstance(recorded, (int, float)) and np.isfinite(recorded)
                and abs(actual - recorded) < 1e-12, context + ": metric differs")


def expected_references():
    keys = set()
    for model in MODELS:
        for dataset in DATASETS:
            keys.add((model, dataset, 42, "original"))
            if model in ADDED_MODELS:
                keys.add((model, dataset, 42, "aggregation_only"))
            for seed in (42, 43, 44):
                keys.add((model, dataset, seed, "pre_mlp"))
                if model in POST_MODELS:
                    keys.add((model, dataset, seed, "post_mlp"))
    return keys


def key(entry):
    return tuple(entry[k] for k in ("model", "dataset", "seed", "method"))


def checkpoint_shapes(dim, both):
    single = {"down.weight": (64, dim), "down.bias": (64,),
              "up.weight": (dim, 64), "up.bias": (dim,)}
    return {prefix + name: shape for prefix in (("pre.", "post.") if both else ("",))
            for name, shape in single.items()}


def verify_val(value, labels, classes, context):
    require(value["total_classes"] == value["valid_classes"] == len(classes), context + ": class count")
    require(set(value["per_class"]) == set(classes), context + ": class names")
    for i, name in enumerate(classes):
        row = value["per_class"][name]
        require(row["positives"] == int(labels[:, i].sum()) and
                row["negatives"] == int(len(labels) - labels[:, i].sum()), context + ": class counts")
        for metric in ("auroc", "auprc"):
            require(np.isfinite(row[metric]) and 0 <= row[metric] <= 1, context + ": invalid class metric")
    for aggregate, per_class in (("macro_auroc", "auroc"), ("macro_auprc", "auprc")):
        same(float(np.mean([value["per_class"][c][per_class] for c in classes])), value[aggregate], context)


def verify_prediction(folder, dataset):
    expected, spec = dataset["expected"], dataset["spec"]
    with np.load(folder / "test_predictions.npz", allow_pickle=False) as prediction:
        require(set(prediction.files) == set(expected) | {"logits"}, "Prediction fields differ")
        for name, value in expected.items():
            require(np.array_equal(prediction[name], value), "Test identity/order differs: " + name)
        logits = prediction["logits"].copy()
    require(logits.shape == expected["labels"].shape and np.isfinite(logits).all(), "Invalid prediction array")
    require(np.isin(expected["labels"], (0, 1)).all(), "Nonbinary labels")
    return metrics(expected["labels"], logits, spec["classes"])


def verify_sources(record, run, source_cache=None):
    """Match each record to a once-per-process verified immutable snapshot.

    Every fit uses the same frozen manifest and source archives. Reading and
    hashing those archives once is sufficient within a report invocation;
    every result/training record is still compared independently with the
    verified maps, manifest digest and protocol digest. The cache belongs to
    main(), never to another report process or persisted on disk.
    """
    sources = record["sources_sha256"]
    require(sources, "Runtime sources are not recorded")
    snapshot = run / "source_manifest.json"
    if source_cache is None:
        source_cache = {}
    snapshot_key = str(snapshot)
    if snapshot_key not in source_cache:
        require(snapshot.is_file(), "Missing frozen source manifest")
        manifest = read(snapshot)
        manifest_digest = sha256(snapshot)
        require(manifest["sources_sha256"], "Frozen runtime source map is empty")
        require(set(manifest["archives"]) == set(manifest["sources_sha256"]), "Incomplete archived source map")
        checked = []
        for name, digest in manifest["sources_sha256"].items():
            archive = Path(manifest["archives"][name])
            require(archive.is_relative_to(run / "source"), "Unexpected runtime archive location")
            require(archive.is_file() and sha256(archive) == digest, "Runtime source archive mismatch: " + name)
            checked.append(dict(source=name, sha256=digest, verified_copy=str(archive)))
        require(any(Path(p).name == "train.py" for p in manifest["sources_sha256"]), "Trainer source absent")
        stat = snapshot.stat()
        source_cache[snapshot_key] = dict(manifest=manifest, digest=manifest_digest,
            stat_identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
            evidence=dict(manifest_sha256=manifest_digest, files=checked,
                archive_content_hashes_checked_once_per_report_process=True))
    verified = source_cache[snapshot_key]
    stat = snapshot.stat()
    require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == verified["stat_identity"],
            "Frozen source manifest changed during this report process")
    manifest = verified["manifest"]
    require(record["source_manifest_sha256"] == verified["digest"], "Runtime source manifest changed")
    require(manifest["protocol_sha256"] == record["protocol_sha256"], "Source manifest protocol differs")
    require(sources == manifest["sources_sha256"], "Runtime source map differs from frozen manifest")
    return verified["evidence"]


def scaled_error_check(error, reference, context):
    """Validate a recorded max error against the worker's scale-aware bound.

    The worker performed elementwise assert_allclose. Only its maximum error
    was stored, so this checks a necessary aggregate bound, not a repeat of
    the elementwise assertion or an independent new forward pass.
    """
    require(np.isfinite(error) and error >= 0, context + ": invalid error")
    require(np.isfinite(reference).all(), context + ": nonfinite reference")
    largest_reference = float(np.max(np.abs(reference)))
    maximum_tolerance = 3e-4 + 3e-4 * largest_reference
    require(error <= maximum_tolerance, context + ": error exceeds scale-aware bound")
    return dict(max_abs_error=float(error), max_abs_reference=largest_reference,
                atol=3e-4, rtol=3e-4, largest_elementwise_tolerance=maximum_tolerance,
                independent_forward_recomputed=False,
                interpretation="The worker checked all elements; the reporter verifies saved evidence and its necessary scalar bound.")


def verify_runtime_validation(model, dataset_name, protocol, dataset, run):
    """Audit saved full-cache and old-checkpoint replay evidence on the CPU."""
    directory = run / "validation" / model / dataset_name
    frozen_file = directory / "frozen_path.json"
    frozen = read(frozen_file)
    spec = dataset["spec"]
    total_rows = len(dataset["rows"])
    caches = protocol["caches"][f"{model}/{dataset_name}"]
    require(frozen["status"] == "passed", "Frozen-path verification failed")
    require(frozen["rows"] == total_rows == sum(spec["counts"].values()), "Frozen-path coverage differs")
    require(frozen["cache_manifest"] == caches, "Frozen-path cache manifest differs from protocol")
    require(np.isfinite(frozen["seconds"]) and frozen["seconds"] >= 0, "Invalid frozen-path verification time")
    cache_records = []
    for item in caches:
        require(sha256(item["path"]) == item["sha256"], "Frozen cache JSON digest changed")
        cache = read(item["path"])
        require(cache["status"] == "complete", "Incomplete frozen cache")
        require(cache["dataset_sha256"] == spec["sha256"] and cache["samples_sha256"] == spec["samples_sha256"],
                "Frozen cache dataset differs")
        for name, identity in item["files"].items():
            stat = Path(name).stat()
            require(stat.st_size == identity["bytes"] and stat.st_mtime_ns == identity["mtime_ns"],
                    "Frozen cache file size/mtime changed: " + name)
        cache_records.append(cache)
    if model in ("AlphaRAD", "RadZero"):
        require(len(cache_records) == cache_records[0]["shards"], "Frozen cache shards missing")
        aggregation = np.empty((total_rows, len(spec["classes"])), dtype=np.float32)
        seen, token_seen, provenance = set(), set(), []
        for cache in cache_records:
            indices = cache["indices"]
            require(cache["processed"] == len(indices), "Frozen shard processed count differs")
            require(not seen.intersection(indices), "Frozen shard overlap")
            seen.update(indices)
            logits = np.load(cache["logits_path"], mmap_mode="r")
            require(logits.shape == (len(indices), len(spec["classes"])), "Frozen logit shape differs")
            aggregation[indices] = logits
            require(sha256(cache["head_path"]) == cache["head_sha256"] == cache_records[0]["head_sha256"],
                    "Frozen head digest differs across shards")
            require(read(cache["head_path"])["class_names"] == spec["classes"], "Frozen head class order differs")
            require(cache["checkpoint_sha256"] == cache_records[0]["checkpoint_sha256"], "Mixed checkpoint shards")
            provenance.append(dict(head_path=cache["head_path"], head_sha256=cache["head_sha256"],
                                   checkpoint_sha256=cache["checkpoint_sha256"], logits_sha256=sha256(cache["logits_path"])))
            for token_spec in cache["token_shards"]:
                require(not token_seen.intersection(token_spec["indices"]), "Frozen token shard overlap")
                token_seen.update(token_spec["indices"])
                tokens = np.load(token_spec["path"], mmap_mode="r")
                require(list(tokens.shape) == token_spec["shape"] and tokens.dtype == np.float32,
                        "Frozen token shape/dtype differs")
        require(seen == token_seen == set(range(total_rows)), "Frozen row/token coverage incomplete")
        require(frozen["cache_provenance"] == provenance, "Frozen native-head provenance differs")
    else:
        require(len(cache_records) == 1, "Unexpected frozen token cache shard count")
        cache = cache_records[0]
        require(cache["class_names"] == spec["classes"] and cache["counts"] == spec["counts"], "Frozen cache classes/counts differ")
        tokens = np.load(cache["files"]["tokens.npy"]["path"], mmap_mode="r")
        require(list(tokens.shape) == cache["token_shape"] and tokens.shape[0] == total_rows,
                "Frozen token row count/shape differs")
        aggregation_spec = cache["files"]["aggregation_logits.npy"]
        require(sha256(aggregation_spec["path"]) == aggregation_spec["sha256"], "Frozen aggregation logit digest changed")
        aggregation = np.load(aggregation_spec["path"], mmap_mode="r")
        require(aggregation.shape == (total_rows, len(spec["classes"])), "Frozen aggregation row shape differs")
        require(len(frozen["cache_provenance"]) == 1, "Frozen provenance count differs")
        provenance = frozen["cache_provenance"][0]
        for field, expected in dict(path=caches[0]["path"], sha256=caches[0]["sha256"],
                query_sha256=cache["query_sha256"], checkpoint_sha256=cache["checkpoint_sha256"]).items():
            require(provenance[field] == expected, "Frozen cache provenance differs: " + field)
    error_evidence = scaled_error_check(frozen["max_abs_logit_error"], aggregation, "Frozen raw path")
    references = [r for r in protocol["references"] if r["model"] == model and r["dataset"] == dataset_name
                  and r["method"] == "pre_mlp"]
    require(sorted(r["seed"] for r in references) == protocol["seeds"], "Missing old-pre replay references")
    test_rows = len(dataset["expected"]["indices"])
    locations = [0, test_rows // 2, test_rows - 1]
    replays = []
    for reference in references:
        path = directory / f"reused_pre_seed{reference['seed']}.json"
        replay = read(path)
        require(replay["status"] == "passed" and replay["source"] == reference["source"], "Old-pre replay identity/status differs")
        require(replay["files_sha256"] == reference["files_sha256"], "Old-pre replay file manifest differs")
        require(replay["test_row_locations"] == locations, "Old-pre replay test rows differ")
        for name, digest in reference["files_sha256"].items():
            require(sha256(Path(reference["source"]) / name) == digest, "Old-pre replay reference changed: " + name)
        with np.load(Path(reference["source"]) / "test_predictions.npz", allow_pickle=False) as prediction:
            for name, expected in dataset["expected"].items():
                require(np.array_equal(prediction[name], expected), "Old-pre replay TEST identity differs: " + name)
            replay_error = scaled_error_check(replay["max_abs_error"], prediction["logits"][locations], "Old-pre checkpoint replay")
        replays.append(dict(seed=reference["seed"], path=str(path), sha256=sha256(path), source=reference["source"],
                            files_sha256=reference["files_sha256"], test_row_locations=locations, error_check=replay_error))
    return dict(frozen_path=dict(path=str(frozen_file), sha256=sha256(frozen_file), status="passed", rows=total_rows,
                cache_manifest=frozen["cache_manifest"], cache_provenance=frozen["cache_provenance"], error_check=error_evidence),
                reused_pre_replays=replays)


def audit_entry(entry, protocol, protocol_hash, datasets, run, initialization_cache, order_cache,
                validation_cache=None, source_cache=None):
    folder = Path(entry["source"])
    model, dataset_name, seed, method = key(entry)
    new = method == "both_mlp"
    fit = method in ("pre_mlp", "post_mlp", "both_mlp")
    dataset = datasets[dataset_name]
    spec = dataset["spec"]
    dim = protocol["model_specs"][model]["dim"]
    single_parameters = dim * 64 + 64 + 64 * dim + dim
    parameter_count = single_parameters * (2 if new else 1) if fit else 0
    if not new:
        require(entry["files_sha256"], "Reference hashes not frozen")
        for name, digest in entry["files_sha256"].items():
            require(sha256(folder / name) == digest, "Reference changed: " + str(folder / name))
        needed = {"result.json", "test_predictions.npz"}
        if fit:
            needed |= {"adapter.pt", "training.json"}
        require(needed <= set(entry["files_sha256"]), "Reference manifest omits required artifact")
    result = read(folder / "result.json")
    records = [result]
    training = read(folder / "training.json") if fit else None
    if training:
        records.append(training)
    for record in records:
        require(record["status"] == "complete", "Incomplete artifact")
        require(record["model"].lower() == model.lower() and record["seed"] == seed, "Run identity mismatch")
        require(record["class_names"] == spec["classes"] and record["counts"] == spec["counts"], "Class/split mismatch")
        require(record["dataset_sha256"] == spec["sha256"] and
                record["samples_sha256"] == spec["samples_sha256"], "Dataset provenance mismatch")
        require(record["trainable_parameters"] == parameter_count, "Parameter budget mismatch")
        if fit:
            for field in ("epochs", "min_epochs", "patience", "lr", "weight_decay"):
                require(record["config"][field] == protocol["train"][field], "Training config mismatch: " + field)
            config = record["config"]
            require(config.get("effective_batch_size", config.get("batch_size")) == 32, "Effective batch size mismatch")
            require(not config.get("test_selection", False), "Test-based selection")
    wanted_old_method = {"original": "baseline", "aggregation_only": "aggregation_only", "post_mlp": "post_mlp",
                         "pre_mlp": "existing" if model == "RadZero" else "token_mlp"}
    require(result["method"] == ("both_mlp" if new else wanted_old_method[method]), "Method identity mismatch")
    values = verify_prediction(folder, dataset)
    for record in records:
        same(values, record["test"], "recomputed TEST")
    details = dict(key=list(key(entry)), source=str(folder), new_fit=new, supervised_fit=fit,
                   recomputed_test={m: values[m] for m in METRICS})
    epoch, epochs_run = 0, 0
    if fit:
        checkpoint = torch.load(folder / "adapter.pt", map_location="cpu", weights_only=True)
        state = checkpoint["adapter_state_dict"]
        shapes = checkpoint_shapes(dim, new)
        require(set(state) == set(shapes), "Unexpected trainable checkpoint tensors")
        require(all(tuple(state[k].shape) == shape for k, shape in shapes.items()), "Checkpoint tensor shape mismatch")
        require(sum(v.numel() for v in state.values()) == parameter_count, "Checkpoint parameter count mismatch")
        require(all(torch.isfinite(v).all().item() for v in state.values()), "Nonfinite checkpoint weights")
        require(checkpoint["dim"] == dim and checkpoint["bottleneck"] == 64, "Checkpoint architecture mismatch")
        checkpoint_classes = checkpoint.get("class_names", checkpoint.get("classes"))
        require(checkpoint_classes == spec["classes"], "Checkpoint class order mismatch")
        epoch = selection_audit(training, result, checkpoint, protocol["train"])
        epochs_run = len(training["history"])
        train_y = np.asarray([dataset["rows"][i]["labels"] for i in dataset["indices"]["train"]], dtype=np.float32)
        val_y = np.asarray([dataset["rows"][i]["labels"] for i in dataset["indices"]["val"]], dtype=np.float32)
        weights = np.minimum((len(train_y) - train_y.sum(0)) / np.maximum(train_y.sum(0), 1), 20.)
        require(np.array_equal(np.asarray(training["pos_weight"], dtype=np.float32), weights), "pos_weight is not TRAIN-only")
        initial = training.get("initial_val", training.get("baseline_val"))
        verify_val(initial, val_y, spec["classes"], "epoch-zero VAL")
        for row in training["history"]:
            require(np.isfinite(row["train_bce"]), "Nonfinite training loss")
            verify_val(row["val"], val_y, spec["classes"], "epoch VAL")
        if epoch == 0:
            up_keys = [k for k in state if k.endswith("up.weight") or k.endswith("up.bias")]
            require(all(torch.count_nonzero(state[k]).item() == 0 for k in up_keys), "Epoch-zero adapter is not identity")
        details.update(best_epoch=epoch, epochs_run=epochs_run, checkpoint_tensors=list(state),
                       identity_checkpoint_verified=epoch == 0)
    if new:
        if validation_cache is None:
            validation_cache = {}
        validation_key = model, dataset_name
        if validation_key not in validation_cache:
            validation_cache[validation_key] = verify_runtime_validation(model, dataset_name, protocol, dataset, run)
        runtime_validation = validation_cache[validation_key]
        for record in records:
            require(record["cache_provenance"] == runtime_validation["frozen_path"]["cache_provenance"],
                    "New result frozen-cache provenance differs from validation evidence")
        require(parameter_count == protocol["model_specs"][model]["trainable_parameters"], "Protocol Both parameter count")
        for record in records:
            require(record["protocol_sha256"] == protocol_hash, "Protocol hash mismatch")
            require(record["adapter_position"] == "both", "Wrong position")
            for field in ("joint_training", "independent_adapters", "frozen_backbone_verified"):
                require(record[field] is True, "Missing positive verification: " + field)
            require(record.get("test_selection") is False, "Test-selection verification missing")
        require(checkpoint["protocol_sha256"] == protocol_hash, "Checkpoint protocol mismatch")
        require(checkpoint["method"] == "both_mlp" and checkpoint["model"] == model, "Checkpoint run identity")
        for field in ("joint_training", "independent_adapters"):
            require(checkpoint[field] is True, "Checkpoint architecture verification missing: " + field)
        for record in records:
            require(record["adapter_sha256"] == sha256(folder / "adapter.pt"), "Recorded checkpoint digest differs")
            require(record["test_predictions_sha256"] == sha256(folder / "test_predictions.npz"), "Recorded prediction digest differs")
            require(record["adapter_architecture"] == dict(dim=dim, bottleneck=64, activation="GELU", linear_bias=True,
                    zero_initialized_up=True, copies=2), "Both architecture record differs")
        initialization_key = dim, seed
        if initialization_key not in initialization_cache:
            torch.manual_seed(seed)
            modules = torch.nn.ModuleDict({"pre": residual_mlp(dim, 64), "post": residual_mlp(dim, 64)})
            initialization_cache[initialization_key] = {
                k: hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k, v in modules.state_dict().items()}
        require(training["initialization_sha256"] == initialization_cache[initialization_key], "Seeded joint initialization differs")
        require(result["initialization_sha256"] == training["initialization_sha256"], "Initialization record differs")
        require(any(training["initialization_sha256"]["pre." + k] != training["initialization_sha256"]["post." + k]
                    for k in ("down.weight", "down.bias")), "MLPs initialized as identical copies")
        order_key = dataset_name, seed
        if order_key not in order_cache:
            order_cache[order_key] = expected_order_hashes(dataset["indices"]["train"], seed, protocol["train"]["epochs"])
        for row in training["history"]:
            require(row["train_order_sha256"] == order_cache[order_key][row["epoch"] - 1], "Seeded training order differs")
        gradient = training["gradient_evidence"]
        require(gradient == result["gradient_evidence"], "Gradient verification records differ")
        for branch in ("pre", "post"):
            record = gradient[branch]
            require(record["up_weight_nonzero"] is True and record["down_weight_nonzero"] is True,
                    branch + " MLP did not receive gradient in both layers")
            require(record["first_up_step"] is not None and record["first_down_step"] is not None,
                    branch + " missing first gradient step")
            require(record["first_down_step"] >= record["first_up_step"], branch + " inconsistent gradient order")
            require(np.isfinite(record["max_grad"]) and record["max_grad"] > 0, branch + " invalid gradient magnitude")
        replay = result.get("checkpoint_replay", result.get("checkpoint_validation"))
        require(replay is not None and replay.get("status", "passed") == "passed", "Missing checkpoint replay")
        require(replay["test_row_locations"] == [0, len(dataset["expected"]["indices"]) // 2,
                                                  len(dataset["expected"]["indices"]) - 1], "Checkpoint replay coverage differs")
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as prediction:
            checkpoint_error = scaled_error_check(replay["max_abs_error"], prediction["logits"][replay["test_row_locations"]],
                                                  "Both checkpoint replay")
        if source_cache is None:
            source_cache = {}
        source_evidence = verify_sources(result, run, source_cache)
        verify_sources(training, run, source_cache)
        details.update(sources=source_evidence, gradient_evidence=gradient,
                       checkpoint_replay=replay, checkpoint_replay_error_check=checkpoint_error,
                       runtime_validation=runtime_validation, result_current_sha256=sha256(folder / "result.json"),
                       frozen_backbone_verified=True,
                       initialization_reconstructed=True, train_order_reconstructed=True)
    filenames = ["result.json", "test_predictions.npz"] + (["training.json", "adapter.pt"] if fit else [])
    details["files_sha256"] = {name: sha256(folder / name) for name in filenames}
    row = dict(model=model, dataset=dataset_name, seed=seed, method=method, new_fit=new,
               macro_auroc_pct=100 * values["macro_auroc"], macro_auprc_pct=100 * values["macro_auprc"],
               best_epoch=epoch, epochs_run=epochs_run, trainable_parameters=parameter_count, source=str(folder))
    return row, details


def summarize(rows, protocol):
    lookup = defaultdict(dict)
    for row in rows:
        lookup[(row["model"], row["dataset"], row["method"])][row["seed"]] = row
    summaries = []
    for dataset in DATASETS:
        for model in MODELS:
            row = dict(model=model, dataset=dataset, classes=len(protocol["datasets"][dataset]["classes"]),
                       test_images=protocol["datasets"][dataset]["counts"]["test"],
                       both_trainable_parameters=protocol["model_specs"][model]["trainable_parameters"])
            for method in METHODS:
                runs = lookup[(model, dataset, method)]
                expected = 1 if method == "original" else (0 if method == "post_mlp" and model not in POST_MODELS else 3)
                row[method + "_runs"] = len(runs)
                row[method + "_expected_runs"] = expected
                row[method + "_status"] = "unavailable" if expected == 0 else "complete" if len(runs) == expected else "partial"
                row[method + "_seeds"] = ",".join(str(s) for s in sorted(runs))
                row[method + "_best_epochs"] = ",".join(str(runs[s]["best_epoch"]) for s in sorted(runs))
                row[method + "_epoch_zero_count"] = sum(r["best_epoch"] == 0 for r in runs.values())
                for metric in METRICS:
                    values = [r[metric + "_pct"] for r in runs.values()]
                    row[method + "_" + metric + "_mean_pct"] = float(np.mean(values)) if values else None
                    row[method + "_" + metric + "_sd_pct"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            for reference in ("pre_mlp", "post_mlp"):
                both, other = lookup[(model, dataset, "both_mlp")], lookup[(model, dataset, reference)]
                paired = sorted(set(both) & set(other))
                prefix = "both_minus_" + reference
                row[prefix + "_paired_runs"] = len(paired)
                for metric in METRICS:
                    differences = [both[s][metric + "_pct"] - other[s][metric + "_pct"] for s in paired]
                    row[prefix + "_" + metric + "_mean_pp"] = float(np.mean(differences)) if differences else None
                    row[prefix + "_" + metric + "_sd_pp"] = float(np.std(differences, ddof=1)) if len(differences) > 1 else None
            summaries.append(row)
    return summaries


def formatted(row, method):
    if row[method + "_status"] == "unavailable":
        return "— (not run)"
    count = row[method + "_runs"]
    if not count:
        return "pending (0/3)"
    result = []
    for metric in METRICS:
        mean, sd = row[method + "_" + metric + "_mean_pct"], row[method + "_" + metric + "_sd_pct"]
        result.append(f"{mean:.2f}" + (f"±{sd:.2f}" if sd is not None else ""))
    text = " / ".join(result)
    if row[method + "_status"] != "complete":
        text += f" [partial {count}/3]"
    return text


def report_text(audit, summaries, controls):
    lines = ["Both joint-adaptation classification results", "=" * 44,
             f"Status: {audit['status'].upper()}; new Both fits: {audit['completed_new_fits']}/63; "
             f"historical evaluations: {audit['completed_reference_evaluations']}/120.",
             "Cells: macro AUROC (%) / macro AUPRC (average precision, %).",
             "Fitted methods: mean ± sample SD over seeds 42,43,44; partial seed sets explicitly marked.",
             "Original: deterministic evaluation, no fabricated seed SD. No cross-dataset averages.", ""]
    for dataset in DATASETS:
        lines += [NAMES[dataset], "-" * len(NAMES[dataset]),
                  "| Model | Original | Pre Token MLP | Post MLP (available only) | Both MLP |",
                  "|---|---|---|---|---|"]
        for row in summaries:
            if row["dataset"] == dataset:
                lines.append("| " + row["model"] + " | " + " | ".join(formatted(row, m) for m in METHODS) + " |")
        lines.append("")
    lines += ["Added fixed-pooling controls (no MLP, no fitted parameters)",
              "| Model | Dataset | Original | Fixed aggregation only |", "|---|---|---|---|"]
    for row in controls:
        def values(prefix):
            return " / ".join(f"{row[prefix + '_' + m + '_pct']:.2f}" for m in METRICS)
        lines.append(f"| {row['model']} | {NAMES[row['dataset']]} | {values('original')} | {values('aggregation_only')} |")
    lines += ["", "Interpretation boundaries:",
              "- Both jointly trains two independent rank-64 residual GELU MLPs and has twice the single-position parameters.",
              "- Higher Both scores cannot alone distinguish complementary positions from increased capacity.",
              "- BiomedCLIP/BioViL-T/MaCo identity adapters equal fixed aggregation-only, not the original native classifier.",
              "- Existing Post is unavailable for RadZero/CARZero/BiomedCLIP/BioViL-T; Direct-anchor is not substituted.",
              "- Three-seed SD describes training variability, not patient-level confidence intervals or significance.",
              "- Local custom cohorts are retained; CheXpert TEST has 67 images; OpenI patient independence remains unverified.",
              "- Historical pre initialization/order hashes were not logged; stored files, configs, labels and metrics are audited.",
              "- New-fit checkpoint replay is sampled at first/middle/last TEST rows, not an independent full model rerun."]
    if audit["errors"]:
        lines += ["", "AUDIT ERRORS (affected runs excluded):"] + [json.dumps(e, ensure_ascii=False) for e in audit["errors"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol_path = run / "PROTOCOL.json"
    protocol, protocol_hash = read(protocol_path), sha256(protocol_path)
    require(protocol["models"] == list(MODELS), "Unexpected model set/order")
    require(protocol["seeds"] == [42, 43, 44], "Unexpected seed set")
    require(set(protocol["datasets"]) == set(DATASETS), "Unexpected datasets")
    references = protocol["references"]
    require(len({key(r) for r in references}) == len(references), "Duplicate reference entries")
    require({key(r) for r in references} == expected_references(), "Historical reference matrix differs")
    datasets = load_datasets(protocol)
    entries = list(references)
    for model in MODELS:
        for dataset in DATASETS:
            for seed in protocol["seeds"]:
                entries.append(dict(model=model, dataset=dataset, seed=seed, method="both_mlp",
                                    source=str(run / "results" / model / dataset / f"seed{seed}" / "both_mlp")))
    rows, audits, pending, errors = [], [], [], []
    initialization_cache, order_cache, validation_cache, source_cache = {}, {}, {}, {}
    for entry in entries:
        result_path = Path(entry["source"]) / "result.json"
        if entry["method"] == "both_mlp" and not result_path.exists():
            pending.append(dict(key=list(key(entry)), source=entry["source"]))
            continue
        try:
            row, evidence = audit_entry(entry, protocol, protocol_hash, datasets, run,
                                        initialization_cache, order_cache, validation_cache, source_cache)
            rows.append(row)
            audits.append(evidence)
        except Exception as error:
            errors.append(dict(key=list(key(entry)), source=entry["source"], error=f"{type(error).__name__}: {error}"))
    completed_new = sum(row["new_fit"] for row in rows)
    status = "failed" if errors else "partial" if pending else "passed"
    audit = dict(status=status, generated_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=protocol_hash,
                 expected_new_fits=63, completed_new_fits=completed_new,
                 expected_reference_evaluations=120, completed_reference_evaluations=len(rows) - completed_new,
                 historical_supervised_fits=90, deterministic_reference_evaluations=30,
                 pending=pending, errors=errors, runs=audits)
    summaries = summarize(rows, protocol)
    controls, lookup = [], {(r["model"], r["dataset"], r["method"]): r for r in rows if r["method"] in ("original", "aggregation_only")}
    for dataset in DATASETS:
        for model in ADDED_MODELS:
            if all((model, dataset, method) in lookup for method in ("original", "aggregation_only")):
                control = dict(model=model, dataset=dataset, trainable_parameters=0)
                for method in ("original", "aggregation_only"):
                    for metric in METRICS:
                        control[method + "_" + metric + "_pct"] = lookup[(model, dataset, method)][metric + "_pct"]
                for metric in METRICS:
                    control["aggregation_minus_original_" + metric + "_pp"] = control["aggregation_only_" + metric + "_pct"] - control["original_" + metric + "_pct"]
                controls.append(control)
    write_csv(run / "per_seed.csv", rows)
    write_csv(run / "summary.csv", summaries)
    write_csv(run / "aggregation_controls.csv", controls)
    atomic_text(run / "AUDIT.json", json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    atomic_text(run / "REPORT.txt", report_text(audit, summaries, controls))
    print(json.dumps({k: v for k, v in audit.items() if k not in ("runs", "pending")}, indent=2))
    if errors or (pending and not args.allow_partial):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
