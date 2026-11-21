"""Audit and summarize the locked 72-fit adapter-placement experiment.

Run with a model's python_local.sh (NumPy, scikit-learn, and torch required).
--allow-partial reports completed fits without treating pending fits as errors;
failed arithmetic, provenance, or matching checks always return a failure.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import residual_mlp, sha256

METRICS = ("macro_auroc", "macro_auprc")
NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_csv(path, rows):
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(path, output.getvalue())


def recalculate(labels, logits):
    require(labels.shape == logits.shape, "Prediction/label shape mismatch")
    require(np.isfinite(logits).all(), "Nonfinite logits")
    require(np.isin(labels, [0, 1]).all(), "Labels must be binary")
    auc, ap = [], []
    for c in range(labels.shape[1]):
        require(len(np.unique(labels[:, c])) == 2, f"Undefined AUROC for class {c}")
        auc.append(float(roc_auc_score(labels[:, c], logits[:, c])))
        ap.append(float(average_precision_score(labels[:, c], logits[:, c])))
    return dict(macro_auroc=float(np.mean(auc)), macro_auprc=float(np.mean(ap)))


def close_metrics(actual, recorded, context):
    for key in METRICS:
        require(abs(actual[key] - recorded[key]) < 1e-12, f"{context}: {key} mismatch")


def load_datasets(protocol):
    datasets = {}
    for name, spec in protocol["datasets"].items():
        require(sha256(spec["path"]) == spec["sha256"], name + " dataset hash changed")
        meta = read(spec["path"])
        require(sha256(meta["samples_file"]) == spec["samples_sha256"] == meta["samples_sha256"],
                name + " samples hash changed")
        rows = read(meta["samples_file"])
        require(meta["class_names"] == spec["classes"], name + " class order changed")
        counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
        require(counts == spec["counts"], name + " split count changed")
        require(len({r["id"] for r in rows}) == len(rows), name + " duplicate IDs")
        for field in ("group_id", "patient_id", "uid"):
            seen = {}
            for row in rows:
                if field in row:
                    require(seen.setdefault(row[field], row["split"]) == row["split"],
                            name + " split overlap in " + field)
        indices = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s], dtype=np.int64)
                   for s in ("train", "val", "test")}
        test = [rows[i] for i in indices["test"]]
        datasets[name] = dict(spec=spec, rows=rows, indices=indices,
            expected=dict(indices=indices["test"], labels=np.asarray([r["labels"] for r in test], dtype=np.float32),
                          ids=np.asarray([r["id"] for r in test]), groups=np.asarray([r["group_id"] for r in test])))
    return datasets


def selection_audit(training, result, checkpoint, cfg):
    initial = training.get("initial_val", training.get("baseline_val"))
    require(initial is not None, "Missing epoch-zero validation metrics")
    best_epoch, best = 0, initial
    stale = 0
    history = training["history"]
    require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "Nonconsecutive history")
    require(cfg["min_epochs"] <= len(history) <= cfg["epochs"], "Invalid training length")
    for index, record in enumerate(history):
        require(np.isfinite(record["val"]["macro_auroc"]), "Nonfinite validation AUROC")
        if record["val"]["macro_auroc"] > best["macro_auroc"]:
            best_epoch, best, stale = record["epoch"], record["val"], 0
        else:
            stale += 1
        should_stop = record["epoch"] >= cfg["min_epochs"] and stale >= cfg["patience"]
        require(not should_stop or index == len(history) - 1, "Training continued after early-stop rule")
    require(len(history) == cfg["epochs"] or should_stop, "Training stopped before budget/patience")
    require(best_epoch == training["best_epoch"] == result["best_epoch"] == checkpoint["best_epoch"],
            "Selected checkpoint is not strict earliest validation maximum")
    close_metrics(best, training["best_val"], "training selected validation")
    close_metrics(best, result["best_val"], "result selected validation")
    return best_epoch


def audit_fit(entry, protocol, protocol_hash, datasets, run):
    folder = Path(entry["source"])
    result, training = read(folder / "result.json"), read(folder / "training.json")
    require(result["status"] == training["status"] == "complete", "Result/training not complete")
    spec = datasets[entry["dataset"]]["spec"]
    for record in (result, training):
        require(record["seed"] == entry["seed"], "Seed mismatch")
        require(record["model"].lower() == entry["model"].lower(), "Model mismatch")
        require(record["class_names"] == spec["classes"], "Class order mismatch")
        require(record["dataset_sha256"] == spec["sha256"], "Dataset hash mismatch")
        require(record["samples_sha256"] == spec["samples_sha256"], "Samples hash mismatch")
        require(record["counts"] == spec["counts"], "Counts mismatch")
        require(record["trainable_parameters"] == protocol["trainable_parameters"] == 99136, "Parameter budget mismatch")
        config = record["config"]
        for key in ("epochs", "min_epochs", "patience", "lr", "weight_decay"):
            require(config[key] == protocol["train"][key], "Training config mismatch: " + key)
        require(config.get("effective_batch_size", config.get("batch_size")) == 32, "Effective batch size mismatch")
        require(not config.get("test_selection", False), "Test-based selection")
        architecture = record["adapter_architecture"]
        for key, value in dict(dim=768, bottleneck=64, activation="GELU", linear_bias=True, zero_initialized_up=True).items():
            require(architecture[key] == value, "Adapter architecture mismatch: " + key)
    if not entry["reused"]:
        require(result["protocol_sha256"] == training["protocol_sha256"] == protocol_hash, "Protocol hash mismatch")
        require(result["method"] == training["method"] == entry["method"], "Method mismatch")
        require(result["test_selection"] is False, "Test-based selection")
        expected_position = "pre" if entry["method"] == "native_mean_pre" else "post"
        require(result["adapter_position"] == expected_position, "Adapter position mismatch")
        require(result["pooling"] == ("native_mean" if entry["method"].startswith("native_mean") else "text_guided"),
                "Pooling mismatch")
        trainer = Path(__file__).with_name("train.py")
        trainer_hash = result["sources_sha256"][str(trainer)]
        candidates = [trainer] + sorted((run / "source_revisions").glob("*/train.py"))
        matched_source = next((p for p in candidates if sha256(p) == trainer_hash), None)
        require(matched_source is not None, "Recorded trainer source has no hash-matching current/archive copy")
        trainer_text = matched_source.read_text()
        require("assert all(p.grad is None and not p.requires_grad for m in frozen_modules for p in m.parameters())" in trainer_text,
                "Recorded completed trainer lacks frozen-backbone gradient assertion")
        explicit_frozen = result.get("frozen_backbone_verified")
        if explicit_frozen is None:
            require(entry["model"] == "AlphaRAD" and entry["dataset"] == "chexpert" and matched_source.parent.name == "revision_0",
                    "Missing frozen verification outside the three preserved revision-0 AlphaRAD fits")
        else:
            require(explicit_frozen is True and training["frozen_backbone_verified"] is True,
                    "Frozen-backbone verification did not pass")
    else:
        for filename, digest in entry["files_sha256"].items():
            require(sha256(folder / filename) == digest, "Reused pre file hash changed: " + filename)

    checkpoint = torch.load(folder / "adapter.pt", map_location="cpu", weights_only=True)
    state = checkpoint["adapter_state_dict"]
    shapes = {"down.weight": (64, 768), "down.bias": (64,), "up.weight": (768, 64), "up.bias": (768,)}
    require(set(state) == set(shapes), "Unexpected trainable parameters in adapter checkpoint")
    require(all(tuple(state[k].shape) == shape for k, shape in shapes.items()), "MLP shape mismatch")
    require(sum(value.numel() for value in state.values()) == 99136, "Checkpoint parameter count mismatch")
    require(all(torch.isfinite(value).all().item() for value in state.values()), "Nonfinite adapter parameter")
    epoch = selection_audit(training, result, checkpoint, protocol["train"])
    if epoch == 0:
        require(torch.count_nonzero(state["up.weight"]).item() == torch.count_nonzero(state["up.bias"]).item() == 0,
                "Epoch-zero checkpoint is not identity initialized")
    with np.load(folder / "test_predictions.npz", allow_pickle=False) as predictions:
        for name, expected in datasets[entry["dataset"]]["expected"].items():
            require(np.array_equal(predictions[name], expected), "Test membership/order mismatch: " + name)
        values = recalculate(predictions["labels"], predictions["logits"])
    close_metrics(values, result["test"], "recomputed result")
    close_metrics(values, training["test"], "recomputed training")
    train_labels = np.asarray([datasets[entry["dataset"]]["rows"][i]["labels"]
                               for i in datasets[entry["dataset"]]["indices"]["train"]], dtype=np.float32)
    weights = np.minimum((len(train_labels) - train_labels.sum(0)) / np.maximum(train_labels.sum(0), 1), 20.)
    require(np.array_equal(np.asarray(training["pos_weight"], dtype=np.float32), weights), "Positive weights not train-only")
    files = {name: sha256(folder / name) for name in ("result.json", "training.json", "adapter.pt", "test_predictions.npz")}
    row = dict(model=entry["model"], dataset=entry["dataset"], seed=entry["seed"], method=entry["method"],
               reused=entry["reused"], macro_auroc_pct=100 * values["macro_auroc"],
               macro_auprc_pct=100 * values["macro_auprc"], best_epoch=epoch, epochs_run=len(training["history"]),
               trainable_parameters=99136, source=str(folder))
    details = dict(key=[entry[k] for k in ("model", "dataset", "seed", "method")],
                   source=str(folder), files_sha256=files, test_recomputed=values, best_epoch=epoch,
                   checkpoint_only_shared_mlp=True, identity_checkpoint_verified=epoch == 0,
                   frozen_evidence="Completed worker freezes backend and asserts no backbone gradients; saved checkpoint contains only four shared MLP tensors.")
    if not entry["reused"]:
        details.update(frozen_backbone_verified=result.get("frozen_backbone_verified"),
                       frozen_assertion_source=str(matched_source), trainer_sha256=trainer_hash)
    return row, training, details


def expected_order_hashes(indices, seed, epochs):
    """Replay the persistent DataLoader's seeded sampler, without GPU/workers."""
    generator = torch.Generator().manual_seed(seed)
    # A persistent iterator consumes its worker base_seed exactly once.
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    sampler = torch.utils.data.RandomSampler(range(len(indices)), generator=generator)
    result = []
    for _ in range(epochs):
        order = indices[np.asarray(list(sampler), dtype=np.int64)]
        result.append(hashlib.sha256(order.astype("<i8").tobytes()).hexdigest())
    return result


def matching_audit(completed, datasets):
    init_expected, observed_init, observed_orders, expected_orders = {}, {}, {}, {}
    checked = []
    for entry, training in completed:
        if entry["reused"]:
            continue
        seed, dataset = entry["seed"], entry["dataset"]
        if seed not in init_expected:
            torch.manual_seed(seed)
            model = residual_mlp(768, 64)
            init_expected[seed] = {k: hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()
                                   for k, v in model.state_dict().items()}
        initialization = training["initialization_sha256"]
        require(initialization == init_expected[seed], "Initialization differs from seeded shared MLP")
        require(observed_init.setdefault(seed, initialization) == initialization, "Same-seed initialization mismatch")
        key = dataset, seed
        if key not in expected_orders:
            expected_orders[key] = expected_order_hashes(datasets[dataset]["indices"]["train"], seed, 20)
        for record in training["history"]:
            order = record["train_order_sha256"]
            epoch = record["epoch"]
            require(order == expected_orders[key][epoch - 1], "Training order differs from persistent seeded DataLoader")
            require(observed_orders.setdefault((dataset, seed, epoch), order) == order,
                    "Same-seed same-epoch training orders differ across methods/models")
        checked.append(dict(model=entry["model"], dataset=dataset, seed=seed, method=entry["method"],
                            epochs=len(training["history"])))
    return dict(new_runs_checked=checked, initialization_hashes_by_seed=init_expected,
                observed_order_hashes_compared=len(observed_orders),
                expected_order_hashes={f"{d}/seed{s}": v for (d, s), v in expected_orders.items()},
                reused_pre_limitation="Historical pre runs did not log initialization or epoch order hashes. Their stored artifacts are hash-verified; seed, sampler implementation, and protocol match, but historical order hashes are not retrospectively observed.")


def maco_pooling_controls(protocol, datasets):
    """Recompute frozen no-MLP controls without touching existing artifacts."""
    controls, evidence = [], []
    for dataset, locked in datasets.items():
        spec = locked["spec"]
        references = [r for r in protocol["reused_pre_runs"] if r["model"] == "MaCo" and r["dataset"] == dataset]
        require(len(references) == 3, "Expected three frozen MaCo pre references")
        cache_paths, cache_hashes = set(), set()
        reference_evidence = []
        for reference in references:
            result_path = Path(reference["source"]) / "result.json"
            require(sha256(result_path) == reference["files_sha256"]["result.json"], "MaCo pre reference changed")
            result = read(result_path)
            cache_paths.add(result["cache_path"])
            cache_hashes.add(result["cache_sha256"])
            reference_evidence.append(dict(path=str(result_path), sha256=reference["files_sha256"]["result.json"]))
        require(len(cache_paths) == len(cache_hashes) == 1, "MaCo pre references disagree on immutable cache")
        cache_path, cache_hash = next(iter(cache_paths)), next(iter(cache_hashes))
        require(sha256(cache_path) == cache_hash, "MaCo cache metadata changed")
        cache = read(cache_path)
        require(cache["status"] == "complete" and cache["model"] == "MaCo", "Invalid MaCo cache")
        require(cache["dataset"] == dataset, "MaCo cache dataset mismatch")
        require(cache["dataset_sha256"] == spec["sha256"] and cache["samples_sha256"] == spec["samples_sha256"],
                "MaCo cache samples/dataset hash mismatch")
        require(cache["class_names"] == spec["classes"] and cache["counts"] == spec["counts"], "MaCo cache class/count mismatch")
        files = {}
        for name, record in cache["files"].items():
            stat = Path(record["path"]).stat()
            require(stat.st_size == record["bytes"] and stat.st_mtime_ns == record["mtime_ns"],
                    "MaCo cache file bytes/mtime changed: " + name)
            files[name] = dict(record, stat_verified=True, sha256_recomputed=False)
        calculated = {}
        for method, filename in (("native_mean", "native_logits.npy"), ("text_guided", "aggregation_logits.npy")):
            record = cache["files"][filename]
            require(sha256(record["path"]) == record["sha256"], "MaCo cached logits hash changed: " + filename)
            files[filename]["sha256_recomputed"] = True
            logits = np.load(record["path"], mmap_mode="r", allow_pickle=False)
            require(logits.shape == (len(locked["rows"]), len(spec["classes"])), "MaCo cache logits shape mismatch")
            calculated[method] = recalculate(locked["expected"]["labels"], logits[locked["indices"]["test"]])
        control = dict(model="MaCo", dataset=dataset, trainable_parameters=0)
        for metric in METRICS:
            control["native_mean_" + metric + "_pct"] = 100 * calculated["native_mean"][metric]
            control["text_guided_" + metric + "_pct"] = 100 * calculated["text_guided"][metric]
            control["text_guided_minus_native_" + metric + "_pp"] = 100 * (calculated["text_guided"][metric] - calculated["native_mean"][metric])
        controls.append(control)
        expected = locked["expected"]
        membership = {"indices_sha256": hashlib.sha256(expected["indices"].astype("<i8").tobytes()).hexdigest(),
                      "labels_sha256": hashlib.sha256(expected["labels"].astype("<f4").tobytes()).hexdigest(),
                      "ids_sha256": hashlib.sha256(json.dumps(expected["ids"].tolist()).encode()).hexdigest(),
                      "groups_sha256": hashlib.sha256(json.dumps(expected["groups"].tolist()).encode()).hexdigest()}
        evidence.append(dict(dataset=dataset, cache_path=cache_path, cache_sha256=cache_hash,
                             hash_locked_pre_references=reference_evidence, files=files,
                             dataset_sha256=spec["sha256"], samples_sha256=spec["samples_sha256"],
                             test_count=len(expected["indices"]), test_membership=membership,
                             metrics_recomputed=calculated,
                             scope="All cache files checked by stored bytes/mtime; both score files additionally checked by SHA256. Test labels and row indices come from hash-locked sample rows. No fitting or test-based selection."))
    return controls, evidence


def average(rows, metric):
    values = [r[metric] for r in rows]
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None) if values else (None, None)


def summarize(rows, protocol):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["dataset"], row["method"]].append(row)
    summary = []
    cases = [(m, d, "text_guided", "pre_mlp", "post_mlp")
             for m in protocol["models"] for d in protocol["datasets"]]
    cases += [("MaCo", d, "native_mean", "native_mean_pre", "native_mean_post") for d in protocol["datasets"]]
    for model, dataset, pooling, pre_method, post_method in cases:
        pre, post = groups[model, dataset, pre_method], groups[model, dataset, post_method]
        pre_seeds, post_seeds = {r["seed"]: r for r in pre}, {r["seed"]: r for r in post}
        paired = sorted(pre_seeds.keys() & post_seeds.keys())
        row = dict(model=model, dataset=dataset, pooling=pooling, pre_method=pre_method, post_method=post_method,
                   pre_n=len(pre), post_n=len(post), paired_n=len(paired), complete=len(paired) == len(protocol["seeds"]))
        for metric in METRICS:
            key = metric + "_pct"
            row["pre_" + metric + "_mean_pct"], row["pre_" + metric + "_sd_pct"] = average(pre, key)
            row["post_" + metric + "_mean_pct"], row["post_" + metric + "_sd_pct"] = average(post, key)
            differences = [{key: pre_seeds[s][key] - post_seeds[s][key]} for s in paired]
            row["pre_minus_post_" + metric + "_mean_pp"], row["pre_minus_post_" + metric + "_sd_pp"] = average(differences, key)
        row["pre_epoch0_seeds"] = ";".join(str(r["seed"]) for r in pre if r["best_epoch"] == 0)
        row["post_epoch0_seeds"] = ";".join(str(r["seed"]) for r in post if r["best_epoch"] == 0)
        summary.append(row)
    return summary


def cell(row, prefix):
    if row[prefix + "_n"] != 3:
        return f"pending ({row[prefix + '_n']}/3)"
    return " / ".join(f"{row[prefix + '_' + m + '_mean_pct']:.2f}±{row[prefix + '_' + m + '_sd_pct']:.2f}" for m in METRICS)


def difference(row):
    if not row["complete"]:
        return "pending"
    return " / ".join(f"{row['pre_minus_post_' + m + '_mean_pp']:+.2f}" for m in METRICS)


def make_report(audit, rows, summary, protocol):
    lines = ["Adapter placement experiment", f"Status: {audit['status']}; audited {len(rows)}/{audit['expected_fits']} fits "
             f"({audit['completed_new_fits']}/45 new, {audit['completed_reused_fits']}/27 reused).",
             "Cells: macro AUROC / macro AUPRC (average precision), percentages; mean ± sample SD over seeds 42,43,44.",
             "Each dataset is reported separately. No average across datasets. Δ = pre minus post, in percentage points.",
             "Partial cells are marked pending; incomplete averages are not used for conclusions.", "",
             "Text-guided/native disease-fusion placement comparison",
             "| Model | Dataset | Pre MLP | Post MLP | Δ pre−post |",
             "| --- | --- | --- | --- | --- |"]
    for row in summary:
        if row["pooling"] == "text_guided":
            lines.append(f"| {row['model']} | {NAMES[row['dataset']]} | {cell(row, 'pre')} | {cell(row, 'post')} | {difference(row)} |")
    lines += ["", "MaCo pooling × placement control",
              "| Dataset | Native mean: pre | Native mean: post | Text-guided: pre | Text-guided: post |",
              "| --- | --- | --- | --- | --- |"]
    for dataset in protocol["datasets"]:
        native = next(r for r in summary if r["model"] == "MaCo" and r["dataset"] == dataset and r["pooling"] == "native_mean")
        guided = next(r for r in summary if r["model"] == "MaCo" and r["dataset"] == dataset and r["pooling"] == "text_guided")
        lines.append(f"| {NAMES[dataset]} | {cell(native, 'pre')} | {cell(native, 'post')} | {cell(guided, 'pre')} | {cell(guided, 'post')} |")
    lines += ["", "MaCo no-MLP pooling controls (zero trainable parameters; one deterministic result, no seed SD)",
              "| Dataset | Original native mean | Added text-guided pooling only | Δ text-guided−native (pp) |",
              "| --- | --- | --- | --- |"]
    for control in audit["maco_pooling_controls"]:
        native = " / ".join(f"{control['native_mean_' + m + '_pct']:.2f}" for m in METRICS)
        guided = " / ".join(f"{control['text_guided_' + m + '_pct']:.2f}" for m in METRICS)
        delta = " / ".join(f"{control['text_guided_minus_native_' + m + '_pp']:+.4f}" for m in METRICS)
        lines.append(f"| {NAMES[control['dataset']]} | {native} | {guided} | {delta} |")
    lines += ["These frozen controls separate the change in pooling from the fitted MLP; the preceding 2×2 table compares MLP placement under each pooling rule."]
    lines += ["", "Epoch-zero selections (identity MLP):"]
    epoch0 = [r for r in rows if r["best_epoch"] == 0]
    lines += [f"- {r['model']} / {NAMES[r['dataset']]} / {r['method']} / seed{r['seed']}" for r in epoch0] or ["- None among audited fits."]
    lines += ["", "Results that do not support a universal pre-aggregation advantage:"]
    negative = []
    for row in summary:
        if not row["complete"]:
            continue
        for metric in METRICS:
            value = row["pre_minus_post_" + metric + "_mean_pp"]
            if value <= 0:
                negative.append(f"- {row['model']} / {NAMES[row['dataset']]} / {row['pooling']}: {metric}, pre−post {value:+.4f} pp.")
    lines += negative or ["- None among complete three-seed comparisons; this is not a significance test."]
    lines += ["", "Audit scope:", "- Recomputed both macro metrics from all available test prediction archives; verified IDs, indices, labels, groups, and class order against locked datasets.",
              "- Verified reused pre artifact hashes, train-only class weights, identical 99,136-parameter MLP shape, and strict earliest validation-AUROC checkpoint selection including epoch zero.",
              "- New fit initialization hashes match seed-reconstructed MLPs; logged epoch-order hashes match a replay of the persistent seeded DataLoader and agree across matching new fits.",
              "- Historical pre fits lack initialization/order hashes; their observed histories cannot provide retrospective hash equality. Source/protocol/seed matching and saved-checkpoint replay provide separate evidence.",
              "- Backbone freezing is supported by completed worker assertions and adapter-only checkpoints; this report does not reload every pretrained backbone.",
              "- MaCo no-MLP controls are recomputed from cache score arrays: cache metadata is hash-locked by preserved pre results, all file bytes/mtimes are checked, and both logit files additionally pass SHA256 verification.",
              "- Seed SD describes fitting variability; it is not a patient-level confidence interval or evidence of statistical significance.", "", "Protocol limitations:"]
    lines += ["- " + item for item in protocol["limitations"]]
    lines += ["- All placements use one fixed learning rate (1e-4), weight decay, and epoch/selection budget. No separate hyperparameter optimization was performed for each placement; the results establish performance under this shared protocol, not each placement's best achievable performance."]
    if audit["missing_fits"]:
        lines += ["", "Pending/incomplete fits:"] + ["- " + item for item in audit["missing_fits"]]
    if audit["errors"]:
        lines += ["", "AUDIT ERRORS (do not use these tables as final results):"] + ["- " + item for item in audit["errors"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).parent / "runs/20261119")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = args.run_dir.resolve()
    protocol = read(run / "PROTOCOL.json")
    protocol_hash = sha256(run / "PROTOCOL.json")
    datasets = load_datasets(protocol)
    entries = [dict(entry, reused=True) for entry in protocol["reused_pre_runs"]]
    entries += [dict(model=model, dataset=dataset, seed=seed, method=method, reused=False,
                     source=str(run / "results" / model / dataset / f"seed{seed}" / method))
                for model in protocol["models"] for dataset in protocol["datasets"]
                for seed in protocol["seeds"] for method in protocol["new_methods"][model]]
    rows, completed, checks, missing, errors = [], [], [], [], []
    for entry in entries:
        label = "/".join(str(entry[k]) for k in ("model", "dataset", "seed", "method"))
        if not (Path(entry["source"]) / "result.json").exists():
            missing.append(label)
            continue
        try:
            row, training, details = audit_fit(entry, protocol, protocol_hash, datasets, run)
            rows.append(row)
            completed.append((entry, training))
            checks.append(details)
        except Exception as exc:
            errors.append(label + ": " + str(exc))
    try:
        matching = matching_audit(completed, datasets)
    except Exception as exc:
        matching = {"status": "failed", "reason": str(exc)}
        errors.append("Matching audit: " + str(exc))
    try:
        pooling_controls, pooling_evidence = maco_pooling_controls(protocol, datasets)
    except Exception as exc:
        pooling_controls, pooling_evidence = [], []
        errors.append("MaCo no-MLP controls audit: " + str(exc))
    expected = protocol["new_training_runs"] + protocol["reused_training_runs"]
    require(len(entries) == expected == 72, "Unexpected protocol fit count")
    audit = dict(status="failed" if errors else "partial" if missing else "passed",
                 timestamp_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=protocol_hash,
                 report_source_sha256=sha256(__file__), expected_fits=expected, completed_fits=len(rows),
                 completed_new_fits=sum(not r["reused"] for r in rows), completed_reused_fits=sum(r["reused"] for r in rows),
                 missing_fits=missing, errors=errors, matching=matching, fit_checks=checks,
                 maco_pooling_controls=pooling_controls, maco_pooling_controls_audit=pooling_evidence,
                 sd="sample standard deviation (ddof=1) over fitting seeds; not a confidence interval",
                 aggregation="No cross-dataset averaging; pre-minus-post deltas are paired by seed.",
                 optimization_limitation="One fixed lr=1e-4, weight decay, and training/selection budget; no per-placement hyperparameter optimization.")
    summary = summarize(rows, protocol)
    write_csv(run / "per_seed.csv", rows)
    write_csv(run / "summary.csv", summary)
    atomic_text(run / "REPORT.txt", make_report(audit, rows, summary, protocol))
    atomic_text(run / "AUDIT.json", json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: audit[k] for k in ("status", "completed_fits", "expected_fits", "completed_new_fits", "completed_reused_fits", "errors")}, indent=2))
    if errors:
        return 1
    return 0 if not missing or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
