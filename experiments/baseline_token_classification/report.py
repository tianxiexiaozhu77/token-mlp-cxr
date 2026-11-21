"""Audit and summarize the prespecified baseline/token-MLP classification matrix.

Default reporting requires all 108 evaluations.  --allow-partial labels every
artifact as partial and computes seed summaries only from completed, audited
runs.  Prediction identities/labels and all reported metrics are checked again.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np

from common import MODELS, metrics, read_dataset, save, sha256


NATIVE_MODELS = {"CARZero", "KAD", "MedKLIP", "MAVL"}
METHODS = ("baseline", "aggregation_only", "token_mlp")
METRICS = ("macro_auroc", "macro_auprc", "micro_auprc")
LABELS = {"baseline": "Original native classifier",
          "aggregation_only": "Added aggregation only (no fitted parameters)",
          "token_mlp": "Shared residual Token MLP"}
ROW_FIELDS = ["model", "dataset", "aggregation_origin", "method", "method_label",
              "runs", "expected_runs", "complete", "seeds", "classes", "train_images",
              "val_images", "test_images", "trainable_parameters", "best_epochs"]
ROW_FIELDS += [field for key in METRICS for field in
               (key + "_mean", key + "_std", "delta_" + key + "_vs_baseline_pp",
                "delta_" + key + "_vs_aggregation_control_pp")]
CLASS_FIELDS = ["model", "dataset", "aggregation_origin", "method", "seed", "disease",
                "positives", "negatives", "auroc", "auprc"]


def same_metrics(actual, recorded, location):
    """Check every aggregate/per-class value, including exact field coverage."""
    if isinstance(actual, dict):
        if not isinstance(recorded, dict) or set(actual) != set(recorded):
            raise ValueError(f"Metric fields differ: {location}")
        for key, value in actual.items():
            same_metrics(value, recorded[key], f"{location}/{key}")
    elif actual is None:
        if recorded is not None:
            raise ValueError(f"Undefined metric differs: {location}")
    elif not isinstance(recorded, (int, float)) or not np.isfinite(recorded) or abs(actual - recorded) > 1e-12:
        raise ValueError(f"Metric differs: {location}: {actual!r} != {recorded!r}")


def exact(actual, expected, location):
    if actual != expected:
        raise ValueError(f"Provenance/identity differs: {location}")


def comparable_backend_metadata(model, value):
    """Ignore only BiomedCLIP's process-local function address in transform repr.

    Frozen native metadata records repr(Compose), whose _convert_to_rgb callable
    includes its memory address. This address changes across processes without
    changing preprocessing. Preserve all other fields/text and original records.
    """
    if model != "BiomedCLIP":
        return value
    result = dict(value)
    result["preprocess"] = re.sub(
        r"(<function _convert_to_rgb at )0x[0-9a-fA-F]+(>)",
        r"\1<process-local-address>\2", result["preprocess"])
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def gpu_index(record):
    return int(record["physical_index"] if isinstance(record, dict) else record)


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_best_epoch(initial_val, history, config):
    """Re-enact strict-improvement selection, including epoch0 and early stop."""
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Training history must contain consecutive epochs starting at 1")
    if len(history) > config["epochs"]:
        raise ValueError("Training exceeded the prespecified epoch budget")
    best, best_epoch, best_val, stale = initial_val["macro_auroc"], 0, initial_val, 0
    if not np.isfinite(best):
        raise ValueError("Nonfinite epoch0 validation metric")
    stop = False
    for i, row in enumerate(history):
        value = row["val"]["macro_auroc"]
        if not np.isfinite(value):
            raise ValueError("Nonfinite validation metric")
        if value > best:
            best, best_epoch, best_val, stale = value, row["epoch"], row["val"], 0
        else:
            stale += 1
        stop = row["epoch"] >= config["min_epochs"] and stale >= config["patience"]
        if stop and i != len(history) - 1:
            raise ValueError("Training continued after the prespecified early stopping condition")
    if len(history) < config["epochs"] and not stop:
        raise ValueError("Training ended before either max epochs or early stopping")
    return best_epoch, best_val


def verify_prediction(path, expected, classes):
    with np.load(path, allow_pickle=False) as data:
        exact(set(data.files), {"logits", "labels", "indices", "ids", "groups"}, str(path) + "/fields")
        for key, value in expected.items():
            np.testing.assert_array_equal(data[key], value, err_msg=f"{path}: {key}")
        logits = np.asarray(data["logits"]).copy()
    if logits.shape != expected["labels"].shape or not np.isfinite(logits).all():
        raise ValueError(f"Invalid logits: {path}")
    return logits, metrics(expected["labels"], logits, classes)


def audit_val_counts(value, labels, classes, location):
    exact(value["valid_classes"], len(classes), location + "/valid_classes")
    exact(value["total_classes"], len(classes), location + "/total_classes")
    exact(set(value["per_class"]), set(classes), location + "/classes")
    for key in METRICS:
        if not np.isfinite(value[key]) or not 0 <= value[key] <= 1:
            raise ValueError(f"Invalid validation metric: {location}/{key}")
    for i, name in enumerate(classes):
        row = value["per_class"][name]
        exact(row["positives"], int(labels[:, i].sum()), location + "/positives/" + name)
        exact(row["negatives"], int(len(labels) - labels[:, i].sum()), location + "/negatives/" + name)
        for key in ("auroc", "auprc"):
            if not isinstance(row[key], (int, float)) or not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid per-class validation metric: {location}/{name}/{key}")
    for aggregate, per_class in (("macro_auroc", "auroc"), ("macro_auprc", "auprc")):
        calculated = float(np.mean([value["per_class"][name][per_class] for name in classes]))
        same_metrics(calculated, value[aggregate], location + "/" + aggregate)


def fmt(row, key):
    mean, sd = row[key + "_mean"], row[key + "_std"]
    return f"{mean:.2f}" if sd is None else f"{mean:.2f} +/- {sd:.2f}"


def plot_results(root, rows, status, completed, expected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    panels = (
        ("openi", "OpenI18\n(18 classes)", tuple(m for m in MODELS if m not in ("MedKLIP", "MAVL"))),
        ("openi_common16", "OpenI common16\n(16 classes)", ("MedKLIP", "MAVL")),
        ("chexpert", "CheXpert\n(12 classes)", MODELS),
        ("chestxray14", "ChestXray14\n(14 classes)", MODELS),
    )
    figure, axes = plt.subplots(2, 4, figsize=(21.4, 9.3), sharey="row",
                                gridspec_kw={"width_ratios": (6, 2.8, 8, 8)})
    colors = {"baseline": "#64748b", "aggregation_only": "#e5a445", "token_mlp": "#267eae"}
    lookup = {(r["model"], r["dataset"], r["method"]): r for r in rows}
    for col, (dataset, title, panel_models) in enumerate(panels):
        for row_i, metric in enumerate(("macro_auroc", "macro_auprc")):
            axis = axes[row_i, col]
            axis.set_title(title if row_i == 0 else "", fontsize=11)
            for method_i, method in enumerate(METHODS):
                for model_i, model in enumerate(panel_models):
                    result = lookup.get((model, dataset, method))
                    if result is None:
                        continue
                    x = model_i + (method_i - 1) * .25
                    axis.bar(x, result[metric + "_mean"], width=.23, color=colors[method],
                             yerr=result[metric + "_std"] or 0, capsize=2,
                             hatch=None if result["complete"] else "//", linewidth=.7)
            labels = [m + (" +" if m not in NATIVE_MODELS else "") for m in panel_models]
            common16_panel = dataset == "openi_common16"
            axis.set_xticks(np.arange(len(panel_models)), labels,
                           rotation=0 if common16_panel else 40,
                           ha="center" if common16_panel else "right", fontsize=9)
            axis.set_ylim(0, 100)
            axis.set_xlim(-.6, len(panel_models)-.4)
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=.22)
            if col == 0:
                axis.set_ylabel("Macro AUROC (%)" if row_i == 0 else "Macro average precision (%)")
    legend = [Patch(facecolor=colors[m], label={"baseline": "Original native classifier",
              "aggregation_only": "Added aggregation only", "token_mlp": "Shared Token MLP"}[m]) for m in METHODS]
    if status != "complete":
        legend.append(Patch(facecolor="white", edgecolor="gray", hatch="//", label="Incomplete seed set"))
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .945), ncol=len(legend), frameon=False)
    figure.suptitle(f"Baseline models: token adaptation before disease-conditioned aggregation | {status.upper()} {completed}/{expected}", fontsize=13)
    figure.text(.5, .027,
                "Fixed custom splits; error bars: sample SD across training seeds. OpenI18 and OpenI common16 are separate tasks; their macro metrics are not directly comparable.\n"
                "+ Added aggregation: Token MLP is compared with both the native classifier and aggregation-only control. Missing bars are unavailable.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .073, 1, .90))
    figure.savefig(root / "comparison.png", dpi=180)
    figure.savefig(root / "comparison.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol_path = root / "PROTOCOL.json"
    protocol = read_json(protocol_path)
    exact(protocol["models"], list(MODELS), "protocol/models")
    exact(protocol["seeds"], [42, 43, 44], "protocol/seeds")
    exact(protocol["allowed_physical_gpus"], [4, 5, 6, 7], "protocol/gpus")
    cfg = protocol["train"]
    for model in MODELS:
        wanted = ["chexpert", "openi_common16" if model in ("MedKLIP", "MAVL") else "openi", "chestxray14"]
        exact(protocol["model_datasets"][model], wanted, "protocol/datasets/" + model)

    datasets, dataset_audits = {}, []
    for name, spec in protocol["datasets"].items():
        path = Path(spec["path"])
        exact(sha256(path), spec["sha256"], "dataset metadata/" + name)
        meta, samples = read_dataset(path)
        exact(meta["samples_sha256"], spec["samples_sha256"], "sample checksum/" + name)
        exact(meta["class_names"], spec["classes"], "class order/" + name)
        counts = dict(Counter(r["split"] for r in samples))
        exact(counts, spec["counts"], "split counts/" + name)
        labels = np.asarray([r["labels"] for r in samples], dtype=np.float32)
        if labels.shape != (len(samples), len(meta["class_names"])) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Nonbinary or malformed labels: " + name)
        split_indices = {split: np.asarray([i for i, r in enumerate(samples) if r["split"] == split], dtype=np.int64)
                         for split in ("train", "val", "test")}
        if set(counts) != set(split_indices):
            raise ValueError("Unexpected/missing split: " + name)
        for split, indices in split_indices.items():
            positives = labels[indices].sum(0)
            if not ((positives > 0) & (positives < len(indices))).all():
                raise ValueError("Some classes are undefined in " + name + "/" + split)
        test_ids = split_indices["test"]
        expected = {"labels": labels[test_ids], "indices": test_ids,
                    "ids": np.asarray([samples[i]["id"] for i in test_ids]),
                    "groups": np.asarray([samples[i]["group_id"] for i in test_ids])}
        datasets[name] = dict(meta=meta, samples=samples, labels=labels, indices=split_indices,
                              counts=counts, expected=expected, spec=spec)
        dataset_audits.append(dict(dataset=name, path=str(path), dataset_sha256=spec["sha256"],
            samples_sha256=spec["samples_sha256"], class_names=meta["class_names"], counts=counts,
            split_positive_counts={s: labels[idx].sum(0).astype(int).tolist() for s, idx in split_indices.items()}))
    common, original = datasets["openi_common16"], datasets["openi"]
    exact(common["meta"]["excluded_classes"], ["Calcified Granuloma", "Granuloma"], "common16 exclusions")
    columns = common["meta"]["parent_column_indices"]
    exact(len(columns), 16, "common16 class count")
    exact(common["meta"]["parent_sha256"], original["spec"]["sha256"], "common16 parent checksum")
    exact(common["meta"]["class_names"], [original["meta"]["class_names"][i] for i in columns], "common16 class mapping")
    exact(len(common["samples"]), len(original["samples"]), "common16 images")
    for reduced, full in zip(common["samples"], original["samples"]):
        exact({k:v for k,v in reduced.items() if k != "labels"},
              {k:v for k,v in full.items() if k != "labels"}, "common16 image/split identity")
        exact(reduced["labels"], [full["labels"][i] for i in columns], "common16 unchanged labels")
    image_audit = None
    image_audit_path = root / "image_audit.json"
    if image_audit_path.is_file():
        record = read_json(image_audit_path)
        exact(record["status"], "complete", "image-file audit status")
        actual_images = {r["image"] for ds in datasets.values() for r in ds["samples"]}
        audited_images = {entry["path"] for entry in record["files"]}
        exact(audited_images, actual_images, "image-file audit coverage")
        exact(record["images"], len(actual_images), "image-file audit count")
        exact(len(record["files"]), len(actual_images), "image-file audit uniqueness")
        image_audit = dict(path=str(image_audit_path), sha256=sha256(image_audit_path),
                           status=record["status"], images=record["images"],
                           note="Full image checksums were verified by the separate image audit; this report checks its coverage/checksum.")

    expected_runs = []
    for model in MODELS:
        origin = "native" if model in NATIVE_MODELS else "added"
        for dataset in protocol["model_datasets"][model]:
            for method in METHODS:
                if method == "aggregation_only" and origin == "native":
                    continue
                for seed in protocol["seeds"] if method == "token_mlp" else (42,):
                    folder = root / "results" / model / dataset / f"seed{seed}" / method
                    expected_runs.append((model, dataset, origin, method, seed, folder))
    exact(len(expected_runs), 108, "expected evaluation count")
    expected_paths = {folder / "result.json" for *_, folder in expected_runs}
    extra = set((root / "results").glob("*/*/seed*/*/result.json")) - expected_paths
    if extra:
        raise ValueError("Unexpected result paths outside the protocol: " + str(sorted(map(str, extra))))
    missing, ready = [], []
    for record in expected_runs:
        model, dataset, origin, method, seed, folder = record
        needed = ["result.json", "test_predictions.npz"]
        if method == "token_mlp":
            needed += ["training.json", "adapter.pt", "checkpoint_validation.json"]
        absent = [name for name in needed if not (folder / name).is_file()]
        if not absent and method == "token_mlp" and read_json(folder / "training.json").get("status") != "complete":
            absent = ["training.json status is not complete"]
        if absent:
            missing.append(dict(model=model, dataset=dataset, method=method, seed=seed,
                                folder=str(folder), missing=absent))
        else:
            ready.append(record)
    if missing and not args.allow_partial:
        raise FileNotFoundError(f"Missing/incomplete {len(missing)} of 108 evaluations; no report written. "
                                "Use --allow-partial for an explicitly partial monitoring report.\n" +
                                "\n".join(item["folder"] for item in missing))
    status = "partial" if missing else "complete"
    if status == "partial" and (root / "summary.json").is_file() and read_json(root / "summary.json").get("status") == "complete":
        raise ValueError("Refusing to overwrite a complete report with a partial report")

    cache_records, cache_audits = {}, []
    runs, per_class, versions, checkpoint_files = [], [], defaultdict(set), {}
    for model, dataset, origin, method, seed, folder in ready:
        ds = datasets[dataset]
        meta, labels, ids = ds["meta"], ds["labels"], ds["indices"]
        classes = meta["class_names"]
        if (model, dataset) not in cache_records:
            path = root / "cache" / model / dataset / "cache.json"
            cache = read_json(path)
            exact(cache["status"], "complete", str(path) + "/status")
            for key, value in (("model", model), ("dataset", dataset), ("class_names", classes),
                               ("dataset_sha256", ds["spec"]["sha256"]), ("samples_sha256", meta["samples_sha256"]),
                               ("counts", ds["counts"])):
                exact(cache[key], value, str(path) + "/" + key)
            exact(Path(cache["dataset_path"]).resolve(), Path(ds["spec"]["path"]).resolve(), "cache dataset path")
            exact(cache["backend_metadata"]["aggregation_origin"], origin, "native/added aggregation")
            if gpu_index(cache["gpu"]) not in protocol["allowed_physical_gpus"]:
                raise ValueError("Cache used a GPU outside authorized 4--7")
            arrays = {}
            for name in ("tokens.npy", "native_logits.npy", "aggregation_logits.npy"):
                spec = cache["files"][name]
                stat = Path(spec["path"]).stat()
                exact((stat.st_size, stat.st_mtime_ns), (spec["bytes"], spec["mtime_ns"]), "cache file metadata/" + name)
                arrays[name] = np.load(spec["path"], mmap_mode="r", allow_pickle=False)
            exact(list(arrays["tokens.npy"].shape), cache["token_shape"], "cached token shape")
            exact(cache["token_shape"][0], len(labels), "cached sample count")
            for name in ("native_logits.npy", "aggregation_logits.npy"):
                exact(arrays[name].shape, labels.shape, "cached prediction shape/" + name)
                if not np.isfinite(arrays[name]).all():
                    raise ValueError("Nonfinite cached predictions")
            if origin == "native":
                np.testing.assert_allclose(arrays["native_logits.npy"], arrays["aggregation_logits.npy"], atol=3e-4, rtol=3e-4)
            smoke = read_json(path.parent / "smoke.json")
            exact(smoke["status"], "passed", "cache smoke status")
            exact(smoke["class_names"], classes, "cache smoke class order")
            exact(smoke["frozen_parameters_with_gradients"], 0, "cache smoke frozen gradients")
            for source, digest in cache["source_sha256"].items():
                if "/baselines/" in source:
                    exact(sha256(source), digest, "cached baseline/author source/" + source)
            cache_records[model, dataset] = (cache, arrays, sha256(path))
            cache_audits.append(dict(model=model, dataset=dataset, aggregation_origin=origin,
                token_shape=cache["token_shape"], query_sha256=cache["query_sha256"],
                checkpoint_sha256=cache["checkpoint_sha256"], cache_sha256=sha256(path),
                smoke_sha256=sha256(path.parent / "smoke.json"), backend_metadata=cache["backend_metadata"]))
        cache, arrays, cache_hash = cache_records[model, dataset]
        data = read_json(folder / "result.json")
        for key, value in (("status", "complete"), ("model", model), ("dataset", dataset), ("method", method),
                           ("seed", seed), ("class_names", classes), ("counts", ds["counts"]),
                           ("dataset_sha256", ds["spec"]["sha256"]), ("samples_sha256", meta["samples_sha256"]),
                           ("cache_sha256", cache_hash), ("query_sha256", cache["query_sha256"]),
                           ("checkpoint_sha256", cache["checkpoint_sha256"]), ("config", cfg)):
            exact(data[key], value, str(folder) + "/" + key)
        exact(Path(data["dataset_path"]).resolve(), Path(ds["spec"]["path"]).resolve(), "result dataset path")
        exact(Path(data["cache_path"]).resolve(), root / "cache" / model / dataset / "cache.json", "result cache path")
        exact(data["backend_metadata"]["aggregation_origin"], origin, "result aggregation origin")
        exact(comparable_backend_metadata(model, data["backend_metadata"]),
              comparable_backend_metadata(model, cache["backend_metadata"]),
              "result/cache backend metadata")
        if gpu_index(data["gpu"]) not in protocol["allowed_physical_gpus"]:
            raise ValueError("Result used a GPU outside authorized 4--7")
        for path, digest in data["source_sha256"].items():
            versions[path].add(digest)
            if "/baselines/" in path:
                exact(sha256(path), digest, "current baseline source/" + path)
                if path in cache["source_sha256"]:
                    exact(digest, cache["source_sha256"][path], "cache/result baseline source/" + path)
        for path, digest in cache["checkpoint_sha256"].items():
            if path in checkpoint_files:
                exact(checkpoint_files[path], digest, "shared released checkpoint/" + path)
            else:
                exact(sha256(path), digest, "released checkpoint/" + path)
                checkpoint_files[path] = digest

        logits, audited = verify_prediction(folder / "test_predictions.npz", ds["expected"], classes)
        same_metrics(audited, data["test"], str(folder) + "/test")
        artifact_files = ["result.json", "test_predictions.npz"]
        if method != "token_mlp":
            exact(data["trainable_parameters"], 0, "control parameters")
            exact(data["best_epoch"], 0, "control epoch")
            control = arrays["native_logits.npy" if method == "baseline" else "aggregation_logits.npy"]
            np.testing.assert_array_equal(logits, control[ids["test"]], err_msg="Control differs from cached native output")
            same_metrics(metrics(labels[ids["val"]], control[ids["val"]], classes), data["best_val"], "control validation")
        else:
            training = read_json(folder / "training.json")
            for key in ("model", "dataset", "method", "seed", "config", "class_names", "dataset_sha256", "samples_sha256",
                        "cache_sha256", "query_sha256", "checkpoint_sha256", "source_sha256", "backend_metadata", "best_epoch",
                        "trainable_parameters", "adapter_architecture", "microbatch_size", "initial_max_abs_error"):
                exact(training[key], data[key], str(folder) + "/training/" + key)
            exact(training["status"], "complete", "training status")
            exact(training["effective_batch_size"], cfg["effective_batch_size"], "effective batch size")
            if not 0 < training["microbatch_size"] <= cfg["effective_batch_size"]:
                raise ValueError("Invalid training microbatch size")
            train_y = labels[ids["train"]]
            weights = np.minimum((len(train_y)-train_y.sum(0)) / np.maximum(train_y.sum(0), 1), 20.)
            np.testing.assert_allclose(training["pos_weight"], weights, atol=1e-7, rtol=1e-7)
            initial_val = training["initial_val"]
            same_metrics(initial_val, data["initial_val"], "result/training initial validation")
            audit_val_counts(initial_val, labels[ids["val"]], classes, "epoch0 validation")
            for item in training["history"]:
                if not np.isfinite(item["train_bce"]) or item["train_bce"] < 0:
                    raise ValueError("Invalid training BCE")
                audit_val_counts(item["val"], labels[ids["val"]], classes, f"epoch{item['epoch']} validation")
            selected_epoch, selected_val = first_best_epoch(initial_val, training["history"], cfg)
            exact(data["best_epoch"], selected_epoch, "earliest-best validation epoch including epoch0")
            same_metrics(selected_val, data["best_val"], "selected result validation")
            same_metrics(selected_val, training["best_val"], "selected training validation")
            same_metrics(audited, training["test"], "training test metrics")
            dim, bottleneck = cache["token_shape"][-1], protocol["bottleneck"]
            architecture = dict(dim=dim, bottleneck=bottleneck, activation="GELU", linear_bias=True, zero_initialized_up=True)
            exact(data["adapter_architecture"], architecture, "adapter architecture")
            parameter_count = 2 * dim * bottleneck + bottleneck + dim
            exact(data["trainable_parameters"], parameter_count, "adapter parameter count")
            import torch
            payload = torch.load(folder / "adapter.pt", map_location="cpu", weights_only=True)
            for key, value in (("model", model), ("method", method), ("class_names", classes),
                               ("dim", dim), ("bottleneck", bottleneck), ("best_epoch", selected_epoch)):
                exact(payload[key], value, "checkpoint/" + key)
            for key in ("model", "dataset", "seed", "class_names", "dataset_sha256", "samples_sha256", "cache_sha256",
                        "query_sha256", "checkpoint_sha256", "source_sha256", "config"):
                exact(payload["provenance"][key], data[key], "checkpoint provenance/" + key)
            state = payload["adapter_state_dict"]
            shapes = {"down.weight": (bottleneck, dim), "down.bias": (bottleneck,),
                      "up.weight": (dim, bottleneck), "up.bias": (dim,)}
            exact(set(state), set(shapes), "checkpoint parameter names")
            for key, shape in shapes.items():
                exact(tuple(state[key].shape), shape, "checkpoint parameter shape/" + key)
                if not torch.isfinite(state[key]).all():
                    raise ValueError("Nonfinite saved adapter weights")
            if selected_epoch == 0 and (torch.count_nonzero(state["up.weight"]) or torch.count_nonzero(state["up.bias"])):
                raise ValueError("Epoch0 checkpoint lost zero-initialized up projection")
            validation = read_json(folder / "checkpoint_validation.json")
            exact(validation, data["checkpoint_validation"], "checkpoint validation copy")
            exact(validation["status"], "passed", "checkpoint reload status")
            locations = [0, len(logits)//2, len(logits)-1]
            exact(validation["test_row_locations"], locations, "checkpoint reload positions")
            exact(validation["indices"], ids["test"][locations].tolist(), "checkpoint reload indices")
            if not np.isfinite(validation["max_abs_error"]) or validation["max_abs_error"] < 0:
                raise ValueError("Invalid checkpoint reload numerical error")
            artifact_files += ["training.json", "adapter.pt", "checkpoint_validation.json"]
        run = dict(model=model, dataset=dataset, aggregation_origin=origin, method=method, seed=seed,
                   folder=str(folder), best_epoch=data["best_epoch"], trainable_parameters=data["trainable_parameters"],
                   gpu=data["gpu"], test=audited, result_sha256=sha256(folder / "result.json"),
                   artifacts_sha256={name: sha256(folder / name) for name in artifact_files})
        runs.append(run)
        for name, value in audited["per_class"].items():
            per_class.append(dict(model=model, dataset=dataset, aggregation_origin=origin, method=method,
                                  seed=seed, disease=name, **value))

    rows = []
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["model"], run["dataset"], run["method"]].append(run)
    for model in MODELS:
        for dataset in protocol["model_datasets"][model]:
            cell_rows = []
            for method in METHODS:
                values = grouped.get((model, dataset, method), [])
                if not values:
                    continue
                values.sort(key=lambda v: v["seed"])
                target = 3 if method == "token_mlp" else 1
                counts = datasets[dataset]["counts"]
                row = dict(model=model, dataset=dataset, aggregation_origin=values[0]["aggregation_origin"],
                           method=method, method_label=LABELS[method], runs=len(values), expected_runs=target,
                           complete=len(values) == target, seeds="/".join(str(v["seed"]) for v in values),
                           classes=len(datasets[dataset]["meta"]["class_names"]),
                           train_images=counts["train"], val_images=counts["val"], test_images=counts["test"],
                           trainable_parameters=values[0]["trainable_parameters"],
                           best_epochs="/".join(str(v["best_epoch"]) for v in values))
                for key in METRICS:
                    scores = [v["test"][key] * 100 for v in values]
                    row[key + "_mean"] = float(np.mean(scores))
                    row[key + "_std"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
                cell_rows.append(row)
            controls = {r["method"]: r for r in cell_rows}
            aggregation_control = "baseline" if model in NATIVE_MODELS else "aggregation_only"
            for row in cell_rows:
                for key in METRICS:
                    for method, suffix in (("baseline", "baseline"), (aggregation_control, "aggregation_control")):
                        row[f"delta_{key}_vs_{suffix}_pp"] = (row[key+"_mean"] - controls[method][key+"_mean"]
                                                              if method in controls else None)
                rows.append(row)
    if status == "complete":
        exact(Counter(r["method"] for r in runs), Counter(baseline=24, aggregation_only=12, token_mlp=72), "completed method counts")
        exact(len(rows), 60, "complete summary row count")
    completed_method_counts = dict(Counter(r["method"] for r in runs))
    summary = dict(status=status, run_dir=str(root), protocol_sha256=sha256(protocol_path),
        evaluations=len(runs), expected_evaluations=len(expected_runs),
        completed_method_counts=completed_method_counts,
        expected_method_counts=dict(baseline=24, aggregation_only=12, token_mlp=72),
        rows=rows, runs=runs, missing=missing, dataset_audits=dataset_audits, cache_audits=cache_audits,
        image_audit=image_audit,
        source_versions={path: sorted(digests) for path, digests in sorted(versions.items())},
        released_checkpoint_files_sha256=checkpoint_files,
        audited_test_indices_ids_groups_labels_match_metadata=True,
        audited_metrics_independently_recomputed=True,
        audited_training_selection_includes_epoch0_and_keeps_earliest_tie=True,
        backend_metadata_comparison="Exact fields and values; only BiomedCLIP preprocess repr's _convert_to_rgb process-local memory address is normalized. Original metadata records are unchanged.",
        audit_scope="Every completed result listed in runs; missing evaluations are not inferred or imputed.",
        cache_integrity="Cache JSON SHA256, data/query/checkpoint provenance, file sizes/mtime and NPY shapes checked. Token-file SHA256 recorded at creation, not re-read by this report.",
        allowed_physical_gpus=protocol["allowed_physical_gpus"],
        actual_physical_gpus=sorted({gpu_index(r["gpu"]) for r in runs}),
        metric_units="Summary/RESULTS.csv percentages; deltas percentage points; individual/per-class metrics fractions.",
        metric_definition="AUPRC denotes sklearn average_precision_score (stepwise average precision), not trapezoidal PR integration.",
        score_policy="Rank metrics use unsaturated native logits or binary logit margins. Floating-point native probabilities can saturate into ties, so old probability-based reports need not be numerically identical.",
        uncertainty="Sample SD (ddof=1) across all completed training seeds on one fixed split; no patient confidence intervals or significance claims.",
        controls="Native aggregation: original vs Token MLP. Added aggregation: original vs aggregation-only vs Token MLP; zero-init Token MLP equals aggregation-only.",
        native_baseline_protocols="BiomedCLIP and BioViL-T use the existing alpharad_binary paired-prompt classifiers, not the earlier positive-prompt diagnostic protocols.",
        scope_notes=protocol["scope_notes"])

    lines = [f"Baseline classification token MLP experiments: {status.upper()}",
             f"Audited completed evaluations: {len(runs)}/108; missing/incomplete: {len(missing)}.",
             "Expected: 24 original baselines + 12 added-aggregation controls + 72 trained evaluations (24 cells x 3 seeds).",
             f"Completed by method: {json.dumps(completed_method_counts, sort_keys=True)}", "",
             "Metrics below are percentages. +/- is sample SD over seeds42/43/44 on the same fixed data split.",
             "Deterministic original and aggregation-only controls are evaluated once. Partial seed sets are explicitly marked.",
             "AUPRC is sklearn average_precision_score, not trapezoidal PR area. Saved class-level values are fractions.",
             "The seed SD is not a patient confidence interval, split uncertainty estimate or significance test.",
             "Native aggregation models: CARZero, KAD, MedKLIP, MAVL. Their original disease-conditioned aggregation/readout stays frozen.",
             "Added aggregation models: BiomedCLIP, BioViL-T, CheXzero, MaCo. Fixed cosine attention (temperature1) is controlled separately.",
             "BiomedCLIP/BioViL-T originals use the existing alpharad_binary positive/negative prompt protocol, not the earlier positive-only diagnostic classifier.",
             "Added aggregation is not AlphaRAD FLaS. Zero-init Token MLP equals aggregation-only for these four models.",
             "Shared adapter: x + Linear_up(GELU(Linear_down(x))); hidden width64, biases in both layers, zero-initialized up projection.",
             "The adapter is shared over tokens/diseases, trained with labels; the visual/text backbone, projections and native readout are frozen.",
             "Feature dimensions differ by model, so adapter parameter budgets are not identical across models.",
             "OpenI18 is used for six models. MedKLIP/MAVL use the separately named OpenI common16 task.",
             "Common16 removes only Calcified Granuloma and Granuloma label columns; all images, identities, labels and splits are otherwise preserved.",
             "Common16 metrics cannot be directly ranked against full OpenI18 macros. The unsupported full task is not represented as completed.",
             "OpenI: existing report/duplicate-group split 5966/755/749; longitudinal patient independence unverified; unmentioned findings treated as zero.",
             "CheXpert: local original official-test cohort repartitioned by patient into 534/67/67, 12 selected classes including No Finding.",
             "ChestXray14: local original official-test cohort repartitioned by patient into 18140/2097/2196, 14 classes.",
             "These custom supervised adaptation splits are not official zero-shot benchmark results. CheXpert test is small; inspect rare-class counts.",
             "AdamW lr1e-4 wd1e-4; effective batch32 with documented microbatch accumulation; max20 epochs, min8, patience5, gradient clip1.",
             "Positive weights use train-only negatives/positives capped20. Validation macro AUROC selects earliest-best checkpoint, including epoch0.",
             "Adapters are fitted separately per dataset. Test labels are not used for checkpoint or hyperparameter selection.",
             "Rank metrics use native logits/logit margins: sigmoid or softmax is mathematically monotone but float32 probabilities may saturate into ties.",
             "Therefore some metrics can differ from previous probability-based native reports despite preserving the underlying classifier.",
             "Audit: saved test labels/IDs/groups/indices matched fixed metadata; all aggregate/per-class metrics recomputed; best checkpoint and provenance checked.",
             "Cache token-file contents are not re-hashed by this report; creation SHA256, immutable size/mtime and array shape provenance are retained.", "",
             "model,dataset,aggregation_origin,method,runs/expected,macro_AUROC,macro_AP,micro_AP,parameters,best_epochs"]
    for row in rows:
        lines.append(f"{row['model']},{row['dataset']},{row['aggregation_origin']},{row['method']},"
                     f"{row['runs']}/{row['expected_runs']},{fmt(row,'macro_auroc')},{fmt(row,'macro_auprc')},"
                     f"{fmt(row,'micro_auprc')},{row['trainable_parameters']},{row['best_epochs']}")
    lines += ["", "Token MLP mean deltas (percentage points):", "model,dataset,metric,vs_original,vs_aggregation_control"]
    for row in rows:
        if row["method"] == "token_mlp":
            for key in METRICS:
                values = [row[f"delta_{key}_vs_{suffix}_pp"] for suffix in ("baseline", "aggregation_control")]
                formatted = ["unavailable" if v is None else f"{v:+.2f}" for v in values]
                lines.append(f"{row['model']},{row['dataset']},{key},{','.join(formatted)}")
    if missing:
        lines += ["", "PARTIAL REPORT: missing/incomplete runs (no values imputed):"]
        lines += [f"{v['model']}/{v['dataset']}/seed{v['seed']}/{v['method']}: {', '.join(v['missing'])}" for v in missing]
    write_csv(root / "RESULTS.csv", rows, ROW_FIELDS)
    write_csv(root / "per_class_results.csv", per_class, CLASS_FIELDS)
    (root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    plot_results(root, rows, status, len(runs), len(expected_runs))
    summary["outputs_sha256"] = {name: sha256(root / name) for name in
                                 ("RESULTS.txt", "RESULTS.csv", "per_class_results.csv", "comparison.png", "comparison.pdf")}
    save(root / "summary.json", summary)
    print(json.dumps(dict(status=status, evaluations=len(runs), expected_evaluations=108,
                          completed_method_counts=completed_method_counts, summary=str(root / "summary.json")), indent=2))


if __name__ == "__main__":
    main()
