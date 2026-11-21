"""CPU audit of 252 fresh fits in the gating x loss factorial experiment.

Historical fits are audited separately and never fill a current control cell.
--allow-partial permits pending files only; failed arithmetic, provenance,
checkpoint, loss, initialization, selection or matching checks always fail.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
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
from experiments.both_adapter_classification.report import (
    audit_entry as audit_historical, expected_references, checkpoint_shapes,
    same, scaled_error_check, verify_runtime_validation, verify_sources, verify_val,
)
from experiments.gated_asl_classification.common import (
    MODELS, DIMS, METHODS, SEEDS, ASL_CONFIG, GATE_CONFIG, parameter_count,
)
from experiments.gated_asl_classification.migrate import verify_migration

DATASETS = ("openi", "chexpert", "chestxray14")
NAMES = {"openi": "OpenI", "chexpert": "CheXpert", "chestxray14": "ChestXray14"}
METRICS = ("macro_auroc", "macro_auprc")
SHARED_KEYS = ("down.weight", "down.bias", "up.weight", "up.bias")
FILES = ("result.json", "training.json", "adapter.pt", "validation_predictions.npz", "test_predictions.npz")
EFFECTS = {
    "gating_effect_bce": {"gated_bce": 1, "pre_bce": -1},
    "gating_effect_asl": {"gated_asl": 1, "pre_asl": -1},
    "asl_effect_plain": {"pre_asl": 1, "pre_bce": -1},
    "asl_effect_gated": {"gated_asl": 1, "gated_bce": -1},
    "interaction": {"gated_asl": 1, "pre_asl": -1, "gated_bce": -1, "pre_bce": 1},
}


def entry_key(entry):
    return tuple(entry[k] for k in ("model", "dataset", "seed", "method"))


def verify_fresh_sources(record, run, caches):
    """Verify the unchanged manifest using an explicitly audited relocation."""
    migration = caches["migration"]
    require((run / "migration.json").exists() == (migration is not None),
            "Migration presence changed during the report")
    if migration is None:
        return verify_sources(record, run, caches["sources"])
    require(sha256(run / "migration.json") == migration["evidence"]["manifest_sha256"],
            "Migration manifest changed during the report")
    manifest = caches["runtime_manifest"]
    require(sha256(run / "source_manifest.json") == caches["runtime_manifest_sha256"],
            "Frozen runtime manifest changed during the report")
    require(record["source_manifest_sha256"] == caches["runtime_manifest_sha256"]
            and record["protocol_sha256"] == manifest["protocol_sha256"]
            and record["sources_sha256"] == manifest["sources_sha256"],
            "Migrated/current fit is not bound to the unchanged runtime manifest")
    return dict(manifest_sha256=caches["runtime_manifest_sha256"],
        files=[dict(source=name, sha256=digest, verified_copy=migration["archives"][name])
               for name, digest in manifest["sources_sha256"].items()],
        archive_content_hashes_checked_once_per_report_process=True,
        migration=migration["evidence"])


def check_protocol(protocol):
    require(protocol["models"] == list(MODELS), "Unexpected models")
    require(protocol["methods"] == list(METHODS), "Unexpected fresh groups")
    require(protocol["seeds"] == list(SEEDS), "Unexpected seed set")
    require(set(protocol["datasets"]) == set(DATASETS), "Unexpected datasets")
    require(protocol["new_training_runs"] == 252 and protocol["all_methods_fresh"] is True
            and protocol["reused_training_runs"] == 0, "Fresh-fit accounting differs")
    require(protocol["placement"] == "pre" and protocol["bottleneck"] == 64, "Wrong placement/bottleneck")
    require(protocol["allowed_physical_gpus"] == [4, 5, 6, 7], "Unexpected authorized GPUs")
    require(protocol["asl_config"] == ASL_CONFIG, "Fixed ASL configuration differs")
    require(protocol["gate_config"] == GATE_CONFIG, "Fixed gate configuration differs")
    require(ASL_CONFIG == dict(gamma_neg=4., gamma_pos=0., clip=.05, eps=1e-8,
                             reduction="mean", detach_focal=False), "ASL constants changed")
    require(GATE_CONFIG == dict(input="nonaffine_layer_norm", layer_norm_eps=1e-5,
            output="per_token_scalar_sigmoid", shared_across_tokens=True, weight_init="zeros",
            initial_probability=.1), "Gate constants changed")
    for name, value in dict(optimizer="AdamW", lr=1e-4, weight_decay=1e-4,
            effective_batch_size=32, epochs=20, min_epochs=8, patience=5, grad_clip=1.).items():
        require(protocol["train"][name] == value, "Fixed training configuration differs: " + name)
    for model in MODELS:
        spec = protocol["model_specs"][model]
        require(spec["dim"] == DIMS[model] and spec["bottleneck"] == 64, "Model dimensions differ")
        for method in METHODS:
            require(spec["trainable_parameters_by_method"][method] == parameter_count(DIMS[model], 64, method),
                    "Frozen parameter budget differs")
    for method in METHODS:
        spec = protocol["method_specs"][method]
        require(spec["gated"] == method.startswith("gated_") and spec["loss"] ==
                ("asl" if method.endswith("_asl") else "weighted_bce") and spec["fresh_fit"] is True
                and spec["placement"] == "pre", "Factor identity differs")
    references = protocol["references"]
    require(len(references) == 120 and {entry_key(r) for r in references} == expected_references(),
            "Historical reference matrix differs")
    require(len({entry_key(r) for r in references}) == len(references), "Duplicate reference")
    parent = protocol["parent_protocol"]
    require(sha256(parent["path"]) == parent["sha256"], "Parent protocol changed")
    old = read(parent["path"])
    require(protocol["references"] == old["references"] and protocol["caches"] == old["caches"]
            and protocol["datasets"] == old["datasets"], "Historical comparison/data/cache provenance changed")


def load_data(protocol):
    datasets = load_datasets(protocol)
    for dataset in datasets.values():
        dataset["split_expected"] = {}
        for split, indices in dataset["indices"].items():
            rows = [dataset["rows"][i] for i in indices]
            dataset["split_expected"][split] = dict(indices=indices,
                labels=np.asarray([r["labels"] for r in rows], dtype=np.float32),
                ids=np.asarray([r["id"] for r in rows]), groups=np.asarray([r["group_id"] for r in rows]),
                class_names=np.asarray(dataset["spec"]["classes"]))
        labels = dataset["split_expected"]["train"]["labels"]
        dataset["weights"] = np.minimum((len(labels)-labels.sum(0))/np.maximum(labels.sum(0), 1), 20.)
    return datasets


def prediction_audit(folder, split, dataset):
    filename = "validation_predictions.npz" if split == "val" else "test_predictions.npz"
    expected = dataset["split_expected"][split]
    with np.load(folder / filename, allow_pickle=False) as prediction:
        require(set(prediction.files) == set(expected) | {"logits"}, split + " prediction fields differ")
        for name, value in expected.items():
            require(np.array_equal(prediction[name], value), split + " sample identity/labels differ: " + name)
        logits = prediction["logits"].copy()
    require(logits.shape == expected["labels"].shape and np.isfinite(logits).all(), split + " invalid predictions")
    return logits, metrics(expected["labels"], logits, dataset["spec"]["classes"])


def init_hashes(dim, seed, gated, cache):
    identity = dim, seed, gated
    if identity not in cache:
        torch.manual_seed(seed)
        state = residual_mlp(dim, 64).state_dict()
        if gated:
            state["gate.weight"] = torch.zeros((1, dim), dtype=torch.float32)
            state["gate.bias"] = torch.full((1,), math.log(.1 / .9), dtype=torch.float32)
        cache[identity] = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()}
    return cache[identity]


def token_count(protocol, model, dataset_name, cache):
    key = model, dataset_name
    if key not in cache:
        sizes = set()
        for spec in protocol["caches"][f"{model}/{dataset_name}"]:
            metadata = read(spec["path"])
            if "token_shards" in metadata:
                sizes.update(s["shape"][1] for s in metadata["token_shards"])
            else:
                sizes.add(metadata["token_shape"][1])
        require(len(sizes) == 1 and next(iter(sizes)) > 0, "Inconsistent token counts")
        cache[key] = next(iter(sizes))
    return cache[key]


def frozen_split_logits(protocol, model, dataset_name, dataset, cache):
    """Read only small frozen logit arrays; never load or rerun image tokens."""
    key = model, dataset_name
    if key not in cache:
        spec = dataset["spec"]
        rows, classes = len(dataset["rows"]), len(spec["classes"])
        records = []
        for item in protocol["caches"][f"{model}/{dataset_name}"]:
            require(sha256(item["path"]) == item["sha256"], "Frozen logit cache manifest changed")
            record = read(item["path"])
            require(record["dataset_sha256"] == spec["sha256"] and record["samples_sha256"] == spec["samples_sha256"],
                    "Frozen logit dataset differs")
            records.append(record)
        if model in ("AlphaRAD", "RadZero"):
            aggregation = np.empty((rows, classes), dtype=np.float32)
            seen = set()
            for record in records:
                indices = record["indices"]
                require(not seen.intersection(indices), "Frozen logit shard overlap")
                seen.update(indices)
                logits = np.load(record["logits_path"], mmap_mode="r")
                require(logits.shape == (len(indices), classes), "Frozen logit shard shape differs")
                aggregation[indices] = logits
            require(seen == set(range(rows)), "Frozen logit coverage differs")
        else:
            require(len(records) == 1, "Unexpected frozen logit shard count")
            logit_spec = records[0]["files"]["aggregation_logits.npy"]
            require(sha256(logit_spec["path"]) == logit_spec["sha256"], "Frozen aggregation logit digest changed")
            aggregation = np.load(logit_spec["path"], mmap_mode="r")
            require(aggregation.shape == (rows, classes), "Frozen aggregation logit shape differs")
        require(np.isfinite(aggregation).all(), "Nonfinite frozen aggregation logits")
        cache[key] = {}
        for split in ("val", "test"):
            selected = np.array(aggregation[dataset["indices"][split]], copy=True)
            selected.setflags(write=False)
            cache[key][split] = selected
    return cache[key]


def gate_statistics_audit(statistics, gated, count, context, initial=False):
    if not gated:
        require(statistics is None, context + ": plain adapter has gate statistics")
        return
    require(isinstance(statistics, dict), context + ": gate observations missing")
    require(statistics["count"] == count, context + ": observed token count differs")
    for name in ("mean", "std", "min", "max"):
        require(np.isfinite(statistics[name]), context + ": nonfinite gate statistic")
    lo, mean, hi, sd = (statistics[k] for k in ("min", "mean", "max", "std"))
    require(0 <= lo <= mean + 1e-12 and mean <= hi + 1e-12 <= 1 + 1e-12
            and 0 <= sd <= .5 + 1e-8, context + ": gate statistics out of range")
    require(sd * sd <= max(0., (hi-mean)*(mean-lo)) + 1e-8, context + ": impossible gate variance")
    require(statistics["std_definition"] == "population across observed token gates", context + ": wrong gate SD definition")
    require(statistics["observation"] == "sigmoid of actual gate linear outputs; no labels used",
            context + ": gate observation meaning differs")
    if initial:
        require(abs(mean - .1) < 1e-7 and abs(lo - .1) < 1e-7 and abs(hi - .1) < 1e-7 and sd < 1e-7,
                context + ": initial gate is not constant 0.1")


def audit_fresh(entry, protocol, protocol_hash, datasets, run, caches):
    folder = Path(entry["source"])
    model, dataset_name, seed, method = entry_key(entry)
    result, training = read(folder / "result.json"), read(folder / "training.json")
    require(result == {k: v for k, v in training.items() if k != "history"}, "Result differs from complete training record")
    spec, model_spec = datasets[dataset_name]["spec"], protocol["model_specs"][model]
    dataset = datasets[dataset_name]
    gated, dim = method.startswith("gated_"), model_spec["dim"]
    loss_name = "asl" if method.endswith("_asl") else "weighted_bce"
    gate_config = protocol["gate_config"] if gated else None
    count = parameter_count(dim, 64, method)
    require(result["status"] == "complete", "Fresh result is not complete")
    require(all(result[k] == entry[k] for k in ("model", "dataset", "seed", "method")), "Fresh run identity differs")
    require(result["protocol_sha256"] == protocol_hash and result["config"] == protocol["train"], "Fresh protocol/config differs")
    require(result["method_spec"] == protocol["method_specs"][method], "Method factor specification differs")
    require(result["fresh_fit"] is True and result["historical_reference_used_for_training"] is False,
            "Historical results used as a fresh control")
    require(result["adapter_position"] == "pre" and result["post_adapter"] is None, "Wrong insertion position")
    require(result["pooling"] == ("text_guided" if model_spec["aggregation_origin"] == "added" else "native"), "Pooling changed")
    require(result["position_description"] == model_spec["position"], "Insertion boundary differs")
    require(result["class_names"] == spec["classes"] and result["counts"] == spec["counts"], "Class/split counts differ")
    require(result["dataset_sha256"] == spec["sha256"] and result["samples_sha256"] == spec["samples_sha256"],
            "Dataset provenance differs")
    require(result["trainable_parameters"] == count == model_spec["trainable_parameters_by_method"][method],
            "Adapter parameter count differs")
    require(result["adapter_architecture"] == dict(dim=dim, bottleneck=64, activation="GELU", linear_bias=True,
            zero_initialized_up=True, copies=1, gated=gated, gate_config=gate_config), "Adapter architecture differs")
    require(result["gate_config"] == gate_config and result["loss_name"] == loss_name, "Gate/loss identity differs")
    weights = dataset["weights"]
    require(np.array_equal(np.asarray(result["train_pos_weight"], dtype=np.float32), weights), "TRAIN-derived diagnostic weights differ")
    expected_loss = (dict(name="weighted_bce", reduction="mean", pos_weight=weights.tolist(),
                         pos_weight_source="TRAIN negatives/positives capped at20") if loss_name == "weighted_bce"
                     else dict(name="asl", pos_weight=None, **protocol["asl_config"]))
    require(result["loss_config"] == expected_loss, "Loss configuration/positive weighting differs")
    if loss_name == "weighted_bce":
        require(np.array_equal(np.asarray(result["pos_weight"], dtype=np.float32), weights), "BCE pos_weight is not TRAIN-only")
    else:
        require(result["pos_weight"] is None, "ASL incorrectly applies positive-class weights")
    require(result["effective_batch_size"] == 32, "Effective batch budget differs")
    require(result["microbatch_size"] == (8 if model in ("CARZero", "KAD") else 32), "Model microbatch differs")
    require(result["test_selection"] is False, "TEST used for selection")
    for flag in ("frozen_backbone_verified", "matched_initialization_verified", "matched_overlapping_epoch_orders_verified"):
        require(result[flag] is True, "Missing training verification: " + flag)
    require(result["gpu"]["physical_index"] in protocol["allowed_physical_gpus"], "Unapproved physical GPU")

    digests = {name: sha256(folder / name) for name in FILES}
    for name, field in (("adapter.pt", "adapter_sha256"), ("validation_predictions.npz", "validation_predictions_sha256"),
                        ("test_predictions.npz", "test_predictions_sha256")):
        require(result[field] == digests[name], "Saved artifact digest differs: " + name)
    checkpoint = torch.load(folder / "adapter.pt", map_location="cpu", weights_only=True)
    expected_metadata = dict(model=model, method=method, dim=dim, bottleneck=64, class_names=spec["classes"],
        adapter_position="pre", gated=gated, gate_config=gate_config, loss_name=loss_name,
        loss_config=expected_loss, fresh_fit=True, protocol_sha256=protocol_hash)
    require(set(checkpoint) == set(expected_metadata) | {"best_epoch", "adapter_state_dict"}, "Checkpoint metadata fields differ")
    for name, value in expected_metadata.items():
        require(checkpoint[name] == value, "Checkpoint metadata differs: " + name)
    state = checkpoint["adapter_state_dict"]
    shapes = checkpoint_shapes(dim, False)
    if gated:
        shapes.update({"gate.weight": (1, dim), "gate.bias": (1,)})
    require(set(state) == set(shapes), "Unexpected checkpoint tensors")
    require(all(tuple(state[k].shape) == shape for k, shape in shapes.items()), "Checkpoint tensor shapes differ")
    require(all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in state.values()),
            "Invalid checkpoint dtype/value")
    require(sum(value.numel() for value in state.values()) == count, "Checkpoint parameter count differs")
    expected_init = init_hashes(dim, seed, gated, caches["initialization"])
    require(result["initialization_sha256"] == expected_init, "Seeded adapter initialization differs")
    shared_init = {k: expected_init[k] for k in SHARED_KEYS}
    require(result["shared_initialization_sha256"] == shared_init, "Shared four-group MLP initialization differs")
    epoch = selection_audit(training, result, checkpoint, protocol["train"])
    if epoch == 0:
        actual_init = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()}
        require(actual_init == expected_init, "Epoch-zero checkpoint differs from full initialized state")
    val_labels = dataset["split_expected"]["val"]["labels"]
    verify_val(result["initial_val"], val_labels, spec["classes"], "initial VAL")
    order_key = dataset_name, seed
    if order_key not in caches["orders"]:
        caches["orders"][order_key] = expected_order_hashes(dataset["indices"]["train"], seed, protocol["train"]["epochs"])
    tokens = token_count(protocol, model, dataset_name, caches["tokens"])
    gate_statistics_audit(result["initial_gate_statistics"], gated, spec["counts"]["val"] * tokens, "initial gates", initial=True)
    for record in training["history"]:
        require(np.isfinite(record["train_loss"]) and record["train_loss"] >= 0, "Invalid training loss")
        require(record["loss_name"] == loss_name, "Epoch loss identity differs")
        verify_val(record["val"], val_labels, spec["classes"], "epoch VAL")
        require(record["train_order_sha256"] == caches["orders"][order_key][record["epoch"] - 1], "Seeded epoch sample order differs")
        gate_statistics_audit(record["train_gate_statistics"], gated, spec["counts"]["train"] * tokens, "TRAIN gates")
        gate_statistics_audit(record["val_gate_statistics"], gated, spec["counts"]["val"] * tokens, "VAL gates")
    gate_statistics_audit(result["selected_val_gate_statistics"], gated, spec["counts"]["val"] * tokens, "selected VAL gates",
                          initial=epoch == 0)
    if gated:
        expected_gate = result["initial_gate_statistics"] if epoch == 0 else training["history"][epoch-1]["val_gate_statistics"]
        for field in ("mean", "std", "min", "max"):
            require(abs(expected_gate[field] - result["selected_val_gate_statistics"][field]) < 1e-7,
                    "Reloaded selected VAL gate statistics differ")

    gradient = result["gradient_evidence"]
    require(set(gradient) == ({"up", "down", "gate"} if gated else {"up", "down"}), "Gradient branch evidence differs")
    total_steps = len(training["history"]) * math.ceil(spec["counts"]["train"] / 32)
    for name, evidence in gradient.items():
        require(evidence["nonzero"] is True and isinstance(evidence["first_nonzero_step"], int)
                and 1 <= evidence["first_nonzero_step"] <= total_steps, "Missing nonzero gradient: " + name)
        require(np.isfinite(evidence["max_abs_grad"]) and evidence["max_abs_grad"] > 0
                and np.isfinite(evidence["first_step_max_abs_grad"])
                and 0 <= evidence["first_step_max_abs_grad"] <= evidence["max_abs_grad"], "Invalid gradient magnitude")
        if name in ("down", "gate"):
            require(evidence["first_step_max_abs_grad"] == 0 and evidence["first_nonzero_step"] > 1,
                    "Zero-up initialization did not block first-step down/gate gradient")
        elif evidence["first_step_max_abs_grad"] > 0:
            require(evidence["first_nonzero_step"] == 1, "First-step up-gradient evidence inconsistent")

    frozen_logits = frozen_split_logits(protocol, model, dataset_name, dataset, caches["frozen_logits"])
    initial_error_evidence = scaled_error_check(result["initial_max_abs_error"], frozen_logits["val"],
                                                "Initial identity adapter VAL parity")
    require(set(result["checkpoint_replay"]) == {"val", "test"}, "Missing VAL/TEST checkpoint replay")
    replay_evidence, recomputed, epoch_zero_predictions = {}, {}, {}
    for split in ("val", "test"):
        logits, values = prediction_audit(folder, split, dataset)
        recomputed[split] = values
        if split == "val":
            same(values, result["selected_val"], "recomputed selected VAL")
            same(values, result["best_val"], "selected checkpoint VAL")
        else:
            same(values, result["test"], "recomputed TEST")
        if epoch == 0:
            # Here both full arrays are saved, so independently check every
            # selected VAL/TEST score, not only a worker's scalar error bound.
            np.testing.assert_allclose(logits, frozen_logits[split], atol=3e-4, rtol=3e-4,
                                       err_msg=split + " epoch-zero predictions differ from frozen aggregation")
            epoch_zero_predictions[split] = dict(status="passed", rows=len(logits),
                max_abs_error=float(np.max(np.abs(logits-frozen_logits[split]))),
                atol=3e-4, rtol=3e-4, all_saved_elements_independently_compared=True,
                reference="frozen aggregation logits; added-pooling models are not compared with Original")
        locations = [0, len(logits)//2, len(logits)-1]
        replay = result["checkpoint_replay"][split]
        require(replay["row_locations"] == locations, split + " replay coverage differs")
        replay_evidence[split] = scaled_error_check(replay["max_abs_error"], logits[locations], split + " checkpoint replay")
    validation_key = model, dataset_name
    if validation_key not in caches["runtime"]:
        caches["runtime"][validation_key] = verify_runtime_validation(model, dataset_name, protocol, dataset, run)
    runtime_validation = caches["runtime"][validation_key]
    require(result["cache_provenance"] == runtime_validation["frozen_path"]["cache_provenance"], "Frozen path provenance differs")
    sources = verify_fresh_sources(result, run, caches)
    row = dict(model=model, dataset=dataset_name, seed=seed, method=method, fresh_fit=True,
        gated=gated, loss_name=loss_name, trainable_parameters=count, best_epoch=epoch,
        epochs_run=len(training["history"]), macro_auroc_pct=100*recomputed["test"]["macro_auroc"],
        macro_auprc_pct=100*recomputed["test"]["macro_auprc"],
        selected_val_macro_auroc_pct=100*recomputed["val"]["macro_auroc"],
        selected_val_macro_auprc_pct=100*recomputed["val"]["macro_auprc"],
        selected_val_gate_mean=result["selected_val_gate_statistics"]["mean"] if gated else None,
        selected_val_gate_std=result["selected_val_gate_statistics"]["std"] if gated else None,
        migrated_complete_fit=bool(caches["migration"] and
            str((folder / "result.json").relative_to(run)) in caches["migration"]["files_sha256"]),
        source=str(folder))
    details = dict(key=list(entry_key(entry)), source=str(folder), fresh_fit=True, files_sha256=digests,
        result_current_sha256=digests["result.json"], checkpoint_tensors=list(state), trainable_parameters=count,
        best_epoch=epoch, epochs_run=len(training["history"]), epoch_zero_full_state_verified=epoch == 0,
        initial_val_identity_error_check=initial_error_evidence,
        epoch_zero_prediction_checks=epoch_zero_predictions,
        recomputed_val=recomputed["val"], recomputed_test=recomputed["test"],
        loss_name=loss_name, loss_config=expected_loss, gradient_evidence=gradient,
        frozen_backbone_verified=True, checkpoint_replay=replay_evidence,
        shared_initialization_reconstructed=True, full_initialization_reconstructed=True,
        training_orders_reconstructed=True, gate_statistics_validated=gated,
        source_verification=sources, runtime_validation=runtime_validation)
    return row, details, training


def matching_audit(completed, protocol, run):
    """Independent pair matching and saved worker summary verification."""
    grouped = defaultdict(list)
    for entry, training in completed:
        grouped[(entry["model"], entry["dataset"])].append((entry, training))
    evidence, pending, errors = [], [], []
    for (model, dataset), fits in grouped.items():
        try:
            observed = {str(seed): dict(initialization=None, epoch_orders={}) for seed in SEEDS}
            identities = set()
            for entry, training in fits:
                identities.add((entry["seed"], entry["method"]))
                current = observed[str(entry["seed"])]
                initialization = training["shared_initialization_sha256"]
                if current["initialization"] is None:
                    current["initialization"] = initialization
                require(current["initialization"] == initialization, "Same-seed MLP tensors differ across fresh groups")
                for epoch in training["history"]:
                    epoch_key = str(epoch["epoch"])
                    previous = current["epoch_orders"].setdefault(epoch_key, epoch["train_order_sha256"])
                    require(previous == epoch["train_order_sha256"], "Same-seed overlapping epoch order differs")
            all_fits = len(identities) == 12
            path = run / "validation" / model / dataset / "matched_four_groups.json"
            group_evidence = dict(model=model, dataset=dataset, completed_fresh_fits=len(identities),
                                  expected_fresh_fits=12, observed_matching=observed)
            if all_fits:
                if not path.is_file():
                    pending.append(dict(kind="group_matching_evidence", model=model, dataset=dataset, source=str(path)))
                    group_evidence["saved_worker_summary"] = "pending"
                else:
                    saved = read(path)
                    require(saved["status"] == "passed" and saved["methods"] == list(METHODS)
                            and saved["seeds"] == list(SEEDS), "Saved matching matrix identity differs")
                    require(saved["shared_initialization_keys"] == list(SHARED_KEYS), "Saved shared state keys differ")
                    require(saved["matching"] == observed, "Saved matching evidence differs from actual fit records")
                    group_evidence.update(saved_worker_summary="passed", path=str(path), sha256=sha256(path))
            else:
                group_evidence["saved_worker_summary"] = "group incomplete; completed fits independently checked"
            evidence.append(group_evidence)
        except Exception as error:
            errors.append(dict(model=model, dataset=dataset, scope="cross_group_matching", error=f"{type(error).__name__}: {error}"))
    return evidence, pending, errors


def stats(values):
    return (float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None)


def summarize(rows, historical, protocol):
    lookup = defaultdict(dict)
    for row in rows:
        lookup[(row["model"], row["dataset"], row["method"])][row["seed"]] = row
    old = defaultdict(dict)
    for row in historical:
        old[(row["model"], row["dataset"], row["method"])][row["seed"]] = row
    summaries, historical_comparisons, effects, paired_effects = [], [], [], []
    for dataset in DATASETS:
        for model in MODELS:
            row = dict(model=model, dataset=dataset, classes=len(protocol["datasets"][dataset]["classes"]),
                       test_images=protocol["datasets"][dataset]["counts"]["test"])
            original = old[(model, dataset, "original")].get(42)
            row["original_runs"] = 1 if original else 0
            row["original_expected_runs"] = 1
            row["original_status"] = "complete" if original else "partial"
            row["original_seeds"] = "42" if original else ""
            for metric in METRICS:
                row["original_" + metric + "_pct"] = original[metric + "_pct"] if original else None
                row["original_" + metric + "_mean_pct"] = original[metric + "_pct"] if original else None
                row["original_" + metric + "_sd_pct"] = None
            for method in METHODS:
                fits = lookup[(model, dataset, method)]
                row[method + "_runs"] = len(fits)
                row[method + "_expected_runs"] = 3
                row[method + "_status"] = "complete" if len(fits) == 3 else "partial"
                row[method + "_seeds"] = ",".join(str(s) for s in sorted(fits))
                row[method + "_best_epochs"] = ",".join(str(fits[s]["best_epoch"]) for s in sorted(fits))
                row[method + "_epoch_zero_count"] = sum(f["best_epoch"] == 0 for f in fits.values())
                row[method + "_trainable_parameters"] = protocol["model_specs"][model]["trainable_parameters_by_method"][method]
                for metric in METRICS:
                    mean, sd = stats([r[metric + "_pct"] for r in fits.values()])
                    row[method + "_" + metric + "_mean_pct"] = mean
                    row[method + "_" + metric + "_sd_pct"] = sd
                paired = sorted(set(fits) & set(lookup[(model, dataset, "pre_bce")]))
                row[method + "_minus_pre_bce_paired_runs"] = len(paired)
                for metric in METRICS:
                    differences = [fits[s][metric + "_pct"] - lookup[(model, dataset, "pre_bce")][s][metric + "_pct"] for s in paired]
                    mean, sd = stats(differences)
                    row[method + "_minus_pre_bce_" + metric + "_mean_pp"] = mean
                    row[method + "_minus_pre_bce_" + metric + "_sd_pp"] = sd
            summaries.append(row)
            current, previous = lookup[(model, dataset, "pre_bce")], old[(model, dataset, "pre_mlp")]
            pairs = sorted(set(current) & set(previous))
            comparison = dict(model=model, dataset=dataset, current_method="fresh pre_bce", historical_method="historical pre_mlp",
                              paired_runs=len(pairs), paired_seeds=",".join(map(str, pairs)),
                              status="complete" if len(pairs) == 3 else "partial")
            for metric in METRICS:
                for prefix, source in (("fresh_pre_bce", current), ("historical_pre_mlp", previous)):
                    mean, sd = stats([r[metric + "_pct"] for r in source.values()])
                    comparison[prefix + "_" + metric + "_mean_pct"] = mean
                    comparison[prefix + "_" + metric + "_sd_pct"] = sd
                mean, sd = stats([current[s][metric + "_pct"]-previous[s][metric + "_pct"] for s in pairs])
                comparison["fresh_minus_historical_" + metric + "_mean_pp"] = mean
                comparison["fresh_minus_historical_" + metric + "_sd_pp"] = sd
            historical_comparisons.append(comparison)
            for effect, coefficients in EFFECTS.items():
                pairs = sorted(set.intersection(*(set(lookup[(model, dataset, m)]) for m in coefficients)))
                effect_row = dict(model=model, dataset=dataset, effect=effect, paired_runs=len(pairs), expected_runs=3,
                    paired_seeds=",".join(map(str, pairs)), status="complete" if len(pairs) == 3 else "partial",
                    definition=json.dumps(coefficients, sort_keys=True))
                per_seed = []
                for seed in pairs:
                    pair_row = dict(model=model, dataset=dataset, seed=seed, effect=effect)
                    for metric in METRICS:
                        pair_row[metric + "_effect_pp"] = sum(coefficient * lookup[(model, dataset, method)][seed][metric + "_pct"]
                                                                for method, coefficient in coefficients.items())
                    per_seed.append(pair_row)
                    paired_effects.append(pair_row)
                for metric in METRICS:
                    mean, sd = stats([r[metric + "_effect_pp"] for r in per_seed])
                    effect_row[metric + "_mean_pp"] = mean
                    effect_row[metric + "_sd_pp"] = sd
                effects.append(effect_row)
    return summaries, historical_comparisons, effects, paired_effects


def format_cell(row, method):
    count = row[method + "_runs"]
    if count == 0:
        return "pending (0/3)"
    parts = []
    for metric in METRICS:
        mean, sd = row[method + "_" + metric + "_mean_pct"], row[method + "_" + metric + "_sd_pct"]
        parts.append(f"{mean:.2f}" + (f"±{sd:.2f}" if sd is not None else ""))
    return " / ".join(parts) + (f" [partial {count}/3]" if count != 3 else "")


def report_text(audit, summaries):
    lines = ["Fresh gating x loss factorial classification experiment", "=" * 54,
        f"Status: {audit['status'].upper()}; fresh fits {audit['completed_new_fits']}/252; "
        f"historical reference evaluations {audit['completed_reference_evaluations']}/120.",
        "All four current groups are newly trained. Historical pre/post fits never replace current controls.",
        "Cells: macro AUROC (%) / macro AUPRC (average precision, %); mean ± sample SD, seeds 42/43/44.",
        "Original is a deterministic historical reference, with no fabricated seed SD. Datasets are separate.", ""]
    if audit.get("migration"):
        lines += ["This campaign resumed in an independent output directory. Completed current-campaign fits",
            "were copied byte-for-byte with their original protocol/source provenance; interrupted fits restart",
            "from their original seeds. Historical reference fits are still excluded from the 252 fresh fits.", ""]
    for dataset in DATASETS:
        lines += [NAMES[dataset], "-" * len(NAMES[dataset]),
            "| Model | Original | Fresh Pre+BCE | Fresh Gated+BCE | Fresh Pre+ASL | Fresh Gated+ASL |",
            "|---|---|---|---|---|---|"]
        for row in summaries:
            if row["dataset"] != dataset:
                continue
            original = " / ".join(f"{row['original_' + metric + '_pct']:.2f}" if row['original_' + metric + '_pct'] is not None
                                   else "unavailable" for metric in METRICS)
            lines.append("| " + row["model"] + " | " + original + " | " + " | ".join(format_cell(row, m) for m in METHODS) + " |")
        lines.append("")
    lines += ["Factorial effects (paired within model, dataset and seed; percentage points):",
        "gating_effect_bce = gated_bce - pre_bce; gating_effect_asl = gated_asl - pre_asl.",
        "asl_effect_plain = pre_asl - pre_bce; asl_effect_gated = gated_asl - gated_bce.",
        "interaction = (gated_asl - pre_asl) - (gated_bce - pre_bce).",
        "See factorial_effects.csv and factorial_effects_per_seed.csv. Missing pairs remain partial.",
        "Historical pre comparison is separately recorded in historical_pre_comparison.csv.", "",
        "Interpretation boundaries:",
        "- The gate adds D+1 trainable parameters and starts at 0.1; the MLP branch starts at zero output.",
        "- Gating changes both token-dependent weighting and initial residual gradient scale; scores alone do not isolate these causes.",
        "- ASL has gamma_pos=0, gamma_neg=4, negative clip=0.05, eps=1e-8, mean reduction, full focal autograd and no pos_weight.",
        "- Weighted BCE uses TRAIN-only class positive weights; ASL replaces that weighting/loss combination.",
        "- All groups use supervised target-dataset adaptation, fixed hyperparameters and validation-only checkpoint selection.",
        "- Epoch zero is eligible and explicitly counted; for BiomedCLIP/BioViL-T/MaCo it equals added pooling-only, not Original.",
        "- Paired seed SD is not a patient-level confidence interval or a test of statistical significance.",
        "- Local custom cohorts remain unchanged: CheXpert TEST has 67 images; OpenI patient independence is unverified.",
        "- No cross-dataset averages or claims of universal improvement are produced.",
        "- VAL/TEST metrics are fully recomputed; forward replay evidence is sampled and is not an independent full-model rerun."]
    if audit["errors"]:
        lines += ["", "AUDIT ERRORS (affected fits excluded):"] + [json.dumps(e, ensure_ascii=False) for e in audit["errors"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol_path = run / "PROTOCOL.json"
    protocol, protocol_hash = read(protocol_path), sha256(protocol_path)
    check_protocol(protocol)
    runtime_manifest = read(run / "source_manifest.json")
    migration = verify_migration(run, runtime_manifest)
    datasets = load_data(protocol)
    historical, historical_audits, errors = [], [], []
    # These 120 entries remain historical evaluations, outside the 252 fits.
    for entry in protocol["references"]:
        try:
            row, evidence = audit_historical(entry, protocol, protocol_hash, datasets, run, {}, {})
            historical.append(row)
            historical_audits.append(evidence)
        except Exception as error:
            errors.append(dict(key=list(entry_key(entry)), scope="historical_reference", source=entry["source"],
                               error=f"{type(error).__name__}: {error}"))
    caches = {name: {} for name in ("initialization", "orders", "tokens", "runtime", "sources", "frozen_logits")}
    caches.update(migration=migration, runtime_manifest=runtime_manifest,
                  runtime_manifest_sha256=sha256(run / "source_manifest.json"))
    rows, audits, completed, pending = [], [], [], []
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                for method in METHODS:
                    entry = dict(model=model, dataset=dataset, seed=seed, method=method,
                        source=str(run / "results" / model / dataset / f"seed{seed}" / method))
                    if not (Path(entry["source"]) / "result.json").is_file():
                        pending.append(dict(kind="fresh_fit", key=list(entry_key(entry)), source=entry["source"]))
                        continue
                    try:
                        row, evidence, training = audit_fresh(entry, protocol, protocol_hash, datasets, run, caches)
                        rows.append(row)
                        audits.append(evidence)
                        completed.append((entry, training))
                    except Exception as error:
                        errors.append(dict(key=list(entry_key(entry)), scope="fresh_fit", source=entry["source"],
                                           error=f"{type(error).__name__}: {error}"))
    matching, matching_pending, matching_errors = matching_audit(completed, protocol, run)
    pending.extend(matching_pending)
    errors.extend(matching_errors)
    status = "failed" if errors else "partial" if pending else "passed"
    summaries, history_comparisons, effects, paired_effects = summarize(rows, historical, protocol)
    reporter_sources = [Path(__file__), ROOT / "experiments/both_adapter_classification/report.py",
        ROOT / "experiments/adapter_placement_classification/report.py", ROOT / "experiments/baseline_token_classification/common.py",
        ROOT / "experiments/gated_asl_classification/common.py", Path(__file__).with_name("migrate.py")]
    audit = dict(status=status, generated_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=protocol_hash,
        expected_new_fits=252, completed_new_fits=len(rows), expected_reference_evaluations=120,
        completed_reference_evaluations=len(historical), historical_supervised_fits=90,
        deterministic_reference_evaluations=30, old_fits_counted_as_new=0,
        migration=migration["evidence"] if migration else None,
        reporter_sources_sha256={str(p): sha256(p) for p in reporter_sources},
        pending=pending, errors=errors, matching_groups=matching, runs=audits, historical_references=historical_audits)
    require(sha256(protocol_path) == protocol_hash
            and sha256(run / "source_manifest.json") == caches["runtime_manifest_sha256"],
            "Frozen protocol/runtime manifest changed while auditing")
    require(verify_migration(run, runtime_manifest) == migration,
            "Migration artifacts changed while auditing; no report written")
    write_csv(run / "summary.csv", summaries)
    write_csv(run / "per_seed.csv", rows)
    write_csv(run / "historical_results.csv", historical)
    write_csv(run / "historical_pre_comparison.csv", history_comparisons)
    write_csv(run / "factorial_effects.csv", effects)
    write_csv(run / "factorial_effects_per_seed.csv", paired_effects)
    atomic_text(run / "AUDIT.json", json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    atomic_text(run / "REPORT.txt", report_text(audit, summaries))
    print(json.dumps({k: v for k, v in audit.items() if k not in
                     ("runs", "historical_references", "pending", "matching_groups", "reporter_sources_sha256")}, indent=2))
    if errors or (pending and not args.allow_partial):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
