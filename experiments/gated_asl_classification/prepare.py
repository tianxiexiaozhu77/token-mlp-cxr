"""Freeze 252 fresh fits; historical results are provenance/comparison only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gated_asl_classification.common import (
    MODELS, DIMS, METHODS, METHOD_NAMES, SEEDS, ASL_CONFIG, GATE_CONFIG,
    BASE, DIRECT, ALPHA, PLACEMENT, DATASETS, read_dataset, save, sha256,
    parameter_count, reference_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "PROTOCOL.json").exists() or (run / "results").exists():
        raise FileExistsError("Use a fresh campaign directory: " + str(run))
    parent_path = ROOT / "experiments/both_adapter_classification/runs/20261119/PROTOCOL.json"
    parent = json.loads(parent_path.read_text())
    datasets, caches, references = {}, {}, []
    for dataset, path in DATASETS.items():
        meta, rows = read_dataset(path)
        spec = dict(path=str(path), sha256=sha256(path), samples_sha256=meta["samples_sha256"],
            classes=meta["class_names"], counts={s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")})
        if spec != parent["datasets"][dataset]:
            raise ValueError("Dataset differs from the frozen preceding protocol: " + dataset)
        datasets[dataset] = spec
    for model in MODELS:
        for dataset, spec in datasets.items():
            paths = (sorted((DIRECT / "cache" / model.lower() / dataset).glob("cache_*.json"))
                     if model in ("AlphaRAD", "RadZero") else [BASE / "cache" / model / dataset / "cache.json"])
            if not paths:
                raise ValueError("No existing frozen cache: " + model + "/" + dataset)
            cache_specs = []
            for path in paths:
                c = json.loads(path.read_text())
                if c["status"] != "complete" or c["dataset_sha256"] != spec["sha256"] or c["samples_sha256"] != spec["samples_sha256"]:
                    raise ValueError("Incomplete/mismatched cache: " + str(path))
                files = ([t["path"] for t in c["token_shards"]] + [c["head_path"], c["context_path"], c["logits_path"]]
                         if "token_shards" in c else [f["path"] for f in c["files"].values()])
                cache_specs.append(dict(path=str(path), sha256=sha256(path),
                    files={f: dict(bytes=Path(f).stat().st_size, mtime_ns=Path(f).stat().st_mtime_ns) for f in files}))
            if cache_specs != parent["caches"][f"{model}/{dataset}"]:
                raise ValueError("Cache record/stat provenance changed: " + model + "/" + dataset)
            caches[f"{model}/{dataset}"] = cache_specs
            reference_methods = {"original": [42], "pre_mlp": SEEDS}
            if model in ("AlphaRAD", "KAD", "MaCo"):
                reference_methods["post_mlp"] = SEEDS
            if model in ("BiomedCLIP", "BioViL-T", "MaCo"):
                reference_methods["aggregation_only"] = [42]
            for method, seeds in reference_methods.items():
                for seed in seeds:
                    folder = reference_folder(model, dataset, seed, method)
                    r = json.loads((folder / "result.json").read_text())
                    if (r["status"] != "complete" or r["seed"] != seed or r["class_names"] != spec["classes"]
                            or r["dataset_sha256"] != spec["sha256"] or r["samples_sha256"] != spec["samples_sha256"]):
                        raise ValueError("Historical reference identity differs: " + str(folder))
                    if method in ("pre_mlp", "post_mlp"):
                        if r["trainable_parameters"] != parameter_count(DIMS[model]):
                            raise ValueError("Historical single-MLP budget differs: " + str(folder))
                        for key, val in dict(epochs=20, min_epochs=8, patience=5, lr=1e-4, weight_decay=1e-4).items():
                            if r["config"][key] != val:
                                raise ValueError("Historical training protocol differs: " + str(folder))
                        if r["config"].get("batch_size", r["config"].get("effective_batch_size")) != 32:
                            raise ValueError("Historical effective batch size differs: " + str(folder))
                    references.append(dict(model=model, dataset=dataset, seed=seed, method=method, source=str(folder),
                        files_sha256={name: sha256(folder / name) for name in ("result.json", "training.json", "adapter.pt", "test_predictions.npz") if (folder / name).exists()}))
    if references != parent["references"] or len(references) != 120:
        raise ValueError("Historical reference artifacts changed from the preceding frozen protocol")
    positions = {
        "AlphaRAD": "raw projected CLS/patch tokens before FLaS head split and token L2",
        "RadZero": "aligned CLS/patch tokens before native LayerNorm/CLS selection and VL-CABS",
        "CARZero": "projected global/patch tokens before bidirectional DQN; both i2t visual memory and t2i global query",
        "KAD": "projected patch tokens before native disease-query decoder_norm",
        "BiomedCLIP": "projected CLS/patch tokens before unchanged added fixed text-guided pooling",
        "BioViL-T": "projected patch tokens before unchanged added fixed text-guided pooling",
        "MaCo": "projected patch tokens before unchanged added fixed text-guided pooling and raw-dot margin"}
    protocol = dict(status="frozen", campaign="gated_asl_classification", models=list(MODELS), methods=list(METHODS),
        datasets=datasets, seeds=list(SEEDS), new_training_runs=252, all_methods_fresh=True, reused_training_runs=0,
        allowed_physical_gpus=[4, 5, 6, 7], dataset_schedule=["chestxray14", "openi", "chexpert"], bottleneck=64,
        model_specs={model: dict(dim=DIMS[model], bottleneck=64, position=positions[model],
            aggregation_origin="added" if model in ("BiomedCLIP", "BioViL-T", "MaCo") else "native",
            trainable_parameters_by_method={method: parameter_count(DIMS[model], 64, method) for method in METHODS}) for model in MODELS},
        method_specs={method: dict(name=METHOD_NAMES[method], gated=method.startswith("gated_"),
            loss="asl" if method.endswith("_asl") else "weighted_bce", fresh_fit=True, placement="pre") for method in METHODS},
        asl_config=ASL_CONFIG, gate_config=GATE_CONFIG,
        train=dict(optimizer="AdamW", lr=1e-4, weight_decay=1e-4, effective_batch_size=32, epochs=20, min_epochs=8,
            patience=5, grad_clip=1., pos_weight="BCE only: TRAIN negatives/positives capped at20; ASL: no pos_weight",
            selection="VAL macro AUROC; strict improvement, earliest tie; epoch0 eligible"),
        placement="pre", initialization="Reset each fit to its seed. Shared residual rank64 GELU MLP; up weight/bias zero. Gated models add per-token scalar sigmoid gate with zero gate weights and bias logit(0.1).",
        loss_semantics=dict(bce="BCEWithLogitsLoss; TRAIN-only class pos_weight capped20; mean over batch and classes",
            asl="sigmoid probabilities; negative probability clipped with +0.05; gamma_neg4/gamma_pos0; eps1e-8; mean over batch and classes; full focal autograd; no class pos_weight"),
        references=references, reference_role="Historical provenance/replay/comparison only. All four current groups, including pre_bce, are fitted freshly; no old fitted result counts toward the 252 fits.",
        caches=caches, reference_roots=dict(baselines=str(BASE), direct=str(DIRECT), alpha_token=str(ALPHA), placement=str(PLACEMENT)),
        parent_protocol=dict(path=str(parent_path), sha256=sha256(parent_path)),
        pooling="Retain earlier Pre Token MLP aggregation and frozen readout; this campaign changes only pre adapter gating and training loss.",
        limitations=["Gated adapters add dim+1 parameters; this is not exact parameter matching.",
            "All groups are supervised target-dataset adaptation, not training-free or zero-shot adaptation.",
            "CheXpert TEST has 67 images; OpenI patient independence remains unverified.",
            "BiomedCLIP/BioViL-T/MaCo epoch0 equals fixed aggregation-only, not the original global classifier.",
            "Seed sample SD is not a patient-level confidence interval. Test does not select epochs/hyperparameters."])
    save(run / "PROTOCOL.json", protocol)
    print(json.dumps(dict(run=str(run), new_fits=252, all_methods_fresh=True, historical_references=len(references), methods=list(METHODS)), indent=2))


if __name__ == "__main__":
    main()
