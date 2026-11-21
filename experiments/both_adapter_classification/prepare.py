"""Freeze all 63 Both fits and existing comparison sources before training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.both_adapter_classification.common import (
    MODELS, DIMS, BASE, DIRECT, ALPHA, PLACEMENT, DATASETS, read_dataset, save, sha256,
    parameter_count, reference_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "PROTOCOL.json").exists():
        raise FileExistsError(run / "PROTOCOL.json")
    datasets, references, caches = {}, [], {}
    for name, path in DATASETS.items():
        meta, rows = read_dataset(path)
        datasets[name] = dict(path=str(path), sha256=sha256(path), samples_sha256=meta["samples_sha256"],
            classes=meta["class_names"], counts={s:sum(r["split"] == s for r in rows) for s in ("train","val","test")})
    for model in MODELS:
        for dataset, spec in datasets.items():
            paths = sorted((DIRECT / "cache" / model.lower() / dataset).glob("cache_*.json")) if model in ("AlphaRAD","RadZero") else [BASE/"cache"/model/dataset/"cache.json"]
            assert paths, (model, dataset)
            cache_specs = []
            for path in paths:
                c = json.loads(path.read_text())
                assert c["status"] == "complete" and c["dataset_sha256"] == spec["sha256"] and c["samples_sha256"] == spec["samples_sha256"]
                files = [t["path"] for t in c["token_shards"]] + [c["head_path"],c["context_path"],c["logits_path"]] if "token_shards" in c else [f["path"] for f in c["files"].values()]
                cache_specs.append(dict(path=str(path),sha256=sha256(path), files={f:dict(bytes=Path(f).stat().st_size,mtime_ns=Path(f).stat().st_mtime_ns) for f in files}))
            caches[f"{model}/{dataset}"] = cache_specs
            methods = {"original":[42],"pre_mlp":[42,43,44]}
            if model in ("AlphaRAD","KAD","MaCo"):
                methods["post_mlp"] = [42,43,44]
            if model in ("BiomedCLIP","BioViL-T","MaCo"):
                methods["aggregation_only"] = [42]
            for method, seeds in methods.items():
                for seed in seeds:
                    folder = reference_folder(model,dataset,seed,method)
                    r = json.loads((folder/"result.json").read_text())
                    assert r["status"] == "complete" and r["seed"] == seed
                    assert r["dataset_sha256"] == spec["sha256"] and r["samples_sha256"] == spec["samples_sha256"]
                    if method in ("pre_mlp","post_mlp"):
                        assert r["trainable_parameters"] == parameter_count(DIMS[model])//2
                        for key,val in dict(epochs=20,min_epochs=8,patience=5,lr=1e-4,weight_decay=1e-4).items():
                            assert r["config"][key] == val, (folder,key)
                        assert r["config"].get("batch_size",r["config"].get("effective_batch_size")) == 32
                    references.append(dict(model=model,dataset=dataset,seed=seed,method=method,source=str(folder),
                        files_sha256={n:sha256(folder/n) for n in ("result.json","training.json","adapter.pt","test_predictions.npz") if (folder/n).exists()}))
    positions = {
        "AlphaRAD":"pre: raw projected CLS/patch tokens before FLaS split/token L2; post: concatenated 8x96 disease contexts before native per-head L2/cosine/scale/head mean",
        "RadZero":"pre: raw projected CLS/patch tokens before native VL-CABS normalization; post: pooled 768D disease context before native L2/cosine/logit_scale",
        "CARZero":"pre: native global+patch tokens before bidirectional DQN; post: same shared 768D MLP applied independently to both decoded i2t/t2i branches before their native shared classifier and arithmetic mean",
        "KAD":"pre: projected patch tokens before native decoder_norm; post: final decoder LayerNorm/eval dropout disease features before native 768-to-2 margin head",
        "BiomedCLIP":"pre: existing projected CLS+patch tokens; post: raw positive/negative text-guided contexts before native final L2/cosine margin",
        "BioViL-T":"pre: existing raw projected patch tokens; post: raw positive/negative text-guided contexts before native final L2/cosine margin",
        "MaCo":"pre: existing raw projected patch tokens; post: raw positive/negative text-guided contexts before unchanged raw-dot margin"}
    protocol = dict(status="frozen",method="both_mlp",models=list(MODELS),datasets=datasets,seeds=[42,43,44],
        new_training_runs=63,allowed_physical_gpus=[4,5,6,7],bottleneck=64,
        model_specs={m:dict(dim=DIMS[m],bottleneck=64,trainable_parameters=parameter_count(DIMS[m]),position=positions[m],
            aggregation_origin="added" if m in ("BiomedCLIP","BioViL-T","MaCo") else "native") for m in MODELS},
        train=dict(optimizer="AdamW",lr=1e-4,weight_decay=1e-4,effective_batch_size=32,epochs=20,min_epochs=8,patience=5,
            grad_clip=1.,pos_weight="TRAIN negatives/positives capped at20",selection="VAL macro AUROC; strict improvement, earliest tie; epoch0 eligible"),
        references=references,caches=caches,reference_roots=dict(baselines=str(BASE),direct=str(DIRECT),alpha_token=str(ALPHA),placement=str(PLACEMENT)),
        joint_training=True,independent_adapters=True,initialization="Seed once; construct pre then post; each up weight/bias zero; independent down random draws",
        pooling="Retain previous Token MLP aggregation; no new pooling changes in this supplement",
        limitations=["Both rank64+64 doubles the single rank64 adapter parameter count; this is not a parameter-matched placement comparison.",
            "Reuse the existing local datasets; CheXpert test has only 67 images and OpenI patient independence is not established here.",
            "BiomedCLIP/BioViL-T/MaCo identity adapters reproduce aggregation_only, which differs from the original classifier.",
            "Test does not select epochs or hyperparameters. Three-seed SD is not a patient-level confidence interval.",
            "Only Both is newly trained. Existing pre/post comparisons are reused explicitly; post exists only for AlphaRAD/KAD/MaCo."])
    save(run/"PROTOCOL.json",protocol)
    print(json.dumps(dict(run=str(run),new_fits=63,references=len(references),datasets={d:s['counts'] for d,s in datasets.items()}),indent=2))


if __name__ == "__main__":
    main()
