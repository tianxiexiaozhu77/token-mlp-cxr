"""Freeze the placement-ablation plan before looking at new test results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import DATASETS, read_dataset, save, sha256

BASE = ROOT / "experiments/baseline_token_classification/runs/20261119"
ALPHA = ROOT / "experiments/direct_anchor_classification/runs/20261118_alpha_token_mlp"
DIRECT = ROOT / "experiments/direct_anchor_classification/runs/20261118"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "PROTOCOL.json").exists():
        raise FileExistsError(run / "PROTOCOL.json")
    datasets = {}
    for name, path in DATASETS.items():
        meta, rows = read_dataset(path)
        datasets[name] = dict(path=str(path), sha256=sha256(path), samples_sha256=meta["samples_sha256"],
            classes=meta["class_names"], counts={s:sum(r["split"] == s for r in rows) for s in ("train", "val", "test")})
    reuse = []
    for model in ("AlphaRAD", "KAD", "MaCo"):
        for dataset in DATASETS:
            for seed in (42, 43, 44):
                parent = ALPHA / "results/alpharad" if model == "AlphaRAD" else BASE / "results" / model
                folder = parent / dataset / f"seed{seed}" / "token_mlp"
                result = json.loads((folder / "result.json").read_text())
                assert result["status"] == "complete" and result["seed"] == seed
                assert result["trainable_parameters"] == 99136
                assert result["dataset_sha256"] == datasets[dataset]["sha256"]
                assert result["samples_sha256"] == datasets[dataset]["samples_sha256"]
                cfg = result["config"]
                for key, expected in {"epochs":20, "min_epochs":8, "patience":5, "lr":1e-4, "weight_decay":1e-4}.items():
                    assert cfg[key] == expected, (folder, key)
                assert cfg.get("batch_size", cfg.get("effective_batch_size")) == 32
                reuse.append(dict(model=model, dataset=dataset, seed=seed, method="pre_mlp",
                    source=str(folder), files_sha256={name:sha256(folder/name) for name in
                    ("result.json", "training.json", "adapter.pt", "test_predictions.npz")}))
    protocol = dict(status="frozen", models=["AlphaRAD", "KAD", "MaCo"], datasets=datasets,
        seeds=[42,43,44], allowed_physical_gpus=[4,5,6,7], bottleneck=64, dim=768, trainable_parameters=99136,
        train=dict(optimizer="AdamW", lr=1e-4, weight_decay=1e-4, effective_batch_size=32,
            epochs=20, min_epochs=8, patience=5, grad_clip=1.,
            pos_weight="TRAIN negatives/positives capped at20", selection="VAL macro AUROC; exact ties keep earliest; epoch0 eligible"),
        new_methods={"AlphaRAD":["post_mlp"], "KAD":["post_mlp"],
            "MaCo":["post_mlp", "native_mean_pre", "native_mean_post"]},
        reused_pre_runs=reuse, new_training_runs=45, reused_training_runs=27,
        reference_roots=dict(baselines=str(BASE), alpha_token=str(ALPHA), direct=str(DIRECT)),
        positions={"AlphaRAD":"post: concatenate 8x96 FLaS contexts, shared MLP, then original per-head L2/cosine/scale/mean",
            "KAD":"post: frozen decoder final norm/eval dropout, shared MLP on each disease feature, frozen768-to2 readout",
            "MaCo":"post: shared MLP on positive/negative raw contexts before raw-dot margin; native_mean variants change pooling only"},
        matching="Same feature cache, labels, seed, MLP initialization, DataLoader train generator, batch budget and validation selection as existing pre runs. No disease-specific parameters or added trainable normalization.",
        limitations=["This ablation reuses the manuscript's local splits; it does not repair the small CheXpert test cohort or establish patient independence for OpenI.",
            "Pre/post placements encounter different feature distributions and, in native models, different surrounding normalization operations. The comparison tests the complete insertion points.",
            "Shared batch size and epoch budget do not equalize FLOPs: post aggregation can cache frozen contexts.",
            "Test labels are used only for final evaluation and reporting, never selection. Existing pre results are reused transparently, not counted as new fits."])
    save(run / "PROTOCOL.json", protocol)
    print(json.dumps(dict(run=str(run), new_fits=45, reused_fits=len(reuse), datasets=datasets), indent=2))


if __name__ == "__main__":
    main()
