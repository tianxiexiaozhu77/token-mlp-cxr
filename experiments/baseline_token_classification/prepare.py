"""Preserve all existing splits; derive explicitly separate OpenI common16 task."""
import argparse
from collections import Counter
from pathlib import Path
from common import ROOT, DATASETS, MODELS, read_dataset, save, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "PROTOCOL.json").exists():
        raise FileExistsError("Protocol already prepared")
    datasets = {}
    for name, path in DATASETS.items():
        meta, rows = read_dataset(path)
        datasets[name] = dict(path=str(path), sha256=sha256(path), samples_sha256=meta["samples_sha256"],
            classes=meta["class_names"], counts=dict(Counter(r["split"] for r in rows)))
    meta, rows = read_dataset(DATASETS["openi"])
    excluded = ["Calcified Granuloma", "Granuloma"]
    indices = [i for i, c in enumerate(meta["class_names"]) if c not in excluded]
    common_rows = [dict(r, labels=[r["labels"][i] for i in indices]) for r in rows]
    folder = run / "data/openi_common16"
    save(folder / "samples.json", common_rows)
    common_meta = dict(meta, dataset="openi_common16", class_names=[meta["class_names"][i] for i in indices],
        prompts=[meta["prompts"][i] for i in indices] if "prompts" in meta else [],
        samples_file=str(folder / "samples.json"), samples_sha256=sha256(folder / "samples.json"),
        parent_dataset=str(DATASETS["openi"]), parent_sha256=sha256(DATASETS["openi"]),
        excluded_classes=excluded, parent_column_indices=indices,
        scope="Same images, labels and splits; only two unavailable class columns removed. NOT full OpenI18.")
    if "split_stats" in common_meta:
        common_meta["split_stats"] = {k: dict(v, positives=[v["positives"][i] for i in indices])
            if "positives" in v else v for k,v in common_meta["split_stats"].items()}
    save(folder / "dataset.json", common_meta)
    datasets["openi_common16"] = dict(path=str(folder / "dataset.json"), sha256=sha256(folder / "dataset.json"),
        samples_sha256=common_meta["samples_sha256"], classes=common_meta["class_names"],
        counts=dict(Counter(r["split"] for r in rows)))
    save(run / "PROTOCOL.json", dict(models=list(MODELS), datasets=datasets,
        model_datasets={m: ["chexpert", "openi_common16" if m in ("MedKLIP", "MAVL") else "openi", "chestxray14"] for m in MODELS},
        seeds=[42,43,44], allowed_physical_gpus=[4,5,6,7], bottleneck=64,
        adapter="x + Linear_up(GELU(Linear_down(x))); shared over tokens/classes; linear biases; up initialized zero",
        methods_native_aggregation=["baseline", "token_mlp"],
        methods_added_aggregation=["baseline", "aggregation_only", "token_mlp"],
        added_aggregation="Fixed cosine attention with temperature=1, original text queries/readout; not FLaS. MLP zero initialization equals aggregation_only, not native global baseline.",
        train=dict(optimizer="AdamW", lr=1e-4, weight_decay=1e-4, effective_batch_size=32,
            epochs=20, min_epochs=8, patience=5, grad_clip=1., pos_weight="train negatives/positives capped20",
            selection="validation macro AUROC; earliest tie; epoch0 eligible", test_selection=False),
        metric="AUROC and average precision (called AUPRC in prior classification experiment); seed sample SD ddof1",
        scope_notes=["Supervised adaptation, not training-free or zero-shot after fitting.",
            "Custom patient splits of locally available CheXpert and ChestXray14 original test cohorts; NOT official benchmark results.",
            "OpenI report/duplicate split reused; longitudinal patient independence unverified.",
            "MedKLIP/MAVL full OpenI18 unsupported by existing disease-description resources; user authorized separately labeled common16.",
            "TaskRes and SS-Text are adaptation algorithms, not additional pretrained medical backbones.",
            "Native aggregation retained when present; otherwise added aggregation separately controlled.",
            "Hidden width64 fixed, native feature dimensions differ; cross-model parameter counts are not equal."]))
    print(run, flush=True)


if __name__ == "__main__":
    main()
