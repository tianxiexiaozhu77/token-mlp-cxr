"""Reuse OpenI and make immutable grouped local-cohort classification splits."""
import argparse
import ast
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

from experiment_utils import ROOT, read_dataset, save, sha256

sys.path.insert(0, str(ROOT / "RadZero/scripts"))
from prepare_openi_adapter import split_groups, UnionFind


def prepare(name, catalog, key, out):
    entry = json.loads(catalog.read_text())["datasets"][key]
    source = Path(entry["csv_path"])
    if sha256(source) != entry["csv_sha256"]:
        raise ValueError("Source manifest hash differs")
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    classes = entry["class_names"]
    samples = []
    for row in rows:
        path = Path(row["image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        patient = row.get("patient_id") or path.stem.split("_")[0]
        if name == "chexpert" and not re.fullmatch(r"patient\d+", patient):
            raise ValueError("Unrecognized CheXpert patient")
        if name == "chestxray14" and not re.fullmatch(r"\d{8}", patient):
            raise ValueError("Unrecognized NIH patient")
        labels = ast.literal_eval(row["label"])
        if len(labels) != len(classes) or any(y not in (0, 1) for y in labels):
            raise ValueError("Invalid binary labels")
        samples.append(dict(id=str(path.relative_to(ROOT / "data")), image=str(path),
                            patient_id=patient, labels=labels))
    print(f"{name}: verifying {len(samples)} image files and exact-byte duplicates", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        hashes = list(pool.map(sha256, [r["image"] for r in samples]))
    union = UnionFind({r["patient_id"] for r in samples})
    by_hash = {}
    for row, digest in zip(samples, hashes):
        row["file_sha256"] = digest
        if digest in by_hash:
            union.union(row["patient_id"], by_hash[digest])
        by_hash[digest] = row["patient_id"]
    groups = {}
    for row in samples:
        group = union.find(row["patient_id"])
        row["group_id"] = f"{name}_{group}"
        item = groups.setdefault(row["group_id"], {"labels": np.zeros(len(classes), dtype=int)})
        item["labels"] = np.maximum(item["labels"], row["labels"])
    assignments, targets = split_groups(groups, 42)
    for row in samples:
        row["split"] = assignments[row["group_id"]]
    stats = {}
    for split in ("train", "val", "test"):
        selected = [r for r in samples if r["split"] == split]
        y = np.array([r["labels"] for r in selected])
        if np.any(y.sum(0) == 0) or np.any(y.sum(0) == len(y)):
            raise ValueError(f"Undefined {split} AUROC: insufficient class support")
        stats[split] = dict(images=len(y), patients=len({r["patient_id"] for r in selected}),
                            groups=len({r["group_id"] for r in selected}), positives=y.sum(0).tolist())
    for field in ("patient_id", "group_id", "file_sha256"):
        seen = {}
        for row in samples:
            previous = seen.setdefault(row[field], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Cross-split leakage: {field}")
    out.mkdir(parents=True, exist_ok=False)
    save(out / "samples.json", samples)
    meta = dict(dataset=name, class_names=classes,
                prompts=[f"There is {c}" for c in classes],
                samples_file=str((out / "samples.json").resolve()),
                samples_sha256=sha256(out / "samples.json"), seed=42,
                split_method="Iterative multilabel 80/10/10 stratification over patient/exact-file-duplicate groups",
                split_stats=stats, patient_independence={"verified": True, "identifier": "source patient ID"},
                original_cohort="Locally available original benchmark test cohort; repartitioned for supervised adaptation. NOT the official benchmark train/test protocol.",
                source_manifest=str(source), source_manifest_sha256=sha256(source),
                source_catalog=str(catalog), source_catalog_sha256=sha256(catalog),
                targets=targets, duplicate_check="Exact file SHA256; reencoded near-duplicates not exhaustively audited")
    save(out / "dataset.json", meta)
    print(json.dumps({"dataset": name, "stats": stats}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = ROOT / "RadZero/runs/openi_adapter_20261020_152558/inputs/dataset.json"
    meta, rows = read_dataset(original)
    seen = {}
    for row in rows:
        if not Path(row["image"]).is_file():
            raise FileNotFoundError(row["image"])
        for field in ("uid", "group_id"):
            key = field, row[field]
            if seen.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("OpenI split leakage")
    # Keep original path/hash to enable exact reuse of provenance-checked tokens.
    save(args.output / "openi.json", {"dataset_file": str(original),
         "dataset_sha256": sha256(original), "images": len(rows),
         "patient_independence": meta["patient_independence"], "split_reused": True})
    for name, catalog, key in [
        ("chexpert", ROOT / "AlphaRAD/local_setup/chexpert_query_data_20261022_shot5_seed42/full_catalog.json", "chexpert"),
        ("chestxray14", ROOT / "AlphaRAD/local_setup/data_20261021_complete/catalog.json", "chest14"),
    ]:
        if (args.output / name / "dataset.json").exists():
            read_dataset(args.output / name / "dataset.json")
            continue
        prepare(name, catalog, key, args.output / name)


if __name__ == "__main__":
    main()
