#!/usr/bin/env python3
"""Prepare a grouped OpenI supervised-adaptation split without altering author data."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/radzero"
OUT = DATA / "adapter_openi"
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "Fracture", "Opacity", "Lesion",
    "Calcified Granuloma", "Granuloma",
]
SYNONYMS = {
    "Pleural_Thickening": ["pleural thickening"],
    "Infiltration": ["Infiltrate"],
    "Atelectasis": ["Atelectases"],
}
DISPLAY = {
    "Opacity": "Lung Opacity", "Lesion": "Lung Lesion",
    "Pleural_Thickening": "pleural thickening",
    "Infiltration": "Infiltrate", "Atelectasis": "Atelectases",
}
SPLITS = ["train", "val", "test"]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def author_labels(automatic):
    """Match the author's case-sensitive pandas str.contains expressions exactly."""
    automatic = automatic or "-1"
    return [
        int(any(re.search(term.lower(), automatic) is not None
                for term in [name] + SYNONYMS.get(name, [])))
        for name in PATHOLOGIES
    ]


def inspect_image(path):
    stat = path.stat()
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height, mode = image.width, image.height, image.mode
        pixels = np.asarray(image)
        descriptor = f"{mode}:{pixels.dtype}:{pixels.shape}".encode()
        pixel_hash = hashlib.sha256(descriptor + pixels.tobytes()).hexdigest()
        gray = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        spectrum = dctn(gray, type=2, norm="ortho")[:8, :8].reshape(-1)
        # The DC coefficient is excluded from the threshold; the resulting hash
        # only proposes near-duplicate candidates, never establishes identity.
        bits = spectrum > np.median(spectrum[1:])
        phash = sum(int(bit) << i for i, bit in enumerate(bits))
        metadata_keys = sorted(image.info)
    return {
        "image": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_sha256": pixel_hash, "width": width, "height": height, "mode": mode,
        "phash": f"{phash:016x}", "png_metadata_keys": metadata_keys,
    }


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def candidate_near_duplicates(records, samples, max_distance=4):
    """BK-tree search of 64-bit pHashes; candidate pairs are not merged."""
    tree = None
    pairs = []
    for index, record in enumerate(records):
        value = int(record["phash"], 16)
        if tree is not None:
            pending = [tree]
            while pending:
                node = pending.pop()
                distance = (value ^ node[0]).bit_count()
                if distance <= max_distance:
                    for previous in node[1]:
                        if samples[index]["uid"] == samples[previous]["uid"]:
                            continue
                        if record["pixel_sha256"] == records[previous]["pixel_sha256"]:
                            continue
                        pairs.append({
                            "id_a": samples[previous]["id"], "id_b": samples[index]["id"],
                            "uid_a": samples[previous]["uid"], "uid_b": samples[index]["uid"],
                            "phash_distance": distance,
                            "same_split": samples[previous]["split"] == samples[index]["split"],
                            "merged": False,
                        })
                for edge, child in node[2].items():
                    if distance - max_distance <= edge <= distance + max_distance:
                        pending.append(child)
        if tree is None:
            tree = [value, [index], {}]
        else:
            node = tree
            while True:
                distance = (value ^ node[0]).bit_count()
                if distance == 0:
                    node[1].append(index)
                    break
                if distance not in node[2]:
                    node[2][distance] = [value, [index], {}]
                    break
                node = node[2][distance]
    return pairs


def score_near_duplicates(pairs, records, samples):
    """Add coarse pixel similarity for review; never infer patient identity."""
    lookup = {sample["id"]: index for index, sample in enumerate(samples)}
    thumbnails = {}
    for pair in pairs:
        arrays = []
        for name in [pair["id_a"], pair["id_b"]]:
            if name not in thumbnails:
                with Image.open(samples[lookup[name]]["image"]) as image:
                    thumbnails[name] = np.asarray(
                        image.convert("L").resize((256, 256), Image.Resampling.LANCZOS),
                        dtype=np.float32,
                    ) / 255.
            arrays.append(thumbnails[name])
        a, b = arrays
        centered_a, centered_b = a - a.mean(), b - b.mean()
        denominator = float(np.sqrt((centered_a**2).sum() * (centered_b**2).sum()))
        pair["thumbnail_pearson"] = (
            float((centered_a * centered_b).sum()) / denominator if denominator else None
        )
        pair["thumbnail_mean_absolute_error_0_1"] = float(np.abs(a - b).mean())
        ra, rb = records[lookup[pair["id_a"]]], records[lookup[pair["id_b"]]]
        pair["same_dimensions"] = (ra["width"], ra["height"]) == (rb["width"], rb["height"])
    return pairs


def split_groups(groups, seed):
    """Deterministic iterative multilabel stratification over indivisible groups."""
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    count = len(keys)
    capacity = np.array([count - 2 * round(count * .1),
                         round(count * .1), round(count * .1)], dtype=int)
    target_sizes = capacity.copy()
    labels = np.array([groups[key]["labels"] for key in keys], dtype=int)
    targets = labels.sum(axis=0)[None, :] * np.array([.8, .1, .1])[:, None]
    # With only three positive groups (Lung Lesion), all three partitions should
    # contain a positive group rather than silently having undefined test AUROC.
    for column, positives in enumerate(labels.sum(axis=0)):
        if positives >= 3:
            targets[1:, column] = np.maximum(1., targets[1:, column])
            targets[0, column] = positives - targets[1:, column].sum()
    remaining = set(range(count))
    assignment = {}
    priority = rng.permutation(count)
    rank = np.empty(count, dtype=int)
    rank[priority] = np.arange(count)
    while remaining:
        ordered = sorted(remaining)
        positives = labels[ordered].sum(axis=0)
        available = np.flatnonzero(positives)
        if not len(available):
            for index in sorted(remaining, key=lambda i: rank[i]):
                candidates = np.flatnonzero(capacity == capacity.max())
                chosen = int(rng.choice(candidates))
                assignment[keys[index]] = SPLITS[chosen]
                capacity[chosen] -= 1
            break
        rarest = int(available[np.argmin(positives[available])])
        candidates = sorted(
            (index for index in remaining if labels[index, rarest]),
            key=lambda i: rank[i],
        )
        for index in candidates:
            slots = np.flatnonzero(capacity > 0)
            demand = targets[slots, rarest]
            slots = slots[np.isclose(demand, demand.max())]
            slots = slots[capacity[slots] == capacity[slots].max()]
            chosen = int(rng.choice(slots))
            assignment[keys[index]] = SPLITS[chosen]
            capacity[chosen] -= 1
            targets[chosen] -= labels[index]
            remaining.remove(index)
    if np.any(capacity):
        raise ValueError(f"Group splitting did not fill target sizes: {capacity}")
    return assignment, dict(zip(SPLITS, target_sizes.tolist()))


def set_link(source, destination):
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        temporary = destination.with_name(destination.name + ".adapter_link_tmp")
        temporary.symlink_to(source, target_is_directory=source.is_dir())
        temporary.replace(destination)
    elif destination.exists():
        if destination.resolve() != source:
            raise ValueError(f"Refusing to replace real path: {destination}")
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def repair_prepared(openi_records):
    """Reconnect moved raw data; report only verification actually performed."""
    validations = {}
    for name, csv_name, expected in [
        ("OpenI", "openi_multi_label_image.csv", 7470),
        ("ChestXray14", "chestxray14_test_image.csv", 22433),
    ]:
        metadata = DATA / "metadata" / name
        prepared = DATA / "prepared" / name
        rows = read_csv(metadata / csv_name)
        if len(rows) != expected or len({row["Path"] for row in rows}) != expected:
            raise ValueError(f"Unexpected {name} author membership")
        if name == "OpenI":
            set_link(DATA / "raw/OpenI/iu_xray_kaggle", prepared / "iu_xray_kaggle")
        else:
            for directory in sorted({Path(row["Path"]).parts[1] for row in rows}):
                set_link(DATA / "raw" / name / directory, prepared / directory)
        for source in metadata.iterdir():
            if source.is_file():
                set_link(source, prepared / source.name)
        paths = [DATA / "prepared" / row["Path"] for row in rows]
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            raise ValueError(f"Missing or empty {name} images after reconnecting")
        checks = [
            "All original author image paths exist and are nonempty",
            "Author row count, unique paths and original metadata preserved",
            "Prepared image links resolve to the current raw directory",
        ]
        extra = {}
        if name == "OpenI":
            if len(openi_records) != expected:
                raise ValueError("OpenI decode audit incomplete")
            checks += ["All 7470 PNGs fully decoded", "Per-file and decoded-pixel SHA-256 recorded"]
            extra["audit"] = str(OUT / "image_audit.json")
        else:
            manifest_path = DATA / "raw/ChestXray14/download_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            entries = {entry["path"]: entry for entry in manifest["images"]}
            if set(entries) != {row["Path"] for row in rows}:
                raise ValueError("ChestXray14 download manifest differs from author paths")
            if manifest["test_csv_sha256"] != sha256_file(metadata / csv_name):
                raise ValueError("ChestXray14 download manifest uses a different author CSV")
            for row, path in zip(rows, paths):
                if path.stat().st_size != entries[row["Path"]]["bytes"]:
                    raise ValueError(f"ChestXray14 byte size differs from manifest: {path}")
            checks.append("All file byte sizes match the supplied download manifest")
            extra.update({
                "download_manifest": str(manifest_path),
                "current_audit_limit": "No PNG decoding, CRC or image SHA-256 recomputation in this repair.",
            })
        validation = {
            "dataset": name, "status": "complete", "images": expected,
            "validated_images": expected, "checks": checks,
            "metadata_sha256": {p.name: sha256_file(p) for p in metadata.iterdir() if p.is_file()},
            "completed": time.time(), **extra,
        }
        save_json(prepared / "validation.json", validation)
        validations[name] = validation
    return validations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = DATA / "metadata/OpenI"
    raw = DATA / "raw/OpenI/iu_xray_kaggle"
    rows = read_csv(metadata / "custom.csv")
    image_rows = read_csv(metadata / "openi_multi_label_image.csv")
    reports = read_csv(raw / "indiana_reports.csv")
    projections = read_csv(raw / "indiana_projections.csv")
    report_map = {row["uid"]: row for row in reports}
    projection_map = {row["filename"]: row for row in projections}
    if len(report_map) != len(reports) or len(projection_map) != len(projections):
        raise ValueError("Duplicate report/projection identifiers")
    if len(rows) != 7470 or len(image_rows) != len(rows):
        raise ValueError("Unexpected author OpenI row count")
    if len({row["file_name"] for row in rows}) != len(rows):
        raise ValueError("Duplicate author image filenames")
    samples, uid_rows = [], defaultdict(list)
    for index, (row, image_row) in enumerate(zip(rows, image_rows)):
        uid = row["uid"]
        filename = row["file_name"]
        numeric_uid = uid.removeprefix("CXR")
        if not uid.startswith("CXR") or filename.split("_")[0] != numeric_uid:
            raise ValueError(f"Author UID/filename mismatch: {uid}/{filename}")
        if numeric_uid not in report_map:
            raise ValueError(f"UID has no report: {uid}")
        if Path(image_row["Path"]).name != filename:
            raise ValueError("Author image CSV and custom CSV order differs")
        projection = projection_map.get(filename)
        if projection and projection["uid"] != numeric_uid:
            raise ValueError(f"Projection UID differs for {filename}")
        image = (DATA / "raw" / image_row["Path"]).resolve(strict=True)
        samples.append({
            "id": filename, "image": str(image), "uid": uid,
            "labels": author_labels(row["labels_automatic"]),
            "projection": projection["projection"] if projection else "unknown",
            "author_index": index,
        })
        uid_rows[uid].append(index)
    for uid, indexes in uid_rows.items():
        if len({tuple(samples[index]["labels"]) for index in indexes}) != 1:
            raise ValueError(f"Conflicting author labels within report UID {uid}")

    # Cache permits resuming the CPU audit; changing a file invalidates its row.
    audit_path = OUT / "image_audit.json"
    prior = json.loads(audit_path.read_text()) if audit_path.exists() else []
    cached = {entry["image"]: entry for entry in prior}
    records = [None] * len(samples)
    pending = []
    for index, sample in enumerate(samples):
        path = Path(sample["image"])
        stat = path.stat()
        entry = cached.get(str(path))
        if entry and entry["bytes"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
            records[index] = entry
        else:
            pending.append((index, path))
    print(f"OpenI: {len(samples)} images, {len(uid_rows)} report UIDs; "
          f"decoding/hashing {len(pending)} images with {args.workers} CPU workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inspect_image, path): index for index, path in pending}
        completed = 0
        for future in as_completed(futures):
            records[futures[future]] = future.result()
            completed += 1
            if completed % 250 == 0:
                save_json(audit_path, [entry for entry in records if entry is not None])
                print(f"Decoded and hashed {completed}/{len(pending)}", flush=True)
    save_json(audit_path, records)

    union = UnionFind(uid_rows)
    pixel_groups = defaultdict(list)
    for index, record in enumerate(records):
        pixel_groups[record["pixel_sha256"]].append(index)
    duplicate_groups = []
    conflicting_duplicates = []
    for digest, indexes in pixel_groups.items():
        if len(indexes) < 2:
            continue
        uids = sorted({samples[index]["uid"] for index in indexes})
        for uid in uids[1:]:
            union.union(uids[0], uid)
        entry = {
            "pixel_sha256": digest, "ids": [samples[index]["id"] for index in indexes],
            "uids": uids, "cross_uid": len(uids) > 1,
            "label_conflict": len({tuple(samples[index]["labels"]) for index in indexes}) > 1,
        }
        duplicate_groups.append(entry)
        if entry["label_conflict"]:
            conflicting_duplicates.append(entry)
    reviewed_path = OUT / "reviewed_duplicate_merges.json"
    reviewed_merges = json.loads(reviewed_path.read_text()) if reviewed_path.exists() else []
    image_index = {sample["id"]: index for index, sample in enumerate(samples)}
    for pair in reviewed_merges:
        a, b = image_index[pair["id_a"]], image_index[pair["id_b"]]
        if (records[a]["file_sha256"] != pair["file_sha256_a"]
                or records[b]["file_sha256"] != pair["file_sha256_b"]):
            raise ValueError("Reviewed duplicate merge refers to changed image files")
        if not pair.get("reason"):
            raise ValueError("Reviewed duplicate merge must explain the evidence")
        union.union(samples[a]["uid"], samples[b]["uid"])
    members = defaultdict(list)
    for uid in uid_rows:
        members[union.find(uid)].append(uid)
    group_by_uid, groups = {}, {}
    for uids in members.values():
        uids = sorted(uids)
        group_id = "openi_" + hashlib.sha256("|".join(uids).encode()).hexdigest()[:16]
        indexes = [index for uid in uids for index in uid_rows[uid]]
        labels = np.asarray([samples[index]["labels"] for index in indexes]).max(axis=0).tolist()
        groups[group_id] = {"uids": uids, "indexes": indexes, "labels": labels}
        for uid in uids:
            group_by_uid[uid] = group_id
    assignment, target_group_sizes = split_groups(groups, args.seed)
    for sample in samples:
        sample["group_id"] = group_by_uid[sample["uid"]]
        sample["split"] = assignment[sample["group_id"]]
    for indexes in pixel_groups.values():
        if len({samples[index]["split"] for index in indexes}) != 1:
            raise ValueError("Exact image duplicate crossed partitions")
    near_duplicates = score_near_duplicates(
        candidate_near_duplicates(records, samples), records, samples
    )
    save_json(OUT / "exact_duplicates.json", duplicate_groups)
    save_json(OUT / "near_duplicate_candidates.json", near_duplicates)

    class_names = [DISPLAY.get(name, name) for name in PATHOLOGIES]
    prompt_dict = json.loads((metadata / "openi_multi_label_text.json").read_text())
    prompts = [prompt_dict[str(index)][0] for index in range(len(PATHOLOGIES))]
    if len(prompt_dict) != 18 or any(len(value) != 1 for value in prompt_dict.values()):
        raise ValueError("Unexpected author prompt layout")
    split_stats = {}
    for split in SPLITS:
        selected = [sample for sample in samples if sample["split"] == split]
        labels = np.asarray([sample["labels"] for sample in selected])
        positives = labels.sum(axis=0).tolist()
        split_stats[split] = {
            "images": len(selected), "report_uids": len({sample["uid"] for sample in selected}),
            "groups": len({sample["group_id"] for sample in selected}),
            "positive_images": dict(zip(class_names, positives)),
            "positive_groups": {
                name: len({sample["group_id"] for sample in selected if sample["labels"][column]})
                for column, name in enumerate(class_names)
            },
            "all_zero_images": int((labels.sum(axis=1) == 0).sum()),
            "missing_positive_classes": [name for name, n in zip(class_names, positives) if n == 0],
            "missing_negative_classes": [name for name, n in zip(class_names, positives) if n == len(selected)],
        }
    if any(stats["missing_positive_classes"] or stats["missing_negative_classes"]
           for stats in split_stats.values()):
        raise ValueError(f"Undefined class AUROC in a split: {split_stats}")

    source_paths = [
        metadata / "custom.csv", metadata / "openi_multi_label_image.csv",
        metadata / "openi_multi_label_text.json", raw / "indiana_reports.csv",
        raw / "indiana_projections.csv",
        ROOT / "RadZero/external/CARZero/inference.py",
        Path(__file__).resolve(),
    ]
    if reviewed_path.exists():
        source_paths.append(reviewed_path)
    save_json(OUT / "samples.json", samples)
    dataset = {
        "schema_version": 1, "dataset": "OpenI", "seed": args.seed,
        "class_names": class_names, "label_pathologies": PATHOLOGIES, "prompts": prompts,
        "samples_file": str(OUT / "samples.json"),
        "samples_sha256": sha256_file(OUT / "samples.json"),
        "split_method": "Deterministic iterative multilabel stratification of report-UID groups, "
                        "merging report UIDs connected by exactly identical decoded image pixels",
        "requested_group_ratios": {"train": .8, "val": .1, "test": .1},
        "target_group_sizes": target_group_sizes, "split_stats": split_stats,
        "label_source": "Exact labels_automatic substring/regex mapping from "
                        "external/CARZero/inference.py::tripple_openi_rusult_merge; "
                        "the author's No_Finding column is omitted, yielding 18 classes",
        "all_zero_label_note": "All-zero rows are retained exactly. They mean no matched automatic "
                               "label among these 18 classes, not independently confirmed normality.",
        "label_uncertainty": "Report-level automatically extracted labels are inherited by all "
                             "views; missing mentions are treated as 0 to match the author, "
                             "without proving absence of disease in each image.",
        "patient_independence": {
            "verified": False,
            "reason": "Available CSVs identify reports (uid), images and projections but provide "
                      "no verified longitudinal patient identifier. Grouping reports and exact "
                      "duplicates cannot prove independence between patients across reports.",
            "filename_IM_token_used_as_patient_id": False,
            "png_metadata_keys_observed": sorted({key for record in records
                                                  for key in record["png_metadata_keys"]}),
        },
        "duplicates": {
            "exact_pixel_groups": len(duplicate_groups),
            "cross_uid_exact_groups": sum(entry["cross_uid"] for entry in duplicate_groups),
            "conflicting_label_exact_groups": len(conflicting_duplicates),
            "conflict_handling": "Preserve original labels, keep connected duplicates in one split",
            "reviewed_near_duplicate_merges": len(reviewed_merges),
            "near_candidate_pairs": len(near_duplicates),
            "near_candidate_cross_split_pairs": sum(not entry["same_split"] for entry in near_duplicates),
            "near_candidate_handling": "pHash Hamming <=4 and thumbnail pixel similarity propose "
                                       "review candidates only. Only explicitly recorded reviewed "
                                       "image duplicates are merged; similarity does not establish "
                                       "patient identity.",
        },
        "unknown_projection_images": [sample["id"] for sample in samples if sample["projection"] == "unknown"],
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "verification": {
            "images_exist": len(samples), "images_fully_decoded": len(records),
            "image_audit_sha256": sha256_file(audit_path),
            "author_row_order_preserved": True, "report_and_filename_uids_consistent": True,
            "within_uid_label_conflicts": 0, "cross_split_report_uid_overlap": 0,
            "cross_split_exact_decoded_image_overlap": 0,
        },
        "evaluation_protocol": "OpenI is a supervised adaptation experiment on this new grouped split. "
                               "Original author metadata are unchanged. Other four datasets remain "
                               "external tests and are not used to train or choose hyperparameters.",
    }
    save_json(OUT / "dataset.json", dataset)
    validations = repair_prepared(records)
    save_json(OUT / "prepared_repair.json", validations)
    print(json.dumps({
        "dataset_file": str(OUT / "dataset.json"), "samples_file": str(OUT / "samples.json"),
        "split_stats": split_stats, "duplicates": dataset["duplicates"],
        "prepared_validation": {name: value["status"] for name, value in validations.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
