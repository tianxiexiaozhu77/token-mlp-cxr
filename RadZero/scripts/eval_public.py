"""Evaluate RadZero, optionally with an OpenI-trained adapter, on author test splits."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
DATA = WORKSPACE / "data/radzero/prepared"
CHECKPOINT = REPO / "checkpoints/RadZero"
DATASETS = {"chestxdet10": "ChestXDet10", "siim": "SIIM", "rsna": "RSNA", "openi": "OpenI", "chestxray14": "ChestXray14"}
COUNTS = {"ChestXDet10": 542, "SIIM": 1704, "RSNA": 5337, "OpenI": 7470, "ChestXray14": 22433}
REFERENCE = {
    "ChestXDet10": {"classification.Total AUC": .787, "grounding.mean_pointing_score": .622},
    "SIIM": {"segmentation.auc": .924, "segmentation.dice": .171, "segmentation.pixel_level_auroc": .947},
    "RSNA": {"segmentation.auc": .834, "segmentation.dice": .546},
    "OpenI": {"classification.Total AUC": .847},
    "ChestXray14": {"classification.Total AUC": .804},
}
PAPER = "https://arxiv.org/html/2504.07416v3"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def check_data(names):
    reports, missing = {}, []
    for name in names:
        path = DATA / name / "validation.json"
        state = json.loads(path.read_text()) if path.exists() else {"status": "not_prepared"}
        print(f"{name}: {state.get('status')} (expected {COUNTS[name]} images)", flush=True)
        if state.get("status") != "complete":
            missing.append(name)
            continue
        count = state.get("images", state.get("validated_images", state.get("completed_images")))
        if count != COUNTS[name]:
            raise ValueError(f"{name}: unexpected validated image count {count}")
        hashes = {}
        for original in sorted((WORKSPACE / "data/radzero/metadata" / name).iterdir()):
            if original.is_file():
                actual = DATA / name / original.name
                digest = sha256(actual)
                if digest != sha256(original):
                    raise ValueError(f"Author metadata differs: {actual}")
                hashes[original.name] = digest
        reports[name] = {"validation": state, "metadata_sha256": hashes}
    if missing:
        raise RuntimeError("Datasets are not ready: " + ", ".join(missing) + ". Wait for download/validation to finish, or select one complete dataset.")
    return reports


def reserve_gpu(index):
    if index < 0:
        raise ValueError("GPU index must be a non-negative physical index")
    lock_path = REPO / "runs/gpu_locks" / f"eval_gpu_{index}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    # A just-exited training process can leave an inherited DataLoader fd or a
    # briefly stale NFS lock. Wait for release; never steal another user's GPU.
    for attempt in range(30):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if attempt == 29:
                lock.close()
                raise
            time.sleep(2)
    # fork() inherits flock descriptors even when close-on-exec is set.
    # Workers must not extend the parent's GPU lease beyond its lifetime.
    def close_in_child():
        if not lock.closed:
            lock.close()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=close_in_child)
    query = subprocess.check_output([
        "nvidia-smi", "-i", str(index), "--query-gpu=uuid,name,memory.used", "--format=csv,noheader,nounits"
    ], text=True).strip()
    uuid, name, memory = [part.strip() for part in query.split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    busy = [line for line in processes.splitlines() if line.split(",")[0].strip() == uuid]
    if busy or int(memory) > 512:
        raise RuntimeError(f"Physical GPU {index} is busy ({memory} MiB; {busy}). Select an idle GPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return lock, {"physical_index": index, "uuid": uuid, "name": name, "memory_used_before_mib": int(memory)}


def load_model(device):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = json.loads((CHECKPOINT / "config.json").read_text())
    cfg["vision_config"]["pretrained_name_or_path"] = str(REPO / "dependencies/dinov2-base-xray-224")
    cfg["text_config"]["pretrained_name_or_path"] = str(REPO / "dependencies/all-mpnet-base-v2")
    cfg["text_config"]["pretrained_tokenizer_name_or_path"] = cfg["text_config"]["pretrained_name_or_path"]
    config_cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoConfig"], str(CHECKPOINT), local_files_only=True)
    model_cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoModel"], str(CHECKPOINT), local_files_only=True)
    base = model_cls(config_cls(**cfg)).float().eval()
    base.load_state_dict(load_file(str(CHECKPOINT / "model.safetensors"), device="cpu"), strict=True)
    base.to(device)

    class EvaluationModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @property
        def device(self):
            return self.inner.device

        def compute_logits(self, pixel_values, encoded_key_phrases, **kwargs):
            output = self.inner.compute_logits(pixel_values, encoded_key_phrases, **kwargs)
            # The published loss squeezes singleton batch/class dimensions.
            # Restore only dimensions; all numerical scores remain unchanged.
            shape = (pixel_values.shape[0], encoded_key_phrases[0]["input_ids"].shape[0])
            if output["logits"].ndim == 2 and tuple(output["logits"].shape) != shape:
                raise ValueError(f"Unexpected logit axes: {output['logits'].shape}, expected {shape}")
            output["logits"] = output["logits"].reshape(shape)
            return output

    model = EvaluationModel(base).eval()
    processor = AutoImageProcessor.from_pretrained(str(CHECKPOINT), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), local_files_only=True)
    if processor.size != {"height": 518, "width": 518}:
        raise ValueError("Expected the published 518x518 checkpoint processor")
    return model, processor, tokenizer


def evaluate(name, model, processor, tokenizer, args, output):
    import numpy as np
    import pandas as pd
    from exp.cxr_pt.inference.utils import get_infer_dirs, eval_classification, calculate_similarities, process_class_prompts
    from exp.cxr_pt.inference.grounding_utils import chestXDet10_eval_grounding
    paths = get_infer_dirs(str(DATA))[name]
    result = {}
    if name in {"ChestXDet10", "OpenI", "ChestXray14"}:
        cls_dir = output / "classification"
        cls_dir.mkdir(parents=True)
        image_csv = paths["image_path"]
        test_json = DATA / name / "test.json"
        if args.smoke:
            # Three images + batch size 2 exercise the singleton final-batch fix.
            subset = pd.read_csv(image_csv).iloc[:3]
            image_csv = output / "smoke_images.csv"
            subset.to_csv(image_csv, index=False)
            text = json.loads(Path(paths["text_path"]).read_text())
            tokens = process_class_prompts(text, tokenizer, model)
            scores = calculate_similarities(subset.Path.tolist(), tokens, model, processor, 2, 0, str(DATA))
            if scores.shape != (3, len(text)) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid smoke logits: {scores.shape}")
            np.save(cls_dir / "smoke_scores.npy", scores)
            result["smoke_classification"] = {"images": 3, "logits_shape": list(scores.shape)}
        else:
            result["classification"] = eval_classification(
                model, processor, tokenizer, [name], args.batch_size, args.num_workers,
                str(DATA), str(cls_dir), [str(image_csv)], [paths["text_path"]],
            )[name]
        if name == "ChestXDet10":
            if args.smoke:
                labels = json.loads(test_json.read_text())[:3]
                test_json = output / "smoke_grounding.json"
                save(test_json, labels)
            result["grounding"] = chestXDet10_eval_grounding(
                model, processor, tokenizer, 2 if args.smoke else args.batch_size,
                0 if args.smoke else args.num_workers, str(image_csv), paths["text_path"], str(test_json), str(DATA),
            )
    else:
        from eval_segmentation_local import eval_segmentation_local
        data = pd.read_csv(paths["data_path"])
        if args.smoke:
            if name == "SIIM":
                positives = data.loc[data[" EncodedPixels"].astype(str).str.strip() != "-1", "ImageId"]
                negatives = data.loc[data[" EncodedPixels"].astype(str).str.strip() == "-1", "ImageId"]
                data = data[data.ImageId.isin([positives.iloc[0], negatives.iloc[0]])].copy()
            else:
                positive = data["boxes"].notna()
                data = pd.concat([data[positive].iloc[:1], data[~positive].iloc[:1]]).copy()
        result["segmentation"] = eval_segmentation_local(
            model, processor, tokenizer, data, name, str(DATA), output,
            compute_pixel_level_auroc=(name == "SIIM" and not args.skip_pixel_auc),
        )
    return serializable(result)


def write_summary(output, state):
    save(output / "run.json", state)
    rows = []
    adapted = bool(state.get("adapter"))
    title = "RadZero + OpenI-supervised adapter external evaluation" if adapted else "RadZero published 518px checkpoint evaluation"
    lines = [title, f"Status: {state['status']}", f"Paper: {PAPER}",
             "Main Tables 1/2/4; scores in [0,1]; delta in percentage points.",
             "Published checkpoint seed is unspecified; exact paper equality is not guaranteed.",
             "Dice uses the author's positive-only mean and best TEST-set threshold (0.01 grid).", ""]
    if adapted:
        lines += [
            f"Adapter: {state['adapter']}",
            "Supervised adaptation on OpenI; NOT an unchanged author zero-shot reproduction.",
            "External test labels are used only for evaluation; paper scores are contextual references.",
            "The primary comparison is the frozen baseline on the same test split and metric protocol.", "",
        ]
    if state["smoke"]:
        lines += ["SMOKE TEST ONLY: subset outputs are not full-dataset paper results."]
    else:
        lines.append("Dataset\tMetric\tLocal\tPaper\tDelta_pp")
        for name, values in state["results"].items():
            for key, paper_value in REFERENCE[name].items():
                section, metric = key.split(".", 1)
                value = values.get(section, {}).get(metric)
                row = {"dataset": name, "metric": key, "local": value, "paper": paper_value,
                       "delta_percentage_points": None if value is None else (value - paper_value) * 100}
                rows.append(row)
                if value is None:
                    lines.append(f"{name}\t{key}\tNOT COMPUTED\t{paper_value:.3f}\t--")
                else:
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite metric: {name} {key}")
                    lines.append(f"{name}\t{key}\t{value:.6f}\t{paper_value:.3f}\t{(value-paper_value)*100:+.3f}")
    if state.get("error"):
        lines += ["", "ERROR: " + state["error"]]
    save(output / "summary.json", {"status": state["status"], "smoke": state["smoke"], "paper_url": PAPER,
                                  "protocol": state.get("protocol", "published_checkpoint_zero_shot"),
                                  "adapter": state.get("adapter"), "comparisons": rows})
    (output / "RESULTS.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", type=str.lower, choices=["all", *DATASETS])
    parser.add_argument("--gpu", type=int, default=4, help="Physical NVIDIA GPU index (default: 4)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--adapter", type=Path, help="OpenI-trained adapter.pt; restricted to the four external test datasets")
    parser.add_argument("--check-only", action="store_true", help="Validate readiness; do not load the model or access CUDA")
    parser.add_argument("--skip-pixel-auc", action="store_true", help="Omit the expensive exact SIIM pixel AUROC; mark it NOT COMPUTED")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Small diagnostic subset; never report as paper accuracy")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be >=1 and workers >=0")
    names = list(DATASETS.values()) if args.dataset == "all" else [DATASETS[args.dataset]]
    if args.adapter:
        args.adapter = args.adapter.resolve()
        if not args.adapter.is_file():
            parser.error(f"Adapter checkpoint does not exist: {args.adapter}")
        if "OpenI" in names:
            parser.error("The full author OpenI set overlaps adapter training. Evaluate the custom held-out OpenI split with the adapter training tools; choose one external dataset here.")
    try:
        data_checks = check_data(names)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(2, f"Data check failed: {error}\n")
    if args.check_only:
        print("All selected datasets are ready. No model loaded and no GPU used.")
        return 0
    lock, gpu = None, None
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        try:
            lock, gpu = reserve_gpu(args.gpu)
        except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
            parser.exit(2, f"GPU check failed: {error}\n")
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cpu" if args.cpu else "cuda:0"
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output_base = REPO / ("runs/setup_validation/public_eval_smoke" if args.smoke else "runs/public_eval")
    output = args.output_dir or output_base / f"{stamp}_{args.dataset}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing outputs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started": now(), "smoke": args.smoke, "datasets": names,
             "device": device, "gpu": gpu, "batch_size": args.batch_size, "num_workers": args.num_workers,
             "dtype": "float32", "tf32": False, "torch": torch.__version__, "checkpoint": str(CHECKPOINT),
             "adapter": str(args.adapter) if args.adapter else None,
             "protocol": "openi_supervised_adapter_external_evaluation" if args.adapter else "published_checkpoint_zero_shot",
             "skip_pixel_auc": args.skip_pixel_auc, "data_checks": data_checks, "results": {}}
    write_summary(output, state)
    print(f"Output directory: {output}", flush=True)
    try:
        state["checkpoint_sha256"] = sha256(CHECKPOINT / "model.safetensors")
        model, processor, tokenizer = load_model(device)
        if args.adapter:
            from adapter_model import attach_adapter
            adapter = attach_adapter(model, args.adapter)
            model.eval()
            state["adapter_sha256"] = sha256(args.adapter)
            state["adapter_parameters"] = sum(parameter.numel() for parameter in adapter.parameters())
            write_summary(output, state)
        with torch.inference_mode():
            for name in names:
                folder = output / name
                folder.mkdir()
                print(f"\nEvaluating {name} ...", flush=True)
                result = evaluate(name, model, processor, tokenizer, args, folder)
                save(folder / "result.json", result)
                state["results"][name] = result
                write_summary(output, state)
        state["status"] = "smoke_passed" if args.smoke else "complete_selected_metrics" if args.skip_pixel_auc and "SIIM" in names else "complete"
        state["finished"] = now()
        write_summary(output, state)
        print(f"Finished. Results: {output / 'RESULTS.txt'}", flush=True)
        return 0
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}", ended=now())
        write_summary(output, state)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
