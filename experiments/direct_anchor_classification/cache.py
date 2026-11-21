"""Cache native FP32 tokens and pre-normalization anchors; audit native logits."""
import argparse
import csv
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

from experiment_utils import ROOT, read_dataset, save, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["radzero", "alpharad"], required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--source-cache", type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / "RadZero/scripts"))
    from eval_public import reserve_gpu
    lock, gpu = reserve_gpu(args.gpu)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from model import FrozenClassificationHead
    meta, samples = read_dataset(args.dataset)
    indices = list(range(args.shard, len(samples), args.shards))
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / f"cache_{args.shard:02d}.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status="running", model=args.model, dataset=str(args.dataset),
                 dataset_sha256=sha256(args.dataset), samples_sha256=meta["samples_sha256"],
                 indices=indices, shard=args.shard, shards=args.shards, processed=0,
                 gpu=gpu, dtype="float32", native_max_abs_error=0.)
    save(state_path, state)
    start = time.time()
    source_tokens = []
    if args.model == "radzero":
        from eval_public import load_model, CHECKPOINT
        from adapter_model import encode_prompts, match_tokens
        from direct_anchor_model import export_head
        model, processor, tokenizer = load_model("cuda:0")
        base = model.inner.requires_grad_(False).eval()
        prompts = meta["prompts"]
        if meta["dataset"] == "chestxray14":
            native_prompts = json.loads((ROOT / "data/radzero/prepared/ChestXray14/chestxray14_test_text.json").read_text())
            prompts = [native_prompts[str(i)][0] for i in range(len(meta["class_names"]))]
        elif meta["dataset"] == "chexpert":
            terms = {"No Finding": "No Finding", "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum", "Edema": "Pulmonary Edema", "Consolidation": "Pulmonary Consolidation", "Fracture": "Bone Fracture"}
            prompts = ["There is " + terms.get(c, c) for c in meta["class_names"]]
        queries, _ = encode_prompts(base, tokenizer, prompts)
        config = export_head(base, queries)
        state["checkpoint_sha256"] = sha256(CHECKPOINT / "model.safetensors")
        if args.source_cache:
            arrays, locations = [], {}
            for path in sorted(args.source_cache.glob("shard_*.json")):
                old = json.loads(path.read_text())
                if old["status"] != "complete" or old["processed"] != len(old["indices"]):
                    raise ValueError("Incomplete source cache")
                for key in ("dataset_sha256", "samples_sha256", "checkpoint_sha256"):
                    if old[key] != state[key]:
                        raise ValueError(f"Source cache mismatch: {key}")
                array = np.load(path.with_suffix(".npy"), mmap_mode="r")
                if list(array.shape) != old["shape"] or array.dtype != np.float32:
                    raise ValueError("Source cache shape/dtype mismatch")
                for local, index in enumerate(old["indices"]):
                    if index in locations:
                        raise ValueError("Duplicate source cache member")
                    locations[index] = len(arrays), local
                arrays.append(array)
                source_tokens.append(dict(path=str(path.with_suffix(".npy").resolve()), indices=old["indices"], shape=list(array.shape)))
            if set(locations) != set(range(len(samples))):
                raise ValueError("Source cache coverage differs")

            class Cached(Dataset):
                def __len__(self): return len(indices)
                def __getitem__(self, i):
                    a, j = locations[indices[i]]
                    return torch.from_numpy(np.array(arrays[a][j], copy=True))
            loader = DataLoader(Cached(), batch_size=32, shuffle=False, num_workers=0)
            get_tokens = lambda batch: batch.cuda()
        else:
            from exp.cxr_pt.inference.dataset import InferDataset, collate_fn
            dataset = InferDataset(processor, [samples[i]["image"] for i in indices], "")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                collate_fn=partial(collate_fn, image_processor=processor))
            get_tokens = lambda batch: base.forward_vision_model(batch.cuda())["vision_tokens"]
        native = lambda tokens: match_tokens(base, tokens, queries)["logits"]
    else:
        sys.path.insert(0, str(ROOT / "AlphaRAD/local_setup"))
        sys.path.insert(0, str(ROOT / "AlphaRAD"))
        # Avoid resolving the identically named RadZero export module.
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("alpha_direct_anchor", ROOT / "AlphaRAD/local_setup/direct_anchor_model.py")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
        from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator
        checkpoint = ROOT / "AlphaRAD/local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
        csv_path = args.output / f"images_{args.shard:02d}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image"])
            writer.writerows([[samples[i]["image"]] for i in indices])
        cfg = {"base": {"seed": 42}, "test": {"checkpoint": [str(checkpoint)]},
               "data_test": {"anchor_images": {"name": "anchor_images", "data_type": "image_classification", "data_path": str(csv_path), "n_class": len(meta["class_names"]), "normalize": None}},
               "dataloader": {"test": {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True, "shuffle": False, "drop_last": False}}}
        evaluator = Evaluator(cfg, [str(checkpoint)], "test")
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _strict_load_published_checkpoint(evaluator.model, weights["model"])
        del weights
        evaluator.model.requires_grad_(False).eval()
        queries, prompts = original_queries_from_evaluator(evaluator, meta["class_names"])
        matcher = QueryRefinementMatcher(evaluator.model, queries[None], (0.,), use_all_image_tokens=evaluator.use_all_image_tokens)
        config = module.export_head(evaluator.model, queries)
        loader = evaluator.data_loader_dict["anchor_images"]
        def get_tokens(batch):
            vision = matcher.encode_vision(batch["images"])
            return torch.cat([vision["image_cls_raw_features"][0][:, None], vision["image_patch_raw_features"][0]], 1)
        def native(tokens):
            vision = {"image_cls_raw_features": [tokens[:, 0]], "image_patch_raw_features": [tokens[:, 1:]]}
            return matcher.match_prepared(vision, queries)["logits"]
        state["checkpoint_sha256"] = sha256(checkpoint)
    state["prompts"] = prompts
    config.update(class_names=meta["class_names"], prompts=prompts)
    head_path = args.output / f"head_{args.shard:02d}.json"
    save(head_path, config)
    state["head_path"] = str(head_path.resolve())
    state["head_sha256"] = sha256(head_path)
    head = FrozenClassificationHead(config).cuda().eval()
    contexts = logits = token_storage = None
    try:
        with torch.no_grad():
            for batch_id, batch in enumerate(loader):
                tokens = get_tokens(batch)
                h = head.context(tokens)
                scores = head.score(h)
                if batch_id == 0 or batch_id == len(loader) - 1:
                    reference = native(tokens)
                    error = float((reference - scores).abs().max())
                    state["native_max_abs_error"] = max(state["native_max_abs_error"], error)
                    torch.testing.assert_close(scores, reference, atol=2e-5, rtol=2e-5)
                    print(f"native logit parity: max_abs_error={error:g}", flush=True)
                if contexts is None:
                    context_path = args.output / f"contexts_{args.shard:02d}.npy"
                    logits_path = args.output / f"logits_{args.shard:02d}.npy"
                    contexts = np.lib.format.open_memmap(context_path, mode="w+", dtype=np.float32, shape=(len(indices), *h.shape[1:]))
                    logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(indices), *scores.shape[1:]))
                    state.update(context_path=str(context_path.resolve()), logits_path=str(logits_path.resolve()))
                    if not args.source_cache:
                        token_path = args.output / f"tokens_{args.shard:02d}.npy"
                        token_storage = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.float32, shape=(len(indices), *tokens.shape[1:]))
                        source_tokens = [dict(path=str(token_path.resolve()), indices=indices, shape=list(token_storage.shape))]
                offset, count = state["processed"], len(tokens)
                if not all(torch.isfinite(t).all() for t in (tokens, h, scores)):
                    raise ValueError("Nonfinite cache tensors")
                contexts[offset:offset+count] = h.cpu().numpy()
                logits[offset:offset+count] = scores.cpu().numpy()
                if token_storage is not None:
                    token_storage[offset:offset+count] = tokens.cpu().numpy()
                state.update(processed=offset+count, seconds=time.time()-start)
                if batch_id % 10 == 0:
                    save(state_path, state)
                    print(f"{args.model} shard={args.shard} {state['processed']}/{len(indices)} elapsed={state['seconds']:.1f}s", flush=True)
        if state["processed"] != len(indices):
            raise ValueError("Incomplete output")
        for array in (contexts, logits, token_storage):
            if array is not None:
                array.flush()
        state.update(status="complete", token_shards=source_tokens, seconds=time.time()-start)
        save(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        save(state_path, state)
        raise
    finally:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
