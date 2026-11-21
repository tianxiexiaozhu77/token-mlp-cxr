"""Native parity, gradient checks, and immutable float32 projected-token caches."""
import argparse
import json
from pathlib import Path
import time

from common import (ROOT, load_backend, read_dataset, save, sha256, setup_gpu,
                    release_gpu, residual_mlp, source_hashes, freeze_backend, query_hashes)


def smoke(backend, paths, class_names):
    import numpy as np
    import torch
    modules = freeze_backend(backend)
    backend.prepare_classes(class_names)
    modules = freeze_backend(backend)
    with torch.no_grad():
        tokens = backend.encode_images(paths).detach().clone().float()
        native = backend.native_scores(paths)
        recovered = backend.native_scores_from_tokens(tokens)
        scores = backend.score_tokens(tokens)
    np.testing.assert_allclose(recovered.cpu(), native.cpu(), atol=3e-4, rtol=3e-4)
    if backend.metadata["aggregation_origin"] == "native":
        np.testing.assert_allclose(scores.cpu(), native.cpu(), atol=3e-4, rtol=3e-4)
    torch.manual_seed(17)
    adapter = residual_mlp(backend.dim).to(tokens.device).eval()
    with torch.no_grad():
        zero = backend.score_tokens(tokens, adapter)
    np.testing.assert_allclose(zero.cpu(), scores.cpu(), atol=2e-5, rtol=2e-5)
    initial = backend.score_tokens(tokens, adapter)
    initial.square().mean().backward()
    first_up = float(adapter.up.weight.grad.abs().sum())
    if first_up == 0:
        raise ValueError("No adapter up-projection gradient at zero initialization")
    with torch.no_grad():
        adapter.up.weight.normal_(std=.003)
    adapter.zero_grad(set_to_none=True)
    nonzero = backend.score_tokens(tokens, adapter)
    nonzero.square().mean().backward()
    gradients = {name: float(p.grad.abs().sum()) if p.grad is not None else None for name,p in adapter.named_parameters()}
    if not all(v is not None and v > 0 for v in gradients.values()):
        raise ValueError("Disconnected/nonfinite adapter gradient: " + str(gradients))
    if not torch.isfinite(nonzero).all() or not torch.isfinite(adapter.down.weight.grad).all():
        raise ValueError("Nonfinite output/gradient")
    frozen_gradients = sum(p.grad is not None for m in modules for p in m.parameters())
    if frozen_gradients:
        raise ValueError("Backbone received gradients")
    report = dict(status="passed", class_names=class_names, token_shape=list(tokens.shape),
        native_reconstruction_max_abs=float((native-recovered).abs().max()),
        zero_adapter_max_abs=float((zero-scores).abs().max()),
        nonzero_effect_max_abs=float((nonzero-scores).abs().max()),
        first_step_up_gradient_l1=first_up, nonzero_gradients_l1=gradients,
        frozen_parameters_with_gradients=frozen_gradients)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    lock, gpu = setup_gpu(args.gpu)
    try:
        import numpy as np
        import torch
        run = args.run_dir.resolve()
        protocol = json.loads((run / "PROTOCOL.json").read_text())
        if protocol.get("query_resource") and sha256(protocol["query_resource"]) != protocol["query_resource_sha256"]:
            raise ValueError("Fixed query resource changed after protocol freeze")
        variant = protocol.get("backend_variants", {}).get(args.model, "native")
        backend = load_backend(args.model, "cuda", variant=variant)
        freeze_backend(backend)
        sources = source_hashes(args.model)
        for source in backend.metadata.get("source_files", []):
            if Path(source).is_file():
                sources[str(Path(source).resolve())] = sha256(source)
        checkpoints = {}
        def collect(value):
            if isinstance(value, str) and Path(value).suffix in (".pt", ".pth", ".bin", ".safetensors", ".ckpt") and Path(value).is_file():
                if value not in checkpoints:
                    checkpoints[value] = sha256(value)
            elif isinstance(value, dict):
                for v in value.values(): collect(v)
            elif isinstance(value, (tuple,list)):
                for v in value: collect(v)
        collect(backend.metadata)
        for dataset in protocol["model_datasets"][args.model]:
            entry = protocol["datasets"][dataset]
            path = Path(entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError("Changed protocol dataset")
            meta, rows = read_dataset(path)
            out = run / "cache" / args.model / dataset
            out.mkdir(parents=True, exist_ok=True)
            done = out / "cache.json"
            if done.exists():
                existing=json.loads(done.read_text())
                if existing["status"] == "complete":
                    if existing["model"]!=args.model or existing["dataset"]!=dataset or existing["dataset_sha256"]!=sha256(path):
                        raise ValueError("Existing cache identity differs")
                    if existing["checkpoint_sha256"]!=checkpoints:
                        raise ValueError("Existing cache checkpoint differs")
                    for file,digest in existing["source_sha256"].items():
                        if "/baselines/" in file and sha256(file)!=digest:
                            raise ValueError("Existing cache backend differs")
                    for spec in existing["files"].values():
                        if sha256(spec["path"])!=spec["sha256"]:
                            raise ValueError("Existing cache content differs")
                    print("REUSE", args.model, dataset, flush=True)
                    continue
                raise FileExistsError(done)
            example = [r["image"] for r in rows if r["split"] == "train"][:2]
            report = smoke(backend, example, meta["class_names"])
            report.update(model=args.model, dataset=dataset, gpu=gpu, backend=backend.metadata,
                          source_sha256=sources)
            save(out / "smoke.json", report)
            print("SMOKE PASS", args.model, dataset, report["token_shape"], flush=True)
            if args.smoke_only:
                continue
            token_file, native_file, aggregation_file = [out / name for name in ("tokens.npy", "native_logits.npy", "aggregation_logits.npy")]
            if any(f.exists() for f in (token_file,native_file,aggregation_file)):
                raise FileExistsError("Partial cache exists; inspect/archive before retry")
            count, classes = len(rows), len(meta["class_names"])
            bs = int(backend.metadata.get("cache_batch_size", 16))
            tokens_array = None
            native_array = np.lib.format.open_memmap(native_file, mode="w+", dtype=np.float32, shape=(count,classes))
            aggregation_array = np.lib.format.open_memmap(aggregation_file, mode="w+", dtype=np.float32, shape=(count,classes))
            started = last = time.time()
            for start in range(0, count, bs):
                batch = rows[start:start+bs]
                with torch.no_grad():
                    tokens = backend.encode_images([r["image"] for r in batch]).float()
                    native = backend.native_scores_from_tokens(tokens)
                    aggregation = native if backend.metadata["aggregation_origin"]=="native" else backend.score_tokens(tokens)
                for value in (tokens,native,aggregation):
                    if not torch.isfinite(value).all():
                        raise ValueError("Nonfinite cache values")
                if tokens_array is None:
                    tokens_array = np.lib.format.open_memmap(token_file, mode="w+", dtype=np.float32,
                        shape=(count,*tokens.shape[1:]))
                tokens_array[start:start+len(batch)] = tokens.cpu().numpy()
                native_array[start:start+len(batch)] = native.cpu().numpy()
                aggregation_array[start:start+len(batch)] = aggregation.cpu().numpy()
                if time.time()-last > 20 or start+len(batch) == count:
                    state = dict(status="caching", model=args.model, dataset=dataset, completed=start+len(batch),
                        total=count, seconds=time.time()-started, gpu=gpu)
                    save(out / "progress.json", state)
                    print(json.dumps(state), flush=True)
                    last = time.time()
            for array in (tokens_array,native_array,aggregation_array):
                array.flush()
            shape = list(tokens_array.shape)
            del tokens_array,native_array,aggregation_array
            file_records = {}
            for file in (token_file,native_file,aggregation_file):
                file_records[file.name] = dict(path=str(file), sha256=sha256(file),
                    bytes=file.stat().st_size, mtime_ns=file.stat().st_mtime_ns)
            record = dict(status="complete", model=args.model, dataset=dataset,
                dataset_path=str(path), dataset_sha256=sha256(path), samples_sha256=meta["samples_sha256"],
                class_names=meta["class_names"], counts=entry["counts"], token_shape=shape,
                files=file_records, backend_metadata=backend.metadata,
                query_sha256=query_hashes(backend), source_sha256=sources,
                checkpoint_sha256=checkpoints, gpu=gpu, seconds=time.time()-started)
            save(done, record)
            print("CACHE COMPLETE", args.model, dataset, shape, flush=True)
        completion_name = "smoke_completion.json" if args.smoke_only else "completion.json"
        save(run / "cache" / args.model / completion_name, dict(status="complete", gpu=gpu,
             model=args.model, smoke_only=args.smoke_only, datasets=protocol["model_datasets"][args.model]))
    finally:
        release_gpu(lock)


if __name__ == "__main__":
    main()
