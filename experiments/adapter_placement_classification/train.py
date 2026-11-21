"""Train all new positions/seeds for one model/dataset on a leased GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import (
    load_backend, read_dataset, save, sha256, setup_gpu, release_gpu,
    residual_mlp, freeze_backend, query_hashes, metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("AlphaRAD", "KAD", "MaCo"), required=True)
    p.add_argument("--dataset", choices=("openi", "chexpert", "chestxray14"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    lock, gpu = setup_gpu(args.gpu)
    try:
        run_worker(args, gpu)
    finally:
        release_gpu(lock)


def run_worker(args, gpu):
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    run = args.run_dir.resolve()
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    cfg = protocol["train"]
    spec = protocol["datasets"][args.dataset]
    assert sha256(spec["path"]) == spec["sha256"]
    meta, rows = read_dataset(spec["path"])
    labels = np.asarray([r["labels"] for r in rows], dtype=np.float32)
    split_ids = {s:np.array([i for i,r in enumerate(rows) if r["split"] == s], dtype=np.int64)
                 for s in ("train", "val", "test")}
    classes = meta["class_names"]
    cache_info = []
    tokens = None
    token_arrays, token_locations = [], {}
    feature_sets = {}
    refs = {k:Path(v) for k,v in protocol["reference_roots"].items()}
    if args.model == "AlphaRAD":
        from experiments.adapter_placement_classification.alpharad_backend import AlphaRADPlacementBackend
        cache_dir = refs["direct"] / "cache/alpharad" / args.dataset
        records = [json.loads(x.read_text()) for x in sorted(cache_dir.glob("cache_*.json"))]
        assert records and len(records) == records[0]["shards"]
        seen = set()
        contexts = np.empty((len(rows),len(classes),768), dtype=np.float32)
        native = np.empty_like(labels)
        for record in records:
            assert record["status"] == "complete" and record["processed"] == len(record["indices"])
            assert record["dataset_sha256"] == spec["sha256"] and record["samples_sha256"] == spec["samples_sha256"]
            assert sha256(record["head_path"]) == record["head_sha256"] == records[0]["head_sha256"]
            assert record["checkpoint_sha256"] == records[0]["checkpoint_sha256"]
            ids = record["indices"]
            assert not seen.intersection(ids)
            seen.update(ids)
            contexts[ids] = np.load(record["context_path"], mmap_mode="r")
            native[ids] = np.load(record["logits_path"], mmap_mode="r")
            for token_spec in record["token_shards"]:
                array = np.load(token_spec["path"], mmap_mode="r")
                assert list(array.shape) == token_spec["shape"] and array.dtype == np.float32
                for local, index in enumerate(token_spec["indices"]):
                    assert index not in token_locations
                    token_locations[index] = len(token_arrays), local
                token_arrays.append(array)
            cache_info.append(dict(record_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
                context_path=record["context_path"], context_sha256=sha256(record["context_path"]),
                head_sha256=record["head_sha256"], checkpoint_sha256=record["checkpoint_sha256"]))
        assert seen == set(range(len(rows))) == set(token_locations)
        head_cfg = json.loads(Path(records[0]["head_path"]).read_text())
        assert head_cfg["class_names"] == classes
        backend = AlphaRADPlacementBackend(head_cfg).cuda().eval()
        feature_sets["text_guided"] = contexts
        aggregation = native
        microbatch = 32
    else:
        cache_path = refs["baselines"] / "cache" / args.model / args.dataset / "cache.json"
        cache = json.loads(cache_path.read_text())
        assert cache["status"] == "complete" and cache["dataset_sha256"] == spec["sha256"]
        assert cache["samples_sha256"] == spec["samples_sha256"] and cache["class_names"] == classes
        source_updates, historical_script_updates = {}, {}
        for filename,digest in cache["source_sha256"].items():
            if sha256(filename) != digest:
                # Historical orchestration scripts do not execute when reading
                # an existing cache. Record their drift separately; native
                # backbone sources remain strictly checked as in the old trainer.
                if "/baselines/" not in filename:
                    historical_script_updates[filename] = dict(original_sha256=digest,
                        current_sha256=sha256(filename), role="historical_cache_orchestration")
                    continue
                backup = Path(filename).parent / "placement_source_backup" / Path(filename).name
                if Path(filename).name != "token_mlp_adapter.py" or not backup.exists() or sha256(backup) != digest:
                    raise ValueError("Unaudited cache source change: " + filename)
                source_updates[filename] = dict(original_sha256=digest, current_sha256=sha256(filename), backup=str(backup))
        for filename,digest in cache["checkpoint_sha256"].items():
            assert sha256(filename) == digest, filename
        for f in cache["files"].values():
            stat = Path(f["path"]).stat()
            assert stat.st_size == f["bytes"] and stat.st_mtime_ns == f["mtime_ns"], f["path"]
        tokens = np.load(cache["files"]["tokens.npy"]["path"], mmap_mode="r")
        native = np.load(cache["files"]["native_logits.npy"]["path"], mmap_mode="r")
        aggregation = np.load(cache["files"]["aggregation_logits.npy"]["path"], mmap_mode="r")
        backend = load_backend(args.model, "cuda")
        freeze_backend(backend)
        backend.prepare_classes(classes)
        freeze_backend(backend)
        assert query_hashes(backend) == cache["query_sha256"], "Frozen query state changed"
        microbatch = int(backend.metadata.get("train_microbatch_size",32))
        backend.release_image_encoder()
        torch.cuda.empty_cache()
        cache_info.append(dict(path=str(cache_path), sha256=sha256(cache_path), source_updates=source_updates,
            historical_script_updates=historical_script_updates,
            query_sha256=cache["query_sha256"], checkpoint_sha256=cache["checkpoint_sha256"], files=cache["files"]))
    frozen_modules = freeze_backend(backend)
    assert all(not p.requires_grad for m in frozen_modules for p in m.parameters())

    def token_batch(ids):
        if tokens is not None:
            return np.array(tokens[ids], copy=True)
        return np.stack([token_arrays[token_locations[int(i)][0]][token_locations[int(i)][1]] for i in ids])

    def score_tokens(x, adapter=None, method="pre_mlp"):
        if args.model == "AlphaRAD":
            return backend(x, adapter=adapter, placement="post" if method == "post_mlp" else "pre")
        if args.model == "KAD":
            return backend.score_tokens(x, adapter, adapter_position="post" if method == "post_mlp" else "pre")
        return backend.score_tokens(x, adapter, adapter_position="post" if method in ("post_mlp","native_mean_post") else "pre",
            pooling="native_mean" if method.startswith("native_mean") else "text_guided")

    def score_features(x, adapter=None, method="post_mlp"):
        if args.model == "AlphaRAD":
            return backend.forward_context(x, adapter=adapter)
        if args.model == "KAD":
            return backend.score_features(x, token_adapter=adapter)
        return backend.score_features(x, token_adapter=adapter,
            pooling="native_mean" if method.startswith("native_mean") else "text_guided")

    # Frozen post features are computed once. No labels enter this computation.
    cache_started = time.time()
    if args.model != "AlphaRAD":
        for pooling in (["text_guided", "native_mean"] if args.model == "MaCo" else ["text_guided"]):
            values = []
            with torch.no_grad():
                for start in range(0,len(rows),32):
                    x = torch.from_numpy(token_batch(np.arange(start,min(start+32,len(rows))))).cuda()
                    value = backend.encode_contexts(x) if args.model == "KAD" else backend.pooled_features(x,pooling=pooling)
                    values.append(value.cpu().numpy())
            feature_sets[pooling] = np.concatenate(values)
    cache_checks = {}
    with torch.no_grad():
        for pooling,features in feature_sets.items():
            method = "native_mean_post" if pooling == "native_mean" else "post_mlp"
            values = np.concatenate([score_features(torch.from_numpy(features[s:s+32]).cuda(),method=method).cpu().numpy()
                for s in range(0,len(rows),32)])
            expected = native if pooling == "native_mean" else aggregation
            np.testing.assert_allclose(values,expected,atol=3e-4,rtol=3e-4)
            cache_checks[pooling] = dict(max_abs_logit_error=float(np.max(np.abs(values-expected))),
                shape=list(features.shape), sha256=hashlib.sha256(features.tobytes()).hexdigest())
    cache_seconds = time.time()-cache_started
    check_dir = run / "validation" / args.model / args.dataset
    save(check_dir / "frozen_contexts.json", dict(status="passed", checks=cache_checks, seconds=cache_seconds, sources=cache_info))

    # Reuse existing pre results only after checking membership, arithmetic,
    # original metadata hashes and a saved-checkpoint replay with current code.
    for reused in protocol["reused_pre_runs"]:
        if reused["model"] != args.model or reused["dataset"] != args.dataset:
            continue
        folder = Path(reused["source"])
        for name,digest in reused["files_sha256"].items():
            assert sha256(folder/name) == digest
        old = json.loads((folder / "result.json").read_text())
        pred = np.load(folder / "test_predictions.npz",allow_pickle=False)
        np.testing.assert_array_equal(pred["indices"], split_ids["test"])
        np.testing.assert_array_equal(pred["labels"], labels[split_ids["test"]])
        np.testing.assert_array_equal(pred["ids"], np.array([rows[i]["id"] for i in split_ids["test"]]))
        recalculated = metrics(pred["labels"],pred["logits"],classes)
        for key in ("macro_auroc","macro_auprc"):
            assert abs(recalculated[key]-old["test"][key]) < 1e-12
        restored = residual_mlp(768,64).cuda().eval()
        restored.load_state_dict(torch.load(folder/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
        locations = [0,len(split_ids["test"])//2,len(split_ids["test"])-1]
        ids = split_ids["test"][locations]
        with torch.no_grad():
            actual = score_tokens(torch.from_numpy(token_batch(ids)).cuda(),restored).cpu().numpy()
        np.testing.assert_allclose(actual,pred["logits"][locations],atol=3e-4,rtol=3e-4)
        save(check_dir/f"reused_pre_seed{reused['seed']}.json", dict(status="passed",source=str(folder),
            source_hashes=reused["files_sha256"], recomputed_test=recalculated, test_row_locations=locations,
            checkpoint_replay_max_abs_error=float(np.max(np.abs(actual-pred["logits"][locations])))))
    del restored

    targets = torch.from_numpy(labels).cuda()
    train_y = labels[split_ids["train"]]
    weights = np.minimum((len(train_y)-train_y.sum(0))/np.maximum(train_y.sum(0),1),20.)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).cuda())
    source_paths = list(Path(__file__).parent.glob("*.py")) + [ROOT/"experiments/baseline_token_classification/common.py"]
    if args.model != "AlphaRAD":
        source_paths += list((ROOT/"baselines"/args.model/"local_setup").glob("*.py"))
    else:
        source_paths += [ROOT/"experiments/direct_anchor_classification/model.py",ROOT/"RadZero/scripts/adapter_model.py",
            ROOT/"AlphaRAD/local_setup/token_adapter_model.py"]
    sources = {str(path):sha256(path) for path in source_paths}
    for method in protocol["new_methods"][args.model]:
        for seed in protocol["seeds"]:
            out = run / "results" / args.model / args.dataset / f"seed{seed}" / method
            if (out/"result.json").exists():
                previous = json.loads((out/"result.json").read_text())
                assert previous["status"] == "complete" and previous["protocol_sha256"] == sha256(run/"PROTOCOL.json")
                continue
            if out.exists():
                raise FileExistsError("Review incomplete result before retry: " + str(out))
            out.mkdir(parents=True)
            pooling = "native_mean" if method.startswith("native_mean") else "text_guided"
            pre = method == "native_mean_pre"
            features = None if pre else feature_sets[pooling]
            expected = native if pooling == "native_mean" else aggregation
            class FeatureRows(Dataset):
                def __init__(self,ids): self.ids=ids
                def __len__(self): return len(self.ids)
                def __getitem__(self,j):
                    i=int(self.ids[j])
                    value = np.array(tokens[i],copy=True) if pre else np.array(features[i],copy=True)
                    return torch.from_numpy(value), i
            # Separate identically seeded loader generators match the existing
            # pre-MLP samplers, including worker base_seed consumption.
            loaders = {split:DataLoader(FeatureRows(ids), batch_size=32, shuffle=split=="train",num_workers=2,
                pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed))
                for split,ids in split_ids.items()}
            forward = score_tokens if pre else score_features
            def predict(split,adapter):
                values,ids_all=[],[]
                with torch.no_grad():
                    for x,ids in loaders[split]:
                        for start in range(0,len(x),microbatch):
                            values.append(forward(x[start:start+microbatch].cuda(non_blocking=True),adapter,method).cpu().numpy())
                        ids_all.extend(ids.tolist())
                assert ids_all == split_ids[split].tolist()
                return np.concatenate(values)
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            adapter = residual_mlp(768,64).cuda()
            assert sum(p.numel() for p in adapter.parameters()) == 99136
            initialization_sha256 = {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in adapter.state_dict().items()}
            optimizer = torch.optim.AdamW(adapter.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
            initial = predict("val",adapter)
            np.testing.assert_allclose(initial,expected[split_ids["val"]],atol=3e-4,rtol=3e-4)
            baseline_val = metrics(labels[split_ids["val"]],initial,classes)
            state = dict(status="running",model=args.model,dataset=args.dataset,seed=seed,method=method,
                pooling=pooling,adapter_position="pre" if pre else "post",trainable_parameters=99136,
                adapter_architecture=dict(dim=768,bottleneck=64,activation="GELU",linear_bias=True,zero_initialized_up=True),
                class_names=classes,dataset_sha256=spec["sha256"],samples_sha256=spec["samples_sha256"],
                counts=spec["counts"],protocol_sha256=sha256(run/"PROTOCOL.json"),config=cfg,gpu=gpu,
                sources_sha256=sources,cache_provenance=cache_info,initialization_sha256=initialization_sha256,
                initial_val=baseline_val,initial_max_abs_error=float(np.max(np.abs(initial-expected[split_ids["val"]]))),
                best_epoch=0,best_val=baseline_val,history=[],pos_weight=weights.tolist(),microbatch_size=microbatch,
                context_preparation_seconds=cache_seconds, test_selection=False)
            def checkpoint():
                torch.save(dict(adapter_state_dict={k:v.detach().cpu() for k,v in adapter.state_dict().items()},
                    model=args.model,method=method,dim=768,bottleneck=64,class_names=classes,
                    adapter_position=state["adapter_position"],pooling=pooling,best_epoch=state["best_epoch"],
                    protocol_sha256=state["protocol_sha256"]),out/"adapter.pt")
            checkpoint();save(out/"training.json",state)
            best,stale=baseline_val["macro_auroc"],0
            started=last=time.time()
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(1,cfg["epochs"]+1):
                adapter.train();loss_sum=0.;seen=0;epoch_started=time.time();order=hashlib.sha256()
                for x,ids in loaders["train"]:
                    order.update(ids.numpy().astype("<i8").tobytes())
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0,len(x),microbatch):
                        batch=x[start:start+microbatch].cuda(non_blocking=True)
                        output=forward(batch,adapter,method)
                        loss=criterion(output,targets[ids[start:start+microbatch]])
                        if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
                        (loss*(len(batch)/len(x))).backward()
                        loss_sum+=float(loss.detach())*len(batch)
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg["grad_clip"])
                    optimizer.step();seen+=len(x)
                    if time.time()-last>20:
                        save(out/"progress.json",dict(epoch=epoch,processed=seen,total=len(train_y),seconds=time.time()-started))
                        last=time.time()
                adapter.eval();validation=metrics(labels[split_ids["val"]],predict("val",adapter),classes)
                state["history"].append(dict(epoch=epoch,train_bce=loss_sum/seen,val=validation,
                    seconds=time.time()-epoch_started,train_order_sha256=order.hexdigest()))
                if validation["macro_auroc"]>best:
                    best=validation["macro_auroc"];stale=0
                    state.update(best_epoch=epoch,best_val=validation);checkpoint()
                else: stale+=1
                state["seconds"]=time.time()-started;save(out/"training.json",state)
                print(f"{args.model} {args.dataset} {method} seed{seed} epoch{epoch} val={validation['macro_auroc']:.6f} best={best:.6f} sec={time.time()-epoch_started:.1f}",flush=True)
                if epoch>=cfg["min_epochs"] and stale>=cfg["patience"]: break
            adapter.load_state_dict(torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
            adapter.eval();values=predict("test",adapter)
            restored=residual_mlp(768,64).cuda().eval()
            restored.load_state_dict(torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
            locations=[0,len(values)//2,len(values)-1];check_ids=split_ids["test"][locations]
            with torch.no_grad():
                check=score_tokens(torch.from_numpy(token_batch(check_ids)).cuda(),restored,method).cpu().numpy()
            np.testing.assert_allclose(check,values[locations],atol=3e-4,rtol=3e-4)
            assert all(p.grad is None and not p.requires_grad for m in frozen_modules for p in m.parameters())
            np.savez(out/"test_predictions.npz",logits=values,labels=labels[split_ids["test"]],indices=split_ids["test"],
                ids=np.array([rows[i]["id"] for i in split_ids["test"]]),groups=np.array([rows[i]["group_id"] for i in split_ids["test"]]))
            state.update(status="complete",test=metrics(labels[split_ids["test"]],values,classes),seconds=time.time()-started,
                frozen_backbone_verified=True,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                checkpoint_replay=dict(test_row_locations=locations,max_abs_error=float(np.max(np.abs(check-values[locations])))))
            save(out/"training.json",state)
            save(out/"result.json",{k:v for k,v in state.items() if k!="history"})
            print("COMPLETE",args.model,args.dataset,method,seed,state["test"]["macro_auroc"],state["test"]["macro_auprc"],flush=True)
            del loaders,adapter,restored,optimizer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
