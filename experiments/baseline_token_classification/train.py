"""Three-seed supervised token adaptation with frozen native or controlled added aggregation."""
import argparse
import json
from pathlib import Path
import random
import time

from common import (load_backend, read_dataset, save, sha256, setup_gpu, release_gpu,
    residual_mlp, source_hashes, freeze_backend, query_hashes, metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(42,43,44), required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    lock, gpu = setup_gpu(args.gpu)
    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset, DataLoader
        run = args.run_dir.resolve()
        protocol = json.loads((run / "PROTOCOL.json").read_text())
        if protocol.get("query_resource") and sha256(protocol["query_resource"]) != protocol["query_resource_sha256"]:
            raise ValueError("Fixed query resource changed after protocol freeze")
        cfg = protocol["train"]
        cache_path = run / "cache" / args.model / args.dataset / "cache.json"
        cache = json.loads(cache_path.read_text())
        if cache["status"] != "complete": raise ValueError("Cache incomplete")
        if sha256(cache["dataset_path"]) != cache["dataset_sha256"]: raise ValueError("Changed dataset")
        meta, rows = read_dataset(cache["dataset_path"])
        if meta["samples_sha256"] != cache["samples_sha256"] or meta["class_names"] != cache["class_names"]:
            raise ValueError("Changed sample/class provenance")
        for spec in cache["files"].values():
            stat = Path(spec["path"]).stat()
            if stat.st_size != spec["bytes"] or stat.st_mtime_ns != spec["mtime_ns"]:
                raise ValueError("Cached file changed: " + spec["path"])
        for path, digest in cache["source_sha256"].items():
            if "/baselines/" in path and sha256(path) != digest:
                raise ValueError("Backend source changed after cache: " + path)
        for path,digest in cache["checkpoint_sha256"].items():
            if sha256(path)!=digest:
                raise ValueError("Pretrained checkpoint changed after cache: "+path)
        labels = np.asarray([r["labels"] for r in rows], dtype=np.float32)
        split_ids = {s: np.array([i for i,r in enumerate(rows) if r["split"]==s], dtype=np.int64) for s in ("train","val","test")}
        arrays = {key: np.load(spec["path"], mmap_mode="r") for key,spec in cache["files"].items()}
        tokens = arrays["tokens.npy"]
        variant = protocol.get("backend_variants", {}).get(args.model, "native")
        backend = load_backend(args.model, "cuda", variant=variant)
        freeze_backend(backend)
        backend.prepare_classes(meta["class_names"])
        freeze_backend(backend)
        if query_hashes(backend) != cache["query_sha256"]:
            raise ValueError("Frozen text/query state differs from cache")
        if backend.metadata["aggregation_origin"] != cache["backend_metadata"]["aggregation_origin"]:
            raise ValueError("Aggregation changed")
        if hasattr(backend, "release_image_encoder"):
            backend.release_image_encoder()
            torch.cuda.empty_cache()
        microbatch = min(int(backend.metadata.get("train_microbatch_size", 32)), cfg["effective_batch_size"])
        class TokenRows(Dataset):
            def __init__(self, ids): self.ids=ids
            def __len__(self): return len(self.ids)
            def __getitem__(self,j):
                i=int(self.ids[j])
                return torch.from_numpy(np.array(tokens[i], copy=True)), i
        loaders = {split: DataLoader(TokenRows(ids), batch_size=cfg["effective_batch_size"],
                shuffle=split=="train", num_workers=2, pin_memory=True, persistent_workers=True,
                generator=torch.Generator().manual_seed(args.seed)) for split,ids in split_ids.items()}
        targets = torch.from_numpy(labels).cuda()
        train_y=labels[split_ids["train"]]
        weights=np.minimum((len(train_y)-train_y.sum(0))/np.maximum(train_y.sum(0),1),20.)
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).cuda())

        def predict(split, adapter):
            values, indices=[],[]
            with torch.no_grad():
                for x, ids in loaders[split]:
                    for start in range(0,len(x),microbatch):
                        values.append(backend.score_tokens(x[start:start+microbatch].cuda(non_blocking=True),adapter).cpu().numpy())
                    indices.extend(ids.tolist())
            if indices!=split_ids[split].tolist(): raise ValueError("Prediction row reorder")
            return np.concatenate(values)

        provenance=dict(model=args.model,dataset=args.dataset,seed=args.seed,gpu=gpu,
            backend_variant=variant,protocol_sha256=sha256(run / "PROTOCOL.json"),
            class_names=meta["class_names"], counts={s:len(ids) for s,ids in split_ids.items()},
            dataset_path=cache["dataset_path"],dataset_sha256=cache["dataset_sha256"],samples_sha256=cache["samples_sha256"],
            cache_path=str(cache_path),cache_sha256=sha256(cache_path),query_sha256=cache["query_sha256"],
            checkpoint_sha256=cache["checkpoint_sha256"],source_sha256=source_hashes(args.model),
            backend_metadata=backend.metadata, config=cfg,
            metric="macro_AUPRC is sklearn average_precision_score; same as prior Direct-anchor experiments",
            supervision="All adapter fitting uses TRAIN labels; VAL macro AUROC selects; TEST evaluated once after selection")

        def result(out, method, values, extra):
            out.mkdir(parents=True,exist_ok=True)
            if (out/"result.json").exists(): raise FileExistsError(out)
            test_ids=split_ids["test"]
            np.savez(out/"test_predictions.npz",logits=values,labels=labels[test_ids],indices=test_ids,
                ids=np.array([rows[i]["id"] for i in test_ids]),groups=np.array([rows[i]["group_id"] for i in test_ids]))
            final=dict(provenance,method=method,status="complete",test=metrics(labels[test_ids],values,meta["class_names"]),**extra)
            save(out/"result.json",final)
            return final

        if args.seed==42:
            for method, array in [("baseline",arrays["native_logits.npy"])]+(
                [("aggregation_only",arrays["aggregation_logits.npy"])] if backend.metadata["aggregation_origin"]=="added" else []):
                out=run/"results"/args.model/args.dataset/"seed42"/method
                if not (out/"result.json").exists():
                    result(out,method,np.asarray(array[split_ids["test"]]),dict(trainable_parameters=0,best_epoch=0,
                        best_val=metrics(labels[split_ids["val"]],array[split_ids["val"]],meta["class_names"])))
        out=run/"results"/args.model/args.dataset/f"seed{args.seed}"/"token_mlp"
        if out.exists(): raise FileExistsError("Adapter run exists; inspect/archive incomplete run before retry: "+str(out))
        out.mkdir(parents=True)
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        adapter=residual_mlp(backend.dim,protocol["bottleneck"]).cuda()
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
        initial=predict("val",adapter)
        expected=arrays["aggregation_logits.npy"][split_ids["val"]]
        np.testing.assert_allclose(initial,expected,atol=3e-4,rtol=3e-4)
        baseline_val=metrics(labels[split_ids["val"]],initial,meta["class_names"])
        state=dict(provenance,status="running",method="token_mlp",history=[],best_epoch=0,best_val=baseline_val,initial_val=baseline_val,
            initial_max_abs_error=float(np.max(np.abs(initial-expected))),trainable_parameters=sum(p.numel() for p in adapter.parameters()),
            adapter_architecture=dict(dim=backend.dim,bottleneck=protocol["bottleneck"],activation="GELU",linear_bias=True,zero_initialized_up=True),
            microbatch_size=microbatch,effective_batch_size=cfg["effective_batch_size"],pos_weight=weights.tolist())
        def checkpoint():
            torch.save(dict(adapter_state_dict={k:v.detach().cpu() for k,v in adapter.state_dict().items()},
                dim=backend.dim,bottleneck=protocol["bottleneck"],method="token_mlp",model=args.model,
                class_names=meta["class_names"],best_epoch=state["best_epoch"],provenance=provenance),out/"adapter.pt")
        checkpoint();save(out/"training.json",state)
        best,stale=baseline_val["macro_auroc"],0
        started=last=time.time()
        for epoch in range(1,cfg["epochs"]+1):
            adapter.train();loss_sum=0.;seen=0;epoch_started=time.time()
            for x,ids in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                for start in range(0,len(x),microbatch):
                    batch=x[start:start+microbatch].cuda(non_blocking=True)
                    index=ids[start:start+microbatch]
                    output=backend.score_tokens(batch,adapter)
                    loss=criterion(output,targets[index])
                    if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
                    (loss*(len(batch)/len(x))).backward()
                    loss_sum+=float(loss.detach())*len(batch)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg["grad_clip"])
                optimizer.step();seen+=len(x)
                if time.time()-last>20:
                    save(out/"progress.json",dict(status="training",epoch=epoch,processed=seen,total=len(train_y),
                        seconds=time.time()-started,gpu=gpu,best_epoch=state["best_epoch"]))
                    last=time.time()
            adapter.eval();values=predict("val",adapter)
            val=metrics(labels[split_ids["val"]],values,meta["class_names"])
            state["history"].append(dict(epoch=epoch,train_bce=loss_sum/seen,val=val,seconds=time.time()-epoch_started))
            if val["macro_auroc"]>best:
                best=val["macro_auroc"];stale=0
                state.update(best_epoch=epoch,best_val=val);checkpoint()
            else: stale+=1
            state["seconds"]=time.time()-started;save(out/"training.json",state)
            print(f"{args.model} {args.dataset} seed{args.seed} epoch{epoch} val={val['macro_auroc']:.6f} best={best:.6f} sec={time.time()-epoch_started:.1f}",flush=True)
            if epoch>=cfg["min_epochs"] and stale>=cfg["patience"]: break
        payload=torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)
        adapter.load_state_dict(payload["adapter_state_dict"],strict=True);adapter.eval()
        values=predict("test",adapter)
        # Independent module reload on first/middle/last test rows verifies saved checkpoint/prediction correspondence.
        restored=residual_mlp(backend.dim,protocol["bottleneck"]).cuda().eval()
        restored.load_state_dict(torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
        locations=[0,len(values)//2,len(values)-1]
        check_ids=split_ids["test"][locations]
        with torch.no_grad():
            check=backend.score_tokens(torch.from_numpy(np.array(tokens[check_ids],copy=True)).cuda(),restored).cpu().numpy()
        np.testing.assert_allclose(check,values[locations],atol=3e-4,rtol=3e-4)
        validation=dict(status="passed",test_row_locations=locations,indices=check_ids.tolist(),max_abs_error=float(np.max(np.abs(check-values[locations]))))
        save(out/"checkpoint_validation.json",validation)
        final=result(out,"token_mlp",values,dict(trainable_parameters=state["trainable_parameters"],best_epoch=state["best_epoch"],
            best_val=state["best_val"],initial_val=baseline_val,initial_max_abs_error=state["initial_max_abs_error"],
            adapter_architecture=state["adapter_architecture"],microbatch_size=microbatch,
            seconds=time.time()-started,checkpoint_validation=validation))
        state.update(status="complete",test=final["test"],seconds=final["seconds"])
        save(out/"training.json",state)
        print("COMPLETE",args.model,args.dataset,args.seed,final["test"]["macro_auroc"],flush=True)
    finally:
        release_gpu(lock)


if __name__=="__main__":
    main()
