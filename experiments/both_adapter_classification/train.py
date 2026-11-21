"""Train three seeds of joint, independent pre/post MLPs for one model/dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import (
    load_backend, read_dataset, save, sha256, setup_gpu, release_gpu,
    residual_mlp, freeze_backend, query_hashes, metrics)
from experiments.both_adapter_classification.common import MODELS, both_mlp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=("openi","chexpert","chestxray14"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    # Model launchers use an NFS TMPDIR; multiprocessing sockets belong on the
    # local host. No feature files are copied to the small local filesystem.
    os.environ["TMPDIR"] = "/tmp"
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
    protocol = json.loads((run/"PROTOCOL.json").read_text())
    protocol_hash = sha256(run/"PROTOCOL.json")
    cfg, spec, model_spec = protocol["train"], protocol["datasets"][args.dataset], protocol["model_specs"][args.model]
    dim, rank = model_spec["dim"], model_spec["bottleneck"]
    manifest = json.loads((run/"source_manifest.json").read_text())
    sources = manifest["sources_sha256"]
    for filename, digest in sources.items():
        assert sha256(filename) == digest, "Runtime source changed: " + filename
    assert sha256(spec["path"]) == spec["sha256"]
    meta, rows = read_dataset(spec["path"])
    classes = meta["class_names"]
    assert classes == spec["classes"]
    labels = np.asarray([r["labels"] for r in rows], dtype=np.float32)
    assert np.isin(labels,[0,1]).all()
    split_ids = {s:np.asarray([i for i,r in enumerate(rows) if r["split"]==s],dtype=np.int64) for s in ("train","val","test")}
    cache_specs = protocol["caches"][f"{args.model}/{args.dataset}"]
    records, cache_info = [], []
    for c in cache_specs:
        assert sha256(c["path"]) == c["sha256"], c["path"]
        for filename, expected in c["files"].items():
            stat = Path(filename).stat()
            assert stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], filename
        records.append(json.loads(Path(c["path"]).read_text()))
    token_arrays, token_locations = [], {}
    tokens = None
    if args.model in ("AlphaRAD","RadZero"):
        from experiments.both_adapter_classification.native_backends import BothAlphaRAD, BothRadZero
        assert len(records) == records[0]["shards"]
        aggregation = np.empty_like(labels)
        seen = set()
        for record in records:
            assert record["processed"] == len(record["indices"])
            assert record["dataset_sha256"] == spec["sha256"] and record["samples_sha256"] == spec["samples_sha256"]
            assert sha256(record["head_path"]) == record["head_sha256"] == records[0]["head_sha256"]
            assert record["checkpoint_sha256"] == records[0]["checkpoint_sha256"]
            ids = record["indices"]
            assert not seen.intersection(ids)
            seen.update(ids)
            aggregation[ids] = np.load(record["logits_path"], mmap_mode="r")
            for token_spec in record["token_shards"]:
                array = np.load(token_spec["path"],mmap_mode="r")
                assert list(array.shape) == token_spec["shape"] and array.dtype == np.float32
                for local,index in enumerate(token_spec["indices"]):
                    assert index not in token_locations
                    token_locations[index] = len(token_arrays),local
                token_arrays.append(array)
            cache_info.append(dict(head_path=record["head_path"],head_sha256=record["head_sha256"],
                checkpoint_sha256=record["checkpoint_sha256"],logits_sha256=sha256(record["logits_path"])))
        assert seen == set(range(len(rows))) == set(token_locations)
        head_cfg = json.loads(Path(records[0]["head_path"]).read_text())
        assert head_cfg["class_names"] == classes
        backend = (BothAlphaRAD if args.model == "AlphaRAD" else BothRadZero)(head_cfg).cuda().eval()
        microbatch = 32
    else:
        cache = records[0]
        assert cache["class_names"] == classes
        source_updates, historical_script_updates = {}, {}
        for filename,digest in cache["source_sha256"].items():
            current = sha256(filename)
            if current == digest:
                continue
            if "/baselines/" not in filename:
                historical_script_updates[filename] = dict(original_sha256=digest,current_sha256=current)
                continue
            backup = Path(filename).parent/"placement_source_backup"/Path(filename).name
            if Path(filename).name != "token_mlp_adapter.py" or not backup.exists() or sha256(backup) != digest:
                raise ValueError("Unaudited native cache source change: " + filename)
            source_updates[filename] = dict(original_sha256=digest,current_sha256=current,backup=str(backup))
        for filename,digest in cache["checkpoint_sha256"].items():
            assert sha256(filename) == digest, filename
        for f in cache["files"].values():
            stat = Path(f["path"]).stat()
            assert stat.st_size == f["bytes"] and stat.st_mtime_ns == f["mtime_ns"],f["path"]
        tokens = np.load(cache["files"]["tokens.npy"]["path"],mmap_mode="r")
        aggregation = np.load(cache["files"]["aggregation_logits.npy"]["path"],mmap_mode="r")
        old_backend = load_backend(args.model,"cuda")
        freeze_backend(old_backend)
        old_backend.prepare_classes(classes)
        freeze_backend(old_backend)
        assert query_hashes(old_backend) == cache["query_sha256"], "Frozen query state changed"
        microbatch = int(old_backend.metadata.get("train_microbatch_size",32))
        old_backend.release_image_encoder()
        if args.model == "CARZero":
            from experiments.both_adapter_classification.native_backends import BothCARZero
            backend = BothCARZero(old_backend)
        else:
            from experiments.both_adapter_classification.pooling_backends import wrap_backend
            backend = wrap_backend(args.model,old_backend,pooling="text_guided")
        cache_info.append(dict(path=cache_specs[0]["path"],sha256=cache_specs[0]["sha256"],
            query_sha256=cache["query_sha256"],checkpoint_sha256=cache["checkpoint_sha256"],
            source_updates=source_updates,historical_script_updates=historical_script_updates))
        torch.cuda.empty_cache()
    assert backend.dim == dim
    frozen_modules = freeze_backend(backend)
    assert all(not p.requires_grad for m in frozen_modules for p in m.parameters())

    def token_row(i):
        if tokens is not None:
            return np.array(tokens[i],copy=True)
        array,local = token_locations[int(i)]
        return np.array(token_arrays[array][local],copy=True)

    def token_batch(ids):
        return np.stack([token_row(int(i)) for i in ids])

    def score(x,adapter=None):
        return backend.score_tokens(x,pre_adapter=None if adapter is None else adapter["pre"],
            post_adapter=None if adapter is None else adapter["post"])

    validation_dir = run/"validation"/args.model/args.dataset
    # Cover every cached row without using its label. This verifies that the
    # new split readout still reproduces the original aggregation path.
    started = time.time()
    max_error = 0.
    with torch.no_grad():
        for start in range(0,len(rows),microbatch):
            ids = np.arange(start,min(start+microbatch,len(rows)))
            actual = score(torch.from_numpy(token_batch(ids)).cuda()).cpu().numpy()
            np.testing.assert_allclose(actual,aggregation[ids],atol=3e-4,rtol=3e-4)
            max_error = max(max_error,float(np.max(np.abs(actual-aggregation[ids]))))
            if start % (microbatch*200) == 0:
                print("CACHE_PARITY",args.model,args.dataset,start,len(rows),flush=True)
    save(validation_dir/"frozen_path.json",dict(status="passed",rows=len(rows),max_abs_logit_error=max_error,
        seconds=time.time()-started,cache_provenance=cache_info,cache_manifest=cache_specs))
    # Old fitted pre adapters must still reproduce their saved test logits.
    for reference in protocol["references"]:
        if reference["model"] != args.model or reference["dataset"] != args.dataset or reference["method"] != "pre_mlp":
            continue
        folder = Path(reference["source"])
        for name,digest in reference["files_sha256"].items():
            assert sha256(folder/name) == digest
        pred = np.load(folder/"test_predictions.npz",allow_pickle=False)
        np.testing.assert_array_equal(pred["indices"],split_ids["test"])
        np.testing.assert_array_equal(pred["labels"],labels[split_ids["test"]])
        restored = residual_mlp(dim,rank).cuda().eval()
        restored.load_state_dict(torch.load(folder/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
        locations = [0,len(split_ids["test"])//2,len(split_ids["test"])-1]
        with torch.no_grad():
            actual = backend.score_tokens(torch.from_numpy(token_batch(split_ids["test"][locations])).cuda(),pre_adapter=restored).cpu().numpy()
        np.testing.assert_allclose(actual,pred["logits"][locations],atol=3e-4,rtol=3e-4)
        save(validation_dir/f"reused_pre_seed{reference['seed']}.json",dict(status="passed",source=str(folder),
            files_sha256=reference["files_sha256"],test_row_locations=locations,max_abs_error=float(np.max(np.abs(actual-pred["logits"][locations])))))
        del restored,pred

    targets = torch.from_numpy(labels).cuda()
    train_y = labels[split_ids["train"]]
    weights = np.minimum((len(train_y)-train_y.sum(0))/np.maximum(train_y.sum(0),1),20.)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).cuda())
    class TokenRows(Dataset):
        def __init__(self,ids): self.ids=ids
        def __len__(self): return len(self.ids)
        def __getitem__(self,j):
            i=int(self.ids[j])
            return torch.from_numpy(token_row(i)),i

    for seed in protocol["seeds"]:
        out = run/"results"/args.model/args.dataset/f"seed{seed}"/"both_mlp"
        if (out/"result.json").exists():
            previous = json.loads((out/"result.json").read_text())
            assert previous["status"] == "complete" and previous["protocol_sha256"] == protocol_hash
            continue
        if out.exists():
            raise FileExistsError("Review incomplete fit before retry: " + str(out))
        out.mkdir(parents=True)
        loaders = {s:DataLoader(TokenRows(ids),batch_size=cfg["effective_batch_size"],shuffle=s=="train",num_workers=2,
            pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed)) for s,ids in split_ids.items()}

        def predict(split,adapter):
            values,ids_all=[],[]
            with torch.no_grad():
                for x,ids in loaders[split]:
                    for start in range(0,len(x),microbatch):
                        values.append(score(x[start:start+microbatch].cuda(non_blocking=True),adapter).cpu().numpy())
                    ids_all.extend(ids.tolist())
            assert ids_all == split_ids[split].tolist()
            return np.concatenate(values)

        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        adapter = both_mlp(dim,rank).cuda()
        assert sum(p.numel() for p in adapter.parameters()) == model_spec["trainable_parameters"]
        assert not set(p.data_ptr() for p in adapter["pre"].parameters()).intersection(p.data_ptr() for p in adapter["post"].parameters())
        init_hash = {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in adapter.state_dict().items()}
        optimizer = torch.optim.AdamW(adapter.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
        initial = predict("val",adapter)
        np.testing.assert_allclose(initial,aggregation[split_ids["val"]],atol=3e-4,rtol=3e-4)
        initial_val = metrics(labels[split_ids["val"]],initial,classes)
        gradients = {name:dict(up_weight_nonzero=False,down_weight_nonzero=False,first_up_step=None,first_down_step=None,max_grad=0.) for name in ("pre","post")}
        state = dict(status="running",model=args.model,dataset=args.dataset,seed=seed,method="both_mlp",adapter_position="both",
            pooling="text_guided" if model_spec["aggregation_origin"]=="added" else "native",joint_training=True,independent_adapters=True,
            trainable_parameters=model_spec["trainable_parameters"],
            adapter_architecture=dict(dim=dim,bottleneck=rank,activation="GELU",linear_bias=True,zero_initialized_up=True,copies=2),
            class_names=classes,dataset_sha256=spec["sha256"],samples_sha256=spec["samples_sha256"],counts=spec["counts"],
            protocol_sha256=protocol_hash,config=cfg,gpu=gpu,sources_sha256=sources,source_manifest_sha256=sha256(run/"source_manifest.json"),
            cache_provenance=cache_info,initialization_sha256=init_hash,initial_val=initial_val,
            initial_max_abs_error=float(np.max(np.abs(initial-aggregation[split_ids["val"]]))),
            best_epoch=0,best_val=initial_val,history=[],pos_weight=weights.tolist(),microbatch_size=microbatch,
            gradient_evidence=gradients,test_selection=False,position_description=model_spec["position"],
            software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda))

        def checkpoint():
            path = out/"adapter.pt.tmp"
            torch.save(dict(adapter_state_dict={k:v.detach().cpu() for k,v in adapter.state_dict().items()},
                model=args.model,method="both_mlp",dim=dim,bottleneck=rank,class_names=classes,best_epoch=state["best_epoch"],
                adapter_position="both",joint_training=True,independent_adapters=True,protocol_sha256=protocol_hash),path)
            path.replace(out/"adapter.pt")

        checkpoint();save(out/"training.json",state)
        best,stale,step=initial_val["macro_auroc"],0,0
        started=last=time.time()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,cfg["epochs"]+1):
            adapter.train();loss_sum=0.;seen=0;epoch_started=time.time();order=hashlib.sha256()
            for x,ids in loaders["train"]:
                step+=1;order.update(ids.numpy().astype("<i8").tobytes())
                optimizer.zero_grad(set_to_none=True)
                for start in range(0,len(x),microbatch):
                    batch=x[start:start+microbatch].cuda(non_blocking=True)
                    output=score(batch,adapter)
                    loss=criterion(output,targets[ids[start:start+microbatch]])
                    if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
                    (loss*(len(batch)/len(x))).backward()
                    loss_sum+=float(loss.detach())*len(batch)
                if not all(g["up_weight_nonzero"] and g["down_weight_nonzero"] for g in gradients.values()):
                    for name,branch in adapter.items():
                        g=gradients[name]
                        for layer in ("up","down"):
                            grad=getattr(branch,layer).weight.grad
                            assert grad is not None and torch.isfinite(grad).all(), (name,layer,"missing/nonfinite gradient")
                            magnitude=float(grad.abs().max())
                            g["max_grad"]=max(g["max_grad"],magnitude)
                            if magnitude>0 and not g[layer+"_weight_nonzero"]:
                                g[layer+"_weight_nonzero"]=True;g["first_"+layer+"_step"]=step
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
                optimizer.step();seen+=len(x)
                if time.time()-last>20:
                    save(out/"progress.json",dict(epoch=epoch,processed=seen,total=len(train_y),seconds=time.time()-started))
                    last=time.time()
            adapter.eval();validation=metrics(labels[split_ids["val"]],predict("val",adapter),classes)
            state["history"].append(dict(epoch=epoch,train_bce=loss_sum/seen,val=validation,
                seconds=time.time()-epoch_started,train_order_sha256=order.hexdigest()))
            if validation["macro_auroc"]>best:
                best=validation["macro_auroc"];stale=0;state.update(best_epoch=epoch,best_val=validation);checkpoint()
            else: stale+=1
            state["seconds"]=time.time()-started;save(out/"training.json",state)
            print(f"{args.model} {args.dataset} both_mlp seed{seed} epoch{epoch} val={validation['macro_auroc']:.6f} best={best:.6f} sec={time.time()-epoch_started:.1f}",flush=True)
            if epoch>=cfg["min_epochs"] and stale>=cfg["patience"]:break
        assert all(g["up_weight_nonzero"] and g["down_weight_nonzero"] for g in gradients.values()), "One adapter did not train"
        assert all(p.grad is None and not p.requires_grad for m in frozen_modules for p in m.parameters())
        adapter.load_state_dict(torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
        adapter.eval();values=predict("test",adapter)
        restored=both_mlp(dim,rank).cuda().eval()
        restored.load_state_dict(torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
        locations=[0,len(values)//2,len(values)-1]
        with torch.no_grad():
            check=score(torch.from_numpy(token_batch(split_ids["test"][locations])).cuda(),restored).cpu().numpy()
        np.testing.assert_allclose(check,values[locations],atol=3e-4,rtol=3e-4)
        np.savez(out/"test_predictions.npz",logits=values,labels=labels[split_ids["test"]],indices=split_ids["test"],
            ids=np.array([rows[i]["id"] for i in split_ids["test"]]),groups=np.array([rows[i]["group_id"] for i in split_ids["test"]]))
        state.update(status="complete",test=metrics(labels[split_ids["test"]],values,classes),seconds=time.time()-started,
            frozen_backbone_verified=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_replay=dict(test_row_locations=locations,max_abs_error=float(np.max(np.abs(check-values[locations])))),
            adapter_sha256=sha256(out/"adapter.pt"),test_predictions_sha256=sha256(out/"test_predictions.npz"))
        save(out/"training.json",state)
        save(out/"result.json",{k:v for k,v in state.items() if k!="history"})
        print("COMPLETE",args.model,args.dataset,seed,state["test"]["macro_auroc"],state["test"]["macro_auprc"],flush=True)
        del loaders,adapter,restored,optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
