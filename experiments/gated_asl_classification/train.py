"""Fresh four-group pre-adapter fits with matched initialization and sample order."""
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
from experiments.gated_asl_classification.common import MODELS, METHODS
from experiments.gated_asl_classification.components import (
    make_adapter, parameter_count, make_loss, METHOD_SPECS)


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
    assert list(protocol["methods"]) == list(METHODS)
    for method in METHODS:
        assert protocol["method_specs"][method] == METHOD_SPECS[method]
        assert model_spec["trainable_parameters_by_method"][method] == parameter_count(dim, rank, method)
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
    assert all(not m.training for m in frozen_modules)
    assert all(not p.requires_grad for m in frozen_modules for p in m.parameters())

    def token_row(i):
        if tokens is not None:
            return np.array(tokens[i],copy=True)
        array,local = token_locations[int(i)]
        return np.array(token_arrays[array][local],copy=True)

    def token_batch(ids):
        return np.stack([token_row(int(i)) for i in ids])

    def score(x,adapter=None):
        return backend.score_tokens(x, pre_adapter=adapter, post_adapter=None)

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
    # Old pre fits are provenance/behavior checks only. All four study groups,
    # including plain weighted-BCE controls, are freshly trained below.
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
    weights_device = torch.from_numpy(weights).cuda()

    class TokenRows(Dataset):
        def __init__(self, ids): self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, index):
            row = int(self.ids[index])
            return torch.from_numpy(token_row(row)), row

    class GateStatistics:
        """Observe actual token gates without changing model outputs/autograd."""
        def __init__(self):
            self.count = 0
            self.total = torch.zeros((), dtype=torch.float64, device="cuda")
            self.square = torch.zeros_like(self.total)
            self.minimum = torch.full((), float("inf"), device="cuda")
            self.maximum = torch.full((), -float("inf"), device="cuda")
        def hook(self, module, inputs, output):
            with torch.no_grad():
                value = output.detach().sigmoid().float()
                value64 = value.double()
                self.count += value.numel()
                self.total += value64.sum()
                self.square += value64.square().sum()
                self.minimum = torch.minimum(self.minimum, value.min())
                self.maximum = torch.maximum(self.maximum, value.max())
        def result(self):
            if not self.count:
                raise ValueError("No dynamic gates were observed")
            total, square, minimum, maximum = torch.stack((
                self.total, self.square, self.minimum.double(), self.maximum.double())).cpu().tolist()
            mean = total / self.count
            std = 0. if minimum == maximum else float(max(0., square/self.count-mean**2)**.5)
            return dict(count=self.count, mean=mean, std=std,
                        min=minimum, max=maximum, std_definition="population across observed token gates",
                        observation="sigmoid of actual gate linear outputs; no labels used")

    # Audit the shared initial down/up parameters and the overlapping epoch
    # orders across all four groups. Independent loader generators prevent
    # model construction, gate statistics, or evaluation from changing shuffle.
    matching = {int(seed): dict(initialization=None, epoch_orders={}) for seed in protocol["seeds"]}
    shared_state_names = ("down.weight", "down.bias", "up.weight", "up.bias")
    def register_matching(seed, initialization, history):
        record = matching[int(seed)]
        shared = {name: initialization[name] for name in shared_state_names}
        if record["initialization"] is None:
            record["initialization"] = shared
        else:
            assert record["initialization"] == shared, "Shared MLP initialization differs between methods"
        for epoch in history:
            previous = record["epoch_orders"].setdefault(epoch["epoch"], epoch["train_order_sha256"])
            assert previous == epoch["train_order_sha256"], "Training sample order differs between methods"

    for method in protocol["methods"]:
        method_spec = METHOD_SPECS[method]
        gated = method_spec["gated"]
        loss_name = method_spec["loss"]
        gate_config = dict(protocol["gate_config"]) if gated else None
        loss_config = (dict(name="weighted_bce", reduction="mean", pos_weight=weights.tolist(),
                            pos_weight_source="TRAIN negatives/positives capped at20")
                       if loss_name == "weighted_bce" else dict(name="asl", pos_weight=None, **protocol["asl_config"]))
        for seed in protocol["seeds"]:
            out = run/"results"/args.model/args.dataset/f"seed{seed}"/method
            if (out/"result.json").exists():
                previous = json.loads((out/"result.json").read_text())
                assert previous["status"] == "complete" and previous["protocol_sha256"] == protocol_hash
                assert previous["method"] == method and previous["fresh_fit"] is True
                history = json.loads((out/"training.json").read_text())["history"]
                register_matching(seed, previous["initialization_sha256"], history)
                continue
            if out.exists():
                raise FileExistsError("Review incomplete fresh fit before retry: " + str(out))
            out.mkdir(parents=True)
            loaders = {split: DataLoader(TokenRows(ids), batch_size=cfg["effective_batch_size"],
                shuffle=split=="train", num_workers=2, pin_memory=True, persistent_workers=True,
                generator=torch.Generator().manual_seed(seed)) for split,ids in split_ids.items()}

            def predict(split, adapter, observe_gate=True):
                values, ids_all = [], []
                statistics = GateStatistics() if gated and observe_gate else None
                hook = adapter.gate.register_forward_hook(statistics.hook) if statistics is not None else None
                try:
                    with torch.no_grad():
                        for x, ids in loaders[split]:
                            for start in range(0,len(x),microbatch):
                                values.append(score(x[start:start+microbatch].cuda(non_blocking=True),adapter).cpu().numpy())
                            ids_all.extend(ids.tolist())
                finally:
                    if hook is not None:
                        hook.remove()
                assert ids_all == split_ids[split].tolist()
                return np.concatenate(values), statistics.result() if statistics is not None else None

            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            adapter = make_adapter(dim, rank, method).cuda()
            expected_count = model_spec["trainable_parameters_by_method"][method]
            assert sum(p.numel() for p in adapter.parameters()) == expected_count
            init_hash = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                         for name,value in adapter.state_dict().items()}
            register_matching(seed, init_hash, [])
            criterion = make_loss(method, weights_device if loss_name=="weighted_bce" else None,
                                  protocol["asl_config"]).cuda()
            optimizer = torch.optim.AdamW(adapter.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
            adapter.eval()
            initial, initial_gate = predict("val",adapter)
            np.testing.assert_allclose(initial,aggregation[split_ids["val"]],atol=3e-4,rtol=3e-4)
            initial_val = metrics(labels[split_ids["val"]],initial,classes)
            gradient_layers = ("up", "down", "gate") if gated else ("up", "down")
            gradients = {name: dict(nonzero=False,first_nonzero_step=None,max_abs_grad=0.,
                                    first_step_max_abs_grad=None) for name in gradient_layers}
            state = dict(status="running", model=args.model, dataset=args.dataset, seed=seed, method=method,
                method_spec=method_spec, fresh_fit=True, historical_reference_used_for_training=False,
                adapter_position="pre", post_adapter=None,
                pooling="text_guided" if model_spec["aggregation_origin"]=="added" else "native",
                trainable_parameters=expected_count,
                adapter_architecture=dict(dim=dim,bottleneck=rank,activation="GELU",linear_bias=True,
                    zero_initialized_up=True,copies=1,gated=gated,gate_config=gate_config),
                loss_name=loss_name, loss_config=loss_config, gate_config=gate_config,
                class_names=classes,dataset_sha256=spec["sha256"],samples_sha256=spec["samples_sha256"],
                counts=spec["counts"],protocol_sha256=protocol_hash,config=cfg,gpu=gpu,
                sources_sha256=sources,source_manifest_sha256=sha256(run/"source_manifest.json"),
                cache_provenance=cache_info,initialization_sha256=init_hash,
                shared_initialization_sha256={name:init_hash[name] for name in shared_state_names},
                initial_val=initial_val,initial_gate_statistics=initial_gate,
                initial_max_abs_error=float(np.max(np.abs(initial-aggregation[split_ids["val"]]))),
                best_epoch=0,best_val=initial_val,history=[],
                pos_weight=weights.tolist() if loss_name=="weighted_bce" else None,
                train_pos_weight=weights.tolist(),microbatch_size=microbatch,
                effective_batch_size=cfg["effective_batch_size"],gradient_evidence=gradients,
                test_selection=False,selection_metric="VAL macro AUROC; strict improvement; earliest tie; epoch0 eligible",
                position_description=model_spec["position"],
                software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda))

            def checkpoint():
                path = out/"adapter.pt.tmp"
                torch.save(dict(adapter_state_dict={name:value.detach().cpu() for name,value in adapter.state_dict().items()},
                    model=args.model,method=method,dim=dim,bottleneck=rank,class_names=classes,
                    best_epoch=state["best_epoch"],adapter_position="pre",gated=gated,gate_config=gate_config,
                    loss_name=loss_name,loss_config=loss_config,fresh_fit=True,protocol_sha256=protocol_hash),path)
                path.replace(out/"adapter.pt")

            checkpoint(); save(out/"training.json",state)
            save(out/"progress.json",dict(status="running",method=method,epoch=0,processed=0,total=len(train_y),seconds=0.))
            best, stale, step = initial_val["macro_auroc"], 0, 0
            started = last = time.time()
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(1,cfg["epochs"]+1):
                adapter.train(); loss_sum=0.; seen=0; epoch_started=time.time(); order=hashlib.sha256()
                training_gate = GateStatistics() if gated else None
                hook = adapter.gate.register_forward_hook(training_gate.hook) if gated else None
                try:
                    for x,ids in loaders["train"]:
                        step += 1
                        order.update(ids.numpy().astype("<i8").tobytes())
                        optimizer.zero_grad(set_to_none=True)
                        for start in range(0,len(x),microbatch):
                            batch=x[start:start+microbatch].cuda(non_blocking=True)
                            output=score(batch,adapter)
                            loss=criterion(output,targets[ids[start:start+microbatch]])
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite training loss")
                            (loss*(len(batch)/len(x))).backward()
                            loss_sum += float(loss.detach())*len(batch)
                        if step==1 or not all(value["nonzero"] for value in gradients.values()):
                            for name in gradient_layers:
                                grad=getattr(adapter,name).weight.grad
                                assert grad is not None and torch.isfinite(grad).all(), (name,"missing/nonfinite gradient")
                                magnitude=float(grad.abs().max())
                                evidence=gradients[name]
                                evidence["max_abs_grad"]=max(evidence["max_abs_grad"],magnitude)
                                if step==1:
                                    evidence["first_step_max_abs_grad"]=magnitude
                                if magnitude>0 and not evidence["nonzero"]:
                                    evidence.update(nonzero=True,first_nonzero_step=step)
                        torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
                        optimizer.step(); seen+=len(x)
                        if time.time()-last>20:
                            save(out/"progress.json",dict(status="running",method=method,epoch=epoch,processed=seen,
                                total=len(train_y),seconds=time.time()-started))
                            last=time.time()
                finally:
                    if hook is not None:
                        hook.remove()
                adapter.eval()
                validation_predictions, validation_gate = predict("val",adapter)
                validation=metrics(labels[split_ids["val"]],validation_predictions,classes)
                epoch_record=dict(epoch=epoch,train_loss=loss_sum/seen,loss_name=loss_name,val=validation,
                    seconds=time.time()-epoch_started,train_order_sha256=order.hexdigest(),
                    train_gate_statistics=training_gate.result() if training_gate is not None else None,
                    val_gate_statistics=validation_gate)
                state["history"].append(epoch_record)
                register_matching(seed,init_hash,[epoch_record])
                if validation["macro_auroc"]>best:
                    best=validation["macro_auroc"]; stale=0
                    state.update(best_epoch=epoch,best_val=validation); checkpoint()
                else:
                    stale+=1
                state["seconds"]=time.time()-started; save(out/"training.json",state)
                print(f"{args.model} {args.dataset} {method} seed{seed} epoch{epoch} "
                      f"val={validation['macro_auroc']:.6f} best={best:.6f} sec={time.time()-epoch_started:.1f}",flush=True)
                if epoch>=cfg["min_epochs"] and stale>=cfg["patience"]:
                    break
            assert all(value["nonzero"] for value in gradients.values()), "An adapter branch did not train"
            assert gradients["down"]["first_step_max_abs_grad"] == 0., "Zero-up initialization must block first-step down gradient"
            if gated:
                assert gradients["gate"]["first_step_max_abs_grad"] == 0., "Zero-up initialization must block first-step gate gradient"
            assert all(not module.training for module in frozen_modules)
            assert all(p.grad is None and not p.requires_grad for module in frozen_modules for p in module.parameters())
            adapter.load_state_dict(torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
            adapter.eval()
            selected_val_predictions, selected_val_gate = predict("val",adapter)
            selected_val = metrics(labels[split_ids["val"]],selected_val_predictions,classes)
            assert selected_val == state["best_val"], "Saved checkpoint does not reproduce selected VAL metrics"
            values, _ = predict("test",adapter,observe_gate=False)
            restored=make_adapter(dim,rank,method).cuda().eval()
            restored.load_state_dict(torch.load(out/"adapter.pt",map_location="cpu",weights_only=True)["adapter_state_dict"],strict=True)
            replay={}
            for split,predictions in (("val",selected_val_predictions),("test",values)):
                locations=[0,len(predictions)//2,len(predictions)-1]
                with torch.no_grad():
                    check=score(torch.from_numpy(token_batch(split_ids[split][locations])).cuda(),restored).cpu().numpy()
                np.testing.assert_allclose(check,predictions[locations],atol=3e-4,rtol=3e-4)
                replay[split]=dict(row_locations=locations,max_abs_error=float(np.max(np.abs(check-predictions[locations]))))
                filename="validation_predictions.npz" if split=="val" else "test_predictions.npz"
                np.savez(out/filename,logits=predictions,labels=labels[split_ids[split]],indices=split_ids[split],
                    ids=np.array([rows[i]["id"] for i in split_ids[split]]),
                    groups=np.array([rows[i]["group_id"] for i in split_ids[split]]),class_names=np.array(classes))
            state.update(status="complete",test=metrics(labels[split_ids["test"]],values,classes),
                selected_val=selected_val,selected_val_gate_statistics=selected_val_gate,
                seconds=time.time()-started,frozen_backbone_verified=True,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_replay=replay,
                adapter_sha256=sha256(out/"adapter.pt"),test_predictions_sha256=sha256(out/"test_predictions.npz"),
                validation_predictions_sha256=sha256(out/"validation_predictions.npz"),
                matched_initialization_verified=True,matched_overlapping_epoch_orders_verified=True)
            save(out/"training.json",state)
            save(out/"result.json",{key:value for key,value in state.items() if key!="history"})
            save(out/"progress.json",dict(status="complete",method=method,epoch=state["history"][-1]["epoch"],
                processed=len(train_y),total=len(train_y),seconds=state["seconds"],best_epoch=state["best_epoch"]))
            print("COMPLETE",args.model,args.dataset,method,seed,state["test"]["macro_auroc"],state["test"]["macro_auprc"],flush=True)
            del loaders,adapter,restored,optimizer,criterion
            torch.cuda.empty_cache()
    save(validation_dir/"matched_four_groups.json",dict(status="passed",methods=list(protocol["methods"]),
        seeds=protocol["seeds"],shared_initialization_keys=list(shared_state_names),
        matching={str(seed):value for seed,value in matching.items()},
        description="Fresh fits share initial down/up tensors and identical training orders for every overlapping epoch."))


if __name__ == "__main__":
    main()

