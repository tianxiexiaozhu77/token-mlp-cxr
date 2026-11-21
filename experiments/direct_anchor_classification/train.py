"""Classification adapter comparisons with validation-only model selection."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from experiment_utils import ROOT, read_dataset, save, sha256


def metrics(labels, logits, classes):
    from sklearn.metrics import roc_auc_score, average_precision_score
    per_class = {}
    for i, name in enumerate(classes):
        y = labels[:, i]
        valid = len(np.unique(y)) == 2
        per_class[name] = dict(positives=int(y.sum()), negatives=int(len(y)-y.sum()),
            auroc=float(roc_auc_score(y, logits[:, i])) if valid else None,
            auprc=float(average_precision_score(y, logits[:, i])) if valid else None)
    valid = [v for v in per_class.values() if v["auroc"] is not None]
    return dict(macro_auroc=float(np.mean([v["auroc"] for v in valid])),
                macro_auprc=float(np.mean([v["auprc"] for v in valid])),
                micro_auprc=float(average_precision_score(labels.ravel(), logits.ravel())),
                valid_classes=len(valid), total_classes=len(classes), per_class=per_class)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", default="baseline,bias,direct,existing",
                   help="Comma-separated baseline,bias,direct,existing; token_mlp adds AlphaRAD's pre-aggregation residual MLP")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / "RadZero/scripts"))
    from eval_public import reserve_gpu
    lock, gpu = reserve_gpu(args.gpu)
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    from model import DirectAnchor, FrozenClassificationHead
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    meta, samples = read_dataset(args.dataset)
    labels = np.array([r["labels"] for r in samples], dtype=np.float32)
    split_ids = {s: np.array([i for i, r in enumerate(samples) if r["split"] == s]) for s in ("train", "val", "test")}
    records = [json.loads(p.read_text()) for p in sorted(args.cache.glob("cache_*.json"))]
    if not records or len(records) != records[0]["shards"]:
        raise ValueError("Cache shards missing")
    context = baseline = None
    seen = set()
    token_specs = {}
    for record in records:
        if record["status"] != "complete" or record["processed"] != len(record["indices"]):
            raise ValueError("Incomplete feature cache")
        if record["dataset_sha256"] != sha256(args.dataset) or record["samples_sha256"] != meta["samples_sha256"]:
            raise ValueError("Feature provenance mismatch")
        if record["head_sha256"] != records[0]["head_sha256"] or sha256(record["head_path"]) != record["head_sha256"]:
            raise ValueError("Frozen head configuration differs across shards")
        if record["checkpoint_sha256"] != records[0]["checkpoint_sha256"]:
            raise ValueError("Mixed base checkpoints")
        ids = record["indices"]
        if seen.intersection(ids):
            raise ValueError("Duplicate feature indices")
        seen.update(ids)
        h = np.load(record["context_path"], mmap_mode="r")
        scores = np.load(record["logits_path"], mmap_mode="r")
        if context is None:
            context = np.empty((len(samples), *h.shape[1:]), dtype=np.float32)
            baseline = np.empty(labels.shape, dtype=np.float32)
        context[ids] = h
        baseline[ids] = scores
        for spec in record["token_shards"]:
            token_specs[spec["path"]] = spec
    if seen != set(range(len(samples))):
        raise ValueError("Cache does not exactly cover samples")
    cfg = json.loads(Path(records[0]["head_path"]).read_text())
    if cfg["class_names"] != meta["class_names"]:
        raise ValueError("Class order mismatch")
    head = FrozenClassificationHead(cfg).cuda().eval()
    contexts = torch.from_numpy(context).cuda()
    targets = torch.from_numpy(labels).cuda()
    with torch.no_grad():
        for start in range(0, len(samples), 256):
            actual = head.score(contexts[start:start+256]).cpu().numpy()
            np.testing.assert_allclose(actual, baseline[start:start+256], atol=2e-5, rtol=2e-5)
    arrays, locations = [], {}
    for spec in token_specs.values():
        array = np.load(spec["path"], mmap_mode="r")
        if list(array.shape) != spec["shape"] or array.dtype != np.float32:
            raise ValueError("Token shape/dtype differs")
        for local, index in enumerate(spec["indices"]):
            if index in locations:
                raise ValueError("Duplicated token index")
            locations[index] = len(arrays), local
        arrays.append(array)
    if set(locations) != set(range(len(samples))):
        raise ValueError("Incomplete tokens")

    class Tokens(Dataset):
        def __init__(self, split): self.ids = split_ids[split]
        def __len__(self): return len(self.ids)
        def __getitem__(self, j):
            index = int(self.ids[j]); a, local = locations[index]
            return torch.from_numpy(np.array(arrays[a][local], copy=True)), index

    loaders = {}
    def loader(split, shuffle=False):
        key = split, shuffle
        if key not in loaders:
            loaders[key] = DataLoader(Tokens(split), batch_size=args.batch_size, shuffle=shuffle,
                num_workers=2, pin_memory=True, persistent_workers=True,
                generator=torch.Generator().manual_seed(args.seed))
        return loaders[key]

    def predict(split, adapter=None, token_adapter=None, queries=None):
        with torch.no_grad():
            if token_adapter is not None or queries is not None:
                pred, ids = [], []
                for tokens, index in loader(split):
                    pred.append(head(tokens.cuda(non_blocking=True), queries=queries, token_adapter=token_adapter).cpu().numpy())
                    ids.extend(index.tolist())
                if ids != split_ids[split].tolist():
                    raise ValueError("Prediction row order changed")
                values = np.concatenate(pred)
            else:
                values = np.concatenate([head.score(contexts[index], adapter).cpu().numpy()
                    for index in np.array_split(split_ids[split], max(1, (len(split_ids[split])+255)//256))])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite logits")
        return values

    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("cache.py")]
    if "token_mlp" in args.methods.split(","):
        if cfg["model"] != "alpharad":
            raise ValueError("token_mlp names the AlphaRAD supplement; RadZero uses existing")
        source_paths.extend([ROOT / "AlphaRAD/local_setup/token_adapter_model.py",
                             ROOT / "RadZero/scripts/adapter_model.py"])
    provenance = dict(dataset=str(args.dataset.resolve()), dataset_sha256=sha256(args.dataset),
        samples_sha256=meta["samples_sha256"], checkpoint_sha256=records[0]["checkpoint_sha256"],
        head_sha256=records[0]["head_sha256"], class_names=meta["class_names"], prompts=cfg["prompts"],
        seed=args.seed, gpu=gpu, model=cfg["model"], rank=args.rank,
        counts={s: len(ids) for s, ids in split_ids.items()},
        source_sha256={str(path.resolve()): sha256(path) for path in source_paths},
        protocol="Frozen backbone. Fixed grouped split. Train-only fitting. Validation macro AUROC selects checkpoint, ties retain earlier epoch, epoch zero eligible. Test never selects checkpoints or hyperparameters.")
    train_y = labels[split_ids["train"]]
    weights = np.minimum((len(train_y)-train_y.sum(0))/np.maximum(train_y.sum(0), 1), 20.)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).cuda())
    baseline_val = metrics(labels[split_ids["val"]], baseline[split_ids["val"]], meta["class_names"])
    for method in args.methods.split(","):
        out = args.output / method
        if (out / "result.json").exists():
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        state = dict(provenance, method=method, status="running", history=[],
            config=dict(epochs=args.epochs, min_epochs=args.min_epochs, patience=args.patience,
                        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay),
            pos_weight=weights.tolist(), baseline_val=baseline_val)
        save(out / "training.json", state)
        adapter = token_adapter = selected_queries = None
        if method == "baseline":
            state.update(trainable_parameters=0, best_epoch=0, best_val=baseline_val)
        elif method == "existing" and cfg["model"] == "alpharad":
            sys.path.insert(0, str(ROOT / "AlphaRAD/local_setup"))
            from query_refinement import fit_query_directions
            # Exact class_local construction used by the existing native matcher.
            local = contexts[split_ids["train"]].reshape(-1, len(meta["class_names"]), head.heads, head.dim // head.heads)
            local = F.normalize(F.normalize(local, dim=-1).flatten(-2), dim=-1).cpu()
            fit = fit_query_directions(local, train_y, head.queries.cpu(), etas=(0., .3))
            selected_queries = torch.from_numpy(fit["queries"][1]).cuda()
            validation = predict("val", queries=selected_queries)
            np.savez(out / "query_adapter.npz", **fit)
            state.update(trainable_parameters=0, fitted_query_values=int(selected_queries.numel()),
                best_epoch=None, eta=.3, best_val=metrics(labels[split_ids["val"]], validation, meta["class_names"]),
                existing_adapter="Original fit_query_directions, fixed eta=0.3; fit on all TRAIN rows. No gradient training or test/validation eta selection.")
        else:
            if method == "token_mlp":
                sys.path.insert(0, str(ROOT / "AlphaRAD/local_setup"))
                from token_adapter_model import ResidualTokenAdapter
                token_adapter = ResidualTokenAdapter(dim=head.dim, bottleneck=64).cuda()
                module = token_adapter
                state["adapter_description"] = "Shared 768->64->768 GELU residual MLP on projected CLS/patch tokens before native FLaS head split/L2/aggregation; original queries frozen"
                state["adapter_architecture"] = dict(dim=head.dim, bottleneck=64,
                    activation="GELU", linear_bias=True, zero_initialized_up=True,
                    boundary="projected_raw_tokens_before_FLaS_head_split_and_normalization")
            elif method == "existing":
                from adapter_model import ResidualAdapter
                token_adapter = ResidualAdapter(dim=head.dim, bottleneck=64).cuda()
                module = token_adapter
                state["existing_adapter"] = "Original RadZero ResidualAdapter before VL-CABS, shared 768->64->768 GELU token adapter"
            elif method in ("bias", "direct"):
                adapter = DirectAnchor(len(meta["class_names"]), head.dim, 0 if method == "bias" else args.rank).cuda()
                module = adapter
            else:
                raise ValueError(method)
            state["trainable_parameters"] = sum(p.numel() for p in module.parameters())
            optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            initial = predict("val", adapter, token_adapter)
            np.testing.assert_allclose(initial, baseline[split_ids["val"]], atol=2e-5, rtol=2e-5)
            state.update(initial_max_abs_error=float(np.max(np.abs(initial-baseline[split_ids["val"]]))),
                         best_epoch=0, best_val=baseline_val)
            def checkpoint():
                torch.save(dict(adapter_state_dict={k:v.detach().cpu() for k,v in module.state_dict().items()},
                    method=method, dim=head.dim, classes=meta["class_names"], rank=args.rank if method == "direct" else 0,
                    bottleneck=64 if token_adapter is not None else None,
                    head_config=cfg, provenance=provenance, best_epoch=state["best_epoch"]), out / "adapter.pt")
            checkpoint()
            best, stale = baseline_val["macro_auroc"], 0
            generator = torch.Generator().manual_seed(args.seed)
            for epoch in range(1, args.epochs+1):
                epoch_start = time.time(); losses = 0.; seen_train = 0
                module.train()
                if token_adapter is None:
                    order = split_ids["train"][torch.randperm(len(split_ids["train"]), generator=generator).numpy()]
                    batches = [(None, order[start:start+args.batch_size]) for start in range(0, len(order), args.batch_size)]
                else:
                    batches = loader("train", True)
                for tokens, index in batches:
                    optimizer.zero_grad(set_to_none=True)
                    logits = head.score(contexts[index], adapter) if token_adapter is None else head(tokens.cuda(non_blocking=True), token_adapter=token_adapter)
                    loss = criterion(logits, targets[index])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
                    optimizer.step()
                    losses += float(loss) * len(index); seen_train += len(index)
                module.eval()
                preds = predict("val", adapter, token_adapter)
                validation = metrics(labels[split_ids["val"]], preds, meta["class_names"])
                row = dict(epoch=epoch, train_bce=losses/seen_train, val=validation, seconds=time.time()-epoch_start)
                state["history"].append(row)
                if validation["macro_auroc"] > best:
                    best = validation["macro_auroc"]; stale = 0
                    state.update(best_epoch=epoch, best_val=validation)
                    checkpoint()
                else:
                    stale += 1
                state["seconds"] = time.time()-started
                save(out / "training.json", state)
                print(f"{cfg['model']} {meta['dataset']} {method} seed={args.seed} epoch={epoch} val={validation['macro_auroc']:.6f} best={best:.6f} sec={row['seconds']:.1f}", flush=True)
                if epoch >= args.min_epochs and stale >= args.patience:
                    break
            payload = torch.load(out / "adapter.pt", map_location="cpu", weights_only=True)
            module.load_state_dict(payload["adapter_state_dict"])
            module.eval()
        # Evaluate TEST only after this method's selection has ended.
        preds = baseline[split_ids["test"]] if method == "baseline" else predict("test", adapter, token_adapter, selected_queries)
        test = metrics(labels[split_ids["test"]], preds, meta["class_names"])
        np.savez(out / "test_predictions.npz", logits=preds, labels=labels[split_ids["test"]],
                 indices=split_ids["test"], ids=np.array([samples[i]["id"] for i in split_ids["test"]]),
                 groups=np.array([samples[i]["group_id"] for i in split_ids["test"]]))
        state.update(status="complete", seconds=time.time()-started, test=test)
        save(out / "training.json", state)
        save(out / "result.json", {k:v for k,v in state.items() if k != "history"})
        print(f"COMPLETE {cfg['model']} {meta['dataset']} {method} seed={args.seed} test_AUROC={test['macro_auroc']:.6f}", flush=True)
    # Explicit unlock avoids stale NFS lock state between scheduled workers.
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


if __name__ == "__main__":
    main()
