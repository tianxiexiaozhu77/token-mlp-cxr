"""Fit one registered adapter using TRAIN/VAL only; TEST is a separate stage."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *
from experiments.large_patient_multilabel.features import FeatureSet,make_module,forward


def atomic_torch_save(path, payload):
    import torch
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


class IndexDataset:
    def __init__(self, features, indices, use_tokens):
        self.features,self.indices,self.use_tokens=features,indices,use_tokens

    def __len__(self):return len(self.indices)

    def __getitem__(self,j):
        import torch
        i=int(self.indices[j])
        return (torch.from_numpy(self.features.tokens(i)) if self.use_tokens else torch.empty(0),i)


def make_loader(features, split, method, batch_size=32, shuffle=False, generator=None):
    import torch
    from torch.utils.data import DataLoader,RandomSampler
    dataset=IndexDataset(features,features.ids[split],method=='pre_mlp')
    sampler=RandomSampler(dataset,generator=generator) if shuffle else None
    # Worker seeding must not consume the sampler RNG when resuming a process.
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,
        num_workers=2 if method=='pre_mlp' else 0,pin_memory=method=='pre_mlp',
        persistent_workers=method=='pre_mlp',generator=torch.Generator().manual_seed(20261121))


def predict(features, split, method, module, loader=None, selection_lock=None):
    import torch
    if split=='test' and (selection_lock is None or not Path(selection_lock).is_file()):
        raise ValueError('TEST is unavailable until all selected checkpoints are locked')
    if module is not None:module.eval()
    if loader is None:loader=make_loader(features,split,method)
    values,indices=[],[]
    with torch.no_grad():
        for tokens,idx in loader:
            tokens=tokens.to(features.device,non_blocking=True) if method=='pre_mlp' else None
            values.append(forward(features,idx.to(features.device),module,method,tokens).cpu().numpy())
            indices.extend(idx.tolist())
    if indices!=features.ids[split].tolist():raise ValueError('Prediction order changed')
    return np.concatenate(values)


def training_targets(rows, ids):
    train=np.asarray([rows[i]['labels'] for i in ids['train']],dtype=np.float32)
    val=np.asarray([rows[i]['labels'] for i in ids['val']],dtype=np.float32)
    weights=np.minimum((len(train)-train.sum(0))/np.maximum(train.sum(0),1),20.)
    return train,val,weights


def buffer_digest(head):
    h=hashlib.sha256()
    for name,value in head.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit(args,gpu):
    import torch
    run=args.run_dir.resolve()
    p=load_protocol(run);cfg=p['train'];ph=sha256(run/'PROTOCOL.json')
    if (run/'SELECTION_LOCK.json').exists():raise ValueError('All fitting is already locked')
    out=folder(run,args.model,args.method,args.seed)
    if (out/'fit.json').exists():
        state=json.loads((out/'fit.json').read_text())
        if state['protocol_sha256']!=ph or sha256(out/'adapter.pt')!=state['checkpoint_sha256']:
            raise ValueError('Completed fit differs')
        print('Verified completed fit',out,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    features=FeatureSet(run,args.model)
    frozen_hash=buffer_digest(features.head)
    train_y,val_y,weights=training_targets(features.rows,features.ids)
    targets=torch.full((len(features.rows),len(features.meta['class_names'])),float('nan'),device='cuda')
    targets[features.ids['train']]=torch.from_numpy(train_y).cuda()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    generator=torch.Generator().manual_seed(args.seed)
    module=make_module(args.method,features)
    count=sum(x.numel() for x in module.parameters())
    expected=10766 if args.method=='linear_probe' else 294912 if args.method=='clip_global' else 99136
    if count!=expected:raise ValueError('Unexpected parameter count')
    optimizer=torch.optim.AdamW(module.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).cuda())
    val_loader=make_loader(features,'val',args.method)
    initial=predict(features,'val',args.method,module,val_loader)
    initial_val=metrics(val_y,initial,features.meta['class_names'])
    if args.method in ('pre_mlp','post_mlp'):
        np.testing.assert_allclose(initial,features.native[features.ids['val']],atol=3e-5,rtol=3e-5)
    best=initial_val['macro_auroc'];best_epoch=0;best_state=copy.deepcopy(module.state_dict());stale=0;start_epoch=1
    state=dict(status='running',model=args.model,method=args.method,seed=args.seed,protocol_sha256=ph,
        dataset_sha256=p['dataset_sha256'],class_names=features.meta['class_names'],trainable_parameters=count,
        initial_val=initial_val,config=cfg,pos_weight=weights.tolist(),history=[],gpu=gpu,
        train_indices_sha256=hashlib.sha256(features.ids['train'].tobytes()).hexdigest(),
        val_indices_sha256=hashlib.sha256(features.ids['val'].tobytes()).hexdigest(),test_accessed=False,
        initialization='fresh adapter; no previously fitted checkpoint loaded',frozen_head_sha256=frozen_hash)
    elapsed_before=0.
    if (out/'last.pt').exists():
        restored=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if restored['state']['protocol_sha256']!=ph:raise ValueError('Resume protocol mismatch')
        state=restored['state'];module.load_state_dict(restored['module']);optimizer.load_state_dict(restored['optimizer'])
        best_state={k:v.cuda() for k,v in restored['best_state'].items()}
        best=restored['best'];best_epoch=restored['best_epoch'];stale=restored['stale'];start_epoch=restored['epoch']+1
        generator.set_state(restored['generator']);torch.set_rng_state(restored['torch_rng'])
        torch.cuda.set_rng_state(restored['cuda_rng']);np.random.set_state(restored['numpy_rng']);random.setstate(restored['python_rng'])
        elapsed_before=state.get('seconds',0.)
        print('Resuming same seed at epoch',start_epoch,flush=True)
    loader=make_loader(features,'train',args.method,shuffle=True,generator=generator)
    train_set=set(features.ids['train'].tolist())
    started=time.time()
    save(out/'training.json',state)
    for epoch in range(start_epoch,cfg['epochs']+1):
        if epoch>cfg['min_epochs'] and stale>=cfg['patience']:break
        module.train();loss_sum=0.;order=[]
        for tokens,idx in loader:
            order.extend(idx.tolist())
            if not set(idx.tolist())<=train_set:raise ValueError('Non-TRAIN sample in optimization')
            optimizer.zero_grad(set_to_none=True)
            tokens=tokens.cuda(non_blocking=True) if args.method=='pre_mlp' else None
            idx=idx.cuda()
            logits=forward(features,idx,module,args.method,tokens)
            loss=criterion(logits,targets[idx])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss or non-TRAIN targets')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(),cfg['grad_clip'])
            optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        if len(order)!=len(train_set) or set(order)!=train_set:raise ValueError('TRAIN coverage differs')
        val=metrics(val_y,predict(features,'val',args.method,module,val_loader),features.meta['class_names'])
        state['history'].append(dict(epoch=epoch,train_bce=loss_sum/len(train_y),val=val,
            train_order_sha256=hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest()))
        if val['macro_auroc']>best:
            best=val['macro_auroc'];best_epoch=epoch;best_state=copy.deepcopy(module.state_dict());stale=0
        else:stale+=1
        state.update(best_epoch=best_epoch,seconds=elapsed_before+time.time()-started)
        atomic_torch_save(out/'last.pt',dict(epoch=epoch,module=module.state_dict(),optimizer=optimizer.state_dict(),
            best_state=best_state,best=best,best_epoch=best_epoch,stale=stale,state=state,
            generator=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate()))
        save(out/'training.json',state)
        print(args.model,args.method,args.seed,'epoch',epoch,'val',round(val['macro_auroc'],6),'best',best_epoch,flush=True)
    if selected_epoch(initial_val['macro_auroc'],state['history'])!=best_epoch:
        raise ValueError('Selection differs from full validation history')
    module.load_state_dict(best_state);module.eval()
    atomic_torch_save(out/'adapter.pt',dict(module_state_dict={k:v.cpu() for k,v in best_state.items()},
        method=args.method,model=args.model,seed=args.seed,classes=features.meta['class_names'],dim=768,
        bottleneck=64,best_epoch=best_epoch,protocol_sha256=ph,head_config=features.config))
    validation=predict(features,'val',args.method,module,val_loader)
    restored=make_module(args.method,features)
    restored.load_state_dict(torch.load(out/'adapter.pt',map_location='cpu',weights_only=True)['module_state_dict'])
    replay=predict(features,'val',args.method,restored,val_loader)
    np.testing.assert_allclose(replay,validation,atol=1e-6,rtol=1e-6)
    if buffer_digest(features.head)!=frozen_hash:raise ValueError('Frozen scoring state changed')
    np.savez(out/'val_predictions.npz',logits=validation,labels=val_y,indices=features.ids['val'],
        ids=np.asarray([features.rows[i]['id'] for i in features.ids['val']]),class_names=features.meta['class_names'])
    state.update(status='complete',best_epoch=best_epoch,best_val=metrics(val_y,validation,features.meta['class_names']),
        checkpoint_sha256=sha256(out/'adapter.pt'),validation_replay_max_error=float(np.max(np.abs(replay-validation))),
        frozen_head_verified=True,seconds=elapsed_before+time.time()-started)
    save(out/'training.json',state);save(out/'fit.json',{k:v for k,v in state.items() if k!='history'})
    print('FIT COMPLETE',args.model,args.method,args.seed,'TEST remains locked',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();lock,gpu=setup_gpu(args.gpu)
    try:fit(args,gpu)
    finally:release_gpu(lock)


if __name__=='__main__':main()
