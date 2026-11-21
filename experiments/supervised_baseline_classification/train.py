"""Train global baselines without accessing TEST metrics during selection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def worker(args,gpu):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from experiments.supervised_baseline_classification.features import load_global
    from experiments.supervised_baseline_classification.components import make_module,global_scores
    run=args.run_dir.resolve();p=json.loads((run/'PROTOCOL.json').read_text());cfg=p['train'];ph=sha256(run/'PROTOCOL.json')
    for path,digest in p['source_sha256'].items():
        assert sha256(path)==digest, 'Source changed after protocol: '+path
    backend,features,meta,rows=load_global(args.model,args.dataset,run)
    frozen=freeze_backend(backend)
    x=torch.from_numpy(features).cuda()
    labels=np.asarray([r['labels'] for r in rows],dtype=np.float32);y=torch.from_numpy(labels).cuda()
    ids={s:np.asarray([i for i,r in enumerate(rows) if r['split']==s],dtype=np.int64) for s in ('train','val','test')}
    microbatch=8 if args.model=='KAD' else 32
    train_y=labels[ids['train']];weights=np.minimum((len(train_y)-train_y.sum(0))/np.maximum(train_y.sum(0),1),20.)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).cuda())
    def forward(index,module,method):
        features=x[index]
        if method=='linear_probe':return module(features)
        if module is not None:features=module(features)
        return global_scores(args.model,backend,features)
    def predict(split,module,method):
        with torch.no_grad():
            return np.concatenate([forward(index,module,method).cpu().numpy() for index in
                [ids[split][s:s+microbatch] for s in range(0,len(ids[split]),microbatch)]])
    def save_predictions(folder,split,values):
        indices=ids[split]
        np.savez(folder/(split+'_predictions.npz'),logits=values,labels=labels[indices],indices=indices,
            ids=np.asarray([rows[i]['id'] for i in indices]),groups=np.asarray([rows[i]['group_id'] for i in indices]),
            class_names=np.asarray(meta['class_names']))
    nofit=run/'results'/args.model/args.dataset/'seed42/global_frozen'
    if not (nofit/'result.json').exists():
        nofit.mkdir(parents=True,exist_ok=True);values=predict('test',None,'global_frozen')
        save_predictions(nofit,'test',values)
        save(nofit/'result.json',dict(status='complete',model=args.model,dataset=args.dataset,method='global_frozen',seed=42,
            trainable_parameters=0,best_epoch=0,protocol_sha256=ph,class_names=meta['class_names'],test=metrics(labels[ids['test']],values,meta['class_names'])))
    for method in METHODS:
        for seed in SEEDS:
            out=run/'results'/args.model/args.dataset/f'seed{seed}'/method
            if (out/'result.json').exists():
                previous=json.loads((out/'result.json').read_text());assert previous['status']=='complete' and previous['protocol_sha256']==ph
                continue
            if out.exists():raise FileExistsError('Incomplete fit needs explicit recovery: '+str(out))
            out.mkdir(parents=True)
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            module=make_module(method,x,ids['train'],len(meta['class_names'])).cuda()
            count=sum(v.numel() for v in module.parameters())
            expected=769*len(meta['class_names']) if method=='linear_probe' else 294912
            assert count==expected
            loader=DataLoader(TensorDataset(torch.from_numpy(ids['train'])),batch_size=32,shuffle=True,num_workers=2,
                persistent_workers=True,generator=torch.Generator().manual_seed(seed))
            optimizer=torch.optim.AdamW(module.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            initial=predict('val',module,method);val=metrics(labels[ids['val']],initial,meta['class_names'])
            best=val['macro_auroc'];best_epoch=0;best_state=copy.deepcopy(module.state_dict());stale=0
            state=dict(status='running',model=args.model,dataset=args.dataset,method=method,seed=seed,
                protocol_sha256=ph,dataset_sha256=p['datasets'][args.dataset]['sha256'],samples_sha256=meta['samples_sha256'],
                class_names=meta['class_names'],config=cfg,trainable_parameters=count,initial_val=val,history=[],pos_weight=weights.tolist(),
                counts={s:len(v) for s,v in ids.items()},gpu=gpu,source_sha256=p['source_sha256'],test_used_for_selection=False,
                feature_cache_sha256=sha256(run/'features'/args.model/args.dataset/'cache.json'))
            started=time.time();save(out/'training.json',state)
            for epoch in range(1,cfg['epochs']+1):
                module.train();loss_sum=0.;order=hashlib.sha256()
                for (indices,) in loader:
                    order.update(indices.numpy().astype('<i8').tobytes());optimizer.zero_grad(set_to_none=True)
                    for start in range(0,len(indices),microbatch):
                        sub=indices[start:start+microbatch].cuda();loss=criterion(forward(sub,module,method),y[sub])
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                        (loss*len(sub)/len(indices)).backward();loss_sum+=float(loss.detach())*len(sub)
                    torch.nn.utils.clip_grad_norm_(module.parameters(),cfg['grad_clip']);optimizer.step()
                module.eval();val=metrics(labels[ids['val']],predict('val',module,method),meta['class_names'])
                state['history'].append(dict(epoch=epoch,train_bce=loss_sum/len(train_y),val=val,train_order_sha256=order.hexdigest()))
                if val['macro_auroc']>best:
                    best=val['macro_auroc'];best_epoch=epoch;best_state=copy.deepcopy(module.state_dict());stale=0
                else:stale+=1
                state.update(best_epoch=best_epoch,seconds=time.time()-started);save(out/'training.json',state)
                print(args.model,args.dataset,method,seed,epoch,'val',round(val['macro_auroc'],6),'best',best_epoch,flush=True)
                if epoch>=cfg['min_epochs'] and stale>=cfg['patience']:break
            module.load_state_dict(best_state);module.eval()
            torch.save(dict(module_state_dict={k:v.cpu() for k,v in best_state.items()},method=method,model=args.model,
                classes=meta['class_names'],dim=768,best_epoch=best_epoch,protocol_sha256=ph),out/'adapter.pt')
            val_values=predict('val',module,method);test_values=predict('test',module,method)
            save_predictions(out,'val',val_values);save_predictions(out,'test',test_values)
            # Full TEST replay from the serialized checkpoint; no re-selection.
            restored=make_module(method,x,ids['train'],len(meta['class_names'])).cuda().eval()
            restored.load_state_dict(torch.load(out/'adapter.pt',map_location='cpu',weights_only=True)['module_state_dict'])
            replay=predict('test',restored,method);np.testing.assert_allclose(replay,test_values,atol=1e-6,rtol=1e-6)
            assert all(not param.requires_grad and param.grad is None for m in frozen for param in m.parameters())
            state.update(status='complete',best_epoch=best_epoch,best_val=metrics(labels[ids['val']],val_values,meta['class_names']),
                test=metrics(labels[ids['test']],test_values,meta['class_names']),seconds=time.time()-started,
                full_checkpoint_replay_max_error=float(np.max(np.abs(replay-test_values))),frozen_backbone_verified=True)
            save(out/'training.json',state);save(out/'result.json',{k:v for k,v in state.items() if k!='history'})
            print('COMPLETE',args.model,args.dataset,method,seed,flush=True)
            del loader,optimizer,module,restored;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    lock,gpu=setup_gpu(args.gpu)
    try:worker(args,gpu)
    finally:release_gpu(lock)


if __name__=='__main__':main()
