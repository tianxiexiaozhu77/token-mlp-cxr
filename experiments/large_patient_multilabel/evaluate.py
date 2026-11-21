"""Unlock TEST only after every registered fit has a fixed selected checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.large_patient_multilabel.common import *
from experiments.large_patient_multilabel.features import FeatureSet,make_module,forward
from experiments.large_patient_multilabel.train import predict,make_loader


def lock_selections(run):
    p=load_protocol(run);ph=sha256(run/'PROTOCOL.json')
    meta,rows=load_data()
    val_ids=np.asarray([i for i,r in enumerate(rows) if r['split']=='val'])
    val_labels=np.asarray([rows[i]['labels'] for i in val_ids])
    records=[]
    for model,method,seed in expected_jobs():
        out=folder(run,model,method,seed)
        state=json.loads((out/'training.json').read_text())
        fit=json.loads((out/'fit.json').read_text())
        if fit['status']!='complete' or fit['protocol_sha256']!=ph or fit['test_accessed']:
            raise ValueError('Fit incomplete or accessed TEST')
        if (fit['model'],fit['method'],fit['seed'])!=(model,method,seed):
            raise ValueError('Registered job mismatch')
        if selected_epoch(state['initial_val']['macro_auroc'],state['history'])!=fit['best_epoch']:
            raise ValueError('Selected epoch differs from validation history')
        if sha256(out/'adapter.pt')!=fit['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        with np.load(out/'val_predictions.npz') as a:
            np.testing.assert_array_equal(a['indices'],val_ids)
            np.testing.assert_array_equal(a['labels'],val_labels)
            np.testing.assert_array_equal(a['class_names'],meta['class_names'])
            actual=metrics(val_labels,a['logits'],meta['class_names'])
        for name in ('macro_auroc','macro_auprc'):
            if abs(actual[name]-fit['best_val'][name])>1e-12:raise ValueError('VAL metrics mismatch')
        records.append(dict(model=model,method=method,seed=seed,best_epoch=fit['best_epoch'],
            checkpoint=str(out/'adapter.pt'),checkpoint_sha256=fit['checkpoint_sha256'],
            fit_sha256=sha256(out/'fit.json'),history_sha256=sha256(out/'training.json')))
    locked=dict(status='locked',protocol_sha256=ph,selections=records,total_fits=len(records))
    path=run/'SELECTION_LOCK.json'
    if path.exists():
        if json.loads(path.read_text())!=locked:raise ValueError('Existing selection lock differs')
    else:save(path,locked)
    return path


def evaluate_model(args,gpu):
    import torch
    run=args.run_dir.resolve();p=load_protocol(run);ph=sha256(run/'PROTOCOL.json')
    # The controller creates this once; evaluators only verify it.
    lock_path=run/'SELECTION_LOCK.json'
    if not lock_path.is_file():raise ValueError('Missing all-fit selection lock')
    lock=json.loads(lock_path.read_text())
    if lock['total_fits']!=24 or lock['protocol_sha256']!=ph:raise ValueError('Invalid lock')
    for row in lock['selections']:
        if sha256(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Locked checkpoint changed')
    features=FeatureSet(run,args.model)
    ids=features.ids['test']
    labels=np.asarray([features.rows[i]['labels'] for i in ids])
    parity=[]
    with torch.no_grad():
        for split in SPLITS:
            candidates=features.ids[split]
            checks=candidates[[0,len(candidates)//2,-1]]
            tokens=torch.from_numpy(np.stack([features.tokens(i) for i in checks])).cuda()
            contexts=features.head.context(tokens)
            reference=features.contexts[checks]
            torch.testing.assert_close(contexts,reference,atol=3e-5,rtol=3e-5)
            parity.append(dict(split=split,indices=checks.tolist(),max_error=float((contexts-reference).abs().max())))
        # Validate all compact native logits, without computing any model metric here.
        replay=np.concatenate([features.head.score(features.contexts[i:i+128]).cpu().numpy()
            for i in range(0,len(features.rows),128)])
        np.testing.assert_allclose(replay,features.native,atol=3e-5,rtol=3e-5)
    save(run/'evaluation'/f'{args.model}_cache_parity.json',dict(status='passed',sampled_token_context_checks=parity,
        all_native_logits_max_error=float(np.max(np.abs(replay-features.native)))))
    for method in ('original','global_frozen',*METHODS):
        seeds=(42,) if method in ('original','global_frozen') else SEEDS
        loader=make_loader(features,'test',method)
        for seed in seeds:
            out=folder(run,args.model,method,seed);out.mkdir(parents=True,exist_ok=True)
            if (out/'result.json').exists():
                saved=json.loads((out/'result.json').read_text())
                if saved['protocol_sha256']!=ph or saved['selection_lock_sha256']!=sha256(lock_path):
                    raise ValueError('Existing TEST result provenance differs')
                continue
            started=time.time();module=None
            if method in METHODS:
                module=make_module(method,features)
                payload=torch.load(out/'adapter.pt',map_location='cpu',weights_only=True)
                if payload['protocol_sha256']!=ph:raise ValueError('Checkpoint protocol mismatch')
                module.load_state_dict(payload['module_state_dict'])
            values=predict(features,'test',method,module,loader,lock_path)
            if method in METHODS:
                restored=make_module(method,features)
                restored.load_state_dict(torch.load(out/'adapter.pt',map_location='cpu',weights_only=True)['module_state_dict'])
                replay=predict(features,'test',method,restored,loader,lock_path)
                np.testing.assert_allclose(replay,values,atol=1e-6,rtol=1e-6)
                error=float(np.max(np.abs(replay-values)))
                fit=json.loads((out/'fit.json').read_text())
                params=fit['trainable_parameters'];epoch=fit['best_epoch']
                del restored
            else:
                params=0;epoch=0;error=0.
                if method=='original':np.testing.assert_allclose(values,features.native[ids],atol=3e-5,rtol=3e-5)
            np.savez(out/'test_predictions.npz',logits=values,labels=labels,indices=ids,
                ids=np.asarray([features.rows[i]['id'] for i in ids]),
                groups=np.asarray([features.rows[i]['group_id'] for i in ids]),
                patients=np.asarray([features.rows[i]['patient_id'] for i in ids]),class_names=features.meta['class_names'])
            save(out/'result.json',dict(status='complete',model=args.model,method=method,seed=seed,
                protocol_sha256=ph,selection_lock_sha256=sha256(lock_path),trainable_parameters=params,
                best_epoch=epoch,test=metrics(labels,values,features.meta['class_names']),
                full_test_checkpoint_replay_max_error=error,test_used_for_selection=False,
                predictions_sha256=sha256(out/'test_predictions.npz'),seconds=time.time()-started,gpu=gpu))
            print('EVALUATED',args.model,method,seed,flush=True)
        del loader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN)
    parser.add_argument('--lock-only',action='store_true')
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.lock_only:lock_selections(args.run_dir);return
    if args.model is None or args.gpu is None:parser.error('--model and --gpu required')
    lease,gpu=setup_gpu(args.gpu)
    try:evaluate_model(args,gpu)
    finally:release_gpu(lease)


if __name__=='__main__':main()
