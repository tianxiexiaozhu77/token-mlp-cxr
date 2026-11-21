"""Locked RSNA transfer of CheXpert-selected checkpoints; no target fitting."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *
from experiments.baseline_token_classification.common import residual_mlp


def alpha_encoder(source_meta,images,out,batch_size):
    import torch
    import importlib.util
    sys.path.insert(0,str(ROOT/'AlphaRAD/local_setup'));sys.path.insert(0,str(ROOT/'AlphaRAD'))
    from cxrclip.evaluator import Evaluator,_strict_load_published_checkpoint
    from query_refinement import QueryRefinementMatcher,original_queries_from_evaluator
    spec=importlib.util.spec_from_file_location('alpha_transfer_head',ROOT/'AlphaRAD/local_setup/direct_anchor_model.py')
    export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
    checkpoint=ROOT/'AlphaRAD/local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar'
    csv_path=out/'images.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image']);w.writerows([[v] for v in images])
    cfg={'base':{'seed':42},'test':{'checkpoint':[str(checkpoint)]},
        'data_test':{'transfer_images':{'name':'transfer_images','data_type':'image_classification',
            'data_path':str(csv_path),'n_class':len(source_meta['class_names']),'normalize':None}},
        'dataloader':{'test':{'batch_size':batch_size,'num_workers':4,'pin_memory':True,'shuffle':False,'drop_last':False}}}
    evaluator=Evaluator(cfg,[str(checkpoint)],'test')
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    _strict_load_published_checkpoint(evaluator.model,weights['model']);del weights
    evaluator.model.requires_grad_(False).eval()
    queries,prompts=original_queries_from_evaluator(evaluator,source_meta['class_names'])
    matcher=QueryRefinementMatcher(evaluator.model,queries[None],(0.,),use_all_image_tokens=evaluator.use_all_image_tokens)
    config=export.export_head(evaluator.model,queries);config.update(class_names=source_meta['class_names'],prompts=prompts)
    record=json.loads(next((DIRECT/'cache/alpharad/chexpert').glob('cache_*.json')).read_text())
    assert config==json.loads(Path(record['head_path']).read_text()),'Source queries/head changed'
    assert sha256(checkpoint)==record['checkpoint_sha256']
    from experiments.direct_anchor_classification.model import FrozenClassificationHead
    backend=FrozenClassificationHead(config).cuda().eval().requires_grad_(False)
    def batches():
        with torch.no_grad():
            for batch in evaluator.data_loader_dict['transfer_images']:
                vision=matcher.encode_vision(batch['images'])
                yield torch.cat([vision['image_cls_raw_features'][0][:,None],vision['image_patch_raw_features'][0]],1)
    def native(tokens):
        return matcher.match_prepared({'image_cls_raw_features':[tokens[:,0]],'image_patch_raw_features':[tokens[:,1:]]},queries)['logits']
    return backend,batches,native,record


def worker(args,gpu):
    import numpy as np
    import torch
    from experiments.supervised_baseline_classification.components import LinearProbe,ClipGlobalAdapter,global_scores
    run=args.run_dir.resolve();external=run/'external';ep=json.loads((external/'PROTOCOL.json').read_text())
    p=json.loads((run/'PROTOCOL.json').read_text());audit=json.loads((external/'cohort_audit.json').read_text())
    assert audit['status']=='passed' and audit['protocol_sha256']==sha256(external/'PROTOCOL.json')==p['external_protocol_sha256']
    assert sha256(external/'image_hashes.json')==audit['image_hashes_sha256']
    assert sha256(ep['samples_file'])==ep['samples_sha256'];rows=json.loads(Path(ep['samples_file']).read_text())
    assert len(rows)==ep['images'];source_meta,source_rows=read_dataset(DATASETS['chexpert'])
    assert sha256(DATASETS['chexpert'])==p['datasets']['chexpert']['sha256']
    for path,digest in p['source_sha256'].items():assert sha256(path)==digest,'Frozen training source changed'
    out=external/'results'/args.model;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(out/'summary.json')
    target_hashes=json.loads((external/'image_hashes.json').read_text())['target']
    for r,h in zip(rows,target_hashes):
        stat=Path(r['image']).stat();assert r['id']==h['id'] and stat.st_size==h['bytes'] and stat.st_mtime_ns==h['mtime_ns']
    # Re-encode three source TEST images before target inference. Their saved
    # per-method predictions provide an independent check of source preprocessing,
    # frozen queries, checkpoint loading and the new evaluation plumbing.
    test_ids=[i for i,r in enumerate(source_rows) if r['split']=='test'];check_locations=[0,len(test_ids)//2,len(test_ids)-1]
    check_ids=[test_ids[i] for i in check_locations];images=[source_rows[i]['image'] for i in check_ids]+[r['image'] for r in rows]
    batch_size=8 if args.model=='AlphaRAD' else 16
    if args.model=='AlphaRAD':backend,batches,native,cache=alpha_encoder(source_meta,images,out,batch_size)
    else:
        cache=json.loads((BASE/'cache'/args.model/'chexpert/cache.json').read_text())
        for path,digest in cache['checkpoint_sha256'].items():assert sha256(path)==digest
        backend=load_backend(args.model,'cuda');freeze_backend(backend);backend.prepare_classes(source_meta['class_names']);freeze_backend(backend)
        assert query_hashes(backend)==cache['query_sha256']
        def batches():
            for start in range(0,len(images),batch_size):yield backend.encode_images(images[start:start+batch_size])
        native=backend.native_scores_from_tokens
    frozen=freeze_backend(backend)
    modules={};sources={};expected={}
    for method in ep['methods']:
        for seed in ((42,) if method in ('original','global_frozen') else SEEDS):
            key=method+'__seed'+str(seed)
            if method in ('original','pre_mlp','post_mlp'):
                spec=next(r for r in p['references'] if r['model']==args.model and r['dataset']=='chexpert' and r['method']==method and r['seed']==seed)
                folder=Path(spec['source'])
                for f,digest in spec['files_sha256'].items():assert sha256(folder/f)==digest
            else:folder=run/'results'/args.model/'chexpert'/f'seed{seed}'/method
            result=json.loads((folder/'result.json').read_text());assert result['status']=='complete'
            if method in ('linear_probe','clip_global','global_frozen'):assert result['protocol_sha256']==sha256(run/'PROTOCOL.json')
            module=None
            if method not in ('original','global_frozen'):
                saved=torch.load(folder/'adapter.pt',map_location='cpu',weights_only=True)
                if method in ('pre_mlp','post_mlp'):
                    module=residual_mlp(768,64);module.load_state_dict(saved['adapter_state_dict'])
                else:
                    st=saved['module_state_dict']
                    module=LinearProbe(st['feature_mean'],st['feature_std'],len(source_meta['class_names'])) if method=='linear_probe' else ClipGlobalAdapter()
                    module.load_state_dict(st)
                module.cuda().eval().requires_grad_(False)
            modules[key]=(method,seed,module)
            expected[key]=np.load(folder/'test_predictions.npz',allow_pickle=False)['logits'][check_locations]
            sources[key]=dict(folder=str(folder),result_sha256=sha256(folder/'result.json'),trainable_parameters=result['trainable_parameters'],
                checkpoint_sha256=sha256(folder/'adapter.pt') if module is not None else None,best_epoch=result['best_epoch'])
    evaluation_plan=dict(status='locked',model=args.model,protocol_sha256=sha256(external/'PROTOCOL.json'),source_protocol_sha256=sha256(run/'PROTOCOL.json'),
        cohort_audit_sha256=sha256(external/'cohort_audit.json'),sources=sources,source_query_classes=source_meta['class_names'],target_column=ep['source_column'],
        code_sha256={str(f):sha256(f) for f in [Path(__file__),Path(__file__).with_name('components.py')]},gpu=gpu)
    save(out/'evaluation_plan.json',evaluation_plan)
    outputs={key:[] for key in modules};offset=0;started=time.time();native_error=0.
    with torch.no_grad():
        for tokens in batches():
            # NumPy mean duplicates the exact global-feature training reduction.
            glob=tokens[:,0] if args.model=='AlphaRAD' else torch.from_numpy(tokens.cpu().numpy().mean(axis=1)).cuda()
            context=backend.context(tokens) if args.model=='AlphaRAD' else None
            for key,(method,seed,module) in modules.items():
                if method=='linear_probe':values=module(glob)
                elif method in ('clip_global','global_frozen'):values=global_scores(args.model,backend,module(glob) if module is not None else glob)
                elif args.model=='AlphaRAD':
                    values=backend.score(backend.context(module(tokens))) if method=='pre_mlp' else backend.score(context,module)
                elif method=='original':values=backend.native_scores_from_tokens(tokens)
                else:values=backend.score_tokens(tokens,module,adapter_position='pre' if method=='pre_mlp' else 'post')
                assert torch.isfinite(values).all();outputs[key].append(values.cpu().numpy())
                if offset==0:
                    np.testing.assert_allclose(outputs[key][-1][:len(check_ids)],expected[key],atol=3e-4,rtol=3e-4,err_msg=key)
            if offset==0 or offset+len(tokens)==len(images):
                native_values=native(tokens).cpu().numpy();ours=outputs['original__seed42'][-1]
                np.testing.assert_allclose(ours,native_values,atol=3e-4,rtol=3e-4)
                native_error=max(native_error,float(np.abs(ours-native_values).max()))
            offset+=len(tokens)
            if offset%128<batch_size or offset==len(images):
                save(out/'progress.json',dict(status='running',processed=offset,total=len(images),seconds=time.time()-started))
                print(args.model,offset,'/',len(images),'seconds',round(time.time()-started),flush=True)
    assert offset==len(images);checks={};results=[]
    labels=np.asarray([r['labels'] for r in rows],dtype=np.float32)
    for key,(method,seed,module) in modules.items():
        values=np.concatenate(outputs[key]);actual=values[:len(check_ids)]
        np.testing.assert_allclose(actual,expected[key],atol=3e-4,rtol=3e-4,err_msg=key)
        checks[key]=float(np.abs(actual-expected[key]).max());full=values[len(check_ids):];target=full[:,[ep['source_column']]]
        folder=out/f'seed{seed}'/method;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'test_predictions.npz',logits=target,source_logits=full,labels=labels,indices=np.arange(len(rows)),
            ids=[r['id'] for r in rows],groups=[r['group_id'] for r in rows],class_names=['Pneumonia'],source_class_names=source_meta['class_names'])
        result=dict(status='complete',model=args.model,dataset='rsna_transfer',method=method,seed=seed,source=sources[key],
            trainable_parameters=sources[key]['trainable_parameters'],test=metrics(labels,target,['Pneumonia']),
            predictions_sha256=sha256(folder/'test_predictions.npz'),evaluation_plan_sha256=sha256(out/'evaluation_plan.json'))
        save(folder/'result.json',result);results.append(result)
    assert all(not param.requires_grad and param.grad is None for m in frozen for param in m.parameters())
    save(out/'summary.json',dict(status='complete',images=len(rows),patients=ep['patients'],results=results,
        source_reencoding_checks=dict(test_locations=check_locations,max_abs_error=checks),native_readout_max_abs_error=native_error,
        frozen_verified=True,seconds=time.time()-started,evaluation_plan_sha256=sha256(out/'evaluation_plan.json')))
    print('COMPLETE',args.model,'RSNA transfer',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    lock,gpu=setup_gpu(a.gpu)
    try:worker(a,gpu)
    finally:release_gpu(lock)


if __name__=='__main__':main()
