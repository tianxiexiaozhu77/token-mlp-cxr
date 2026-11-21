"""Locked RSNA transfer of CheXpert-selected checkpoints; no target fitting."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.radzero_supervised_classification.common import *
from experiments.baseline_token_classification.common import residual_mlp


def radzero_encoder(source_meta, images, out, batch_size):
    import torch
    from functools import partial
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(ROOT/'RadZero/scripts'))
    from eval_public import load_model, CHECKPOINT
    from adapter_model import encode_prompts, match_tokens
    from direct_anchor_model import export_head
    from exp.cxr_pt.inference.dataset import InferDataset, collate_fn
    from experiments.direct_anchor_classification.model import FrozenClassificationHead
    record = json.loads(next((DIRECT/'cache/radzero/chexpert').glob('cache_*.json')).read_text())
    assert sha256(CHECKPOINT/'model.safetensors') == record['checkpoint_sha256']
    source_config = json.loads(Path(record['head_path']).read_text())
    model, processor, tokenizer = load_model('cuda:0')
    base = model.inner.requires_grad_(False).eval()
    queries, _ = encode_prompts(base, tokenizer, source_config['prompts'])
    config = export_head(base, queries)
    config.update(class_names=source_meta['class_names'], prompts=source_config['prompts'])
    assert config == source_config, 'Frozen source queries/head changed'
    backend = FrozenClassificationHead(config).cuda().eval().requires_grad_(False)
    backend.image_encoder = base
    dataset = InferDataset(processor, images, '')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                        collate_fn=partial(collate_fn, image_processor=processor))
    def batches():
        with torch.no_grad():
            for batch in loader:
                yield base.forward_vision_model(batch.cuda())['vision_tokens']
    def native(tokens):
        return match_tokens(base, tokens, queries)['logits']
    return backend, batches, native, record


def worker(args,gpu):
    import numpy as np
    import torch
    from experiments.radzero_supervised_classification.features import LinearProbe,ClipGlobalAdapter,global_scores
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
    batch_size=8
    backend,batches,native,cache=radzero_encoder(source_meta,images,out,batch_size)
    frozen=freeze_backend(backend)
    modules={};sources={};expected={}
    for method in ep['methods']:
        for seed in ((42,) if method in ('original','global_frozen') else SEEDS):
            key=method+'__seed'+str(seed)
            if method in ('original','pre_mlp'):
                spec=next(r for r in p['references'] if r['model']==args.model and r['dataset']=='chexpert' and r['method']==method and r['seed']==seed)
                folder=Path(spec['source'])
                for f,digest in spec['files_sha256'].items():assert sha256(folder/f)==digest
            else:folder=run/'results'/args.model/'chexpert'/f'seed{seed}'/method
            result=json.loads((folder/'result.json').read_text());assert result['status']=='complete'
            if method in ('linear_probe','clip_global','global_frozen','post_mlp'):assert result['protocol_sha256']==sha256(run/'PROTOCOL.json')
            module=None
            if method not in ('original','global_frozen'):
                saved=torch.load(folder/'adapter.pt',map_location='cpu',weights_only=True)
                if method in ('pre_mlp','post_mlp'):
                    module=residual_mlp(768,64);module.load_state_dict(saved['adapter_state_dict'] if method=='pre_mlp' else saved['module_state_dict'])
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
        code_sha256={str(f):sha256(f) for f in [Path(__file__),Path(__file__).with_name('features.py')]},gpu=gpu)
    save(out/'evaluation_plan.json',evaluation_plan)
    outputs={key:[] for key in modules};offset=0;started=time.time();native_error=0.
    with torch.no_grad():
        for tokens in batches():
            glob=tokens[:,0]
            context=backend.context(tokens)
            for key,(method,seed,module) in modules.items():
                if method=='linear_probe':values=module(glob)
                elif method in ('clip_global','global_frozen'):values=global_scores(args.model,backend,module(glob) if module is not None else glob)
                else:
                    values=backend.score(backend.context(module(tokens))) if method=='pre_mlp' else backend.score(context,module)
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
