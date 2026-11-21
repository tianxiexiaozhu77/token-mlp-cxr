"""Frozen CARZero with its official bidirectional cross-attention scoring.
The caller supplies fixed held-out images. AlphaRAD's common prompt is used;
the public CARZero prompt JSONs are archived separately for provenance.
"""
from pathlib import Path
import sys, contextlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

class CARZeroAdapter:
    def __init__(self,device='cuda'):
        self.device=torch.device(device)
        sys.path.insert(0,str(ROOT))
        import CARZero
        from CARZero import builder
        from CARZero.models import transformer_backbones
        from transformers import AutoModel,AutoConfig
        from omegaconf import OmegaConf
        self.official=CARZero
        self.checkpoint=ROOT/'pretrain_model/CARZero_best_model.ckpt'
        ckpt=torch.load(self.checkpoint,map_location='cpu')
        cfg=ckpt['hyper_parameters']
        original_bert=cfg.model.text.bert_type
        cfg.model.text.bert_type=str(ROOT/'local_assets/BioClinicalMPBERT')
        # The complete author checkpoint supplies all BERT parameters. Instantiate
        # from its architecture config then strict-load the complete checkpoint,
        # avoiding a redundant base-model weights download that is overwritten.
        original_load=AutoModel.from_pretrained
        original_vision_load=transformer_backbones.load_weight
        def from_config_only(path,*args,**kwargs):
            config=AutoConfig.from_pretrained(path,local_files_only=True)
            for key in ['output_attentions','output_hidden_states']:
                if key in kwargs:setattr(config,key,kwargs[key])
            return AutoModel.from_config(config)
        AutoModel.from_pretrained=from_config_only
        transformer_backbones.load_weight=lambda model,path:model
        try:self.model=builder.build_CARZero_dqn_wo_self_atten_mlp_gl_model(cfg)
        finally:
            AutoModel.from_pretrained=original_load
            transformer_backbones.load_weight=original_vision_load
        model_keys=self.model.state_dict()
        sd={k.split('CARZero.')[-1]:v for k,v in ckpt['state_dict'].items() if k.split('CARZero.')[-1] in model_keys}
        load_result=self.model.load_state_dict(sd,strict=True)
        coverage={'strict':True,'model_state_keys':len(model_keys),'loaded_state_keys':len(sd),'checkpoint_state_keys':len(ckpt['state_dict']),'missing_keys':list(load_result.missing_keys),'unexpected_keys':list(load_result.unexpected_keys),'loaded_numel':sum(v.numel() for v in sd.values()),'model_numel':sum(v.numel() for v in model_keys.values())}
        self.model=self.model.to(self.device).eval().requires_grad_(False)
        self.text_cache={};self.native_checked=False
        self.metadata={'model':'CARZero','checkpoint':str(self.checkpoint),'score_kind':'author bidirectional image-to-text/text-to-image fusion logits arithmetic mean','prompts':'AlphaRAD common prompt: There is [disease].','native_prompt_archive':str(ROOT/'local_assets/author_prompts'),'preprocess':'official process_img: OpenCV grayscale, long-side resize with zero padding; checkpoint test CenterCrop/normalization','checkpoint_image_size':int(cfg.data.image.imsize),'checkpoint_transforms':OmegaConf.to_container(cfg.transforms,resolve=True),'original_text_encoder':str(original_bert),'bert_initialization':'local config then complete strict checkpoint load; no randomly initialized inference weights', 'vision_initialization':'skip redundant M3AE initializer; complete strict CARZero checkpoint supplies all vision parameters','supported_tasks':['classification'],'native_score_parity':'pending first prediction','protocol_note':'No gradient updates, no additional sigmoid/temperature/feature normalization. Label names replace underscores with spaces.'}
        self.metadata['checkpoint_key_coverage']=coverage

    @torch.no_grad()
    def _text(self,class_names):
        key=tuple(class_names)
        if key not in self.text_cache:
            prompts={}
            for name in class_names:
                if not isinstance(name,str) or not name.strip():raise ValueError(f'Invalid class {name!r}')
                prompts[name]=[f'There is {name.replace("_"," ")}.']
            processed=self.model.process_class_prompts(prompts,self.device)
            batch={k:torch.cat([v[k] for v in processed.values()],0) for k in ['caption_ids','attention_mask','token_type_ids']}
            local,global_,_=self.model.text_encoder_forward(batch['caption_ids'],batch['attention_mask'],batch['token_type_ids'])
            self.text_cache[key]=(local,global_,processed)
            self.metadata['last_prompts']=prompts
        return self.text_cache[key]

    @torch.no_grad()
    def predict(self,image_paths,class_names):
        if not image_paths:return np.empty((0,len(class_names)),np.float32)
        q_local,q_global,processed=self._text(class_names)
        imgs=self.model.process_img(image_paths,self.device)
        i_local,i_global=self.model.image_encoder_forward(imgs)
        q_local=q_local.view(q_local.size(0),q_local.size(1),-1).permute(0,2,1)
        outputs=[]
        for i in range(len(image_paths)):
            local=i_local[i:i+1].view(i_local[i:i+1].size(0),i_local[i:i+1].size(1),-1).permute(0,2,1)
            global_=i_global[i:i+1]
            i2t,_=self.model.fusion_module(torch.cat([global_.unsqueeze(1),local],1),q_global,return_atten=True)
            t2i,_=self.model.fusion_module(torch.cat([q_global.unsqueeze(1),q_local],1),global_,return_atten=True)
            outputs.append((i2t.squeeze(-1)+t2i.squeeze(-1).transpose(1,0))/2)
        result=torch.cat(outputs,0).float().cpu().numpy()
        if not self.native_checked:
            native=self.official.dqn_shot_classification(self.model,imgs,processed).values
            error=float(np.max(np.abs(result-native)))
            if not np.allclose(result,native,rtol=1e-6,atol=1e-6):raise RuntimeError(f'CARZero author-score parity failed: {error}')
            self.metadata['native_score_parity']={'passed':True,'max_abs_error':error,'batch_size':len(image_paths)}
            self.native_checked=True
        return result

def create_adapter(device='cuda'):
    return CARZeroAdapter(device)
