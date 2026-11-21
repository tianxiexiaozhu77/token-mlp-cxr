"""Author MaCo zero-shot classification, fixed image manifests supplied by caller.
No learned adaptation or support labels are used. This wraps ZeroShot/model_MaCo.py
and ZeroShot/utils/prompts.py; the sole device fix replaces hard-coded .cuda().
"""
from pathlib import Path
import contextlib, os, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
ROOT=Path(__file__).resolve().parents[1]

@contextlib.contextmanager
def _root_cwd():
    old=os.getcwd();os.chdir(ROOT)
    try: yield
    finally: os.chdir(old)

class MaCoAdapter:
    def __init__(self,device='cuda'):
        self.device=torch.device(device)
        sys.path.insert(0,str(ROOT))
        from ZeroShot.model_MaCo import maco
        from ZeroShot.utils.prompts import _get_default_text_promtps
        self.prompt_fn=_get_default_text_promtps
        with _root_cwd(): self.model=maco(bert_type='MaCo',normalize_feature=False,multi_class=True,num_classes=1)
        self.checkpoint=ROOT/'checkpoints/maco.pth'
        checkpoint=torch.load(self.checkpoint,map_location='cpu')
        sd=checkpoint.get('state_dict',checkpoint.get('model'))
        if sd is None: raise ValueError('Checkpoint lacks author model/state_dict')
        # The released checkpoint has a 384->768 pooler projection; the source
        # inherits Hugging Face's 384->384 default. Restore the trained dimensions
        # while retaining the original pooler operation (first token, dense, tanh).
        pooler=self.model.bert_encoder.model.bert.pooler
        out_dim,in_dim=sd['bert_encoder.model.bert.pooler.dense.weight'].shape
        if in_dim!=pooler.dense.in_features or out_dim!=768:
            raise RuntimeError(f'Unexpected MaCo checkpoint pooler shape: {(out_dim,in_dim)}')
        pooler.dense=torch.nn.Linear(in_dim,out_dim,bias=True)
        result=self.model.load_state_dict(sd,strict=False)
        # The pretrained checkpoint includes reconstruction-only heads; its zero-shot
        # positive/negative heads are intentionally constructed from text embeddings.
        invalid=[k for k in result.missing_keys if not k.startswith('cls_head.')]
        if invalid: raise RuntimeError(f'Missing pretrained inference parameters: {invalid}')
        self.pos_weight_img=sd['pos_weight_img.weight'].detach().float().to(self.device)
        self.load_report={'missing_keys':list(result.missing_keys),'unexpected_keys':list(result.unexpected_keys)}
        self.model=self.model.to(self.device).eval().requires_grad_(False)
        self.transform=transforms.Compose([transforms.Resize(224),transforms.CenterCrop((224,224)),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize(mean=[0.4978],std=[0.2449])])
        self.text_cache={};self.native_checked=False
        self.metadata={'model':'MaCo','checkpoint':str(self.checkpoint),'native_score_parity':'pending first prediction','score_kind':'native positive-vs-negative two-way softmax probability','prompts':'author default: one positive and four negative prompts; arithmetic text embedding mean','preprocess':'author Resize(shorter=224), CenterCrop224, grayscale3, ToTensor, Normalize(.4978,.2449)','normalization':'no feature normalization or added temperature','supported_tasks':['classification','grounding','segmentation'], 'grounding_preprocess':'author direct 224x224 bicubic resize (no crop), grayscale3 normalization; native 224 map nearest-resized to each original H/W, returned as list[N] of [C,H,W], full image coverage', 'segmentation_note':'Evaluate the same native zero-shot phrase-grounding heatmap with shared Dice protocol; not the separately fine-tuned SIIM segmentation head.', 'grounding_score':'author per-patch linear projection weighted by softmax(pos_weight_img/.02), raw dot product with text, Gaussian sigma1.5, nearest resize224','load_report':self.load_report,'protocol_note':'AlphaRAD paper common positive prompt There is [disease]. matches author default (trailing whitespace tokenizes identically). Four author native negative prompts retained; AlphaRAD does not specify these negative prompts. No prompt selection on test labels.'}
        self.metadata['checkpoint_compatibility']='Restore BERT pooler dense 384->768 from released checkpoint shape (source HF default is 384->384). Original first-token/Tanh pooling preserved; all projection weights loaded, no new inference parameters.'

    def set_dataset(self,entry):
        self.dataset_entry=entry

    @torch.no_grad()
    def _text(self,class_names):
        key=tuple(class_names)
        if key not in self.text_cache:
            positive=[];negative=[];prompts={}
            for name in class_names:
                if not isinstance(name,str) or not name.strip(): raise ValueError(f'Invalid class: {name!r}')
                # Dataset underscores are spelling separators, not a change of concept.
                name=name.replace('_',' ')
                pos,neg=self.prompt_fn(name);prompts[name]={'pos':pos,'neg':neg}
                def embed(items):
                    all_emb=[]
                    for text in items:
                        inputs=[x.to(self.device) for x in self.model.bert_encoder.encode(text)]
                        all_emb.append(self.model.bert_encoder(*inputs).logits)
                    return torch.stack(all_emb).mean(0).squeeze(0)
                positive.append(embed(pos));negative.append(embed(neg))
            self.text_cache[key]=(torch.stack(positive),torch.stack(negative))
            self.metadata['last_prompts']=prompts
        return self.text_cache[key]

    @torch.no_grad()
    def predict(self,image_paths,class_names):
        if not image_paths: return np.empty((0,len(class_names)),np.float32)
        pos,neg=self._text(class_names)
        tensors=[]
        for path in image_paths:
            with Image.open(path) as image: tensors.append(self.transform(image.convert('RGB')))
        images=torch.stack(tensors).to(self.device)
        feature=self.model.forward_img_encoder_nomask(images)[:,1:,:].mean(1)
        feature=self.model.img_mlp(feature)
        output=torch.stack([feature@pos.T,feature@neg.T],dim=-1).softmax(-1)[...,0]
        result=output.float().cpu().numpy()
        if not self.native_checked:
            from ZeroShot.model_MaCo import ZeroShotClassificationHead
            self.model.cls_head=ZeroShotClassificationHead(768,len(class_names),True).to(self.device)
            self.model.cls_head.set_cls_feature(pos,neg)
            native=self.model(images)[0][...,0].float().cpu().numpy()
            error=float(np.max(np.abs(result-native)))
            if not np.allclose(result,native,rtol=1e-6,atol=1e-6):raise RuntimeError(f'MaCo native scoring parity failed: {error}')
            self.metadata['native_score_parity']={'passed':True,'max_abs_error':error,'batch_size':len(image_paths)}
            self.native_checked=True
        return result

    @torch.no_grad()
    def grounding(self,image_paths,class_names):
        from scipy.ndimage import gaussian_filter
        import torch.nn.functional as F
        transform=transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize([.4978],[.2449])])
        imgs=[];original_sizes=[]
        for path in image_paths:
            with Image.open(path) as image:
                original_sizes.append((image.height,image.width))
                imgs.append(transform(image.resize((224,224),resample=Image.Resampling.BICUBIC)))
        patch=self.model.forward_img_encoder_nomask(torch.stack(imgs).to(self.device))[:,1:,:]
        patch=self.model.img_mlp(patch)
        patch=patch*(self.pos_weight_img/.02).softmax(-1).view(1,196,1)
        texts=[]
        for name in class_names:
            if not isinstance(name,str) or not name.strip(): raise ValueError(f'Invalid class: {name!r}')
            query=f'There is {name.replace("_"," ")}. '
            # Exact single-sentence branch of author Engine._text_process.
            query=''.join(t.strip()+'. ' for t in query.split('.') if len(t.strip()+'. ')>5).lower()
            tokens=[x.to(self.device) for x in self.model.bert_encoder.encode(query)]
            texts.append(self.model.bert_encoder(*tokens).logits.squeeze(0))
        raw=(patch@torch.stack(texts).T).permute(0,2,1).reshape(len(image_paths),len(class_names),14,14).cpu().numpy()
        smooth=np.stack([np.stack([gaussian_filter(m,sigma=(1.5,1.5),order=0) for m in row]) for row in raw])
        native=F.interpolate(torch.from_numpy(smooth),size=(224,224),mode='nearest')
        return [F.interpolate(native[i:i+1],size=hw,mode='nearest')[0].numpy() for i,hw in enumerate(original_sizes)]

def create_adapter(device='cuda'):
    return MaCoAdapter(device)
