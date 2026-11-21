"""Frozen official BiomedCLIP; AlphaRAD paper's common disease prompt."""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]

class Adapter:
    def __init__(self, device='cuda'):
        from open_clip import create_model_and_transforms
        from open_clip.factory import _MODEL_CONFIGS
        from open_clip.tokenizer import HFTokenizer
        self.device = torch.device(device)
        ckpt = ROOT / 'checkpoints'
        cfg = json.loads((ckpt / 'open_clip_config.json').read_text())
        model_cfg = copy.deepcopy(cfg['model_cfg'])
        model_cfg['text_cfg']['hf_model_name'] = str(ckpt / 'text_config')
        model_cfg['text_cfg']['hf_tokenizer_name'] = str(ckpt)
        _MODEL_CONFIGS['biomedclip_local_offline'] = model_cfg
        self.model, _, self.preprocess = create_model_and_transforms(
            'biomedclip_local_offline', pretrained=str(ckpt / 'open_clip_pytorch_model.bin'),
            pretrained_hf=False, precision='fp32', device=self.device,
            image_mean=cfg['preprocess_cfg']['mean'], image_std=cfg['preprocess_cfg']['std'])
        self.model.eval().requires_grad_(False)
        self.tokenizer = HFTokenizer(str(ckpt), context_length=256)
        self._text = {}
        self.metadata = dict(model='BiomedCLIP', score_kind='scaled_cosine_similarity_independent_per_class',
            prompt_template='There is {disease}.', prompt_protocol='AlphaRAD paper section 4.3',
            preprocess=repr(self.preprocess), checkpoint=str(ckpt / 'open_clip_pytorch_model.bin'),
            grounding_available=False, training=False, precision='float32', class_mapping='identity',
            protocol='Official OpenCLIP 2.23.0 preprocessing and released checkpoint. Independent scores for multilabel evaluation; no across-class softmax.')

    @torch.inference_mode()
    def predict(self, image_paths, class_names):
        key = tuple(class_names)
        if not key or any(not isinstance(s,str) or not s.strip() for s in key):
            raise ValueError('Nonempty disease names required')
        if key not in self._text:
            tokens = self.tokenizer([f'There is {s}.' for s in key]).to(self.device)
            self._text[key] = self.model.encode_text(tokens, normalize=True)
        images = []
        for path in image_paths:
            with Image.open(path) as im: images.append(self.preprocess(im.convert('RGB')))
        batch = torch.stack(images).to(self.device)
        feats = self.model.encode_image(batch, normalize=True)
        scores = self.model.logit_scale.exp() * (feats @ self._text[key].T)
        return scores.float().cpu().numpy()

def create_adapter(device='cuda'):
    return Adapter(device)
