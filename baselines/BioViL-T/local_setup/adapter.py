"""Official BioViL-T models, transforms and native global/patch cosine scoring."""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'official/hi-ml-multimodal/src'))

class Adapter:
    def __init__(self, device='cuda'):
        from health_multimodal.image.model.model import ImageModel
        from health_multimodal.image.model.types import ImageEncoderType
        from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
        from health_multimodal.image import ImageInferenceEngine
        from health_multimodal.text import TextInferenceEngine
        from health_multimodal.vlp import ImageTextInferenceEngine
        from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer
        self.device = torch.device(device)
        ckpt = ROOT / 'checkpoints'
        self.model = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
            joint_feature_size=128, pretrained_model_path=ckpt/'biovil_t_image_model_proj_size_128.pt')
        self.model.eval().requires_grad_(False).to(self.device)
        self.transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)
        self.image_engine = ImageInferenceEngine(self.model, self.transform)
        self.text_model = CXRBertModel.from_pretrained(str(ckpt), local_files_only=True)
        self.text_model.eval().requires_grad_(False).to(self.device)
        tokenizer = CXRBertTokenizer.from_pretrained(str(ckpt), local_files_only=True)
        self.text_engine = TextInferenceEngine(tokenizer, self.text_model)
        self.engine = ImageTextInferenceEngine(self.image_engine, self.text_engine)
        self._text = {}
        self.metadata = dict(model='BioViL-T', score_kind='cosine_similarity_independent_per_class',
            prompt_template='There is {disease}.', prompt_protocol='AlphaRAD paper section 4.3',
            preprocess='Official load_image: uint8 full range rescale, grayscale; Resize shorter edge 512, CenterCrop448, ToTensor, ExpandChannels; no ImageNet normalization',
            checkpoint=[str(ckpt/'biovil_t_image_model_proj_size_128.pt'), str(ckpt/'pytorch_model.bin')],
            grounding_available=True, grounding_coordinates='Native original image pixels; NaN outside center crop coverage',
            grounding_return='list of arrays [C,H_i,W_i], since original image sizes can vary',
            training=False, precision='float32', class_mapping='identity',
            protocol='Official native global cosine; official sigma=1.5 patch smoothing and inverse geometry for grounding.')

    def _queries(self,class_names):
        key=tuple(class_names)
        if not key or any(not isinstance(s,str) or not s.strip() for s in key): raise ValueError('Nonempty disease names required')
        if key not in self._text:
            self._text[key]=self.text_engine.get_embeddings_from_prompt([f'There is {s}.' for s in key],verbose=False)
        return self._text[key]

    def _images(self,image_paths):
        from health_multimodal.image.data.io import load_image
        images=[load_image(Path(p)) for p in image_paths]
        return torch.stack([self.transform(im) for im in images]).to(self.device), [im.size for im in images]

    @torch.inference_mode()
    def predict(self,image_paths,class_names):
        queries=self._queries(class_names)
        images,_=self._images(image_paths)
        feats=F.normalize(self.model(images).projected_global_embedding,dim=-1)
        return (feats @ queries.T).float().cpu().numpy()

    @torch.inference_mode()
    def grounding(self,image_paths,class_names):
        queries=self._queries(class_names)
        images,sizes=self._images(image_paths)
        embeddings=self.model.get_patchwise_projected_embeddings(images,normalize=True)
        out=[]
        for i,(width,height) in enumerate(sizes):
            maps=[]
            for q in queries:
                sim=self.engine._get_similarity_map_from_embeddings(embeddings[i],q[None,:])
                maps.append(self.engine.convert_similarity_to_image_size(sim,width=width,height=height,
                    resize_size=512,crop_size=448,val_img_transform=self.transform,interpolation='nearest'))
            out.append(np.stack(maps))
        return out

def create_adapter(device='cuda'):
    return Adapter(device)
