"""KAD-224 released zero-shot inference; official bare disease-name queries."""
from pathlib import Path
import sys, types, json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as tvmodels
from transformers import AutoModel, BertConfig, AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
AUTHOR=ROOT/'A3_CLIP'
sys.path.insert(0,str(AUTHOR))
# The author source imports an unused institutional object-store client at import.
# Execute the same model code with only that unused dependency/initialization removed.
source=(AUTHOR/'models/clip_tqn.py').read_text()
for line in ['from petrel_client.client import Client','client = Client(conf_path)']:
 assert line in source;source=source.replace(line,'# local inference: unused object-store client omitted')
mod=types.ModuleType('kad_local_native');mod.__file__=str(AUTHOR/'models/clip_tqn.py')
exec(compile(source,mod.__file__,'exec'),mod.__dict__)
ALIASES={'no finding':'normal','no findings':'normal','effusion':'pleural effusion','mass':'lung mass',
 'nodule':'lung nodule','pleural thickening':'pleural thicken','pleural_thickening':'pleural thicken',
 'covid':'covid-19'}
class Adapter:
 def __init__(self,device='cuda'):
  self.device=torch.device(device);self.query_cache={}
  self.checkpoint=ROOT/'local_assets/KAD_224_best_valid.pt'
  bert=str(ROOT/'local_assets/Med-KEBERT')
  self.tokenizer=AutoTokenizer.from_pretrained(bert,do_lower_case=True,local_files_only=True)
  # Initialization-only downloads are redundant: every model key is loaded strictly below.
  original_resnet=tvmodels.resnet50;original_automodel=AutoModel.from_pretrained
  try:
   tvmodels.resnet50=lambda **kw:original_resnet(weights=None)
   AutoModel.from_pretrained=lambda name,config=None,**kw:AutoModel.from_config(config or BertConfig.from_pretrained(name,local_files_only=True))
   self.image_encoder=mod.ModelRes(res_base_model='resnet50')
   self.text_encoder=mod.CLP_clinical(bert_model_name=bert)
   self.model=mod.TQN_Model()
  finally:
   tvmodels.resnet50=original_resnet;AutoModel.from_pretrained=original_automodel
  ckpt=torch.load(self.checkpoint,map_location='cpu')
  for name,model in [('image_encoder',self.image_encoder),('text_encoder',self.text_encoder),('model',self.model)]:
   state={k.removeprefix('module.'):v for k,v in ckpt[name].items()}
   if name=='text_encoder' and 'bert_model.embeddings.position_ids' in state:
    # Old transformers saved this deterministic buffer; restore its old persistence
    # so every released checkpoint key is still checked and loaded strictly.
    positions=state['bert_model.embeddings.position_ids']
    expected=torch.arange(model.bert_model.config.max_position_embeddings).expand((1,-1))
    if not torch.equal(positions,expected):raise ValueError('Unexpected BERT position_ids buffer')
    model.bert_model.embeddings._non_persistent_buffers_set.discard('position_ids')
   model.load_state_dict(state,strict=True);model.to(self.device).eval().requires_grad_(False)
  self.transform=transforms.Compose([transforms.Resize([224,224],interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
  self.metadata={'model':'KAD-224','score_kind':'positive_class_softmax_probability','checkpoint':str(self.checkpoint),
   'preprocess':'RGB; bicubic resize 224x224; ImageNet normalization. Matches author CheXpert/PadChest pipeline; NIH images are square so scalar Resize is equivalent.',
   'prompts':'bare disease names (official KAD protocol); max_length=256; pooled BERT -> MLP -> TQN',
   'label_mapping':ALIASES,'transformers_compatibility':'Restore persistence of original deterministic arange position_ids buffer after validating its exact values, then strict-load every checkpoint key.','limitations':['Beyond author evaluated disease lists, bare class names are an explicit zero-shot transfer protocol; no prompt tuning.',
   'CPU construction skips overwritten ImageNet/BERT initialization downloads; all image/text/TQN checkpoint keys loaded strictly.']}
 def _queries(self,class_names):
  names=tuple(ALIASES.get(n.lower(),n.lower()) for n in class_names)
  if names not in self.query_cache:
   tokens=self.tokenizer(list(names),add_special_tokens=True,max_length=256,padding='max_length',return_tensors='pt').to(self.device)
   self.query_cache[names]=self.text_encoder.encode_text(tokens)
  return self.query_cache[names]
 def check_classes(self,class_names):
  if not class_names or any(not isinstance(n,str) or not n.strip() for n in class_names):
   raise ValueError('Expected nonempty disease names')
  return True
 @torch.inference_mode()
 def predict(self,image_paths,class_names):
  imgs=[]
  for p in image_paths:
   with Image.open(p) as im:imgs.append(self.transform(im.convert('RGB')))
  x=torch.stack(imgs).to(self.device)
  features,_=self.image_encoder(x)
  return self.model(features,self._queries(class_names)).softmax(-1)[:,:,1].cpu().numpy()
 @torch.inference_mode()
 def grounding(self,image_paths,class_names):
  images=[];sizes=[]
  for p in image_paths:
   with Image.open(p) as im:
    sizes.append((im.height,im.width));images.append(self.transform(im.convert('RGB')))
  features,_=self.image_encoder(torch.stack(images).to(self.device))
  _,attention=self.model(features,self._queries(class_names),return_atten=True)
  side=int(attention.shape[-1]**.5)
  if side*side!=attention.shape[-1]:raise ValueError('Non-square KAD attention grid')
  maps=attention.reshape(len(images),len(class_names),side,side).cpu().numpy()
  import cv2
  # Native visualization restores the attention grid with OpenCV INTER_LINEAR.
  # Return the continuous scalar map before 8-bit display quantization/colormap.
  self.metadata['grounding']={'source':str(AUTHOR/'plot_visualize_512.py'),
   'method':'final TQN layer averaged-head raw attention; OpenCV INTER_LINEAR to original extent',
   'variant':'KAD224 14x14 (official plotting example is KAD512 16x16)',
   'values':'raw continuous attention before display-only maximum scaling, uint8 quantization and JET colormap','crop':False}
  result=[]
  for maps_i,(height,width) in zip(maps,sizes):
   result.append(np.stack([cv2.resize(m,(width,height),interpolation=cv2.INTER_LINEAR) for m in maps_i]))
  return result
def create_adapter(device='cuda'):return Adapter(device)
