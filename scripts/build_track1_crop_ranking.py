"""CPU semantic quality check across all automatically segmented Track 1 views."""
import ast,json
from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
b.cells=[nb.v4.new_markdown_cell('# Track 1 automatic crop ranking\nCompare released object semantics against automatically segmented crops; no manual boxes, masks, poses or geometry labels.')]
source=(root/'track1/select_seed.py').read_text()
code='''import sys,subprocess,json,hashlib,urllib.request
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','transformers==5.3.0'],check=True)
'''
code+="Path('/tmp/select_seed.py').write_text("+repr(source)+")\nsys.path.insert(0,'/tmp')\n"
code+='''from select_seed import DESCRIPTIONS,DISTRACTORS
from transformers import AutoModel,AutoProcessor
from PIL import Image
import torch,numpy as np
torch.set_num_threads(2)
model_id='google/siglip2-base-patch16-224'
processor=AutoProcessor.from_pretrained(model_id)
model=AutoModel.from_pretrained(model_id).eval()
texts=[];labels=[]
for name,descriptions in DESCRIPTIONS.items():
 for text in descriptions:labels.append(name);texts.append('This is a photo of '+text+'.')
for text in DISTRACTORS:labels.append('distractor:'+text);texts.append('This is a photo of '+text+'.')
url='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_1/meta/episodes_metadata.jsonl'
with urllib.request.urlopen(url) as f:metadata={x['episode_index']:x for x in map(json.loads,f.read().decode().splitlines())}
all_paths=sorted(Path('/kaggle/input').rglob('object_observation.png'))
chosen={}
for path in all_paths:
 episode=int(path.parent.name.split('_')[-1])
 semantic=json.loads(path.with_name('segmentation_report.json').read_text()).get('semantic_selection',False)
 if episode not in chosen or semantic:chosen[episode]=path
paths=[chosen[key] for key in sorted(chosen)]
assert len(paths)==30,len(paths)
records=[]
for start in range(0,len(paths),4):
 batch=paths[start:start+4]
 inputs=processor(text=texts,images=[Image.open(p).convert('RGB') for p in batch],padding='max_length',max_length=64,truncation=True,return_tensors='pt')
 with torch.inference_mode():logits=model(**inputs).logits_per_image.float().numpy()
 for path,row in zip(batch,logits):
  episode=int(path.parent.name.split('_')[-1]);name=metadata[episode]['object']
  own=[i for i,label in enumerate(labels) if label==name];other=[i for i,label in enumerate(labels) if label!=name]
  record={'episode':episode,'object':name,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
   'target_logit':float(row[own].max()),'semantic_margin':float(row[own].max()-row[other].max()),
   'predicted_class':labels[int(row.argmax())],'model':model_id,'model_revision':getattr(model.config,'_commit_hash',None)}
  records.append(record);print('CROP_RESULT',json.dumps(record),flush=True)
Path('/kaggle/working/semantic_crop_ranking.json').write_text(json.dumps(records,indent=2))
'''
ast.parse(code);b.cells.append(nb.v4.new_code_cell(code));nb.validate(b)
dest=root/'notebooks/track1_crop_ranking';dest.mkdir(exist_ok=True);nb.write(b,dest/'rank.ipynb')
meta=json.loads((root/'notebooks/track1_model_preparation/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-track1-crop-ranking',title='V2D Track1 Crop Ranking',code_file='rank.ipynb',dataset_sources=[],kernel_sources=['priyamsaha17/v2d-track1-automatic-segmentation','priyamsaha17/v2d-track1-segments-a','priyamsaha17/v2d-track1-segments-b','priyamsaha17/v2d-track1-segments-semantic'])
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
