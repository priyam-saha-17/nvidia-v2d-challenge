from pathlib import Path
import ast,json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.v4.new_notebook();b.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
b.cells=[nb.v4.new_markdown_cell('''# Track 1 automatic segmentation — episode 0
Use only the released Track 1 video and object text prompt. Grounding DINO seeds
SAM2 automatically; propagate both directions and export the official mask schema.
T4 runs in FP32: SAM2's stored memory precision is adjusted consistently. No hand
annotations, Track 2 assets or confidential references are used.''')]
b.cells.append(nb.v4.new_code_cell('''import os,sys,subprocess,json,urllib.request,hashlib
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','transformers==5.3.0','h5py','opencv-python<4.12'],check=True)
env=os.environ.copy();env['SAM2_BUILD_CUDA']='0'
subprocess.run(['git','clone','--depth','1','https://github.com/facebookresearch/sam2.git','/tmp/sam2_source'],check=True)
sam_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/tmp/sam2_source',text=True).strip()
predictor_source=Path('/tmp/sam2_source/sam2/sam2_video_predictor.py')
original=predictor_source.read_text();replacement=original.replace('torch.bfloat16','torch.float32')
assert replacement!=original,'Recheck SAM2 stored-memory dtype before running T4'
predictor_source.write_text(replacement)
subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','-e','/tmp/sam2_source'],env=env,check=True)
checkpoint=Path('/tmp/sam2.1_hiera_large.pt')
urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',checkpoint)
'''))
b.cells.append(nb.v4.new_code_cell('''base='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_1/'
with urllib.request.urlopen(base+'meta/episodes_metadata.jsonl') as response: metadata=[json.loads(line) for line in response.read().decode().splitlines()]
m=next(x for x in metadata if x['episode_index']==0)
video=Path('/tmp')/(m['sequence_id']+'.0.color.mp4')
urllib.request.urlretrieve(base+'videos/chunk-000/observation.images.exo_camera/episode_000000.mp4',video)
with urllib.request.urlopen(base+'meta/episodes.jsonl') as response: episodes=[json.loads(line) for line in response.read().decode().splitlines()]
episode=next(x for x in episodes if x['episode_index']==0)
print('Track 1 episode metadata:',m,episode)
'''))
source=(root/'track1/segment_video.py').read_text()
b.cells.append(nb.v4.new_code_cell("Path('/tmp/segment_video.py').write_text("+repr(source)+")\nsubprocess.run([sys.executable,'/tmp/segment_video.py','--video',str(video),'--object-prompt',m['object_prompt'],'--sam2-checkpoint',str(checkpoint),'--output','/kaggle/working/episode_000000','--expected-frames',str(episode['length'])],env=env,check=True)"))
b.cells.append(nb.v4.new_code_cell('''report={'sam2_revision':sam_revision,'sam2_weight_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
 'memory_dtype_patch':'torch.bfloat16 -> torch.float32 in sam2_video_predictor.py',
 'patched_predictor_sha256':hashlib.sha256(predictor_source.read_bytes()).hexdigest(),'track':'track_1','episode':0}
Path('/kaggle/working/model_manifest.json').write_text(json.dumps(report,indent=2))
'''))
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b);dest=root/'notebooks/track1_segmentation';dest.mkdir(exist_ok=True)
nb.write(b,dest/'segment.ipynb')
meta=json.loads((root/'notebooks/t4_diagnostic/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-track1-automatic-segmentation',title='V2D Track1 Automatic Segmentation',code_file='segment.ipynb')
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
