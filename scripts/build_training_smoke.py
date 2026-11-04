from pathlib import Path
import json,ast
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.read(root/'notebooks/runtime_probe/probe.ipynb',as_version=4)
# Keep installations outside Kaggle output; save only checkpoints/reports.
for cell in b.cells:
 if cell.cell_type=='code':
  cell.source=cell.source.replace('/kaggle/working/fc-env','/tmp/fc-env').replace('/kaggle/working/video_to_data','/tmp/video_to_data')
b.cells[0]=nb.v4.new_markdown_cell('''# FlashCHORD T4 simulation/training smoke test
Short training on NVIDIA's released tissue-box example. This is an integration test;
it does not cover the challenge roster and produces no leaderboard submission.''')
b.cells.append(nb.v4.new_code_cell('''import hashlib, shutil, urllib.request
from concurrent.futures import ThreadPoolExecutor
rev='8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'
assets_path='robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon'
subprocess.run(['git','sparse-checkout','add',assets_path],cwd=runtime,check=True,env=env)
# Materialize only the LFS assets needed for floating Sharpa and this example.
roots=[fc/'src/flash_chord/assets/meshes/sharpawave',runtime/assets_path]
pointers=[]
for base in roots:
 for f in base.rglob('*'):
  if f.is_file() and f.stat().st_size<1024:
   raw=f.read_bytes()
   if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
    fields=dict(line.split(' ',1) for line in raw.decode().splitlines())
    pointers.append((f,fields['oid'].split(':')[1],int(fields['size'])))
def fetch(item):
 f,digest,size=item
 relative=f.relative_to(runtime).as_posix()
 url='https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/'+rev+'/'+relative
 tmp=f.with_suffix(f.suffix+'.download')
 h=hashlib.sha256()
 with urllib.request.urlopen(url,timeout=180) as response,tmp.open('wb') as out:
  while chunk:=response.read(1024*1024):
   h.update(chunk);out.write(chunk)
 assert tmp.stat().st_size==size and h.hexdigest()==digest, relative
 tmp.replace(f)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,pointers))
shutil.copytree(runtime/assets_path,fc/'src/flash_chord/assets/human_motion_data/ego_recon',dirs_exist_ok=True)
print('Verified LFS assets:',len(pointers))
'''))
b.cells.append(nb.v4.new_code_cell('''reference=fc/'src/flash_chord/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave'
output=Path('/kaggle/working/tissue_smoke')
command=[py,'scripts/train_flash_sac.py','experiment=sharpa_flash_sac',
 f'task.parquet={reference}',f'output_dir={output}',
 'scene.world_count=64','training.total_environment_steps=131072',
 'training.replay.capacity=16384','training.replay.minimum_size=1024',
 'training.replay.batch_size=256','training.replay.observation_storage_dtype=float16',
 'training.checkpoint.save_interval_environment_steps=65536',
 'logging.mode=disabled','reset.seed=42']
train_env=env.copy();train_env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
subprocess.run(command,cwd=fc,env=train_env,check=True)
checkpoints=list(output.glob('policy_*.safetensors'))
assert checkpoints,'Trainer completed without a policy checkpoint'
print('Saved checkpoints:',[str(p) for p in checkpoints])
'''))
for cell in b.cells:
 if cell.cell_type=='code': ast.parse(cell.source)
nb.validate(b)
dest=root/'notebooks/training_smoke';dest.mkdir(exist_ok=True)
nb.write(b,dest/'train.ipynb')
meta=json.loads((root/'notebooks/runtime_probe/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-flashchord-training-smoke',title='V2D FlashCHORD Training Smoke',code_file='train.ipynb')
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
