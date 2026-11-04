from pathlib import Path
import ast,json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.read(root/'notebooks/training_smoke/train.ipynb',as_version=4)
b.cells[0]=nb.v4.new_markdown_cell('''# Track 2 Tier 1 actual G1 policy/evaluation probe
Train episode 0 on its complete retargeted reference, then run the official
unassisted evaluator. This is one episode of the required 30 and is not submitted
alone. A short training budget tests integration; quality is evaluated afterward.''')
b.cells[4]=nb.v4.new_code_cell('''import hashlib, shutil, urllib.request,json
from concurrent.futures import ThreadPoolExecutor
rev='8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'
roots=[fc/'src/flash_chord/assets/meshes/g1',fc/'src/flash_chord/assets/policies/sonic']
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
 url='https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/'+rev+'/'+f.relative_to(runtime).as_posix()
 tmp=f.with_suffix(f.suffix+'.download');h=hashlib.sha256()
 with urllib.request.urlopen(url,timeout=180) as response,tmp.open('wb') as out:
  while chunk:=response.read(1024*1024):h.update(chunk);out.write(chunk)
 assert tmp.stat().st_size==size and h.hexdigest()==digest
 tmp.replace(f)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,pointers))
reports=list(Path('/kaggle/input').rglob('retarget_report.json'))
assert len(reports)==1,reports
report=json.loads(reports[0].read_text());assert not report['failures'] and len(report['parquets'])==30
shutil.copytree(reports[0].parent/'motion_assets','/kaggle/working/motion_assets',dirs_exist_ok=True)
url='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_2/tier_1_multiview_caption/meta/episodes_metadata.jsonl'
with urllib.request.urlopen(url) as response: items=[json.loads(line) for line in response.read().decode().splitlines()]
m=next(x for x in items if x['episode_index']==0)
reference=Path('/kaggle/working/motion_assets/human_motion_data/whole_body/soma')/('sequence_id='+m['sequence_id'])/'robot_name=g1/data.parquet'
assert reference.is_file()
print('Materialized G1 controller/mesh files:',len(pointers),'reference:',reference)
''')
b.cells[5]=nb.v4.new_code_cell('''output=Path('/kaggle/working/policies/episode_000000')
thresholds=[13107,26214,39321,52428,65536,78643,108134,114688,131072]
command=[py,'scripts/train_flash_sac.py','experiment=g1_recon_body_flash_sac',
 f'task.parquet="{reference}"',f'output_dir={output}',
 'scene.world_count=128','training.total_environment_steps=131072',
 'training.replay.capacity=16384','training.replay.minimum_size=4096',
 'training.replay.batch_size=256','training.replay.observation_storage_dtype=float16',
 'training.checkpoint.save_interval_environment_steps=65536',
 'curriculum.thresholds='+str(thresholds).replace(' ',''),
 'logging.mode=disabled','reset.seed=42']
train_env=env.copy();train_env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
subprocess.run(command,cwd=fc,env=train_env,check=True)
checkpoints=list(output.glob('policy_*.safetensors'));assert checkpoints
checkpoint=max(checkpoints,key=lambda p:int(p.stem.split('_')[1]))
print('Actual challenge policy checkpoint:',checkpoint)
''')
b.cells.append(nb.v4.new_code_cell('''import zipfile
kit_zip=Path('/tmp/kit.zip')
urllib.request.urlretrieve('https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip',kit_zip)
with zipfile.ZipFile(kit_zip) as z:z.extractall('/tmp/submission_kit')
evaluator=next(Path('/tmp/submission_kit').rglob('examples/run_policy_evaluation.py'))
mapping=Path('/kaggle/working/checkpoints.json');mapping.write_text(json.dumps({'0':str(checkpoint)},indent=2))
subprocess.run([py,str(evaluator),'--checkpoints',str(mapping),'--results','/kaggle/working/results',
 '--runtime-root',str(fc),'--world-count','1'],cwd=fc,env=train_env,check=True)
assert Path('/kaggle/working/results/episode_000000.parquet').is_file()
print('Official unassisted evaluation complete')
'''))
normalizer=(root/'scripts/normalize_support_stages.py').read_text()
b.cells.insert(5,nb.v4.new_code_cell("Path('/tmp/normalize_support.py').write_text("+repr(normalizer)+")\nsubprocess.run([py,'/tmp/normalize_support.py','/kaggle/working/motion_assets/human_motion_data/whole_body/reconstructed_stage'],check=True)"))
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b);dest=root/'notebooks/g1_policy_probe';dest.mkdir(exist_ok=True)
nb.write(b,dest/'train_eval.ipynb')
meta=json.loads((root/'notebooks/training_smoke/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-g1-policy-evaluation-probe',title='V2D G1 Policy Evaluation Probe',code_file='train_eval.ipynb',kernel_sources=['priyamsaha17/v2d-tier1-retarget-all'])
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
