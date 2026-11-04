from pathlib import Path
import ast,json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.v4.new_notebook();b.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
b.cells=[nb.v4.new_markdown_cell('''# Track 2 Tier 1 G1 retargeting probe
Retarget the first 32 frames of released episode 0 using the official SOMA-X/Pink
pipeline. This validates the challenge data adapter; it is not a full submission.
No SAM 3D Body checkpoint or token is required for this Tier 1 path.''')]
b.cells.append(nb.v4.new_code_cell('''import os,sys,subprocess,json,hashlib,shutil,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
subprocess.run(['uv','python','install','3.11'],check=True)
subprocess.run(['uv','venv','--python','3.11','/tmp/retarget-env'],check=True)
py='/tmp/retarget-env/bin/python'
deps=['torch','py-soma-x==0.2.1','pin==3.7.0','pin-pink==4.2.0','coal==3.0.1',
'eigenpy==3.10.3','cmeel==0.57.3','cmeel-boost==1.87.0.1','cmeel-urdfdom==4.0.1',
'cmeel-console-bridge==1.0.2.3','cmeel-tinyxml2==10.0.0','cmeel-tinyxml==2.6.2.3',
'cmeel-assimp==5.4.3.1','cmeel-octomap==1.10.0','cmeel-qhull==8.0.2.1',
'cmeel-zlib==1.3.1','qpsolvers[daqp]==4.9.0','loop-rate-limiters','viser',
'trimesh<5','scipy','pandas','pyarrow','huggingface_hub','usd-core']
subprocess.run(['uv','pip','install','--python',py,*deps],check=True)
repo=Path('/tmp/video_to_data');rev='8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'
env=os.environ.copy();env['GIT_LFS_SKIP_SMUDGE']='1'
subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/nvidia-isaac/video_to_data.git',str(repo)],check=True,env=env)
subprocess.run(['git','sparse-checkout','set','robotic_grounding'],cwd=repo,check=True,env=env)
subprocess.run(['git','checkout',rev],cwd=repo,check=True,env=env)
rg=repo/'robotic_grounding';assets=rg/'source/robotic_grounding/robotic_grounding/assets'
env['PYTHONPATH']=str(rg/'source/robotic_grounding')
'''))
b.cells.append(nb.v4.new_code_cell('''pointers=[]
for f in (assets/'meshes/g1').rglob('*'):
 if f.is_file() and f.stat().st_size<1024:
  raw=f.read_bytes()
  if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
   fields=dict(line.split(' ',1) for line in raw.decode().splitlines())
   pointers.append((f,fields['oid'].split(':')[1],int(fields['size'])))
def fetch(item):
 f,digest,size=item
 url='https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/'+rev+'/'+f.relative_to(repo).as_posix()
 tmp=f.with_suffix(f.suffix+'.download');h=hashlib.sha256()
 with urllib.request.urlopen(url,timeout=180) as response,tmp.open('wb') as out:
  while chunk:=response.read(1024*1024):h.update(chunk);out.write(chunk)
 assert tmp.stat().st_size==size and h.hexdigest()==digest
 tmp.replace(f)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,pointers))
subprocess.run([py,'scripts/setup_soma_assets.py'],cwd=rg,env=env,check=True)
'''))
b.cells.append(nb.v4.new_code_cell('''source=Path('/tmp/tier1');(source/'meta').mkdir(parents=True)
base='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_2/tier_1_multiview_caption/'
def get(relative):
 dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 urllib.request.urlretrieve(base+relative,dest)
 return dest
get('meta/info.json');metadata=get('meta/episodes_metadata.jsonl')
items=[json.loads(line) for line in metadata.read_text().splitlines()]
m=next(x for x in items if x['episode_index']==0)
metadata.write_text(json.dumps(m)+'\\n')
for relative in ['data/chunk-000/episode_000000.parquet',m['mesh'],m['ground_plane'],m['support_surface']]:get(relative)
'''))
converter=(root/'scripts/export_track2_inputs.py').read_text()
b.cells.append(nb.v4.new_code_cell("Path('/tmp/export_inputs.py').write_text("+repr(converter)+")\nsubprocess.run([py,'/tmp/export_inputs.py','--input',str(source),'--output','/tmp/retarget-inputs'],check=True)"))
b.cells.append(nb.v4.new_code_cell('''motion_root=Path('/kaggle/working/motion_assets/human_motion_data')
subprocess.run([py,'scripts/retarget/soma_to_g1.py','/tmp/retarget-inputs/'+m['sequence_id'],
 '--save','--motion-root',str(motion_root),'--soma-subdir','soma','--end-frame','32'],
 cwd=rg,env=env,check=True)
stage=motion_root/'whole_body/reconstructed_stage';stage.mkdir(parents=True,exist_ok=True)
shutil.copy2(source/m['support_surface'],stage/Path(m['support_surface']).name)
parquets=list(motion_root.rglob('*.parquet'));assert parquets
print('Retargeted 32-frame probe:',[str(p) for p in parquets])
'''))
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b);dest=root/'notebooks/tier1_retarget_probe';dest.mkdir(exist_ok=True)
nb.write(b,dest/'retarget.ipynb')
meta=json.loads((root/'notebooks/runtime_probe/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-tier1-retarget-probe',title='V2D Tier1 Retarget Probe',code_file='retarget.ipynb')
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
