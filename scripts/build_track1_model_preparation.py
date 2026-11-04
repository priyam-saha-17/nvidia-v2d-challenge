"""Prepare approved private/public model files without occupying a GPU slot."""
from pathlib import Path
import ast,json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
template=nb.read(root/'notebooks/track1_inference/inference.ipynb',as_version=4)
setup='''import os,sys,json,subprocess,hashlib,tarfile,shutil
from pathlib import Path
py=sys.executable;env=os.environ.copy()
def file_sha256(path):
 digest=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()
subprocess.run([py,'-m','pip','install','gdown','huggingface_hub'],check=True)
repo=Path('/tmp/video_to_data');env['GIT_LFS_SKIP_SMUDGE']='1'
subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/nvidia-isaac/video_to_data.git',str(repo)],env=env,check=True)
subprocess.run(['git','sparse-checkout','set','reconstruction/modules/v2d_foundation_pose','reconstruction/modules/v2d_cari4d'],cwd=repo,env=env,check=True)
subprocess.run(['git','checkout','8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'],cwd=repo,env=env,check=True)
modules=repo/'reconstruction/modules'
subprocess.run([py,'-m','pip','install','--no-deps','-e',str(modules/'v2d_foundation_pose/lib'),'-e',str(modules/'v2d_cari4d/lib')],env=env,check=True)
'''
cache='''out=Path('/kaggle/working/public_models.bin')
parts=[weights/'hf_home',weights/'sam3d_body/torch_home',weights/'foundationpose']
with tarfile.open(out,'w:gz',compresslevel=1) as archive:
 for part in parts:archive.add(part,arcname=str(part.relative_to(weights)))
Path('/kaggle/working/public_model_cache_manifest.json').write_text(json.dumps({
 'sha256':file_sha256(out),'bytes':out.stat().st_size,
 'paths':[str(p.relative_to(weights)) for p in parts],
 'contains_gated_models':False,'contains_credentials':False},indent=2))
print('Public model preparation complete',out.stat().st_size)
'''
b=nb.v4.new_notebook(metadata=template.metadata)
b.cells=[nb.v4.new_markdown_cell('# Track 1 model preparation\nCPU-only: verify the approved private checkpoint files and fetch/cache public models. No reconstruction or scoring claim.'),nb.v4.new_code_cell(setup),template.cells[-2],nb.v4.new_code_cell(cache)]
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b)
dest=root/'notebooks/track1_model_preparation';nb.write(b,dest/'prepare.ipynb')
meta=json.loads((dest/'kernel-metadata.json').read_text());meta['kernel_sources']=[]
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
