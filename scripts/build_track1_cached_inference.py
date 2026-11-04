"""Build later per-episode jobs using the first job's verified native runtime cache."""
import argparse,ast,json
from pathlib import Path
import nbformat as nb
p=argparse.ArgumentParser()
p.add_argument('--episode',type=int,required=True)
p.add_argument('--segmentation-source',required=True)
p.add_argument('--shape-source',help='Notebook containing episode_XXXXXX/normalized_shape.glb for non-hoop objects')
p.add_argument('--runtime-source',default='priyamsaha17/v2d-track1-model-preparation')
a=p.parse_args()
if a.episode not in range(30):p.error('Track 1 episode must be 0..29')
root=Path(__file__).resolve().parents[1]
template=nb.read(root/'notebooks/track1_inference/inference.ipynb',as_version=4)
bootstrap='''import os,sys,json,subprocess,urllib.request,hashlib,tarfile,shutil
from pathlib import Path
def file_sha256(path):
 digest=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()
manifests=list(Path('/kaggle/input').rglob('runtime_cache_manifest.json'))
assert len(manifests)==1,manifests
manifest=json.loads(manifests[0].read_text())
cache=manifests[0].with_name('track1_runtime.bin')
if not cache.exists():cache=manifests[0].with_name('track1_runtime.tar.gz')
assert cache.stat().st_size==manifest['bytes']
assert file_sha256(cache)==manifest['sha256']
assert manifest['runtime_revision']=='8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'
subprocess.run(['apt-get','update','-qq'],check=True)
subprocess.run(['apt-get','install','-y','-qq','gcc-12','g++-12','cmake','libeigen3-dev','libboost-system-dev','libboost-program-options-dev','libegl1-mesa-dev','libosmesa6-dev','libspatialindex-dev','libgl1','ffmpeg'],check=True)
# Only accept the explicit paths listed in the manifest, preventing arbitrary writes.
allowed=tuple(x.lstrip('/').rstrip('/')+'/' for x in manifest['paths'])
with tarfile.open(cache) as archive:
 for item in archive.getmembers():
  name=item.name.rstrip('/')+'/'
  if not name.startswith(allowed):raise ValueError('Unexpected runtime archive member '+item.name)
 archive.extractall('/',filter='fully_trusted')
py='/tmp/track1-env/bin/python';cuda=Path('/tmp/cuda124')
modules=Path('/tmp/video_to_data/reconstruction/modules')
workspace=Path('/workspace');workspace.mkdir(exist_ok=True)
for name in ['v2d_common','v2d_mv','v2d_mesh','v2d_foundation_pose','v2d_moge','v2d_sam3d_body','v2d_cari4d']:
 path=workspace/name
 if not path.exists():path.symlink_to(modules/name,target_is_directory=True)
env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+':'+env['PATH'],CC='gcc-12',CXX='g++-12',TORCH_CUDA_ARCH_LIST='7.5',FORCE_CUDA='1',MAX_JOBS='2',PYOPENGL_PLATFORM='egl',OPENCV_IO_ENABLE_OPENEXR='1',CUB_HOME=str(cuda/'include'),CPATH=str(cuda/'include')+':/usr/include/eigen3')
env['PYTHONPATH']=':'.join(str(workspace/x) for x in ['v2d_cari4d/lib/cari4d','v2d_sam3d_body/lib','v2d_foundation_pose/lib/FoundationPose'])
print('Verified native runtime cache restored',flush=True)
'''
# These archives are produced by our own explicit runtime builder and hash verified.
# Symlinks point to their original absolute /tmp paths, hence the trusted extraction.
run=template.cells[-1].source
run=run.replace("image=next(inputs.rglob('object_observation.png'))",f"episode_id={a.episode}\nfolder=f'episode_{{episode_id:06d}}'\nimage=next(p for p in inputs.rglob('object_observation.png') if p.parent.name==folder)")
run=run.replace("masks=next(inputs.rglob('masks.h5'))", "masks=next(p for p in inputs.rglob('masks.h5') if p.parent.name==folder)")
old="subprocess.run([py,str(code/'fit_hoop.py'),'--image',str(image),'--output','/kaggle/working/shape'],env=env,check=True)"
if a.episode>=3:
 if not a.shape_source:p.error('Non-hoop episodes require their actual image-conditioned shape source')
 run=run.replace(old,"shape=next(p for p in inputs.rglob('normalized_shape.glb') if p.parent.name==folder)\nPath('/kaggle/working/shape').mkdir(exist_ok=True)\nshutil.copy2(shape,'/kaggle/working/shape/normalized_shape.glb')")
run=run.replace("x['episode_index']==0","x['episode_index']==episode_id")
run=run.replace("base+'videos/chunk-000/observation.images.exo_camera/episode_000000.mp4'", "base+f'videos/chunk-000/observation.images.exo_camera/episode_{episode_id:06d}.mp4'")
run=run.replace("'--episode','0'", "'--episode',str(episode_id)")
# The automatic seed frame differs between clips; use its recorded index.
run=run.replace("'--output','/kaggle/working/reconstruction'", "'--seed-frame',str(json.loads(image.with_name('segmentation_report.json').read_text())['seed']['frame']),\n '--output','/kaggle/working/reconstruction'")
runtime_test=nb.v4.new_code_cell(template.cells[3].source.split("Path('/kaggle/working/track1_compiler_manifest.json')")[0])
model_cell=nb.v4.new_code_cell(template.cells[-2].source)
model_cell.source=model_cell.source.replace("subprocess.run([py,'-c',public_models],env=env,check=True)", """public_caches=list(Path('/kaggle/input').rglob('public_models.bin'))
if public_caches:
 assert len(public_caches)==1,public_caches
 archive_path=public_caches[0]
 cache_manifest=json.loads(archive_path.with_name('public_model_cache_manifest.json').read_text())
 assert archive_path.stat().st_size==cache_manifest['bytes']
 assert file_sha256(archive_path)==cache_manifest['sha256']
 prefixes=tuple(x.rstrip('/')+'/' for x in cache_manifest['paths'])
 with tarfile.open(archive_path) as archive:
  for item in archive.getmembers():
   if not (item.name.rstrip('/')+'/').startswith(prefixes):raise ValueError('Unexpected public-model cache member')
  archive.extractall(weights,filter='data')
 print('Verified public model cache restored',flush=True)
else:
 subprocess.run([py,'-c',public_models],env=env,check=True)""")
b=nb.v4.new_notebook(metadata=template.metadata)
b.cells=[nb.v4.new_markdown_cell(f'# Track 1 reconstructed episode {a.episode}\nRestore a hash-verified runtime and reconstruct the full original timeline.'),nb.v4.new_code_cell(bootstrap),runtime_test,model_cell,nb.v4.new_code_cell(run)]
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b)
dest=root/'notebooks'/f'track1_episode_{a.episode:06d}';dest.mkdir(exist_ok=True)
nb.write(b,dest/'inference.ipynb')
meta=json.loads((root/'notebooks/track1_inference/kernel-metadata.json').read_text())
sources=list(dict.fromkeys([a.runtime_source,a.segmentation_source]+([a.shape_source] if a.shape_source else [])))
meta['dataset_sources'].append('priyamsaha17/v2d-track1-private-native-runtime')
meta.update(id=f'priyamsaha17/v2d-track1-episode-{a.episode:06d}',title=f'V2D Track1 Episode {a.episode:06d}',kernel_sources=sources)
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(dest)
