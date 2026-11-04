"""Extend the validated runtime with a reusable cache and real episode inference."""
import ast,json
from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.read(root/'notebooks/track1_runtime/runtime.ipynb',as_version=4)
b.cells[0]=nb.v4.new_markdown_cell('''# Track 1 reconstruction — episode 0
Build and archive the validated T4 runtime, verify the approved private model
files, estimate a video-derived hoop shape and physical scale, then run the
native MHR reconstruction. No clean object assets or reference poses are used.
Third-party runtime/model files stay in private Kaggle storage.''')
# Do not request the unavailable Kaggle secret service; approved private input is sufficient.
b.cells.pop(1)
b.cells[1].source='hf_token=None\n'+b.cells[1].source
b.cells.append(nb.v4.new_code_cell('''# Archive only explicit runtime directories; no home credentials or model caches.
def file_sha256(path):
 digest=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()
cache=Path('/kaggle/working/track1_runtime.tar.gz')
paths=[Path('/tmp/track1-env'),Path('/tmp/video_to_data'),Path('/tmp/cuda124'),Path('/tmp/nvdiffrast')]
python_base=Path(subprocess.check_output([py,'-c','import sys;print(sys.base_prefix)'],text=True).strip())
paths.append(python_base)
with tarfile.open(cache,'w:gz',compresslevel=1) as archive:
 for path in paths:archive.add(path,arcname=str(path).lstrip('/'))
Path('/kaggle/working/runtime_cache_manifest.json').write_text(json.dumps({
 'sha256':file_sha256(cache),'bytes':cache.stat().st_size,
 'paths':[str(x) for x in paths],'runtime_revision':rev,
 'torch':'2.5.1+cu124','gpu_arch':'7.5','contains_model_weights':False},indent=2))
print('Reusable runtime archived:',cache.stat().st_size,flush=True)
'''))
b.cells.append(nb.v4.new_code_cell('''weights=Path('/tmp/track1_weights');weights.mkdir(exist_ok=True)
manifests=[json.loads(p.read_text()) for p in Path('/kaggle/input').rglob('model_manifest.json')]
manifest=next(value for value in manifests if isinstance(value,list) and len(value)==8 and all('file' in x and 'sha256' in x for x in value))
archives=list(Path('/kaggle/input').rglob('track1_models.tar'))
if len(archives)==1:
 with tarfile.open(archives[0]) as archive:archive.extractall(weights,filter='data')
else:
 # Kaggle expands uploaded .tar files into a directory named after the archive.
 roots=[p.parents[2] for p in Path('/kaggle/input').rglob('step200000.pth') if p.parent.name=='2026-08-25-09-35-57']
 assert len(roots)==1,roots
 for item in manifest:
  source=roots[0]/item['file'];destination=weights/item['file']
  destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
for item in manifest:
 path=weights/item['file'];assert path.stat().st_size==item['bytes']
 assert file_sha256(path)==item['sha256'],path
# T4 lacks bfloat16. Use full precision for the human initializer to avoid FP16 overflow.
config=weights/'sam3d_body/checkpoints/sam-3d-body-dinov3/model_config.yaml'
original_config=config.read_text()
assert 'USE_FP16: true' in original_config
config.write_text(original_config.replace('USE_FP16: true','USE_FP16: false'))
Path('/kaggle/working/t4_precision_manifest.json').write_text(json.dumps({
 'sam3d_precision':'float32','original_config_sha256':hashlib.sha256(original_config.encode()).hexdigest(),
 'effective_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()},indent=2))
env['HF_HOME']=str(weights/'hf_home')
public_models="""
from pathlib import Path
from v2d.cari4d.lib.download_weights import _download_moge2,_download_torch_hub_models
from v2d.foundation_pose.lib.download_weights import download_weights
from v2d.foundation_pose.lib.backends import NVLABS_PYTORCH
weights=Path('/tmp/track1_weights')
_download_moge2(weights)
_download_torch_hub_models(weights)
download_weights(str(weights/'foundationpose'),backend=NVLABS_PYTORCH)
"""
subprocess.run([py,'-c',public_models],env=env,check=True)
'''))
source_files=['fit_hoop.py','calibrate_object.py','run_episode.py','export_cari4d.py']
code="code=Path('/tmp/track1_code');code.mkdir(exist_ok=True)\n"
for file in source_files:
 code+=f"(code/{file!r}).write_text({(root/'track1'/file).read_text()!r})\n"
code+='''inputs=Path('/kaggle/input')
image=next(inputs.rglob('object_observation.png'))
masks=next(inputs.rglob('masks.h5'))
subprocess.run([py,str(code/'fit_hoop.py'),'--image',str(image),'--output','/kaggle/working/shape'],env=env,check=True)
base='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_1/'
with urllib.request.urlopen(base+'meta/episodes_metadata.jsonl') as response: metadata=[json.loads(line) for line in response.read().decode().splitlines()]
with urllib.request.urlopen(base+'meta/episodes.jsonl') as response: episodes=[json.loads(line) for line in response.read().decode().splitlines()]
m=next(x for x in metadata if x['episode_index']==0)
episode=next(x for x in episodes if x['episode_index']==0)
video=Path('/tmp')/(m['sequence_id']+'.0.color.mp4')
urllib.request.urlretrieve(base+'videos/chunk-000/observation.images.exo_camera/episode_000000.mp4',video)
env.update(TORCH_HOME=str(weights/'sam3d_body/torch_home'),MHR_ASSETS_ROOT=str(weights/'sam3d_body'),FOUNDATIONPOSE_WEIGHTS_DIR=str(weights/'foundationpose/nvlabs_pytorch'),HDF5_USE_FILE_LOCKING='FALSE',PYTHONUNBUFFERED='1')
subprocess.run([py,str(code/'run_episode.py'),'--video',str(video),'--masks',str(masks),
 '--mesh','/kaggle/working/shape/normalized_shape.glb','--weights',str(weights),
 '--output','/kaggle/working/reconstruction','--episode','0','--frames',str(episode['length'])],env=env,check=True)
'''
b.cells.append(nb.v4.new_code_cell(code))
for cell in b.cells:
 if cell.cell_type=='code':ast.parse(cell.source)
nb.validate(b)
dest=root/'notebooks/track1_inference';dest.mkdir(exist_ok=True)
nb.write(b,dest/'inference.ipynb')
meta=json.loads((root/'notebooks/track1_runtime/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-track1-reconstruction',title='V2D Track1 Reconstruction',code_file='inference.ipynb',dataset_sources=['priyamsaha17/v2d-track1-private-pretrained-models'],kernel_sources=['priyamsaha17/v2d-track1-automatic-segmentation'])
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
