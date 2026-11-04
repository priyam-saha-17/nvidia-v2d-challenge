"""Prepare a two-device Track 1 inference notebook; launch after pilot validation."""
import argparse,ast,json
from pathlib import Path
import nbformat as nb
p=argparse.ArgumentParser()
p.add_argument('--episodes',type=int,nargs='+',required=True)
p.add_argument('--segmentation-source',required=True)
p.add_argument('--shape-source')
p.add_argument('--name',required=True)
p.add_argument('--extra-segmentation-source',action='append',default=[])
p.add_argument('--semantic-source',default='priyamsaha17/v2d-track1-segments-semantic')
p.add_argument('--recover-source',help='Completed production notebook with cached genuine CoCoNet predictions')
p.add_argument('--recover-episodes',type=int,nargs='+',default=[])
a=p.parse_args()
if not 1<=len(a.episodes)<=30 or any(x not in range(30) for x in a.episodes):p.error('Use actual Track 1 episodes')
if any(x>=3 for x in a.episodes) and not a.shape_source:p.error('Actual image-conditioned shapes are required')
if bool(a.recover_source)!=bool(a.recover_episodes):p.error('Recovery requires source and episode IDs together')
if set(a.recover_episodes)&set(a.episodes):p.error('Recovered episodes must be distinct from new inference')
root=Path(__file__).resolve().parents[1]
template=nb.read(root/'notebooks/track1_episode_000000/inference.ipynb',as_version=4)
b=nb.v4.new_notebook(metadata=template.metadata)
b.cells=[nb.v4.new_markdown_cell('# Track 1 two-device reconstruction\nIndependent actual episodes on both T4 devices. Preserve outputs and intermediate caches; require all 30 validated episodes before scoring.'),*template.cells[1:4]]
code="code=Path('/tmp/track1_code');code.mkdir(exist_ok=True)\n"
for file in ['fit_hoop.py','calibrate_object.py','run_episode.py','export_cari4d.py','run_batch.py','patch_runtime.py','recover_refinement.py']:
 code+=f"(code/{file!r}).write_text({(root/'track1'/file).read_text()!r})\n"
code+="subprocess.run([py,str(code/'patch_runtime.py'),'--report','/kaggle/working/runtime_patch_manifest.json'],env=env,check=True)\n"
code+="""env.update(TORCH_HOME=str(weights/'sam3d_body/torch_home'),MHR_ASSETS_ROOT=str(weights/'sam3d_body'),FOUNDATIONPOSE_WEIGHTS_DIR=str(weights/'foundationpose/nvlabs_pytorch'),HDF5_USE_FILE_LOCKING='FALSE',PYTHONUNBUFFERED='1')
"""
if a.recover_episodes:
 code+=f"subprocess.run([py,str(code/'recover_refinement.py'),'--episodes',*map(str,{a.recover_episodes!r}),'--weights',str(weights),'--output','/kaggle/working/recovered'],env=env,check=True)\n"
code+=f"subprocess.run([py,str(code/'run_batch.py'),'--episodes',*map(str,{a.episodes!r}),'--weights',str(weights),'--output','/kaggle/working'],env=env,check=True)\n"
b.cells.append(nb.v4.new_code_cell(code))
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b)
dest=root/'notebooks'/('track1_inference_'+a.name);dest.mkdir(exist_ok=True)
nb.write(b,dest/'inference.ipynb')
meta=json.loads((root/'notebooks/track1_episode_000000/kernel-metadata.json').read_text())
sources=['priyamsaha17/v2d-track1-model-preparation',a.segmentation_source,*a.extra_segmentation_source]
if a.shape_source and a.shape_source!='private-dataset':sources.append(a.shape_source)
if a.semantic_source:sources.append(a.semantic_source)
if a.recover_source:sources.append(a.recover_source)
meta['dataset_sources'].append('priyamsaha17/v2d-track1-reconstructed-shapes')
meta.update(id='priyamsaha17/v2d-track1-inference-'+a.name,title='V2D Track1 Inference '+a.name,kernel_sources=list(dict.fromkeys(sources)))
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Prepared, not launched:',dest)
