"""Build normalized image-conditioned shapes for a completed segmentation batch."""
import argparse,ast,json
from pathlib import Path
import nbformat as nb
p=argparse.ArgumentParser()
p.add_argument('--segmentation-source',required=True)
p.add_argument('--name',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
template=nb.read(root/'notebooks/track1_shape/shape.ipynb',as_version=4)
setup=template.cells[1].source
setup=setup[:setup.index("images=list(Path('/kaggle/input')")]
code=""
for file in ['reconstruct_object.py','prepare_mesh.py']:
 code+=f"Path('/tmp/{file}').write_text({(root/'track1'/file).read_text()!r})\n"
code+='''import urllib.request,shutil
base='https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_1/'
with urllib.request.urlopen(base+'meta/episodes_metadata.jsonl') as response: metadata={x['episode_index']:x for x in map(json.loads,response.read().decode().splitlines())}
groups={}
for report_path in Path('/kaggle/input').rglob('segmentation_report.json'):
 episode_id=int(report_path.parent.name.split('_')[-1])
 if metadata[episode_id]['object']=='hula_hoop':continue
 report=json.loads(report_path.read_text())
 groups.setdefault(metadata[episode_id]['object'],[]).append((episode_id,report_path,report))
if not groups:raise ValueError('No actual segmented object observations')
results=[]
for object_name,observations in sorted(groups.items()):
 # The same released object is observed in multiple clips. Select the automatic
 # seed with highest object detector confidence; share its normalized geometry.
 chosen=max(observations,key=lambda item:item[2]['seed']['object']['confidence'])
 image=chosen[1].with_name('object_observation.png')
 raw=Path('/kaggle/working/raw_shapes')/object_name
 prepared=Path('/kaggle/working/prepared_shapes')/object_name
 try:
  subprocess.run([py,'/tmp/reconstruct_object.py','--image',str(image),'--output',str(raw),'--weights','/tmp/shape_weights'],env=env,check=True)
  subprocess.run([py,'/tmp/prepare_mesh.py','--mesh',str(raw/'normalized_shape.glb'),'--output',str(prepared)],env=env,check=True)
  for episode_id,_,report in observations:
   dest=Path('/kaggle/working')/f'episode_{episode_id:06d}';dest.mkdir(exist_ok=True)
   shutil.copy2(prepared/'normalized_shape.glb',dest/'normalized_shape.glb')
   provenance={'episode':episode_id,'object':object_name,'shape_source_episode':chosen[0],
      'selection':'maximum automatic object detector seed confidence among released Track 1 clips',
      'metric_scale_estimated':False,'ground_truth_used':False}
   (dest/'shape_provenance.json').write_text(json.dumps(provenance,indent=2))
  result={'object':object_name,'success':True,'episodes':[x[0] for x in observations]}
 except subprocess.CalledProcessError as error:
  result={'object':object_name,'success':False,'error':str(error)}
 results.append(result)
 Path('/kaggle/working/batch_shape_report.json').write_text(json.dumps(results,indent=2))
 print('SHAPE_RESULT',result,flush=True)
Path('/kaggle/working/code_revision.json').write_text(json.dumps({'Hunyuan3D-2':revision}))
if any(not x['success'] for x in results):raise RuntimeError('Some shapes failed; inspect the report')
'''
b=nb.v4.new_notebook(metadata=template.metadata)
b.cells=[nb.v4.new_markdown_cell('# Track 1 batch object shapes\nImage-conditioned reconstruction from automatic masks; no metric-scale claim until depth calibration.'),nb.v4.new_code_cell(setup),nb.v4.new_code_cell(code)]
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b)
dest=root/'notebooks'/('track1_shapes_'+a.name);dest.mkdir(exist_ok=True)
nb.write(b,dest/'shape.ipynb')
meta=json.loads((root/'notebooks/track1_shape/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-track1-shapes-'+a.name,title='V2D Track1 Shapes '+a.name,kernel_sources=[a.segmentation_source])
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(dest)
