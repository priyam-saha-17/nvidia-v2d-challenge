"""Reconstruct the missing iron from the best automatically accepted source crop."""
from pathlib import Path
import ast,json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
template=nb.read(root/'notebooks/track1_shape/shape.ipynb',as_version=4)
setup=template.cells[1].source.split("images=list(Path('/kaggle/input')")[0]
selection='''import hashlib
ranking=json.loads(next(Path('/kaggle/input').rglob('semantic_crop_ranking.json')).read_text())
by_hash={x['input_sha256']:x for x in ranking}
candidates=[]
for path in Path('/kaggle/input').rglob('object_observation.png'):
 if path.parent.name not in {'episode_000006','episode_000007','episode_000008'}:continue
 report=json.loads(path.with_name('segmentation_report.json').read_text())
 if report.get('semantic_selection'):
  margin=float(report['seed']['semantic_margin'])
 else:
  value=by_hash.get(hashlib.sha256(path.read_bytes()).hexdigest())
  if value is None:continue
  margin=float(value['semantic_margin'])
 if margin>0:candidates.append((margin,path,report))
assert candidates,'No automatic iron crop passes semantic verification'
margin,image,segmentation=max(candidates,key=lambda item:item[0])
print('Automatic iron source',image,'semantic margin',margin,flush=True)
'''
code=""
for file in ['reconstruct_object.py','prepare_mesh.py']:
 code+=f"Path('/tmp/{file}').write_text({(root/'track1'/file).read_text()!r})\n"
code+='''subprocess.run([py,'/tmp/reconstruct_object.py','--image',str(image),'--output','/kaggle/working/raw_iron','--weights','/tmp/shape_weights'],env=env,check=True)
subprocess.run([py,'/tmp/prepare_mesh.py','--mesh','/kaggle/working/raw_iron/normalized_shape.glb','--output','/kaggle/working/iron'],env=env,check=True)
Path('/kaggle/working/iron_source.json').write_text(json.dumps({'source_episode':int(image.parent.name.split('_')[-1]),'source_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'semantic_margin':margin,'Hunyuan3D_code_revision':revision,'manual_annotations_used':False},indent=2))
'''
b=nb.v4.new_notebook(metadata=template.metadata)
b.cells=[nb.v4.new_markdown_cell('# Track 1 iron reconstruction\nImage-conditioned geometry from an automatically semantically verified crop. No tripod candidate, manual labels or reference geometry.'),nb.v4.new_code_cell(setup),nb.v4.new_code_cell(selection),nb.v4.new_code_cell(code)]
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b)
dest=root/'notebooks/track1_iron_recovery';dest.mkdir(exist_ok=True);nb.write(b,dest/'iron.ipynb')
meta=json.loads((root/'notebooks/track1_shape/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-track1-iron-reconstruction',title='V2D Track1 Iron Reconstruction',code_file='iron.ipynb',kernel_sources=['priyamsaha17/v2d-track1-segments-a','priyamsaha17/v2d-track1-segments-semantic','priyamsaha17/v2d-track1-crop-ranking'])
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Prepared iron recovery; wait for corrected masks to finish before launching')
