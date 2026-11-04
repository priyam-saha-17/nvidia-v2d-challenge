"""Build a geometry-only Hunyuan3D T4 notebook using automatic video masks."""
from pathlib import Path
import ast, json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.v4.new_notebook()
b.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
b.cells=[nb.v4.new_markdown_cell('''# Track 1 image-conditioned object shape
Generate a normalized geometry hypothesis from the automatically segmented video.
This is not yet a metric reconstruction. Keep external model licenses separate
from this repository's Apache license. No Track 2 assets or reference poses.''')]
b.cells.append(nb.v4.new_code_cell('''import os,sys,subprocess,json
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
subprocess.run(['uv','python','install','3.11'],check=True)
subprocess.run(['uv','venv','--python','3.11','/tmp/shape-env'],check=True)
py='/tmp/shape-env/bin/python'
subprocess.run(['uv','pip','install','--python',py,'torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu124'],check=True)
subprocess.run(['uv','pip','install','--python',py,'transformers==4.48.3','diffusers==0.32.2','accelerate','einops','omegaconf','trimesh','pillow','numpy<2','scikit-image','opencv-python<4.12','pyyaml','tqdm','safetensors','huggingface_hub<1','pygltflib','pymeshlab','ninja','pybind11'],check=True)
repo=Path('/tmp/Hunyuan3D-2')
subprocess.run(['git','clone','--depth','1','https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git',str(repo)],check=True)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
subprocess.run(['uv','pip','install','--python',py,'--no-deps','-e',str(repo)],check=True)
env=os.environ.copy();env['PYTHONPATH']=str(repo)
images=list(Path('/kaggle/input').rglob('object_observation.png'))
assert len(images)==1,images
'''))
source=(root/'track1/reconstruct_object.py').read_text()
b.cells.append(nb.v4.new_code_cell("Path('/tmp/reconstruct_object.py').write_text("+repr(source)+")\nsubprocess.run([py,'/tmp/reconstruct_object.py','--image',str(images[0]),'--output','/kaggle/working/episode_000000','--weights','/tmp/shape_weights'],env=env,check=True)\nPath('/kaggle/working/code_revision.json').write_text(json.dumps({'Hunyuan3D-2':revision}))"))
for cell in b.cells:
 if cell.cell_type=='code':ast.parse(cell.source)
nb.validate(b)
dest=root/'notebooks/track1_shape';dest.mkdir(exist_ok=True)
nb.write(b,dest/'shape.ipynb')
meta=json.loads((root/'notebooks/track1_segmentation/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-track1-object-shape',title='V2D Track1 Object Shape',code_file='shape.ipynb',kernel_sources=['priyamsaha17/v2d-track1-automatic-segmentation'])
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
