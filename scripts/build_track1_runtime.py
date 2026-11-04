from pathlib import Path
import ast,json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
b=nb.v4.new_notebook();b.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
b.cells=[nb.v4.new_markdown_cell('''# Track 1 reconstruction runtime
Track 1 only. Build the official reconstruction dependencies for Tesla T4 (SM 7.5)
using Python 3.11, PyTorch 2.5.1 and CUDA 12.4. This notebook tests the runtime;
video reconstruction and scoring are subsequent stages. Secret label: HF Access Key.''')]
b.cells.append(nb.v4.new_code_cell('''from kaggle_secrets import UserSecretsClient
try:
 hf_token=UserSecretsClient().get_secret('HF Access Key')
except Exception:
 hf_token=None
 print('Secret service unavailable; public runtime compilation can continue')
from huggingface_hub import get_hf_file_metadata,hf_hub_url
for repo,file in [('nvidia/cari4d_commercial','2026-08-25-09-35-57/step200000.pth'),('facebook/sam-3d-body-dinov3','assets/mhr_model.pt')]:
 if hf_token:get_hf_file_metadata(hf_hub_url(repo,file),token=hf_token)
print('Checkpoint access verified' if hf_token else 'Checkpoint access not tested in this run')
'''))
b.cells.append(nb.v4.new_code_cell('''import os,sys,json,subprocess,urllib.request,hashlib,tarfile,shutil
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
subprocess.run(['uv','python','install','3.11'],check=True)
subprocess.run(['uv','venv','--python','3.11','/tmp/track1-env'],check=True)
py='/tmp/track1-env/bin/python'
subprocess.run(['apt-get','update','-qq'],check=True)
subprocess.run(['apt-get','install','-y','-qq','gcc-12','g++-12','cmake','libeigen3-dev','libboost-system-dev','libboost-program-options-dev','libegl1-mesa-dev','libosmesa6-dev','libspatialindex-dev','libgl1','ffmpeg'],check=True)
base='https://developer.download.nvidia.com/compute/cuda/redist/'
with urllib.request.urlopen(base+'redistrib_12.4.1.json') as response: redist=json.load(response)
cuda=Path('/tmp/cuda124');cuda.mkdir(exist_ok=True)
compiler_manifest={}
for name in ['cuda_nvcc','cuda_cudart','cuda_cccl']:
 item=redist[name]['linux-x86_64'];archive=Path('/tmp')/(name+'.tar.xz')
 urllib.request.urlretrieve(base+item['relative_path'],archive)
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['sha256']
 extract=Path('/tmp')/('extract-'+name);extract.mkdir(exist_ok=True)
 with tarfile.open(archive) as t:t.extractall(extract,filter='data')
 contents=next(extract.iterdir());shutil.copytree(contents,cuda,dirs_exist_ok=True,symlinks=True)
 compiler_manifest[name]=item
if not (cuda/'lib64').exists():(cuda/'lib64').symlink_to('lib')
env=os.environ.copy();env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+':'+env['PATH'],CC='gcc-12',CXX='g++-12',TORCH_CUDA_ARCH_LIST='7.5',FORCE_CUDA='1',MAX_JOBS='2',PYOPENGL_PLATFORM='egl',OPENCV_IO_ENABLE_OPENEXR='1')
if hf_token:env['HF_TOKEN']=hf_token
subprocess.run(['uv','pip','install','--python',py,'torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu124'],env=env,check=True)
constraints=Path('/tmp/track1-constraints.txt');constraints.write_text('numpy==1.26.3\\ntorch==2.5.1\\ntorchvision==0.20.1\\nscipy==1.13.1\\npandas==2.2.3\\n')
env['PIP_CONSTRAINT']=str(constraints)
packages=['numpy==1.26.3','scipy==1.13.1','pandas==2.2.3','setuptools','wheel','ninja','pybind11==2.10.0','joblib','scikit-learn','ruamel.yaml','trimesh<5','pyyaml','opencv-python<4.12','imageio','imageio-ffmpeg','open3d','transformations','warp-lang','einops','kornia','pyrender','av','h5py','hdf5plugin','videoio==0.3.0','scikit-image','meshcat','webdataset','omegaconf','pypng','roma','seaborn','openpyxl','wandb','matplotlib','tqdm','pillow','accelerate','timm','yacs','smplx','tensorboardx','pycocotools','rtree','xatlas','pytorch-lightning','hydra-core','hydra-submitit-launcher','hydra-colorlog','pyrootutils','networkx==3.2.1','rich','dill','jsonlines','loguru','optree','fvcore','cython','py-soma-x==0.2.1','fast_simplification','gdown','tensorrt-cu12==10.7.0.post1']
subprocess.run(['uv','pip','install','--python',py,'-c',str(constraints),*packages],env=env,check=True)
site=Path(subprocess.check_output([py,'-c','import site;print(site.getsitepackages()[0])'],text=True).strip())
for include in (site/'nvidia').glob('*/include'):
 shutil.copytree(include,cuda/'include',dirs_exist_ok=True)
env['CUB_HOME']=str(cuda/'include')
env['CPATH']=str(cuda/'include')+':/usr/include/eigen3'
subprocess.run([py,'-m','pip','--version'],env=env,check=False)
'''))
# uv's bare environment needs pip for native setup.py builds.
b.cells[-1].source=b.cells[-1].source.replace("'setuptools','wheel'", "'pip','setuptools','wheel'")
b.cells.append(nb.v4.new_code_cell('''repo=Path('/tmp/video_to_data');rev='8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'
env['GIT_LFS_SKIP_SMUDGE']='1'
subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/nvidia-isaac/video_to_data.git',str(repo)],env=env,check=True)
subprocess.run(['git','sparse-checkout','set','reconstruction'],cwd=repo,env=env,check=True)
subprocess.run(['git','checkout',rev],cwd=repo,env=env,check=True)
modules=repo/'reconstruction/modules';workspace=Path('/workspace');workspace.mkdir(exist_ok=True)
for name in ['v2d_common','v2d_mv','v2d_mesh','v2d_foundation_pose','v2d_moge','v2d_sam3d_body','v2d_cari4d']:
 path=workspace/name
 if not path.exists():path.symlink_to(modules/name,target_is_directory=True)
subprocess.run(['uv','pip','install','--python',py,'kaolin==0.18.0','-f','https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html'],env=env,check=True)
subprocess.run(['uv','pip','install','--python',py,'--no-build-isolation','git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'],env=env,check=True)
subprocess.run(['git','clone','--depth','1','https://github.com/NVlabs/nvdiffrast','/tmp/nvdiffrast'],env=env,check=True)
nvdiffrast_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/tmp/nvdiffrast',text=True).strip()
subprocess.run(['uv','pip','install','--python',py,'--no-build-isolation','/tmp/nvdiffrast'],env=env,check=True)
subprocess.run(['uv','pip','install','--python',py,'--no-deps','git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'],env=env,check=True)
subprocess.run(['uv','pip','install','--python',py,'-c',str(constraints),'git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030'],env=env,check=True)
cpp=workspace/'v2d_foundation_pose/lib/FoundationPose/mycpp'
pybind_dir=subprocess.check_output([py,'-m','pybind11','--cmakedir'],text=True,env=env).strip()
subprocess.run(['cmake','-S',str(cpp),'-B',str(cpp/'build'),'-DPYTHON_EXECUTABLE='+py,'-Dpybind11_DIR='+pybind_dir],env=env,check=True)
subprocess.run(['cmake','--build',str(cpp/'build'),'-j2'],env=env,check=True)
subprocess.run(['uv','pip','install','--python',py,'--no-build-isolation','-e',str(workspace/'v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda')],env=env,check=True)
subprocess.run(['uv','pip','install','--python',py,'-c',str(constraints),'-e',str(workspace/'v2d_common')+'[io]','-e',str(workspace/'v2d_mv')+'[math,vis]','-e',str(workspace/'v2d_mesh/lib'),'-e',str(workspace/'v2d_foundation_pose/lib'),'-e',str(workspace/'v2d_moge/lib'),'-e',str(workspace/'v2d_sam3d_body/lib'),'-e',str(workspace/'v2d_cari4d/lib')],env=env,check=True)
env['PYTHONPATH']=':'.join(str(workspace/x) for x in ['v2d_cari4d/lib/cari4d','v2d_sam3d_body/lib','v2d_foundation_pose/lib/FoundationPose'])
'''))
b.cells.append(nb.v4.new_code_cell('''test="""
import json, torch, kaolin, pytorch3d, nvdiffrast.torch as dr
from pytorch3d.ops import knn_points
from pathlib import Path
from v2d.cari4d.lib.run_inference import run_inference
x=torch.rand(1,32,3,device='cuda');r=knn_points(x,x,K=1)
assert r.dists.max().item()<1e-6
ctx=dr.RasterizeCudaContext()
report={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'knn_gpu':True,'rasterizer_context':True,'cari4d_import':True}
Path('/kaggle/working/track1_runtime_report.json').write_text(json.dumps(report,indent=2));print(report)
"""
subprocess.run([py,'-c',test],env=env,check=True)
Path('/kaggle/working/track1_compiler_manifest.json').write_text(json.dumps({'cuda':compiler_manifest,'runtime_revision':rev,'nvdiffrast_revision':nvdiffrast_revision},indent=2))
'''))
for c in b.cells:
 if c.cell_type=='code':ast.parse(c.source)
nb.validate(b);dest=root/'notebooks/track1_runtime';dest.mkdir(exist_ok=True)
nb.write(b,dest/'runtime.ipynb')
meta=json.loads((root/'notebooks/reconstruction_access/kernel-metadata.json').read_text())
meta.update(code_file='runtime.ipynb',enable_gpu=True,machine_shape='NvidiaTeslaT4')
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
