"""Build a secret-free Kaggle hardware/runtime diagnostic notebook."""
import argparse
import json
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--owner', required=True, help='Authenticated Kaggle username')
args = parser.parse_args()
dest = ROOT / 'notebooks/t4_diagnostic'
dest.mkdir(parents=True, exist_ok=True)
book = nb.v4.new_notebook()
book.cells = [nb.v4.new_markdown_cell('''# V2D T4 runtime diagnostic
This notebook verifies hardware, CUDA and public data access before using GPU
quota for reconstruction or policy training. It produces **no competition submission**.
The official FlashCHORD runtime is pinned to
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. Full training was validated upstream
on a 48 GiB GPU; two T4s do not provide a single combined memory pool.
No API credentials belong in this notebook.'''), nb.v4.new_code_cell('''import json, sys, platform, subprocess, urllib.request
from pathlib import Path
report = {'python': sys.version, 'platform': platform.platform()}
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                      '--format=csv,noheader'], capture_output=True, text=True)
report['gpu'] = gpu.stdout.strip()
report['gpu_error'] = gpu.stderr.strip()
print(json.dumps(report, indent=2))
assert gpu.returncode == 0 and 'T4' in gpu.stdout, 'Requested T4 hardware is not available'
'''), nb.v4.new_code_cell('''import torch
report['torch'] = torch.__version__
report['torch_cuda'] = torch.version.cuda
assert torch.cuda.is_available()
x = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
y = x @ x.T
torch.cuda.synchronize()
report['tensor_test'] = bool(torch.isfinite(y).all().item())
del x, y
torch.cuda.empty_cache()
print('CUDA tensor test:', report['tensor_test'])
'''), nb.v4.new_code_cell('''url = 'https://huggingface.co/api/datasets/nvidia/video_to_data_challenge/tree/main?recursive=true&limit=1000'
with urllib.request.urlopen(url, timeout=60) as response:
    inventory = json.load(response)
report['dataset_files'] = len([x for x in inventory if x['type'] == 'file'])
report['dataset_bytes'] = sum(x.get('size', 0) for x in inventory if x['type'] == 'file')
Path('/kaggle/working/runtime_report.json').write_text(json.dumps(report, indent=2))
Path('/kaggle/working/dataset_inventory.json').write_text(json.dumps(inventory, indent=2))
print(json.dumps(report, indent=2))
''')]
book.metadata['kernelspec'] = {'name':'python3', 'display_name':'Python 3', 'language':'python'}
nb.validate(book)
nb.write(book, dest / 'diagnostic.ipynb')
metadata = {'id': f'{args.owner}/v2d-t4-runtime-diagnostic',
 'title': 'V2D T4 Runtime Diagnostic', 'code_file': 'diagnostic.ipynb',
 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
 'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
 'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}
(dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(dest)
