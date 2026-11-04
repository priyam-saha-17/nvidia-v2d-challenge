from pathlib import Path
import json
import nbformat as nb
root=Path(__file__).resolve().parents[1]
dest=root/'notebooks/runtime_probe'; dest.mkdir(exist_ok=True)
b=nb.v4.new_notebook()
b.cells=[nb.v4.new_markdown_cell('''# V2D isolated FlashCHORD runtime compatibility
Installs the pinned runtime on Python 3.11 and tests JAX and Warp on T4.
This is an infrastructure check, not competition training or a submission.'''),nb.v4.new_code_cell('''import subprocess, sys
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
subprocess.run(['uv','python','install','3.11'],check=True)
subprocess.run(['uv','venv','--python','3.11','/kaggle/working/fc-env'],check=True)
'''),nb.v4.new_code_cell('''import os
runtime=Path('/kaggle/working/video_to_data')
env=os.environ.copy(); env['GIT_LFS_SKIP_SMUDGE']='1'
subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/nvidia-isaac/video_to_data.git',str(runtime)],check=True,env=env)
subprocess.run(['git','sparse-checkout','set','robotic_grounding/flash_chord'],cwd=runtime,check=True,env=env)
subprocess.run(['git','checkout','8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef'],cwd=runtime,check=True,env=env)
fc=runtime/'robotic_grounding/flash_chord'
py='/kaggle/working/fc-env/bin/python'
subprocess.run(['uv','pip','install','--python',py,'-e',str(fc)+'[train]'],check=True)
'''),nb.v4.new_code_cell('''probe = """
import json
from pathlib import Path
import jax, jax.numpy as jnp
import warp as wp
import newton
import flash_chord
wp.init()
a=jnp.arange(256,dtype=jnp.float32)
value=float(jnp.sum(a*a).block_until_ready())
report={'jax_devices':[str(d) for d in jax.devices()],
 'jax_sum_squares':value,'warp_cuda_available':wp.is_cuda_available(),
 'newton_import':True,'flash_chord_import':True}
Path('/kaggle/working/flashchord_runtime_report.json').write_text(json.dumps(report,indent=2))
assert any(d.platform=='gpu' for d in jax.devices())
assert report['warp_cuda_available']
print(json.dumps(report,indent=2))
"""
subprocess.run([py,'-c',probe],check=True)
''')]
b.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
nb.validate(b); nb.write(b,dest/'probe.ipynb')
meta=json.loads((root/'notebooks/t4_diagnostic/kernel-metadata.json').read_text())
meta.update(id='priyamsaha17/v2d-flashchord-runtime-probe',title='V2D FlashCHORD Runtime Probe',code_file='probe.ipynb')
(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(dest)
