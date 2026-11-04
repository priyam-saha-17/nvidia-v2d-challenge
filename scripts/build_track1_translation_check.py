"""Prepare an offline CPU reference check of actual before/after candidates."""
import argparse
import ast
import json
from pathlib import Path
import nbformat as nb

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', required=True, help='Private Kaggle dataset with actual before/after episode directories')
a = p.parse_args()
if not a.dataset.startswith('priyamsaha17/'):
    p.error('Use the authorized Kaggle account')
root = Path(__file__).resolve().parents[1]
source = (root / 'track1/check_common_translation.py').read_text()
code = '''import sys,subprocess,json
from pathlib import Path
inputs=Path('/kaggle/input')
models=list(inputs.rglob('mhr_model.pt'));assert len(models)==1,models
before=list(p.parent for p in inputs.rglob('episode_000000.npz') if p.parent.name=='before')
after=list(p.parent for p in inputs.rglob('episode_000000.npz') if p.parent.name=='after')
assert len(before)==len(after)==1,(before,after)
checker=Path('/kaggle/working/check_common_translation.py')
'''
code += f'checker.write_text({source!r})\n'
code += '''subprocess.run([sys.executable,str(checker),'--before',str(before[0]),'--after',str(after[0]),
 '--mhr-model',str(models[0]),'--report','/kaggle/working/common_translation_decoder_check.json'],check=True)
'''
ast.parse(code)
notebook = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(
    '# Track 1 actual translation candidate verification\n'
    'Read actual before/after predictions and the licensed reference MHR offline on CPU. '
    'This checks geometry invariance; it does not establish held-out accuracy.'), nb.v4.new_code_cell(code)])
nb.validate(notebook)
directory = root / 'notebooks/track1_translation_check'
directory.mkdir(parents=True, exist_ok=True)
nb.write(notebook, directory / 'check.ipynb')
metadata = {'id': 'priyamsaha17/v2d-track1-translation-reference-check',
            'title': 'V2D Track1 Translation Reference Check', 'code_file': 'check.ipynb',
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': False, 'enable_internet': False,
            'dataset_sources': ['priyamsaha17/v2d-track1-private-pretrained-models', a.dataset],
            'competition_sources': [], 'kernel_sources': []}
(directory / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Prepared only; actual candidates must be generated, uploaded privately and verified before scoring:', directory)
