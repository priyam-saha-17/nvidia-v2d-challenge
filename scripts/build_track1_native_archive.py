"""Prepare CPU compression of completed actual native Track 1 outputs."""
import ast
import json
from pathlib import Path
import nbformat as nb

root = Path(__file__).resolve().parents[1]
source = (root / 'track1/archive_native_outputs.py').read_text()
code = "import sys,subprocess\nfrom pathlib import Path\n"
code += f"script=Path('/kaggle/working/archive_native_outputs.py');script.write_text({source!r})\n"
code += "subprocess.run([sys.executable,str(script)],check=True)\n"
ast.parse(code)
notebook = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(
    '# Track 1 generated artifact retention\n'
    'Compress genuine prediction bundles, final NPZ/GLB, previews and provenance on CPU. '
    'Inspect prediction schemas; exclude pretrained weights and credentials.'), nb.v4.new_code_cell(code)])
nb.validate(notebook)
directory = root / 'notebooks/track1_native_archive'
directory.mkdir(parents=True, exist_ok=True)
nb.write(notebook, directory / 'archive.ipynb')
metadata = {'id': 'priyamsaha17/v2d-track1-native-artifact-archive',
            'title': 'V2D Track1 Native Artifact Archive', 'code_file': 'archive.ipynb',
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': False, 'enable_internet': False, 'dataset_sources': [],
            'competition_sources': [], 'kernel_sources': [
                'priyamsaha17/v2d-track1-inference-main-a',
                'priyamsaha17/v2d-track1-inference-main-b']}
(directory / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Prepared only; launch after both complete native production rosters pass:', directory)
