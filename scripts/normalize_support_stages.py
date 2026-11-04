"""Preserve explicitly empty support stages outside the implicit resolver path.

The released ground-only stages are valid USD Xform containers with no shapes.
The pinned FlashCHORD support loader instead rejects a present stage with no
Cube/Cylinder. Its normal no-stage path already creates the scene ground plane.
Rename only truly empty Xform/Scope stages; never omit unsupported geometry.
Run inside the FlashCHORD environment, which supplies usd-core.
"""
import argparse
import hashlib
import json
from pathlib import Path
from pxr import Usd

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
records = []
for path in sorted(a.directory.glob('*_support.usda')):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f'Cannot open support stage: {path}')
    types = {prim.GetTypeName() for prim in stage.Traverse()}
    if types & {'Cube', 'Cylinder'}:
        continue
    if not types.issubset({'Xform', 'Scope', ''}):
        raise ValueError(f'Unsupported geometry must not be omitted: {path}: {types}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = path.with_suffix('.ground_only.usda')
    if saved.exists():
        raise FileExistsError(saved)
    path.rename(saved)
    records.append({'original': str(path), 'preserved': str(saved),
                    'sha256': digest, 'reason': 'released empty ground-only stage'})
report = a.directory/'ground_only_stages.json'
report.write_text(json.dumps(records, indent=2)+'\n')
print(f'Preserved {len(records)} empty stages; simulator ground remains enabled')
