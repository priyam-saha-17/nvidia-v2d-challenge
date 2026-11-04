"""Prepare a credential-free checkpoint archive locally; this never uploads it."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/track1_models'
dest=ROOT/'outputs/track1_model_upload'
dest.mkdir(exist_ok=True)
files=[
 'cari4d/2026-08-25-09-35-57/step200000.pth',
 'cari4d/2026-08-25-09-35-57/resolved_config.yaml',
 'cari4d/2026-08-25-09-35-57/manifest.json',
 'sam3d_body/checkpoints/sam-3d-body-dinov3/LICENSE',
 'sam3d_body/checkpoints/sam-3d-body-dinov3/README.md',
 'sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',
 'sam3d_body/checkpoints/sam-3d-body-dinov3/model_config.yaml',
 'sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt']
manifest=[]
for relative in files:
    path=source/relative
    if not path.is_file():raise FileNotFoundError(path)
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(16*1024*1024):digest.update(chunk)
    manifest.append({'file':relative,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
assert manifest[0]['sha256']=='78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3'
(ROOT/'track1/model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(dest/'track1_models.tar','w') as archive:
    for relative in files:archive.add(source/relative,arcname=relative,recursive=False)
(dest/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
metadata={'title':'V2D Track1 Private Pretrained Models',
 'id':'priyamsaha17/v2d-track1-private-pretrained-models','licenses':[{'name':'other'}],
 'description':'Private compute copy. Original models retain their licenses: https://huggingface.co/nvidia/cari4d_commercial (revision 1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9) and https://huggingface.co/facebook/sam-3d-body-dinov3 (revision 11aaa346c7204874a1cbafe3d39a979080b2c55a). No credentials included. Not a relicensing of these models.'}
(dest/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Local archive prepared:',(dest/'track1_models.tar').stat().st_size,'bytes;',len(files),'explicit files')
