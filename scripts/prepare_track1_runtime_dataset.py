"""Verify a recovered runtime cache and prepare private Kaggle dataset metadata."""
from pathlib import Path
import hashlib,json,os,tarfile
root=Path(__file__).resolve().parents[1]
source=root/'outputs/track1_runtime_cache'
manifest=json.loads((source/'runtime_cache_manifest.json').read_text())
archive=source/'track1_runtime.tar.gz'
if archive.stat().st_size!=manifest['bytes']:raise ValueError('Runtime cache download incomplete')
digest=hashlib.sha256()
with archive.open('rb') as handle:
 for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
if digest.hexdigest()!=manifest['sha256']:raise ValueError('Runtime cache download hash mismatch')
allowed=tuple(path.lstrip('/').rstrip('/')+'/' for path in manifest['paths'])
forbidden={'token','access_token','kaggle.json','.env','credentials'}
count=0
with tarfile.open(archive) as handle:
 for item in handle:
  if not (item.name.rstrip('/')+'/').startswith(allowed):raise ValueError('Unexpected archived path')
  if item.isfile() and Path(item.name).name in forbidden:raise ValueError('Potential credential filename in cache')
  count+=1
out=root/'outputs/track1_runtime_dataset';out.mkdir(exist_ok=True)
# A binary suffix prevents Kaggle's upload service from expanding this tar archive.
path=out/'track1_runtime.bin'
if not path.exists():os.link(archive,path)
(out/'runtime_cache_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'dataset-metadata.json').write_text(json.dumps({
 'title':'V2D Track1 Private Native Runtime',
 'id':'priyamsaha17/v2d-track1-private-native-runtime',
 'licenses':[{'name':'other'}],
 'description':'Private compute cache: Python 3.11, PyTorch 2.5.1 CUDA 12.4, SM7.5 native reconstruction dependencies and source revision 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef. Contains original third-party license files. No pretrained model weights or credentials. Not a relicensing of third-party components.'},indent=2)+'\n')
print('Runtime cache verified:',manifest['bytes'],'bytes;',count,'archive entries')
