"""Download public tables/metadata, preserving track/tier directories.

Videos and mesh assets are excluded. Use the authoritative HF tree inventory;
download to a temporary file and verify the declared size before publishing it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
import shutil

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--workers', type=int, default=4)
p.add_argument('--prefix', default='')
p.add_argument('--assets-only', action='store_true')
a = p.parse_args()
inventory = json.loads((ROOT/'references/dataset_tree.json').read_text())
suffixes = ('.glb', '.usda') if a.assets_only else ('.parquet', '.json', '.jsonl', '.md')
files = [x for x in inventory if x['type'] == 'file' and
         x['path'].startswith(a.prefix) and x['path'].endswith(suffixes)]

def download(item):
    relative = Path(item['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe inventory path')
    target = ROOT/'data/challenge'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == item['size']:
        return
    tmp = target.with_suffix(target.suffix+'.partial')
    url = 'https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/'+quote(item['path'], safe='/')
    with urlopen(url, timeout=120) as response, tmp.open('wb') as out:
        shutil.copyfileobj(response, out)
    if tmp.stat().st_size != item['size']:
        raise ValueError(f'Size mismatch: {relative}')
    tmp.replace(target)

with ThreadPoolExecutor(max_workers=a.workers) as pool:
    list(pool.map(download, files))
print(f'Downloaded or verified {len(files)} public files')
