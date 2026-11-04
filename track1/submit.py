"""Pack, preflight and optionally upload a complete Track 1 reconstruction.

All five leaderboards receive the same file through NVIDIA's official uploader.
Actual reconstruction outputs and a producing GitHub commit are required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
KIT=ROOT/'references/submission_kit/v2d_submission_kit'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--message',default='Track 1 video-derived reconstruction')
    p.add_argument('--upload',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'https://github\.com/priyam-saha-17/nvidia-v2d-challenge/commit/[0-9a-f]{40}',a.commit):
        p.error('Provide the real producing commit URL in the competition repository')
    sample=KIT/'data/track_1_sample_submission.parquet'
    rows=pd.read_parquet(sample,columns=['row_id'])['row_id']
    ids=sorted(rows.str.extract(r'^t1_(\d+)_',expand=False).dropna().astype(int).unique())
    if len(ids)!=30:raise ValueError(f'Unexpected official roster: {ids}')
    episodes=a.episodes.resolve()
    if (episodes/'common_translation_filter_manifest.json').is_file():
        check=json.loads((episodes/'common_translation_decoder_check.json').read_text())
        if check.get('passed') is not True or {x['episode'] for x in check['episodes']}!=set(range(30)):
            raise ValueError('Filtered candidate requires its successful actual reference decoder check')
        for entry in check['episodes']:
            stem=f"episode_{entry['episode']:06d}"
            for suffix,field in [('.npz','output_npz_sha256'),('_object.glb','mesh_sha256')]:
                path=episodes/(stem+suffix)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry[field]:
                    raise ValueError('Filtered candidate differs from the reference-validated artifact: '+path.name)
    for episode in ids:
        stem=episodes/f'episode_{episode:06d}'
        with np.load(stem.with_suffix('.npz'),allow_pickle=False) as values:
            timing=ROOT/f'data/challenge/track_1/data/chunk-000/episode_{episode:06d}.parquet'
            expected=len(pd.read_parquet(timing,columns=['frame_index']))
            if values['pose'].shape!=(expected,136):
                raise ValueError(f'Episode {episode}: incomplete original video timeline')
        if not (episodes/(stem.name+'_object.glb')).is_file():
            raise FileNotFoundError(f'Missing reconstructed mesh for episode {episode}')
    a.output.mkdir(parents=True,exist_ok=True)
    output=a.output.resolve();packed=output/'track1.parquet'
    subprocess.run([sys.executable,str(KIT/'eval_reconstruction.py'),
        '--episodes',str(episodes),'--sample',str(sample),'--commit',a.commit,
        '--out',str(packed)],cwd=KIT,check=True)
    subprocess.run([sys.executable,str(KIT/'tools/preflight.py'),'--track','track_1'],cwd=KIT,check=True)
    upload=[sys.executable,str(KIT/'submit/v2d_submit.py'),str(packed),
            '--track','track_1','-m',a.message]
    subprocess.run([*upload,'--dry-run'],cwd=KIT,check=True)
    digest=hashlib.sha256(packed.read_bytes()).hexdigest()
    manifest={'track':'track_1','code_commit_url':a.commit,'submission_sha256':digest,
              'submission_bytes':packed.stat().st_size,'episodes':list(map(int,ids))}
    (output/'submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not a.upload:
        print('Official packing and preflight passed. No upload requested.');return
    receipt=output/('receipt-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    if receipt.exists():raise FileExistsError(receipt)
    subprocess.run([*upload,'--wait','--receipt',str(receipt)],cwd=KIT,check=True)
    print('Official upload/score receipt:',receipt)


if __name__=='__main__':main()
