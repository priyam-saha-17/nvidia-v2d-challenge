"""Finish genuine cached predictions after the public topology asset repair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def finish(episode, device, args):
    started = time.monotonic()
    name = f'episode_{episode:06d}'
    archives = list(args.input.rglob(name + '_failed_intermediates.tar.gz'))
    if len(archives) != 1:
        raise ValueError(f'Expected one actual cached failed run for {name}')
    root = args.output / 'native_outputs' / name
    root.mkdir(parents=True, exist_ok=True)
    files = {'bundle': root / 'coconet.pth', 'mesh': root / 'output_aligned.glb'}
    found = set()
    with tarfile.open(archives[0], 'r|gz') as archive:
        for member in archive:
            key = ('bundle' if member.name.endswith('/inference/coconet.pth') else
                   'mesh' if member.name.endswith('/object_mesh/output_aligned.glb') else None)
            if key and '/final/' in member.name and member.isfile():
                if key in found:
                    raise ValueError('Ambiguous cached final prediction')
                with archive.extractfile(member) as source, files[key].open('wb') as target:
                    shutil.copyfileobj(source, target)
                found.add(key)
    if found != set(files):
        raise ValueError(f'Cache lacks completed CoCoNet or calibrated mesh: {name}')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(device), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    # Isolate Torch CUDA contexts by launching the real refinement in a child.
    projection = root / 'shared_identity.pth'
    project_code = '''import torch,numpy as np,sys,json,hashlib
from pathlib import Path
source,target=map(Path,sys.argv[1:]);bundle=torch.load(source,map_location='cpu',weights_only=False)
assert bundle['metadata']['ground_truth_used'] is False
frames=len(bundle['frames']);assert list(map(int,bundle['frames']))==list(range(frames))
shared={k:np.median(np.asarray(bundle['pr'][k]),axis=0).astype(np.float32) for k in ('mhr_shape','mhr_scale')}
for section in ('pr','pr_initial'):
 if section not in bundle:continue
 for k,v in shared.items():bundle[section][k]=np.repeat(v[None],frames,axis=0)
 bundle[section]['mhr_face']=np.zeros((frames,72),np.float32)
bundle['metadata']['shared_identity_projection']={'method':'per-clip median identity; zero expression matching scorer','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'ground_truth_used':False}
torch.save(bundle,target)
target.with_suffix('.json').write_text(json.dumps({'frames':frames,'projection':bundle['metadata']['shared_identity_projection']},indent=2))
'''
    subprocess.run([sys.executable, '-c', project_code, str(files['bundle']), str(projection)], env=env, check=True)
    frames = json.loads(projection.with_suffix('.json').read_text())['frames']
    refined = root / 'refined.pth'
    command = [sys.executable, '-m', 'learning.training.mhr_opt_refineout',
               '--bundle', str(projection), '--object-mesh', str(files['mesh']),
               '--out', str(refined), '--mode', 'smplh_parity', '--num-steps', '300',
               '--batch-size', '32', '--device', 'cuda',
               '--postopt-checkpoint', str(root / 'refinement_checkpoint.pth'),
               '--mhr-assets-root', str(args.weights / 'sam3d_body'),
               '--w-temporal', '100', '--w-human-pose-prior', '200',
               '--contact-activation-distance-m', '0.05', '--report-every', '10',
               '--diagnostics-every', '500']
    log = args.output / (name + '.log')
    with log.open('w') as stream:
        subprocess.run(command, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, str(Path(__file__).with_name('export_cari4d.py')),
                        '--bundle', str(refined), '--mesh', str(files['mesh']),
                        '--mhr-assets-root', str(args.weights / 'sam3d_body'),
                        '--episode', str(episode), '--frames', str(frames),
                        '--output', str(args.output / 'episodes')], env=env, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    record = {'episode': episode, 'frames': frames, 'success': True,
              'recovered_stage': 'completed actual CoCoNet prediction',
              'source_archive_sha256': sha256(archives[0]),
              'source_bundle_sha256': sha256(files['bundle']),
              'elapsed_seconds': time.monotonic() - started}
    print('RECOVERED_EPISODE', json.dumps(record), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, nargs='+', required=True)
    p.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    a = p.parse_args()
    if len(a.episodes) > 2 or len(set(a.episodes)) != len(a.episodes):
        p.error('Recover at most two distinct episodes on two separate T4 devices')
    a.output.mkdir(parents=True, exist_ok=True)
    def recover_one(episode, device):
        try:
            return finish(episode, device, a)
        except Exception as error:
            result = {'episode': episode, 'success': False,
                      'error': type(error).__name__ + ': ' + str(error)}
            print('RECOVERY_FAILED', json.dumps(result), flush=True)
            log = a.output / f'episode_{episode:06d}.log'
            if log.exists():
                print(log.read_text(errors='replace')[-5000:], flush=True)
            return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(recover_one, episode, device) for device, episode in enumerate(a.episodes)]
        records = [future.result() for future in futures]
    (a.output / 'batch_reconstruction_report.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
