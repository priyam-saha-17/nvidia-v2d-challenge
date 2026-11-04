"""Compress actual episode artifacts on Kaggle CPU for GitHub publication.

Only authored output directories are read. Pretrained checkpoints, runtime caches
and credentials are excluded. Prediction PTH schemas are checked before archiving.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

import torch


PREDICTIONS = {'coconet.pth', 'shared_identity.pth', 'refined.pth', 'refinement_checkpoint.pth'}
SCHEMAS = {'cari4d.mhr_wild_inference.v1', 'cari4d.mhr_postopt_smplh_parity.v10'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def prediction_schema(path):
    if path.name not in PREDICTIONS:
        raise ValueError('Unexpected PTH file in generated artifacts: ' + path.name)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema') not in SCHEMAS:
        raise ValueError('Expected actual prediction/pose-optimization output, not pretrained weights')
    if any(key in payload for key in ('state_dict', 'model_state_dict', 'model')):
        raise ValueError('Unexpected model weights in generated artifacts')
    result = {'schema': payload['schema'], 'root_keys': sorted(payload)}
    del payload
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    p.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    a = p.parse_args()
    roster = {}
    for report in a.input.rglob('batch_reconstruction_report.json'):
        for row in json.loads(report.read_text()):
            episode = row['episode']
            if row['success'] is not True or episode not in range(30) or episode in roster:
                raise ValueError('Require 30 unique successful actual episode runs')
            roster[episode] = (report.parent, row)
    if set(roster) != set(range(30)):
        raise ValueError('Incomplete actual roster')
    a.output.mkdir(parents=True, exist_ok=True)
    archives = []
    for episode, (source, run) in sorted(roster.items()):
        stem = f'episode_{episode:06d}'
        files = []
        native = source / 'native_outputs' / stem
        if not native.is_dir():
            raise FileNotFoundError(native)
        for path in native.rglob('*'):
            if not path.is_file():
                continue
            if path.is_symlink() or not (path.suffix in {'.pth', '.pkl', '.json', '.glb'} or path.name.endswith('.sam3d.h5')):
                raise ValueError('Unexpected generated-output file: ' + str(path))
            files.append((path, 'native/' + path.relative_to(native).as_posix()))
        for suffix in ('.npz', '_object.glb', '.report.json'):
            path = source / 'episodes' / (stem + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((path, 'episodes/' + path.name))
        for path in (source / 'previews' / stem).glob('*.mp4'):
            files.append((path, 'previews/' + path.name))
        log = source / (stem + '.log')
        if log.is_file():
            files.append((log, log.name))
        identities = {}
        for path, destination in files:
            item = {'bytes': path.stat().st_size, 'sha256': digest(path)}
            if path.suffix == '.pth':
                item['prediction_inspection'] = prediction_schema(path)
            identities[destination] = item
        archive_path = a.output / (stem + '_native_artifacts.tar.gz')
        if archive_path.exists():
            raise FileExistsError(archive_path)
        manifest = {'episode': episode, 'run': run, 'source_output': str(source.relative_to(a.input)),
                    'pretrained_weights_included': False, 'files': identities}
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / 'manifest.json'
            metadata.write_text(json.dumps(manifest, indent=2) + '\n')
            with tarfile.open(archive_path, 'w:gz', compresslevel=1) as archive:
                archive.add(metadata, arcname='manifest.json')
                for path, destination in files:
                    archive.add(path, arcname=destination, recursive=False)
        record = {'episode': episode, 'archive': archive_path.name,
                  'bytes': archive_path.stat().st_size, 'sha256': digest(archive_path),
                  'pretrained_weights_included': False}
        archives.append(record)
        (a.output / 'native_artifact_manifest.json').write_text(json.dumps(archives, indent=2) + '\n')
        print('NATIVE_ARTIFACT_ARCHIVED', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
