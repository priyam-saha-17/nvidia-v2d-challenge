"""Archive authoritative generated masks and geometry, excluding model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mask-source', action='append', required=True,
                   help='Exact Kaggle notebook/version=local download directory')
    p.add_argument('--shapes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    selected = {}
    for specification in a.mask_source:
        notebook, directory = specification.split('=', 1)
        for report in Path(directory).rglob('segmentation_report.json'):
            episode = int(report.parent.name.removeprefix('episode_'))
            value = json.loads(report.read_text())
            candidate = (bool(value.get('semantic_selection')), notebook, report, value)
            if episode in selected and selected[episode][0] == candidate[0]:
                raise ValueError(f'Ambiguous authoritative masks: episode {episode}')
            if episode not in selected or candidate[0]:
                selected[episode] = candidate
    if set(selected) != set(range(30)):
        raise ValueError('Require all 30 actual mask outputs')
    shape_rows = json.loads((a.shapes / 'shape_dataset_manifest.json').read_text())
    if {row['episode'] for row in shape_rows} != set(range(3, 30)):
        raise ValueError('Require all 27 actual non-hoop source meshes')
    records = []
    files = []
    for episode, (_, notebook, report, value) in sorted(selected.items()):
        stem = report.parent.name
        entry = {'episode': episode, 'notebook': notebook, 'frames': value['frames'],
                 'source_video_sha256': value['video_sha256'],
                 'semantic_repair': bool(value.get('semantic_selection')), 'files': {}}
        for name in ('masks.h5', 'segmentation_report.json', 'object_observation.png'):
            source = report.with_name(name)
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            destination = f'masks/{stem}/{name}'
            entry['files'][destination] = {'sha256': digest(source), 'bytes': source.stat().st_size}
            files.append((source, destination))
        records.append(entry)
    for row in shape_rows:
        folder = a.shapes / f"episode_{row['episode']:06d}"
        if digest(folder / 'normalized_shape.glb') != row['mesh_sha256']:
            raise ValueError('Source geometry hash mismatch')
        for name in ('normalized_shape.glb', 'shape_provenance.json'):
            files.append((folder / name, f'shapes/{folder.name}/{name}'))
    files.append((a.shapes / 'shape_dataset_manifest.json', 'shapes/shape_dataset_manifest.json'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / 'manifest.json'
        manifest.write_text(json.dumps({'stage': 'generated preprocessing artifacts; unscored',
                                       'pretrained_weights_included': False,
                                       'raw_videos_included': False, 'masks': records}, indent=2) + '\n')
        with tarfile.open(a.output, 'w:gz', compresslevel=1) as archive:
            archive.add(manifest, arcname='manifest.json')
            for source, destination in files:
                archive.add(source, arcname=destination, recursive=False)
    print(json.dumps({'archive': str(a.output), 'sha256': digest(a.output),
                      'bytes': a.output.stat().st_size, 'episodes': len(records)}))


if __name__ == '__main__':
    main()
