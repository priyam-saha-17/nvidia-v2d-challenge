"""Collect complete, real production episodes without overwriting provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True,
                        help='Kaggle notebook/version=download-directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for specification in args.source:
        notebook, directory = specification.split('=', 1)
        root = Path(directory)
        direct = root / 'batch_reconstruction_report.json'
        reports = [direct] if direct.is_file() else list(root.rglob('batch_reconstruction_report.json'))
        if len(reports) != 1:
            raise ValueError(f'{notebook}: expected one production batch report')
        for entry in json.loads(reports[0].read_text()):
            if not entry['success']:
                print(f"INCOMPLETE {notebook} episode {entry['episode']}: {entry.get('error')}")
                continue
            episode = entry['episode']
            if episode not in range(30) or episode in records:
                raise ValueError(f'Duplicate or invalid episode: {episode}')
            stem = f'episode_{episode:06d}'
            files = {}
            for suffix in ('.npz', '_object.glb', '.report.json'):
                exported = root / 'episodes' / (stem + suffix)
                matches = [exported] if exported.is_file() else list(root.rglob(stem + suffix))
                if len(matches) != 1:
                    raise ValueError(f'{notebook}: missing/ambiguous {stem + suffix}')
                source = matches[0]
                target = args.output / source.name
                checksum = digest(source)
                if target.exists() and digest(target) != checksum:
                    raise ValueError(f'Refusing to overwrite a different artifact: {target}')
                shutil.copy2(source, target)
                files[source.name] = {'sha256': checksum, 'bytes': source.stat().st_size}
            report = json.loads((args.output / (stem + '.report.json')).read_text())
            if report['ground_truth_used'] is not False or report['frames'] != entry['frames']:
                raise ValueError(f'Invalid export provenance for {stem}')
            if report['mesh_sha256'] != files[stem + '_object.glb']['sha256']:
                raise ValueError(f'Mesh differs from validated export: {stem}')
            if not 0 <= report['max_native_round_trip_error_m'] <= 0.0005:
                raise ValueError(f'Native MHR decoder check failed: {stem}')
            records[episode] = {'notebook': notebook, 'batch': entry, 'files': files}
    missing = sorted(set(range(30)) - records.keys())
    manifest = {'episodes': records, 'missing_episodes': missing,
                'complete': not missing, 'official_packing_required': True}
    (args.output / 'collection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Collected {len(records)}/30 actual episodes. Missing: {missing}')
    if missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
