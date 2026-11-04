"""Create a separate candidate by smoothing shared human/object translation.

Both surfaces receive the same per-frame shift; the first frame is fixed. This
preserves relative contact geometry and the scorer's first-frame alignment, up
to float32 rounding. A reference MHR decoder check is required before submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


FLIP = np.array([1., -1., -1.])
# The official MHR converter specifies 100 mm per native root translation unit.
NATIVE_TRANSLATION_TO_METRES = 0.1 * FLIP


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchored_smooth(values, strength):
    length = len(values)
    if length < 3:
        return values.copy()
    difference = sparse.diags([np.ones(length - 2), -2 * np.ones(length - 2),
                               np.ones(length - 2)], [0, 1, 2], shape=(length - 2, length))
    system = (sparse.eye(length) + strength * difference.T @ difference).tocsc()
    result = values.copy()
    right = values[1:] - system[1:, :1].toarray() * values[0]
    result[1:] = spsolve(system[1:, 1:], right)
    return result


def acceleration_energy(values):
    return float(np.square(np.diff(values, n=2, axis=0)).sum(axis=1).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--strength', type=float, default=4., help='Second-difference penalty at the released 30 Hz timeline')
    p.add_argument('--max-correction-m', type=float, default=0.05)
    a = p.parse_args()
    if not np.isfinite(a.strength) or a.strength <= 0 or not 0 < a.max_correction_m <= .1:
        p.error('Use positive finite strength and a correction bound of at most 10 cm')
    if a.episodes.resolve() == a.output.resolve() or (a.output.exists() and any(a.output.iterdir())):
        p.error('Use a separate empty output directory; never overwrite the source')
    for episode in range(30):
        for suffix in ('.npz', '_object.glb'):
            if not (a.episodes / (f'episode_{episode:06d}' + suffix)).is_file():
                raise FileNotFoundError(f'Incomplete actual source roster: episode {episode}')
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for episode in range(30):
        stem = f'episode_{episode:06d}'
        source = a.episodes / (stem + '.npz')
        with np.load(source, allow_pickle=False) as payload:
            values = {key: payload[key].copy() for key in payload.files}
        pose = values['pose']
        object_translation = values['object_translation']
        if pose.shape != (len(pose), 136) or object_translation.shape != (len(pose), 3):
            raise ValueError(f'Unexpected actual trajectory shape: {stem}')
        if not np.isfinite(pose).all() or not np.isfinite(object_translation).all():
            raise ValueError(f'Nonfinite source prediction: {stem}')
        if pose.dtype != np.float32 or object_translation.dtype != np.float32:
            raise ValueError('Require the validated float32 native exporter outputs')
        human_root = pose[:, :3].astype(np.float64) * NATIVE_TRANSLATION_TO_METRES
        object_root = object_translation.astype(np.float64)
        common = (human_root + object_root) / 2
        correction = anchored_smooth(common, a.strength) - common
        maximum = float(np.linalg.norm(correction, axis=1).max())
        gain = min(1., a.max_correction_m / max(maximum, 1e-12))
        correction *= gain
        correction[0] = 0
        filtered_pose = pose.copy()
        filtered_pose[:, :3] = human_root / NATIVE_TRANSLATION_TO_METRES + correction / NATIVE_TRANSLATION_TO_METRES
        filtered_object = (object_root + correction).astype(np.float32)
        np.testing.assert_array_equal(filtered_pose[0], pose[0])
        np.testing.assert_array_equal(filtered_object[0], object_translation[0])
        np.testing.assert_array_equal(filtered_pose[:, 3:], pose[:, 3:])
        actual_human = filtered_pose[:, :3].astype(np.float64) * NATIVE_TRANSLATION_TO_METRES
        np.testing.assert_allclose(filtered_object.astype(np.float64) - actual_human,
                                   object_root - human_root, rtol=0, atol=2e-6)
        before = acceleration_energy(common)
        after = acceleration_energy((actual_human + filtered_object) / 2)
        if after > before + 1e-10:
            raise ValueError('Shared translation acceleration energy increased')
        values['pose'] = filtered_pose
        values['object_translation'] = filtered_object
        target = a.output / source.name
        np.savez_compressed(target, **values)
        mesh = a.episodes / (stem + '_object.glb')
        shutil.copy2(mesh, a.output / mesh.name)
        record = {'episode': episode, 'frames': len(pose), 'source_npz_sha256': digest(source),
                  'output_npz_sha256': digest(target), 'mesh_sha256': digest(mesh),
                  'strength': a.strength, 'correction_gain': gain,
                  'maximum_correction_m': float(np.linalg.norm(correction, axis=1).max()),
                  'median_correction_m': float(np.median(np.linalg.norm(correction, axis=1))),
                  'common_acceleration_energy_before': before, 'common_acceleration_energy_after': after,
                  'first_frame_unchanged': True, 'shared_geometry_translation': True,
                  'reference_decoder_check_required': True, 'ground_truth_used': False}
        (a.output / (stem + '.report.json')).write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
    (a.output / 'common_translation_filter_manifest.json').write_text(json.dumps({
        'method': 'first-frame-anchored common translation smoothing',
        'native_translation_unit_m': .1, 'native_to_camera_axes': FLIP.tolist(),
        'held_out_accuracy_improvement_established': False,
        'reference_decoder_check_required': True, 'episodes': records}, indent=2) + '\n')
    print('Created a separate 30-episode candidate; reference decoder validation and official packing remain required.')


if __name__ == '__main__':
    main()
