"""Validate a real translation-filter candidate with the actual reference MHR."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--mhr-model', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    model_sha = digest(a.mhr_model)
    if model_sha != '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc':
        raise ValueError('Use the exact reference MHR model verified by the native exporter')
    torch.set_num_threads(2)
    model = torch.jit.load(str(a.mhr_model), map_location='cpu').eval()
    flip = torch.tensor([1., -1., -1.])
    records = []
    for episode in range(30):
        stem = f'episode_{episode:06d}'
        before_path, after_path = a.before / (stem + '.npz'), a.after / (stem + '.npz')
        with np.load(before_path, allow_pickle=False) as payload:
            before = {k: payload[k].copy() for k in payload.files}
        with np.load(after_path, allow_pickle=False) as payload:
            after = {k: payload[k].copy() for k in payload.files}
        if set(before) != set(after):
            raise ValueError('Unexpected candidate parameter fields')
        for key in set(before) - {'pose', 'object_translation'}:
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(before['pose'][:, 3:], after['pose'][:, 3:])
        np.testing.assert_array_equal(before['pose'][0], after['pose'][0])
        np.testing.assert_array_equal(before['object_translation'][0], after['object_translation'][0])
        length = len(before['pose'])
        delta = after['object_translation'].astype(np.float64) - before['object_translation']
        indices = sorted({0, length // 2, length - 1, int(np.linalg.norm(delta, axis=1).argmax())})
        identity = torch.tensor(np.repeat(before['shape'][None], len(indices), axis=0), dtype=torch.float32)
        expression = torch.zeros(len(indices), 72)

        def decode(values):
            parameters = np.concatenate([values['pose'][indices],
                                         np.repeat(values['scales'][None], len(indices), axis=0)], axis=1)
            with torch.inference_mode():
                vertices, _ = model(identity, torch.tensor(parameters, dtype=torch.float32), expression, True)
            return vertices * flip / 100.

        actual_delta = (decode(after) - decode(before)).numpy()
        error = np.linalg.norm(actual_delta - delta[indices, None, :], axis=-1)
        maximum = float(error.max())
        if not np.isfinite(error).all() or maximum > 0.00005:
            raise ValueError(f'{stem}: human/object shift differs by {maximum} m')
        before_mesh = a.before / (stem + '_object.glb')
        after_mesh = a.after / (stem + '_object.glb')
        if digest(before_mesh) != digest(after_mesh):
            raise ValueError('Candidate mesh changed')
        records.append({'episode': episode, 'frames': length, 'reference_checked_frames': indices,
                        'maximum_reference_shift_error_m': maximum,
                        'source_npz_sha256': digest(before_path),
                        'output_npz_sha256': digest(after_path), 'mesh_sha256': digest(after_mesh)})
        print('REFERENCE_TRANSLATION_CHECK', episode, maximum, flush=True)
    report = {'passed': True, 'mhr_model_sha256': model_sha,
              'checker_script_sha256': digest(Path(__file__)), 'torch_version': torch.__version__,
              'first_frame_and_nontranslation_parameters_unchanged': True,
              'geometry_claim': 'Shared human/object shifts verified against actual MHR on sampled real frames; all pose controls except translation are exactly unchanged.',
              'held_out_accuracy_improvement_established': False, 'episodes': records}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
