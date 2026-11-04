"""Convert one released Track 2 tier to the official SOMA-to-G1 input layout.

No clean-reference lookup is performed for another tier. Object visibility gaps
are filled by interpolation solely from that tier's observed poses. Human poses
remain in their released SOMA convention; object poses remain OpenCV world poses,
as required by soma_to_g1.py's explicit CV-to-SOMA transform.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
info = json.loads((a.input/'meta/info.json').read_text())
names = info['features']['observation.human.pose']['names']
joint_names = [name.rsplit('.', 1)[0] for name in names[::3]]
assert len(joint_names) == 77
for line in (a.input/'meta/episodes_metadata.jsonl').read_text().splitlines():
    meta = json.loads(line)
    episode = int(meta['episode_index'])
    frame = pd.read_parquet(a.input/f'data/chunk-000/episode_{episode:06d}.parquet')
    dest = a.output/meta['sequence_id']
    # Require the original mesh/ground plane; never silently substitute primitives.
    mesh = a.input/meta['mesh']
    plane = a.input/meta['ground_plane']
    if not mesh.is_file() or not plane.is_file():
        raise FileNotFoundError(f'Missing released mesh or ground plane for episode {episode}')
    dest.mkdir(parents=True, exist_ok=True)
    stack = lambda key: np.stack(frame[key]).astype(np.float32)
    np.savez_compressed(dest/'soma_params.npz',
        poses=stack('observation.human.pose').reshape(-1,77,3),
        transl=stack('observation.human.translation'),
        identity_coeffs=stack('observation.human.identity_coeffs'),
        scale_params=stack('observation.human.scale_params'),
        bone_length_flexibles=stack('observation.human.bone_length_flexibles'),
        joint_names=np.asarray(joint_names), identity_model_type='mhr',
        unit='meters',keep_root=False,absolute_pose=False)
    pose = stack('observation.object.pose')
    valid = frame['observation.object.visible'].to_numpy(bool, copy=True)
    valid &= np.isfinite(pose).all(axis=1) & (np.linalg.norm(pose[:,3:],axis=1)>1e-8)
    ids = np.flatnonzero(valid)
    if len(ids)<2:
        raise ValueError(f'Insufficient object observations for episode {episode}')
    t = frame['timestamp'].to_numpy(float)
    translation=np.column_stack([np.interp(t,t[ids],pose[ids,i]) for i in range(3)])
    rotation=Slerp(t[ids],Rotation.from_quat(pose[ids,3:],scalar_first=True))(np.clip(t,t[ids[0]],t[ids[-1]]))
    transforms=np.tile(np.eye(4,dtype=np.float32),(len(t),1,1))
    transforms[:,:3,:3]=rotation.as_matrix();transforms[:,:3,3]=translation
    np.save(dest/'poses.npy',transforms)
    (dest/'reconstructed_mesh').mkdir(exist_ok=True)
    shutil.copy2(mesh,dest/'reconstructed_mesh/output_aligned.glb')
    shutil.copy2(plane,dest/'ground_plane.json')
    (dest/'source.json').write_text(json.dumps({'episode_index':episode,
        'source_tier':str(a.input.resolve()),'sequence_id':meta['sequence_id'],
        'fps':info['fps'],'interpolated_object_frames':int((~valid).sum()),
        'support_surface':meta['support_surface']},indent=2)+'\n')
    print(f'episode {episode}: {len(frame)} frames; {int((~valid).sum())} missing poses interpolated')
