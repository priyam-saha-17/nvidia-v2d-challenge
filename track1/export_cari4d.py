"""Export an actual native-MHR CARI4D wild-video result to the official kit.

Run in the reconstruction environment with its PYTHONPATH and MHR assets.
This is an adapter, not a reconstruction method. It refuses incomplete timelines
and validates the exported native parameters by decoding the reference model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from lib_mhr import MHRLayer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--mhr-assets-root', type=Path, required=True)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--frames', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.episode < 0 or a.frames < 1 or a.batch_size < 1:
        p.error('Episode, frame count or batch size is invalid')
    if a.mesh.name != 'output_aligned.glb':
        p.error('Use the actual wild-pipeline aligned mesh: output_aligned.glb')
    # This bundle is produced by our pinned inference code, not an untrusted upload.
    bundle = torch.load(a.bundle, map_location='cpu', weights_only=False)
    if bundle.get('metadata', {}).get('ground_truth_used') is not False:
        raise ValueError('Expected explicit no-ground-truth inference provenance')
    frames = [int(f) for f in bundle['frames']]
    if frames != list(range(a.frames)):
        raise ValueError('Bundle must cover every original video frame in order')
    pr = bundle['pr']
    for key in ('mhr_global_rot6d', 'mhr_trans', 'mhr_body_pose_cont',
                'mhr_hand', 'mhr_shape', 'mhr_scale'):
        values = np.asarray(pr[key])
        if len(values) != a.frames or not np.isfinite(values).all():
            raise ValueError(f'Invalid or incomplete parameter array: {key}')
    poses = np.asarray(pr['pose_abs'], dtype=np.float32)
    if poses.shape != (a.frames,4,4) or not np.isfinite(poses).all():
        raise ValueError('Expected finite object poses [T,4,4]')
    np.testing.assert_allclose(poses[:,3,:], np.tile([0,0,0,1],(a.frames,1)), atol=1e-6)
    rotations = poses[:,:3,:3]
    np.testing.assert_allclose(rotations.transpose(0,2,1)@rotations,
                               np.tile(np.eye(3),(a.frames,1,1)), atol=1e-4)
    if (np.linalg.det(rotations) <= 0).any():
        raise ValueError('Object pose contains a reflection')
    metadata = bundle.get('metadata', {})
    transform = np.asarray(metadata.get('object_mesh_to_training_transform', np.eye(4)))
    if not np.allclose(transform, np.eye(4), atol=1e-6):
        raise ValueError('Nonidentity object mesh frame requires explicit geometric conversion')

    layer = MHRLayer.from_mhr_assets(mhr_assets_root=a.mhr_assets_root, device=a.device)
    backend = layer.backend
    model_path = backend.mhr_model_path
    reference = torch.jit.load(str(model_path), map_location=a.device).eval()
    flip = torch.tensor([1.,-1.,-1.], device=a.device)
    # The competition requires one identity for the entire clip.
    shape = np.median(np.asarray(pr['mhr_shape']), axis=0).astype(np.float32)
    latent_scale = np.median(np.asarray(pr['mhr_scale']), axis=0).astype(np.float32)
    exported = []
    scales = None
    errors = []
    for start in range(0,a.frames,a.batch_size):
        stop = min(start+a.batch_size,a.frames)
        params = {k: torch.as_tensor(np.asarray(v)[start:stop],device=a.device,dtype=torch.float32)
                  for k,v in pr.items() if k.startswith('mhr_')}
        count = stop-start
        params['mhr_shape'] = torch.as_tensor(shape,device=a.device).expand(count,-1)
        params['mhr_scale'] = torch.as_tensor(latent_scale,device=a.device).expand(count,-1)
        # Expression is absent from the challenge schema and the scorer uses zero.
        params['mhr_face'] = torch.zeros(count,72,device=a.device)
        with torch.inference_mode():
            context = backend.prepare_vertices_context(params)
            translation,body,identity,scale = backend._mutable_vertices_inputs(params,context)
            zero = torch.zeros_like(translation)
            verts, native = context.head.mhr_forward(
                global_trans=zero,global_rot=context.global_rot,body_pose_params=body,
                hand_pose_params=context.hand,scale_params=scale,shape_params=identity,
                expr_params=params['mhr_face'],return_model_params=True)
            target = verts*flip + translation[:,None,:]
            native = native.clone()
            expr = torch.zeros(count,72,device=a.device)

            def decode(model_params):
                v,_ = reference(identity.contiguous(),model_params.contiguous(),expr,True)
                return v*flip/100.

            base = decode(native)
            # Determine the native root translation units from the model itself.
            columns = []
            for axis in range(3):
                shifted = native.clone();shifted[:,axis] += 1.
                columns.append((decode(shifted)-base).mean(dim=1))
            jacobian = torch.stack(columns,dim=-1)
            delta = torch.linalg.solve(jacobian,(target-base).mean(dim=1).unsqueeze(-1)).squeeze(-1)
            native[:,:3] += delta
            actual = decode(native)
            error = torch.linalg.vector_norm(actual-target,dim=-1).amax(dim=1)
            if not torch.isfinite(error).all() or error.max().item() > 0.0005:
                raise ValueError(f'Native decoder round trip exceeds 0.5 mm: {error.max().item()} m')
        block = native.detach().cpu().numpy()
        if block.shape != (count,204):
            raise ValueError(f'Expected native MHR parameters [B,204], got {block.shape}')
        if scales is None:
            scales = block[0,136:].copy()
        np.testing.assert_allclose(block[:,136:],np.tile(scales,(count,1)),atol=1e-6)
        exported.append(block[:,:136]);errors.extend(error.cpu().tolist())
    a.output.mkdir(parents=True,exist_ok=True)
    stem = a.output/f'episode_{a.episode:06d}'
    np.savez_compressed(stem.with_suffix('.npz'),pose=np.concatenate(exported),
        scales=scales,shape=shape,object_rotation=rotations,
        object_translation=poses[:,:3,3],object_scale=np.float32(1.))
    mesh_out = a.output/(stem.name+'_object.glb')
    shutil.copy2(a.mesh,mesh_out)
    report = {'episode':a.episode,'frames':a.frames,'source_bundle_sha256':sha256(a.bundle),
        'mesh_sha256':sha256(mesh_out),'mhr_model_sha256':sha256(model_path),
        'identity_selection':'per-clip median of predicted identity latents',
        'expression':'zero, matching challenge scorer',
        'max_native_round_trip_error_m':max(errors),'ground_truth_used':False}
    stem.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
