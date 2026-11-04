"""Experimental robust smoothing of reconstructed input poses, never policy rollouts.

Poses use metres and xyzw quaternions. Missing observations have zero confidence.
Contact transitions reduce smoothing across phase boundaries. This module has no
access to clean reference trajectories and makes no leaderboard-quality claim.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def refine_poses(poses, confidence, *, smoothness=8.0, huber_m=0.03,
                 rotation_weight_m=0.05, contact_phase=None, max_nfev=40):
    poses = np.asarray(poses, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 3:
        raise ValueError('Expected [T,7] poses with at least three frames')
    n = len(poses)
    if confidence.shape != (n,) or not np.isfinite(confidence).all() or (confidence < 0).any():
        raise ValueError('Confidence must be finite, nonnegative and length T')
    if smoothness < 0 or huber_m <= 0 or rotation_weight_m <= 0:
        raise ValueError('Invalid regularization parameters')
    valid = (confidence > 0) & np.isfinite(poses).all(axis=1) & (np.linalg.norm(poses[:, 3:], axis=1) > 1e-8)
    if valid.sum() < 2:
        raise ValueError('At least two valid observed poses are required')
    ids = np.flatnonzero(valid)
    times = np.arange(n)
    pos0 = np.column_stack([np.interp(times, ids, poses[valid, i]) for i in range(3)])
    rot0 = Slerp(ids, Rotation.from_quat(poses[valid, 3:]))(np.clip(times, ids[0], ids[-1]))
    obs_pos = pos0.copy()
    obs_pos[valid] = poses[valid, :3]
    obs_rot = rot0.as_matrix()
    obs_rot[valid] = Rotation.from_quat(poses[valid, 3:]).as_matrix()
    weights = np.sqrt(np.where(valid, confidence, 0))[:, None]
    phase_weight = np.ones(n - 2)
    if contact_phase is not None:
        phase = np.asarray(contact_phase)
        if phase.shape != (n,):
            raise ValueError('contact_phase must have length T')
        phase_weight[(phase[:-2] != phase[1:-1]) | (phase[1:-1] != phase[2:])] = 0.1
    temporal_weight = np.sqrt(smoothness * phase_weight)[:, None]

    def residual(x):
        x = x.reshape(n, 6)
        rot = Rotation.from_rotvec(x[:, 3:]) * rot0
        rot_error = (Rotation.from_matrix(obs_rot).inv() * rot).as_rotvec()
        dp = (x[:, :3] - obs_pos) * weights
        dr = rot_error * weights * rotation_weight_m
        # IRLS-style pseudo-Huber residual only for observations.
        observed = np.concatenate([dp, dr], axis=1)
        observed = observed / np.power(1 + (observed / huber_m) ** 2, 0.25)
        angular_velocity = (rot[:-1].inv() * rot[1:]).as_rotvec()
        return np.concatenate([observed.ravel(),
            (np.diff(x[:, :3], n=2, axis=0) * temporal_weight).ravel(),
            (np.diff(angular_velocity, axis=0) * temporal_weight * rotation_weight_m).ravel()])

    # Banded finite-difference sparsity keeps long sequences practical on CPU/T4 hosts.
    from scipy.sparse import lil_matrix
    sparsity = lil_matrix((6*n + 6*(n-2), 6*n), dtype=int)
    for t in range(n):
        sparsity[6*t:6*t+6, 6*t:6*t+6] = 1
    for t in range(n-2):
        for offset in (6*n, 6*n+3*(n-2)):
            sparsity[offset+3*t:offset+3*t+3, 6*t:6*(t+3)] = 1
    initial = np.column_stack([pos0, np.zeros((n, 3))])
    fit = least_squares(residual, initial.ravel(), jac_sparsity=sparsity.tocsr(), max_nfev=max_nfev)
    x = fit.x.reshape(n, 6)
    result = np.column_stack([x[:, :3], (Rotation.from_rotvec(x[:, 3:]) * rot0).as_quat()])
    return result, {'success': bool(fit.success), 'cost': float(fit.cost),
                    'nfev': int(fit.nfev), 'valid_observations': int(valid.sum())}
