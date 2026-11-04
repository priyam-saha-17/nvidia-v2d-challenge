import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trajectory_refinement import refine_poses


class RefinementTests(unittest.TestCase):
    def test_noise_and_occlusion(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 40)
        clean = np.column_stack([t, .1*np.sin(t), np.ones_like(t)])
        noisy = clean + rng.normal(0, .012, clean.shape)
        q = Rotation.from_rotvec(np.column_stack([np.zeros_like(t), np.zeros_like(t), .3*t])).as_quat()
        poses = np.column_stack([noisy, q])
        confidence = np.ones(len(t))
        confidence[16:20] = 0
        poses[16:20] = 0
        result, info = refine_poses(poses, confidence, max_nfev=80)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(np.mean((result[:, :3]-clean)**2), np.mean((noisy-clean)**2))
        np.testing.assert_allclose(np.linalg.norm(result[:, 3:], axis=1), 1, atol=1e-6)
        self.assertEqual(info['valid_observations'], 36)

    def test_quaternion_sign_invariance(self):
        p = np.zeros((12, 7)); p[:, 6] = 1
        p[::2, 6] = -1
        r, _ = refine_poses(p, np.ones(12))
        np.testing.assert_allclose(Rotation.from_quat(r[:, 3:]).magnitude(), 0, atol=1e-8)

    def test_reject_missing(self):
        with self.assertRaises(ValueError):
            refine_poses(np.zeros((8, 7)), np.ones(8))


if __name__ == '__main__':
    unittest.main()
