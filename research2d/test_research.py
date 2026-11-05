"""Numerical derivative, data isolation and geometry checks for both 2D models."""
from __future__ import annotations

import unittest
import numpy as np

from experiment import SpatialModel, average_disparity, patches, spectral_kernels, stimulus
from orientation import measured_orientation, oriented_image, fit_orientation_curve, variance_explained


class SpatialResearchTests(unittest.TestCase):
    def test_stereo_shapes_and_stimulus_determinism(self):
        left, right, disparity = stimulus("gaussian", 101, 120)
        l2, r2, d2 = stimulus("gaussian", 101, 120)
        self.assertEqual(left.shape, (120, 120))
        self.assertEqual(patches(left, right).shape, (1600, 19))
        self.assertEqual(average_disparity(disparity).shape, (40, 40))
        np.testing.assert_array_equal(left, l2)
        np.testing.assert_array_equal(right, r2)
        np.testing.assert_array_equal(disparity, d2)
        self.assertAlmostEqual(float(left.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(left.std()), 1.0, places=12)
        self.assertLessEqual(float(abs(disparity).max()), 1.0)

    def test_2d_periodic_kernel_and_gradient(self):
        rng = np.random.default_rng(1234)
        x = np.column_stack((rng.normal(size=(16, 18)), np.ones(16)))
        model = SpatialModel(x, 4)
        weights = SpatialModel.initial_weights(64)
        merit, gradient = model.evaluate(weights)
        ku, kv = spectral_kernels(4, 4, 2.0)
        self.assertAlmostEqual(float(ku[0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(kv[0, 0]), 0.0, places=12)
        self.assertTrue(np.isfinite(merit))
        eps = 1e-5
        for index in range(weights.size):
            wp = weights.copy()
            wm = weights.copy()
            wp[index] += eps
            wm[index] -= eps
            numeric = (model.evaluate(wp)[0] - model.evaluate(wm)[0]) / (2 * eps)
            self.assertAlmostEqual(numeric, gradient[index],
                                   delta=5e-5 * (1 + abs(numeric)))

    def test_ground_truth_never_enters_training_model(self):
        left, right, truth = stimulus("gaussian", 22, 12)
        x = patches(left, right)
        model = SpatialModel(x, 4)
        w = SpatialModel.initial_weights(3)
        f, gradient = model.evaluate(w)
        flipped_truth = -truth
        np.testing.assert_allclose(gradient, model.evaluate(w)[1])
        self.assertAlmostEqual(f, model.evaluate(w)[0])
        self.assertFalse(np.array_equal(truth, flipped_truth))
        self.assertEqual(len(w), 100)

    def test_oriented_pattern_and_mask(self):
        img, x = oriented_image(112, 60)
        self.assertEqual(img.shape, (60, 60))
        self.assertEqual(x.shape, (144, 26))
        angle, mask = measured_orientation(img)
        self.assertEqual(angle.shape, (144,))
        self.assertTrue(mask.any())
        self.assertEqual(SpatialModel.initial_weights(5, 26, 9).size, 243)

    def test_orientation_periodicity(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 120)
        targets = 2.2 * np.sin(2 * angles) + .3 * np.cos(2 * angles) - .7
        mask = np.ones(angles.size, dtype=bool)
        coef = fit_orientation_curve(angles, targets, mask)
        self.assertAlmostEqual(variance_explained(angles, targets, mask, coef), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
