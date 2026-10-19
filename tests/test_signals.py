import unittest
import numpy as np

from rehab_audit.signals import constant_tail_length, constant_tail_start, interval_overlap_audit, window_features, window_starts


class SignalTests(unittest.TestCase):
    def test_upstream_window_geometry(self):
        self.assertEqual(window_starts(880, 200, 40), list(range(0, 681, 40)))
        self.assertEqual(len(window_starts(880, 200, 40)), 18)

    def test_nonzero_constant_tail_is_only_a_proxy(self):
        trace = np.array([[1., 2.], [2., 3.], [-4., -5.], [-4., -5.]])
        self.assertEqual(constant_tail_start(trace), 2)
        self.assertEqual(constant_tail_start(np.ones((4, 2))), 0)
        self.assertEqual(constant_tail_start(np.arange(8).reshape(4, 2)), 3)
        self.assertEqual(constant_tail_length(np.arange(8).reshape(4, 2)), 0)
        self.assertEqual(constant_tail_length(np.ones((4, 2))), 4)
        self.assertEqual(constant_tail_length(trace), 2)

    def test_tail_uses_final_value_not_adjacent_differences(self):
        trace = np.array([[0.], [.8], [1.6], [2.4]])
        self.assertEqual(constant_tail_start(trace, atol=1.), 2)
        with self.assertRaises(ValueError):
            constant_tail_start(np.empty((4, 0)))

    def test_no_patient_identity_is_inferred_by_overlap(self):
        result = interval_overlap_audit(["a", "a", "b"], [0, 40, 40], [0], [1, 2], 200)
        self.assertEqual(result["test_window_overlap_fraction"], .5)
        self.assertEqual(result["mean_max_pairwise_sample_overlap"], .4)
        self.assertEqual(interval_overlap_audit(["a", "b"], [0, 0], [0], [1], 200)["test_windows_overlapping_training"], 0)

    def test_overlap_boundaries_and_empty_test_set(self):
        result = interval_overlap_audit(["a"] * 3, [0, 200, 199], [0], [1, 2], 200)
        self.assertEqual(result["test_windows_overlapping_training"], 1)
        self.assertEqual(result["mean_max_pairwise_sample_overlap"], .0025)
        result = interval_overlap_audit(["a"], [0], [0], [], 200)
        self.assertIsNone(result["test_window_overlap_fraction"])
        self.assertIsNone(result["mean_max_pairwise_sample_overlap"])

    def test_invalid_overlap_metadata_does_not_silently_coerce(self):
        cases = [(["a", "b"], [0., 40.], [0], [1], 200),
                 (["a", "b"], [0, -1], [0], [1], 200),
                 (["a", "b"], [0, 40], [-1], [1], 200),
                 (["a", "b"], [0, 40], [0], [2], 200),
                 (["a", "b"], [0, 40], [0], [0], 200),
                 (["a", "b"], [0, 40], [0, 0], [1], 200),
                 (["a", "b"], [0, 40], [0.], [1], 200),
                 (["a", "b"], [0, 40], [0], [1], True)]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                interval_overlap_audit(*values)

    def test_feature_definition_and_invalid_values(self):
        result = window_features(np.array([[0., 1.], [2., 3.]]))
        np.testing.assert_allclose(result[:6], [1., 2., 1., 1., 2., 2.])
        self.assertEqual(result.shape, (10,))
        with self.assertRaises(ValueError):
            window_features(np.array([[np.nan], [1.]]))
        with self.assertRaises(ValueError):
            constant_tail_start(np.array([[0.], [1.]]), -1)
        with self.assertRaises(ValueError):
            window_starts(880, 200, 0)
        with self.assertRaisesRegex(ValueError, "overflow"):
            window_features(np.full((2, 1), 1e308))
