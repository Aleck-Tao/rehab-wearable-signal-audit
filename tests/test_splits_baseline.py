import json
import unittest
from unittest.mock import patch

import numpy as np

from rehab_audit.baseline import RidgeClassifier, classification_metrics
from rehab_audit.splits import (Split, content_group_disjoint_split, split_leakage, stratified_record_split,
                                stratified_window_split)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat([2, 7], 50)
        self.records = np.array([f"class{label}_record{index // 10}"
                                 for label in [2, 7] for index in range(50)])

    def assert_partition(self, split):
        np.testing.assert_array_equal(np.sort(np.concatenate(
            (split.train_indices, split.test_indices))), np.arange(100))
        self.assertEqual(set(self.labels[split.train_indices]), {2, 7})
        self.assertEqual(set(self.labels[split.test_indices]), {2, 7})

    def test_seeded_splits_are_complete_and_deterministic(self):
        for function in (stratified_window_split, stratified_record_split):
            for seed in (11, 29, 47):
                with self.subTest(function=function.__name__, seed=seed):
                    split = function(self.labels, self.records, seed=seed)
                    repeated = function(self.labels, self.records, seed=seed)
                    self.assert_partition(split)
                    np.testing.assert_array_equal(split.train_indices, repeated.train_indices)
                    np.testing.assert_array_equal(split.test_indices, repeated.test_indices)
                    self.assertEqual(split.test_indices.size, 20)

    def test_seeds_change_record_selection(self):
        first = stratified_record_split(self.labels, self.records, seed=11)
        second = stratified_record_split(self.labels, self.records, seed=47)
        self.assertFalse(np.array_equal(first.test_indices, second.test_indices))

    def test_optimized_splits_match_original_selection_exactly(self):
        # A deliberately simple small-data reference protects the original
        # sorted-ID/RNG convention while the production grouping is optimized.
        labels, records = [], []
        for label in (-3, 0, 11):
            for record in range(7):
                count = record + 1
                labels.extend([label] * count)
                records.extend([f"class{label}_record{record}"] * count)
        permutation = np.random.default_rng(9).permutation(len(labels))
        labels, records = np.asarray(labels)[permutation], np.asarray(records)[permutation]
        for function in (stratified_window_split, stratified_record_split):
            for seed in (11, 29, 47):
                for fraction in (0.2, 0.4, 0.99):
                    rng = np.random.default_rng(seed)
                    train, test = [], []
                    for label in np.unique(labels):
                        if function is stratified_record_split:
                            units = np.unique(records[labels == label])
                        else:
                            units = np.flatnonzero(labels == label)
                        count = min(len(units) - 1, max(1, int(np.ceil(len(units) * fraction))))
                        units = rng.permutation(units)
                        if function is stratified_record_split:
                            test.extend(np.flatnonzero(np.isin(records, units[:count])))
                            train.extend(np.flatnonzero(np.isin(records, units[count:])))
                        else:
                            test.extend(units[:count])
                            train.extend(units[count:])
                    actual = function(labels, records, test_fraction=fraction, seed=seed)
                    with self.subTest(function=function.__name__, seed=seed, fraction=fraction):
                        np.testing.assert_array_equal(actual.train_indices, np.sort(train))
                        np.testing.assert_array_equal(actual.test_indices, np.sort(test))

    def test_large_window_inventory_has_bounded_grouping_passes(self):
        # Same scale as the planned audit, but synthetic and dependency-free.
        # Count grouping operations instead of imposing a machine-speed limit.
        record_labels = np.arange(4257) % 15
        labels = np.repeat(record_labels, 18)
        records = np.repeat([f"record_{index:05d}" for index in range(4257)], 18)
        with patch("rehab_audit.splits.np.unique", wraps=np.unique) as unique:
            record_split = stratified_record_split(labels, records, seed=11)
            window_split = stratified_window_split(labels, records, seed=11)
            record_report = split_leakage(record_split, records)
            window_report = split_leakage(window_split, records)
            grouping_calls = unique.call_count
        self.assertLess(grouping_calls, 64)
        self.assertEqual(record_split.train_indices.size + record_split.test_indices.size, 76626)
        self.assertEqual(window_split.train_indices.size + window_split.test_indices.size, 76626)
        self.assertEqual(record_report["shared_record_count"], 0)
        self.assertEqual(record_report["test_window_shared_record_fraction"], 0.0)
        self.assertGreater(window_report["test_window_shared_record_fraction"], 0.99)
        self.assertEqual(np.unique(labels[record_split.train_indices]).size, 15)
        self.assertEqual(np.unique(labels[record_split.test_indices]).size, 15)

    def test_content_groups_allow_cross_label_duplicates_and_stay_disjoint(self):
        labels = np.repeat([0, 1], 10)
        contents = np.tile([f"hash{index}" for index in range(10)], 2)
        for seed in (101, 307, 809):
            split = content_group_disjoint_split(labels, contents, seed=seed)
            train_groups = set(contents[split.train_indices])
            test_groups = set(contents[split.test_indices])
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(len(test_groups), 2)
            self.assertEqual(set(labels[split.test_indices]), {0, 1})
            expected_order = np.random.default_rng(seed).permutation(10)
            expected_test = np.flatnonzero(np.isin(contents, np.unique(contents)[expected_order[:2]]))
            np.testing.assert_array_equal(split.test_indices, expected_test)

    def test_content_split_rejects_missing_class_support_without_redraw(self):
        for seed in (101, 307, 809):
            with self.assertRaisesRegex(ValueError, "no redraw"):
                content_group_disjoint_split([0, 0, 1, 1], ["a", "b", "c", "c"], seed=seed)

    def test_record_split_is_disjoint_but_window_comparator_is_not(self):
        record = split_leakage(stratified_record_split(self.labels, self.records), self.records)
        window = split_leakage(stratified_window_split(self.labels, self.records), self.records)
        self.assertEqual(record["shared_record_count"], 0)
        self.assertEqual(record["test_window_shared_record_fraction"], 0.0)
        self.assertEqual(window["test_window_shared_record_fraction"], 1.0)
        self.assertGreater(window["shared_record_count"], 0)
        json.dumps(window, allow_nan=False)

    def test_fraction_is_of_records_not_windows(self):
        labels = np.array([0] * 14 + [1] * 14)
        records = np.array([f"c{label}r{record}" for label in (0, 1)
                            for record in range(5) for _ in range(10 if record == 0 else 1)])
        split = stratified_record_split(labels, records, seed=3)
        self.assertEqual(np.unique(records[split.test_indices]).size, 2)
        self.assertEqual(split_leakage(split, records)["shared_record_count"], 0)

    def test_small_classes_reserve_both_sides(self):
        split = stratified_record_split([0, 0, 1, 1], ["a", "b", "c", "d"], test_fraction=0.99)
        self.assertEqual(split.train_indices.size, 2)
        self.assertEqual(split.test_indices.size, 2)
        for function in (stratified_window_split, stratified_record_split):
            with self.assertRaisesRegex(ValueError, "at least two"):
                function([0, 1, 1], ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "at least two"):
            stratified_record_split([0, 0, 1, 1], ["a", "a", "b", "b"])

    def test_invalid_labels_records_and_fractions_rejected(self):
        for function in (stratified_window_split, stratified_record_split):
            for labels, records in (([], []), ([[0, 1]], ["a", "b"]),
                                     ([0.0, 1.0], ["a", "b"]), ([0, 1], ["a"]),
                                     ([0, 1], ["a", "a"]), ([0, 1], ["", "b"]),
                                     ([0, 1], [1, 2]), ([0, 1], ["a", None])):
                with self.subTest(function=function.__name__, labels=labels, records=records):
                    with self.assertRaises(ValueError):
                        function(labels, records)
            for fraction in (0, 1, -0.1, float("nan"), float("inf"), "0.2", True, 0.2j):
                with self.assertRaises(ValueError):
                    function(self.labels, self.records, test_fraction=fraction)

    def test_leakage_rejects_invalid_partitions(self):
        for split in (Split(np.array([0]), np.array([0, 1])),
                      Split(np.array([0, 0]), np.array([1, 2])),
                      Split(np.array([0]), np.array([3])),
                      Split(np.array([0]), np.array([1])),
                      Split(np.array([], dtype=int), np.array([0, 1, 2]))):
            with self.assertRaises(ValueError):
                split_leakage(split, ["a", "b", "c"])


class BaselineTests(unittest.TestCase):
    def test_separable_data_with_constant_column(self):
        features = np.array([[-2, 5], [-1, 5], [1, 5], [2, 5]], dtype=float)
        labels = np.array([2, 2, 7, 7])
        model = RidgeClassifier().fit(features, labels)
        np.testing.assert_array_equal(model.predict(features), labels)
        self.assertEqual(model.scale_[1], 1.0)
        np.testing.assert_allclose(model.coef_[1], 0)

    def test_test_data_cannot_refit_standardization(self):
        features = np.array([[-2, 3], [-1, 3], [1, 3], [2, 3]])
        model = RidgeClassifier().fit(features, [0, 0, 1, 1])
        mean, scale = model.mean_.copy(), model.scale_.copy()
        individual = model.predict([[1, 3]])[0]
        batch = model.predict([[1, 3], [100000, -100000]])[0]
        self.assertEqual(individual, batch)
        np.testing.assert_array_equal(model.mean_, mean)
        np.testing.assert_array_equal(model.scale_, scale)

    def test_intercept_is_not_regularized_and_ties_use_smallest_label(self):
        model = RidgeClassifier(alpha=1000).fit(np.ones((4, 2)), [2, 2, 2, 7])
        np.testing.assert_allclose(model.intercept_, [0.75, 0.25])
        np.testing.assert_allclose(model.coef_, 0)
        tied = RidgeClassifier().fit(np.ones((4, 1)), [7, 2, 7, 2])
        self.assertEqual(int(tied.predict([[1]])[0]), 2)

    def test_ridge_matches_centered_closed_form(self):
        features = np.array([[0., 1.], [1., 2.], [2., -1.], [3., 4.]])
        labels = np.array([0, 1, 0, 1])
        model = RidgeClassifier().fit(features, labels)
        normalized = (features - features.mean(axis=0)) / features.std(axis=0)
        targets = np.column_stack((labels == 0, labels == 1)).astype(float)
        expected = np.linalg.solve(normalized.T @ normalized + np.eye(2),
                                   normalized.T @ (targets - targets.mean(axis=0)))
        np.testing.assert_allclose(model.coef_, expected, atol=1e-14)
        np.testing.assert_allclose(model.intercept_, targets.mean(axis=0), atol=1e-14)

    def test_fit_and_predict_validation(self):
        for alpha in (0, -1, float("nan"), float("inf"), "1", True, 1j):
            with self.assertRaises(ValueError):
                RidgeClassifier(alpha=alpha)
        for features, labels in (([], []), ([1, 2], [0, 1]), ([[], []], [0, 1]),
                                  ([[1], [float("nan")]], [0, 1]),
                                  ([[1], [float("inf")]], [0, 1]),
                                  ([[1], [2]], [0]), ([[1], [2]], [0.0, 1.0]),
                                  ([[1], [2]], [[0], [1]]), ([["1"]], [0])):
            with self.subTest(features=features, labels=labels):
                with self.assertRaises(ValueError):
                    RidgeClassifier().fit(features, labels)
        with self.assertRaises(RuntimeError):
            RidgeClassifier().predict([[1]])
        model = RidgeClassifier().fit([[1], [2]], [0, 1])
        for features in ([[1, 2]], [[float("nan")]], [], [[float("inf")]]):
            with self.assertRaises(ValueError):
                model.predict(features)

    def test_failed_refit_invalidates_previous_model(self):
        model = RidgeClassifier().fit([[1], [2]], [0, 1])
        with self.assertRaises(ValueError):
            model.fit([[float("nan")], [2]], [0, 1])
        with self.assertRaises(RuntimeError):
            model.predict([[1]])

    def test_finite_but_overflowing_training_features_rejected(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            RidgeClassifier().fit([[1e308], [1e308]], [0, 1])

    def test_metrics_have_hand_checked_confusion_and_macro_scores(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 5) / 2)
        self.assertEqual([row["support"] for row in result["per_class"]], [2, 2])
        json.dumps(result, allow_nan=False)

    def test_metrics_retain_zero_support_classes_in_macro_average(self):
        result = classification_metrics([2, 2], [2, 2], classes=[7, 2])
        self.assertEqual(result["classes"], [2, 7])
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 0]])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["per_class"][1]["support"], 0)
        self.assertEqual(result["per_class"][1]["f1"], 0.0)

    def test_metrics_validation(self):
        for truth, prediction, classes in (([], [], None), ([0], [0, 1], None),
                                            ([0.0], [0], None), ([0], [1], [0]),
                                            ([0], [0], [0, 0]), ([0], [0], [])):
            with self.assertRaises(ValueError):
                classification_metrics(truth, prediction, classes)


if __name__ == "__main__":
    unittest.main()
