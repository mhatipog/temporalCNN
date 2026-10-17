"""Deterministic logic and real-data grouping checks (no neural training)."""
import unittest
import numpy as np
import example
import study


class StudyTests(unittest.TestCase):
    def test_metrics(self):
        y = np.array([0, 1, 1, 0])
        guess = np.array([0, 1, 0, 0])
        groups = np.array([1, 2, 2, 3])
        self.assertEqual(study.scores(y, guess, groups)['accuracy'], 0.75)
        self.assertAlmostEqual(study.scores(y, guess, groups)['polygon_majority_accuracy'], 2 / 3)

    def test_nearest_neighbor(self):
        x = np.array([[[0.0]], [[2.0]]])
        labels = np.array([5, 6])
        self.assertEqual(study.nearest_neighbor(x, labels, np.array([[[0.1]], [[1.9]]])).tolist(), [5, 6])

    def test_paired_interval_null(self):
        y = np.array([0, 1, 1, 0])
        groups = np.array([1, 2, 2, 3])
        self.assertEqual(study.paired_polygon_interval(groups, y, y, y, 42), [0.0, 0.0, 0.0])

    def test_balanced_split_on_original_csv(self):
        x, polygons, labels = example.load('example/train_dataset.csv')
        for seed in (7, 42, 101):
            tr, val, ids = example.balanced_split(polygons, labels, seed=seed)
            self.assertEqual(len(tr), 208)
            self.assertEqual(len(val), 52)
            self.assertEqual(np.bincount(labels[val], minlength=13).tolist(), [4] * 13)
            self.assertFalse(set(polygons[tr]) & set(polygons[val]))
            self.assertEqual(set(ids), set(polygons[val]))


if __name__ == '__main__':
    unittest.main()
