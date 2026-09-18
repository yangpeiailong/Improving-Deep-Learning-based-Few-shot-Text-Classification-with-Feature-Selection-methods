from __future__ import annotations

import unittest

from fs_lrtc.data.splitting import (
    grouped_inner_validation_split,
    grouped_test_folds,
    inner_validation_split,
    stratified_test_folds,
)


class SplittingTests(unittest.TestCase):
    def test_stratified_folds_are_deterministic_partition(self):
        labels = ["a"] * 10 + ["b"] * 10
        folds = stratified_test_folds(labels, 5, 7, "demo")
        self.assertEqual(sorted(index for fold in folds for index in fold), list(range(20)))
        self.assertTrue(all(len(fold) == 4 for fold in folds))
        self.assertEqual(folds, stratified_test_folds(labels, 5, 7, "demo"))

    def test_grouped_folds_never_split_group(self):
        labels = ["a"] * 12 + ["b"] * 12
        groups = ["x", "x"] + [f"a{i}" for i in range(10)] + ["z", "z"] + [f"b{i}" for i in range(10)]
        folds = grouped_test_folds(labels, groups, 3, 11, "demo")
        location = {index: fold for fold, indices in enumerate(folds) for index in indices}
        self.assertEqual(location[0], location[1])
        self.assertEqual(location[12], location[13])
        self.assertTrue(all(fold for fold in folds))
        self.assertLessEqual(max(map(len, folds)) - min(map(len, folds)), 2)

    def test_inner_validation_preserves_rare_class_training_record(self):
        labels = ["rare"] * 4 + ["common"] * 20
        inner_train, validation = inner_validation_split(list(range(24)), labels, 0.15, 1, "demo", 0)
        self.assertGreaterEqual(sum(labels[i] == "rare" for i in inner_train), 1)
        self.assertGreaterEqual(sum(labels[i] == "rare" for i in validation), 1)

    def test_grouped_inner_validation_keeps_groups_atomic(self):
        labels = ["a", "a", "a", "a", "b", "b", "b", "b"]
        groups = ["x", "x", "y", "z", "p", "p", "q", "r"]
        train, validation = grouped_inner_validation_split(
            list(range(8)), labels, groups, 0.25, 9, "demo", 0
        )
        train_groups = {groups[i] for i in train}
        validation_groups = {groups[i] for i in validation}
        self.assertFalse(train_groups & validation_groups)


if __name__ == "__main__":
    unittest.main()
