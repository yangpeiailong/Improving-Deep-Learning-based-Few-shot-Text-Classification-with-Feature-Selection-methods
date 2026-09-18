from __future__ import annotations

import unittest

from fs_lrtc.features.pipeline import fit_selector_at_scope
from fs_lrtc.features.selectors import FilterFeatureSelector, RandomFeatureSelector


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.documents = [["cat"], ["cat", "dog"], ["dog"], ["mouse"]]
        self.labels = ["a", "a", "b", "b"]

    def test_df_scores(self):
        scores = {item.token: item.score for item in FilterFeatureSelector("df").fit(self.documents).ranked_features_}
        self.assertEqual(scores, {"cat": 2.0, "dog": 2.0, "mouse": 1.0})

    def test_information_gain_known_values(self):
        scores = {item.token: item.score for item in FilterFeatureSelector("ig").fit(self.documents, self.labels).ranked_features_}
        self.assertAlmostEqual(scores["cat"], 1.0)
        self.assertAlmostEqual(scores["dog"], 0.0)
        self.assertAlmostEqual(scores["mouse"], 0.31127812445913283)

    def test_dfs_known_values(self):
        scores = {item.token: item.score for item in FilterFeatureSelector("dfs").fit(self.documents, self.labels).ranked_features_}
        self.assertAlmostEqual(scores["cat"], 1.0)
        self.assertAlmostEqual(scores["dog"], 0.5)
        # A one-document class-specific term remains below a frequent perfect
        # discriminator, which is the intended low-frequency penalty of DFS.
        self.assertAlmostEqual(scores["mouse"], 2.0 / 3.0)

    def test_tie_break_is_lexical(self):
        ranked = FilterFeatureSelector("df").fit([["z", "a"]]).ranked_features_
        self.assertEqual([item.token for item in ranked], ["a", "z"])

    def test_transform_preserves_empty_documents(self):
        selector = FilterFeatureSelector("df").fit(self.documents)
        transformed = selector.transform([["unknown"], ["cat", "unknown"]], 1)
        self.assertEqual(len(transformed), 2)
        self.assertEqual(transformed[0], [])

    def test_held_out_labels_cannot_change_supervised_features(self):
        documents = [["a"], ["a", "x"], ["b"], ["b", "y"], ["test1"], ["test2"]]
        labels = ["0", "0", "1", "1", "0", "1"]
        changed = labels[:4] + ["changed", "changed"]
        for method in ("ig", "dfs"):
            first = fit_selector_at_scope(method, documents, labels, [0, 1, 2, 3], [4, 5])
            second = fit_selector_at_scope(method, documents, changed, [0, 1, 2, 3], [4, 5])
            self.assertEqual(first.ranked_features_, second.ranked_features_)

    def test_forbidden_index_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_selector_at_scope("ig", self.documents, self.labels, [0, 1, 2], [2, 3])

    def test_random_control_is_reproducible_and_fit_scoped(self):
        first = RandomFeatureSelector(17).fit(self.documents).top(3)
        second = RandomFeatureSelector(17).fit(self.documents).top(3)
        self.assertEqual(first, second)
        self.assertNotIn("heldout", [item.token for item in first])


if __name__ == "__main__":
    unittest.main()
