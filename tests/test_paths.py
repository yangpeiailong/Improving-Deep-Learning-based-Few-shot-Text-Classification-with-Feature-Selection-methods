from __future__ import annotations

import unittest

from fs_lrtc.utils.paths import compact_artifact_filename


class PathTests(unittest.TestCase):
    def test_short_artifact_name_is_preserved(self):
        self.assertEqual(compact_artifact_filename("20ng_f0_df", ".json"), "20ng_f0_df.json")

    def test_long_artifact_name_is_short_stable_and_distinct(self):
        first = "amazon_review_full_fold0_df_fasttext_residual_word_gcn"
        second = "amazon_review_full_fold0_ig_fasttext_residual_word_gcn"
        first_name = compact_artifact_filename(first, ".json")
        self.assertEqual(first_name, compact_artifact_filename(first, ".json"))
        self.assertLessEqual(len(first_name), 41)
        self.assertNotEqual(first_name, compact_artifact_filename(second, ".json"))


if __name__ == "__main__":
    unittest.main()
