from __future__ import annotations

import unittest

from fs_lrtc.experiments.main_summary import (
    add_family,
    family_summary,
    runtime_summary,
    selector_summary,
    significance_tests,
    union_fieldnames,
)


class MainSummaryTests(unittest.TestCase):
    def setUp(self):
        self.aggregate = [
            {"family": "classic", "model": "random_wa", "selector": "none", "test_accuracy_mean": 0.5},
            {"family": "classic", "model": "random_wa", "selector": "dfs", "test_accuracy_mean": 0.6},
            {"family": "bert", "model": "bert_cls", "selector": "none", "test_accuracy_mean": 0.8},
            {"family": "bert", "model": "bert_cls", "selector": "dfs", "test_accuracy_mean": 0.7},
        ]
        self.effects = [
            {"family": "classic", "model": "random_wa", "selector": "dfs", "accuracy_delta_mean": 0.1},
            {"family": "bert", "model": "bert_cls", "selector": "dfs", "accuracy_delta_mean": -0.1},
        ]

    def test_family_and_overall_summaries(self):
        overall = selector_summary(self.aggregate, self.effects)
        dfs = next(row for row in overall if row["selector"] == "dfs")
        self.assertAlmostEqual(dfs["mean_accuracy_delta_vs_none"], 0.0)
        self.assertEqual((dfs["cell_wins"], dfs["cell_losses"]), (1, 1))
        by_family = family_summary(self.aggregate, self.effects)
        classic_dfs = next(
            row for row in by_family if row["family"] == "classic" and row["selector"] == "dfs"
        )
        self.assertAlmostEqual(classic_dfs["mean_accuracy_delta_vs_none"], 0.1)

    def test_significance_has_holm_and_effect_size(self):
        effects = []
        for family in ("classic", "gcn", "bert"):
            for selector, delta in (("df", -0.01), ("ig", 0.02), ("dfs", 0.03)):
                for index in range(4):
                    effects.append(
                        {
                            "family": family,
                            "selector": selector,
                            "accuracy_delta_mean": delta + index * 0.001,
                        }
                    )
        rows = significance_tests(effects)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(0.0 <= row["p_value_holm"] <= 1.0 for row in rows))
        self.assertTrue(all(-1.0 <= row["rank_biserial_correlation"] <= 1.0 for row in rows))

    def test_runtime_uses_final_stage_components(self):
        rows = []
        for selector, training in (("none", 9.0), ("dfs", 4.0)):
            rows.append(
                {
                    "model": "random_wa",
                    "selector": selector,
                    "final_feature_selection_seconds": 1.0 if selector == "dfs" else 0.0,
                    "final_training_seconds": training,
                    "test_inference_seconds": 1.0,
                }
            )
        summary = runtime_summary(rows)
        dfs = next(row for row in summary if row["selector"] == "dfs")
        self.assertAlmostEqual(dfs["mean_final_pipeline_seconds"], 6.0)
        self.assertAlmostEqual(dfs["relative_final_pipeline_time_vs_none"], 0.6)

    def test_family_column_and_union_schema(self):
        rows = add_family(
            [
                {"model": "random_wa", "a": 1},
                {"model": "bert_cls", "b": 2},
            ]
        )
        self.assertEqual([row["family"] for row in rows], ["classic", "bert"])
        self.assertEqual(union_fieldnames(rows), ["family", "model", "a", "b"])


if __name__ == "__main__":
    unittest.main()
