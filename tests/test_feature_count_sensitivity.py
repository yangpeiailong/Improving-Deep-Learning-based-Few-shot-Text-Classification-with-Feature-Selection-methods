from __future__ import annotations

import unittest
from pathlib import Path

from fs_lrtc.config import load_yaml
from fs_lrtc.experiments.feature_count import (
    aggregate_sensitivity,
    build_sensitivity_configs,
    expected_sensitivity_keys,
    expected_unit_runs,
    paired_sensitivity_effects,
    trend_summary,
    validate_sensitivity_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FeatureCountSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.master = load_yaml(PROJECT_ROOT / "configs" / "feature_count_sensitivity.yaml")
        self.bases = {
            family: load_yaml(PROJECT_ROOT / "configs" / spec["base_config"])
            for family, spec in self.master["families"].items()
        }

    def test_plan_adds_only_1440_new_runs(self):
        units = build_sensitivity_configs(self.master, self.bases)
        self.assertEqual(len(units), 9)
        self.assertEqual(sum(map(expected_unit_runs, units.values())), 1440)
        self.assertEqual(len(expected_sensitivity_keys(self.master)), 2080)
        self.assertEqual(units[("bert", 500)]["models"], ["bert_cls"])
        self.assertEqual(units[("classic", 500)]["feature_selection"]["selectors"], ["df", "ig", "dfs"])

    def test_validation_aggregation_effects_and_trend(self):
        master = {
            "datasets": ["d"], "folds": [0, 1], "selectors": ["df", "ig", "dfs"],
            "analysis_feature_counts": [500, 1000],
            "families": {"classic": {"models": ["m"]}},
        }
        rows = []
        for fold in master["folds"]:
            rows.append(self._row(fold, "none", "all", 0.6 + 0.1 * fold))
            for selector in master["selectors"]:
                for count in master["analysis_feature_counts"]:
                    rows.append(self._row(fold, selector, str(count), 0.7 + 0.1 * fold))
        self.assertEqual(validate_sensitivity_rows(rows, master)["status"], "passed")
        aggregate = aggregate_sensitivity(rows)
        effects = paired_sensitivity_effects(rows)
        trends = trend_summary(aggregate, effects)
        self.assertEqual(len(aggregate), 7)
        self.assertEqual(len(effects), 6)
        self.assertEqual(len(trends), 6)
        self.assertAlmostEqual(effects[0]["accuracy_delta_mean"], 0.1)

    @staticmethod
    def _row(fold, selector, count, accuracy):
        return {
            "family": "classic", "dataset_id": "d", "fold": fold,
            "selector": selector, "feature_count": count, "model": "m",
            "test_accuracy": accuracy, "test_macro_f1": accuracy,
            "test_balanced_accuracy": accuracy, "best_epoch": 2,
            "development_feature_selection_seconds": 0.1,
            "final_feature_selection_seconds": 0.1,
            "development_training_seconds": 1.0,
            "final_training_seconds": 1.0, "test_inference_seconds": 0.1,
            "inner_train_indices_sha256": f"inner-{fold}",
            "validation_indices_sha256": f"val-{fold}",
            "outer_train_indices_sha256": f"outer-{fold}",
            "test_indices_sha256": f"test-{fold}",
            "development_vocabulary_sha256": f"dev-{fold}-{selector}-{count}",
            "final_vocabulary_sha256": f"final-{fold}-{selector}-{count}",
        }


if __name__ == "__main__":
    unittest.main()
