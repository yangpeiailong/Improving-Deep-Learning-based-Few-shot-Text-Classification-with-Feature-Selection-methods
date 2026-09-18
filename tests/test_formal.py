from __future__ import annotations

import unittest

from fs_lrtc.experiments.formal import (
    aggregate_results,
    build_shard_configs,
    expected_run_keys,
    paired_effects,
    validate_formal_rows,
)


def config() -> dict:
    return {
        "datasets": ["sample"],
        "folds": [0, 1],
        "feature_selection": {"selectors": ["none", "dfs"]},
        "models": ["random_wa"],
    }


def row(fold: int, selector: str, accuracy: float) -> dict:
    return {
        "dataset_id": "sample",
        "fold": fold,
        "selector": selector,
        "model": "random_wa",
        "test_accuracy": accuracy,
        "test_macro_f1": accuracy - 0.01,
        "test_balanced_accuracy": accuracy - 0.02,
        "validation_accuracy": accuracy - 0.03,
        "best_epoch": 10 + fold,
        "development_feature_selection_seconds": 0.1,
        "final_feature_selection_seconds": 0.2,
        "development_training_seconds": 1.0,
        "final_training_seconds": 2.0,
        "test_inference_seconds": 0.01,
        "inner_train_indices_sha256": f"inner-{fold}",
        "validation_indices_sha256": f"validation-{fold}",
        "outer_train_indices_sha256": f"outer-{fold}",
        "test_indices_sha256": f"test-{fold}",
        "development_vocabulary_sha256": f"dev-{fold}-{selector}",
        "final_vocabulary_sha256": f"final-{fold}-{selector}",
    }


class FormalResultTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row(0, "none", 0.50),
            row(0, "dfs", 0.60),
            row(1, "none", 0.55),
            row(1, "dfs", 0.70),
        ]

    def test_expected_and_validation(self):
        self.assertEqual(len(expected_run_keys([config()])), 4)
        report = validate_formal_rows(self.rows, [config()])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["actual_rows"], 4)

    def test_master_config_expands_without_protocol_duplication(self):
        master = {
            "experiment_id": "formal",
            "datasets": ["a", "b"],
            "folds": [0],
            "feature_selection": {"selectors": ["none"]},
            "models": ["random_wa"],
            "output": {"directory": "unused"},
            "formal_orchestration": {
                "output_base": "main/classic",
                "shards": {1: ["a"], 2: ["b"]},
            },
        }
        shards = build_shard_configs(master)
        self.assertEqual(shards[1]["datasets"], ["a"])
        self.assertEqual(shards[2]["output"]["directory"], "main/classic/shard_2")
        self.assertNotIn("formal_orchestration", shards[1])

    def test_missing_and_hash_conflicts_are_rejected(self):
        altered = [dict(value) for value in self.rows[:-1]]
        altered[1]["test_indices_sha256"] = "different"
        report = validate_formal_rows(altered, [config()])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["missing_keys"]), 1)
        self.assertTrue(report["split_hash_conflicts"])

    def test_aggregate_and_paired_effect(self):
        aggregate = aggregate_results(self.rows)
        self.assertEqual(len(aggregate), 2)
        effect = paired_effects(self.rows)[0]
        self.assertAlmostEqual(effect["accuracy_delta_mean"], 0.125)
        self.assertEqual((effect["wins"], effect["ties"], effect["losses"]), (2, 0, 0))


if __name__ == "__main__":
    unittest.main()
