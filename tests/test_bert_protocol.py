from __future__ import annotations

import unittest
from pathlib import Path

from fs_lrtc.config import load_yaml
from fs_lrtc.experiments.formal import build_shard_configs, expected_run_keys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BertProtocolTests(unittest.TestCase):
    def test_formal_protocol_is_frozen_at_1280_fp32_runs(self):
        config = load_yaml(PROJECT_ROOT / "configs" / "formal_bert.yaml")
        shards = build_shard_configs(config)
        self.assertEqual(len(shards), 16)
        self.assertEqual(len(expected_run_keys(list(shards.values()))), 1280)
        self.assertEqual(shards[1]["datasets"], ["20ng"])
        self.assertEqual(shards[2]["datasets"], ["amazon_review_full"])
        self.assertEqual(shards[3]["datasets"], ["farm_ads"])
        self.assertEqual(shards[4]["datasets"], ["wos5736"])
        self.assertEqual(config["training"]["maximum_epochs"], 40)
        self.assertEqual(config["training"]["minimum_epochs"], 8)
        self.assertEqual(config["training"]["early_stopping_patience"], 5)
        self.assertEqual(config["training"]["warmup_ratio"], 0.025)
        self.assertFalse(config["training"]["mixed_precision"])
        self.assertEqual(config["formal_orchestration"]["output_base"], "main/bert_v3")

    def test_convergence_pilot_never_requests_full_evaluation(self):
        config = load_yaml(PROJECT_ROOT / "configs" / "bert_convergence.yaml")
        self.assertEqual(config["evaluation_scope"], "development_only")
        self.assertEqual(config["datasets"], ["20ng"])
        self.assertEqual(config["folds"], [0])
        self.assertEqual(config["feature_selection"]["selectors"], ["none"])
        self.assertEqual(len(config["models"]), 4)
        self.assertEqual(config["training"]["maximum_epochs"], 40)
        self.assertEqual(config["training"]["minimum_epochs"], 8)
        self.assertEqual(config["training"]["early_stopping_patience"], 5)
        self.assertEqual(config["training"]["warmup_ratio"], 0.025)
        self.assertFalse(config["output"]["save_predictions"])


if __name__ == "__main__":
    unittest.main()
