import unittest

from fs_lrtc.experiments.baseline import (
    CandidateResult,
    development_search,
    final_evaluation,
    select_candidate,
    validate_partition,
    vocabulary_at_scope,
)


class BaselineTests(unittest.TestCase):
    def test_validate_partition_accepts_frozen_shape(self):
        split = {
            "inner_train": [0, 1],
            "validation": [2],
            "outer_train": [0, 1, 2],
            "test": [3],
        }
        validate_partition(split, 4)

    def test_validate_partition_rejects_overlap(self):
        split = {
            "inner_train": [0, 1],
            "validation": [1],
            "outer_train": [0, 1],
            "test": [2],
        }
        with self.assertRaises(ValueError):
            validate_partition(split, 3)

    def test_none_vocabulary_ignores_held_out_tokens(self):
        documents = [["train", "shared"], ["heldout", "shared"]]
        vocabulary, selector = vocabulary_at_scope(
            "none", documents, ["a", "b"], [0], [1], 100, 1
        )
        self.assertEqual(vocabulary, ["shared", "train"])
        self.assertIsNone(selector)

    def test_candidate_selection_uses_metrics_then_smaller_c(self):
        candidates = [
            CandidateResult(10.0, 0.8, 0.7, 0.7, 1.0, 0.1),
            CandidateResult(1.0, 0.8, 0.7, 0.7, 1.0, 0.1),
            CandidateResult(0.1, 0.7, 0.9, 0.9, 1.0, 0.1),
        ]
        self.assertEqual(select_candidate(candidates).c_value, 1.0)

    def test_development_and_final_evaluation_end_to_end(self):
        documents, labels = [], []
        for index in range(30):
            label = "class_a" if index % 2 == 0 else "class_b"
            labels.append(label)
            documents.append([label, "shared", f"token_{index % 5}"])
        split = {
            "inner_train": list(range(20)),
            "validation": list(range(20, 24)),
            "outer_train": list(range(24)),
            "test": list(range(24, 30)),
        }
        representation = {
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": True,
            "norm": "l2",
        }
        classifier = {
            "c_values": [0.1, 1.0],
            "solver": "lbfgs",
            "max_iter": 200,
            "tolerance": 0.0001,
        }
        best, candidates, _, _, _ = development_search(
            documents, labels, split, "ig", 4, 1,
            representation, classifier, 42,
        )
        scores, truth, prediction, vocabulary, _, _ = final_evaluation(
            documents, labels, split, "ig", 4, 1,
            representation, classifier, best.c_value, 42,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(vocabulary), 4)
        self.assertEqual(truth, prediction)
        self.assertEqual(scores["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
