import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from fs_lrtc.experiments.neural import fasttext_embedding_matrix, label_mapping, model_family
    from fs_lrtc.data.sequences import build_vocabulary


class FakeFastText:
    def get_dimension(self):
        return 3

    def get_word_vector(self, token):
        return np.array([len(token), 1.0, -1.0], dtype=np.float32)


@unittest.skipUnless(torch is not None, "PyTorch experiment dependency is not installed")
class NeuralTrainingTests(unittest.TestCase):
    def test_fasttext_matrix_keeps_special_rows_zero(self):
        vocabulary = build_vocabulary(["hello", "x"])
        matrix = fasttext_embedding_matrix(vocabulary, FakeFastText(), 3)
        self.assertEqual(matrix.shape, (4, 3))
        self.assertEqual(matrix[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(matrix[1].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(matrix[2].tolist(), [5.0, 1.0, -1.0])

    def test_label_mapping_is_sorted_and_fit_scoped(self):
        mapping = label_mapping(["z", "a", "heldout"], [0, 1])
        self.assertEqual(mapping, {"a": 0, "z": 1})

    def test_model_family_validation(self):
        self.assertEqual(model_family("random_lstm"), "lstm")
        with self.assertRaises(ValueError):
            model_family("random_unknown")


if __name__ == "__main__":
    unittest.main()
