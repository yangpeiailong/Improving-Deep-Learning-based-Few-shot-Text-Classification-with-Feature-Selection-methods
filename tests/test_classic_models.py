import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from fs_lrtc.models.classic import BiLSTMClassifier, TextCNNClassifier, WordAveragingClassifier


@unittest.skipUnless(torch is not None, "PyTorch experiment dependency is not installed")
class ClassicModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_word_average_is_invariant_to_right_padding(self):
        model = WordAveragingClassifier(20, 8, 3, 0.0)
        model.eval()
        short = model(torch.tensor([[2, 3]]), torch.tensor([2]))
        padded = model(torch.tensor([[2, 3, 0, 0]]), torch.tensor([2]))
        self.assertTrue(torch.allclose(short, padded, atol=1e-7))

    def test_cnn_masks_extra_right_padding(self):
        model = TextCNNClassifier(20, 8, 3, 4, [2, 3], 0.0)
        model.eval()
        short = model(torch.tensor([[2, 3, 4]]), torch.tensor([3]))
        padded = model(torch.tensor([[2, 3, 4, 0, 0]]), torch.tensor([3]))
        self.assertTrue(torch.allclose(short, padded, atol=1e-7))

    def test_lstm_uses_lengths_and_has_deterministic_zero_state(self):
        model = BiLSTMClassifier(20, 8, 3, 5, 6, 0.0)
        model.eval()
        short = model(torch.tensor([[2, 3, 4]]), torch.tensor([3]))
        padded = model(torch.tensor([[2, 3, 4, 0, 0]]), torch.tensor([3]))
        repeated = model(torch.tensor([[2, 3, 4]]), torch.tensor([3]))
        self.assertTrue(torch.allclose(short, padded, atol=1e-7))
        self.assertTrue(torch.allclose(short, repeated, atol=1e-7))

    def test_models_return_logits_not_probabilities(self):
        model = WordAveragingClassifier(20, 8, 3, 0.0)
        output = model(torch.tensor([[2, 3]]), torch.tensor([2]))
        self.assertEqual(tuple(output.shape), (1, 3))
        self.assertFalse(torch.allclose(output.sum(dim=1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
