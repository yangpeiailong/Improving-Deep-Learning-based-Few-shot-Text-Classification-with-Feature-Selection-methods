import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from torch import nn
    from torch.utils.data import DataLoader

    from fs_lrtc.data.sequences import EncodedDocument, collate_sequences
    from fs_lrtc.experiments.neural import evaluate


class TinyDataset:
    def __init__(self):
        self.records = [
            EncodedDocument([1], 1, 0, 4),
            EncodedDocument([1], 1, 1, 9),
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class FixedClassifier(nn.Module if torch is not None else object):
    def forward(self, input_ids, lengths):
        del lengths
        return torch.tensor([[2.0, -1.0], [-1.0, 2.0]])[: input_ids.size(0)]


@unittest.skipUnless(torch is not None, "PyTorch experiment dependency is not installed")
class TrainingDiagnosticTests(unittest.TestCase):
    def test_evaluation_reports_cross_entropy_loss(self):
        loader = DataLoader(TinyDataset(), batch_size=2, collate_fn=collate_sequences)
        scores, truth, predictions, indices, _, _ = evaluate(
            FixedClassifier(), loader, torch.device("cpu")
        )
        self.assertGreater(scores["loss"], 0.0)
        self.assertEqual(scores["accuracy"], 1.0)
        self.assertEqual(truth, [0, 1])
        self.assertEqual(predictions, [0, 1])
        self.assertEqual(indices, [4, 9])


if __name__ == "__main__":
    unittest.main()
