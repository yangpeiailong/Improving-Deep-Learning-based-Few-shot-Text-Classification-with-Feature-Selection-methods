import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from fs_lrtc.data.sequences import (
        PAD_INDEX,
        UNK_INDEX,
        EncodedDocument,
        build_vocabulary,
        collate_sequences,
        encode_document,
        matched_truncation_length,
        mean_effective_length,
    )


@unittest.skipUnless(torch is not None, "PyTorch experiment dependency is not installed")
class SequenceTests(unittest.TestCase):
    def test_reserved_indices_and_rank_order(self):
        vocabulary = build_vocabulary(["z", "a"])
        self.assertEqual(vocabulary["<pad>"], PAD_INDEX)
        self.assertEqual(vocabulary["<unk>"], UNK_INDEX)
        self.assertEqual(vocabulary["z"], 2)
        self.assertEqual(vocabulary["a"], 3)

    def test_unselected_tokens_are_dropped_and_empty_becomes_unk(self):
        vocabulary = build_vocabulary(["keep"])
        self.assertEqual(encode_document(["drop", "keep", "drop"], vocabulary, 10), ([2], 1))
        self.assertEqual(encode_document(["drop"], vocabulary, 10), ([UNK_INDEX], 1))

    def test_collate_preserves_true_lengths(self):
        records = [
            EncodedDocument([2, 3], 2, 0, 7),
            EncodedDocument([4], 1, 1, 8),
        ]
        batch = collate_sequences(records, minimum_padded_length=4)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertTrue(torch.equal(batch["lengths"], torch.tensor([2, 1])))
        self.assertEqual(batch["input_ids"][1, 1:].tolist(), [0, 0, 0])

    def test_matched_truncation_uses_fit_documents_only(self):
        documents = [["a", "b", "c", "d"], ["a", "b"], ["heldout"] * 100]
        full = build_vocabulary(["a", "b", "c", "d"])
        reference = build_vocabulary(["a"])
        cap = matched_truncation_length(documents, [0, 1], full, reference, 10)
        self.assertEqual(cap, 1)
        self.assertEqual(mean_effective_length(documents, [0, 1], reference, 10), 1.0)


if __name__ == "__main__":
    unittest.main()
