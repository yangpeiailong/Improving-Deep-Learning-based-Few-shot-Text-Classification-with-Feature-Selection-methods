from __future__ import annotations

import types
import unittest


try:
    import torch
    from torch import nn

    from fs_lrtc.data.bert import filtered_bert_texts
    from fs_lrtc.models.bert_heads import BertTextClassifier
except (ImportError, OSError):
    torch = None


if torch is not None:
    class FakeEncoder(nn.Module):
        def __init__(self, hidden_size=6):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=hidden_size)
            self.embedding = nn.Embedding(20, hidden_size)

        def forward(self, input_ids, attention_mask, return_dict=True, **kwargs):
            states = self.embedding(input_ids)
            return types.SimpleNamespace(
                last_hidden_state=states,
                pooler_output=states[:, 0],
            )
else:
    FakeEncoder = object


@unittest.skipIf(torch is None, "PyTorch experiment dependency is not installed")
class BertHeadTests(unittest.TestCase):
    def build(self, head):
        return BertTextClassifier(
            FakeEncoder(), 3, head, 0.0,
            cnn_filter_count=4, cnn_kernel_sizes=(2, 3),
            lstm_hidden_size=5, attention_hidden_size=4,
        )

    def test_all_heads_return_raw_logits(self):
        ids = torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        for head in ("bert_cls", "bert_mean", "bert_cnn", "bert_bilstm_attention"):
            model = self.build(head).eval()
            logits = model(ids, mask)
            self.assertEqual(tuple(logits.shape), (2, 3))
            self.assertFalse(torch.allclose(logits.sum(dim=1), torch.ones(2)))

    def test_masked_mean_ignores_right_padding(self):
        states = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = BertTextClassifier.masked_mean(states, mask)
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 4.0]])))

    def test_bilstm_attention_is_deterministic_in_eval(self):
        model = self.build("bert_bilstm_attention").eval()
        ids = torch.tensor([[1, 2, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        self.assertTrue(torch.allclose(model(ids, mask), model(ids, mask)))

    def test_feature_filtering_preserves_none_and_handles_empty(self):
        documents = [["kept", "outside"], ["outside"]]
        self.assertEqual(
            filtered_bert_texts(documents, [0], None, "[UNK]"),
            ["kept outside"],
        )
        self.assertEqual(
            filtered_bert_texts(documents, [0, 1], ["kept"], "[UNK]"),
            ["kept", "[UNK]"],
        )


if __name__ == "__main__":
    unittest.main()
