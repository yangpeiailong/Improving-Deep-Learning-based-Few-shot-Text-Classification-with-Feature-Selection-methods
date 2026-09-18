from __future__ import annotations

import unittest

from fs_lrtc.data.word_graph import build_word_graph, document_term_matrix


class WordGraphTests(unittest.TestCase):
    def test_held_out_documents_cannot_change_training_graph(self):
        documents = [["a", "b", "c"], ["a", "b"], ["held", "out"]]
        first = build_word_graph(documents, [0, 1], ["a", "b", "c"], window_size=2)
        documents[2] = ["a", "b", "c", "changed"]
        second = build_word_graph(documents, [0, 1], ["a", "b", "c"], window_size=2)
        self.assertEqual(first, second)

    def test_graph_is_symmetric_and_contains_self_loops(self):
        graph = build_word_graph(
            [["a", "b"], ["a", "b"], ["a", "c"]],
            [0, 1, 2], ["a", "b", "c"], window_size=2,
        )
        edges = dict(zip(
            zip(graph.adjacency.rows, graph.adjacency.columns, strict=True),
            graph.adjacency.values,
            strict=True,
        ))
        for node in range(4):
            self.assertIn((node, node), edges)
        for (left, right), value in edges.items():
            self.assertAlmostEqual(value, edges[(right, left)])

    def test_empty_document_uses_isolated_unknown_node(self):
        graph = build_word_graph([["a", "b"]], [0], ["a", "b"], window_size=2)
        matrix = document_term_matrix([["outside"]], [0], graph)
        self.assertEqual(matrix.shape, (1, 3))
        self.assertEqual(matrix.columns, [graph.unknown_node])
        self.assertEqual(matrix.values, [1.0])

    def test_window_count_includes_last_window(self):
        graph = build_word_graph([["a", "b", "c"]], [0], ["a", "b", "c"], 2)
        self.assertEqual(graph.windows, 2)


try:
    import torch

    from fs_lrtc.experiments.gcn import build_gcn, fasttext_node_matrix, sparse_tensor
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch experiment dependency is not installed")
class GCNModelTests(unittest.TestCase):
    def test_model_outputs_logits_for_each_document(self):
        graph = build_word_graph([["a", "b"], ["a", "c"]], [0, 1], ["a", "b", "c"], 2)
        documents = document_term_matrix([["a"], ["b", "c"]], [0, 1], graph)
        adjacency = sparse_tensor(graph.adjacency, torch.device("cpu"))
        document_tensor = sparse_tensor(documents, torch.device("cpu"))
        model = build_gcn(
            4, 2,
            {"embedding_dimension": 8, "hidden_dimension": 6,
             "output_dimension": 5, "dropout": 0.0},
        )
        logits = model(adjacency, document_tensor)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertFalse(torch.allclose(logits.sum(dim=1), torch.ones(2)))

    def test_fasttext_nodes_are_aligned_and_can_be_frozen(self):
        class FakeFastText:
            def get_dimension(self):
                return 3

            def get_word_vector(self, token):
                return {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}[token]

        weights = fasttext_node_matrix(["a", "b"], FakeFastText(), 3)
        self.assertEqual(tuple(weights.shape), (3, 3))
        self.assertTrue(torch.equal(weights[2], torch.zeros(3)))
        model = build_gcn(
            3, 2,
            {"embedding_dimension": 3, "embeddings_trainable": False,
             "hidden_dimension": 4, "output_dimension": 3, "dropout": 0.0,
             "residual_scale_initial": 0.1},
            weights,
        )
        self.assertFalse(model.word_embeddings.requires_grad)
        self.assertAlmostEqual(float(model.residual_scale.detach()), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
