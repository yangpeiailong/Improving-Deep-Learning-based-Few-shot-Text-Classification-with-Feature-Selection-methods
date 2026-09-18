"""Inductive word-graph convolutional document classifier."""

from __future__ import annotations

import torch
from torch import nn


class InductiveWordGCN(nn.Module):
    def __init__(
        self,
        node_count: int,
        class_count: int,
        embedding_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout: float,
        pretrained_weights: torch.Tensor | None = None,
        embeddings_trainable: bool = True,
        residual_scale_initial: float = 0.1,
    ) -> None:
        super().__init__()
        if pretrained_weights is None:
            word_embeddings = torch.empty(node_count, embedding_dimension)
            nn.init.xavier_uniform_(word_embeddings)
        else:
            if tuple(pretrained_weights.shape) != (node_count, embedding_dimension):
                raise ValueError("Pretrained GCN word matrix has an unexpected shape")
            word_embeddings = pretrained_weights.detach().clone().float()
        self.word_embeddings = nn.Parameter(
            word_embeddings, requires_grad=embeddings_trainable
        )
        self.convolution_1 = nn.Linear(embedding_dimension, hidden_dimension, bias=False)
        self.convolution_2 = nn.Linear(hidden_dimension, output_dimension, bias=False)
        self.base_projection = (
            nn.Identity()
            if embedding_dimension == output_dimension
            else nn.Linear(embedding_dimension, output_dimension, bias=False)
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_initial)))
        self.word_normalization = nn.LayerNorm(output_dimension)
        self.classifier = nn.Linear(output_dimension, class_count)
        self.dropout = nn.Dropout(dropout)

    def encode_words(self, adjacency: torch.Tensor) -> torch.Tensor:
        base = self.base_projection(self.word_embeddings)
        hidden = torch.sparse.mm(adjacency, self.word_embeddings)
        hidden = torch.relu(self.convolution_1(hidden))
        hidden = self.dropout(hidden)
        hidden = torch.sparse.mm(adjacency, hidden)
        graph_residual = self.convolution_2(hidden)
        return self.word_normalization(base + self.residual_scale * graph_residual)

    def forward(
        self, adjacency: torch.Tensor, document_terms: torch.Tensor
    ) -> torch.Tensor:
        word_states = self.encode_words(adjacency)
        document_states = torch.sparse.mm(document_terms, word_states)
        return self.classifier(self.dropout(document_states))
