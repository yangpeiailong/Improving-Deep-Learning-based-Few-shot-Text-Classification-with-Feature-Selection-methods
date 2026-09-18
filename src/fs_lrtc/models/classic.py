"""Padding-aware classic neural text classifiers returning raw logits."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _embedding(
    vocabulary_size: int,
    embedding_dimension: int,
    pretrained_weights: torch.Tensor | None,
    trainable: bool,
) -> nn.Embedding:
    layer = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=0)
    if pretrained_weights is not None:
        if tuple(pretrained_weights.shape) != (vocabulary_size, embedding_dimension):
            raise ValueError("Pretrained embedding matrix has an unexpected shape")
        with torch.no_grad():
            layer.weight.copy_(pretrained_weights)
    with torch.no_grad():
        layer.weight[0].zero_()
    layer.weight.requires_grad_(trainable)
    return layer


class WordAveragingClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        class_count: int,
        dropout: float,
        pretrained_weights: torch.Tensor | None = None,
        embeddings_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = _embedding(
            vocabulary_size, embedding_dimension, pretrained_weights, embeddings_trainable
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dimension, class_count)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = input_ids.ne(0).unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        denominator = lengths.clamp_min(1).to(embedded.dtype).unsqueeze(1)
        pooled = summed / denominator
        return self.classifier(self.dropout(pooled))


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        class_count: int,
        filter_count: int,
        kernel_sizes: Sequence[int],
        dropout: float,
        pretrained_weights: torch.Tensor | None = None,
        embeddings_trainable: bool = True,
    ) -> None:
        super().__init__()
        if not kernel_sizes or min(kernel_sizes) < 1:
            raise ValueError("CNN kernel sizes must be positive and non-empty")
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        self.embedding = _embedding(
            vocabulary_size, embedding_dimension, pretrained_weights, embeddings_trainable
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dimension, filter_count, kernel_size)
            for kernel_size in self.kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filter_count * len(self.kernel_sizes), class_count)

    @property
    def minimum_input_length(self) -> int:
        return max(self.kernel_sizes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled: list[torch.Tensor] = []
        for kernel_size, convolution in zip(
            self.kernel_sizes, self.convolutions, strict=True
        ):
            convolved = torch.relu(convolution(embedded))
            available = torch.clamp(lengths - kernel_size + 1, min=1)
            positions = torch.arange(convolved.size(2), device=convolved.device)
            valid = positions.unsqueeze(0) < available.unsqueeze(1)
            convolved = convolved.masked_fill(~valid.unsqueeze(1), float("-inf"))
            pooled.append(convolved.max(dim=2).values)
        representation = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(representation))


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        class_count: int,
        hidden_size: int,
        classifier_hidden_size: int,
        dropout: float,
        pretrained_weights: torch.Tensor | None = None,
        embeddings_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = _embedding(
            vocabulary_size, embedding_dimension, pretrained_weights, embeddings_trainable
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, class_count),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(representation)
