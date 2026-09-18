"""Mask-aware classification heads over a local BERT encoder."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class BertTextClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        class_count: int,
        head: str,
        dropout: float,
        cnn_filter_count: int = 100,
        cnn_kernel_sizes: Sequence[int] = (2, 3, 4),
        lstm_hidden_size: int = 128,
        attention_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        hidden = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        if head in {"bert_cls", "bert_mean"}:
            output_size = hidden
        elif head == "bert_cnn":
            self.convolutions = nn.ModuleList([
                nn.Conv1d(hidden, cnn_filter_count, int(kernel))
                for kernel in cnn_kernel_sizes
            ])
            self.kernel_sizes = tuple(int(value) for value in cnn_kernel_sizes)
            output_size = cnn_filter_count * len(self.kernel_sizes)
        elif head == "bert_bilstm_attention":
            self.lstm = nn.LSTM(
                hidden,
                lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Sequential(
                nn.Linear(lstm_hidden_size * 2, attention_hidden_size),
                nn.Tanh(),
                nn.Linear(attention_hidden_size, 1),
            )
            output_size = lstm_hidden_size * 2
        else:
            raise ValueError(f"Unsupported BERT head: {head}")
        self.classifier = nn.Linear(output_size, class_count)

    @staticmethod
    def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def masked_cnn(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = states.transpose(1, 2)
        lengths = mask.sum(dim=1)
        pooled: list[torch.Tensor] = []
        for convolution, kernel in zip(self.convolutions, self.kernel_sizes, strict=True):
            values = torch.relu(convolution(sequence))
            positions = torch.arange(values.shape[-1], device=values.device).unsqueeze(0)
            valid = positions < (lengths - kernel + 1).clamp_min(1).unsqueeze(1)
            values = values.masked_fill(~valid.unsqueeze(1), torch.finfo(values.dtype).min)
            pooled.append(values.max(dim=-1).values)
        return torch.cat(pooled, dim=1)

    def bilstm_attention(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            states, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=states.shape[1]
        )
        scores = self.attention(output).squeeze(-1)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), output).squeeze(1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in kwargs:
            encoder_inputs["token_type_ids"] = kwargs["token_type_ids"]
        encoded = self.encoder(**encoder_inputs, return_dict=True)
        states = encoded.last_hidden_state
        if self.head == "bert_cls":
            pooled = encoded.pooler_output
            if pooled is None:
                pooled = states[:, 0]
        elif self.head == "bert_mean":
            pooled = self.masked_mean(states, attention_mask)
        elif self.head == "bert_cnn":
            pooled = self.masked_cnn(states, attention_mask)
        else:
            pooled = self.bilstm_attention(states, attention_mask)
        return self.classifier(self.dropout(pooled))
