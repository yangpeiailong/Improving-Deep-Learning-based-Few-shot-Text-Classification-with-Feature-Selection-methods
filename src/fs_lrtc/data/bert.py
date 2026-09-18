"""Dynamic-padding datasets for offline BERT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


class BertTextDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: Sequence[int]) -> None:
        if len(labels) != len(encodings["input_ids"]):
            raise ValueError("BERT encoding and label counts differ")
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


@dataclass
class BertBatchCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([record["labels"] for record in records], dtype=torch.long)
        features = [
            {key: value for key, value in record.items() if key != "labels"}
            for record in records
        ]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def encode_texts(tokenizer: Any, texts: Sequence[str], max_length: int) -> dict[str, list[list[int]]]:
    if max_length < 8:
        raise ValueError("BERT max_length must be at least 8")
    encoded = tokenizer(
        list(texts),
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: list(value) for key, value in encoded.items()}


def filtered_bert_texts(
    tokenized_documents: Sequence[Sequence[str]],
    indices: Sequence[int],
    selected_vocabulary: Sequence[str] | None,
    unknown_token: str,
) -> list[str]:
    """Create normalized BERT inputs; None retains every token in each document."""
    selected = None if selected_vocabulary is None else set(selected_vocabulary)
    output: list[str] = []
    for index in indices:
        tokens = list(tokenized_documents[index])
        if selected is not None:
            tokens = [token for token in tokens if token in selected]
        output.append(" ".join(tokens) if tokens else unknown_token)
    return output
