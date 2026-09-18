"""Vocabulary-scoped integer sequence datasets for classic neural models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_INDEX = 0
UNK_INDEX = 1


@dataclass(frozen=True)
class EncodedDocument:
    token_ids: list[int]
    length: int
    label_id: int
    record_index: int


def build_vocabulary(tokens: Sequence[str]) -> dict[str, int]:
    """Keep feature ranking order while reserving stable special indices."""
    if len(tokens) != len(set(tokens)):
        raise ValueError("Vocabulary tokens must be unique")
    if PAD_TOKEN in tokens or UNK_TOKEN in tokens:
        raise ValueError("Selected tokens collide with reserved special tokens")
    vocabulary = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    vocabulary.update({token: index for index, token in enumerate(tokens, start=2)})
    return vocabulary


def encode_document(
    tokens: Sequence[str], vocabulary: dict[str, int], max_length: int
) -> tuple[list[int], int]:
    """Drop unselected tokens; use UNK only when no selected token remains."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    selected = [vocabulary[token] for token in tokens if token in vocabulary]
    selected = selected[:max_length]
    if not selected:
        selected = [UNK_INDEX]
    return selected, len(selected)


def retained_token_count(tokens: Sequence[str], vocabulary: dict[str, int]) -> int:
    """Count tokens retained by a fitted vocabulary before truncation/fallback."""
    return sum(token in vocabulary for token in tokens)


def mean_effective_length(
    tokenized_documents: Sequence[Sequence[str]],
    indices: Sequence[int],
    vocabulary: dict[str, int],
    max_length: int,
) -> float:
    if not indices:
        raise ValueError("Cannot summarize an empty index set")
    lengths = [
        max(1, min(retained_token_count(tokenized_documents[index], vocabulary), max_length))
        for index in indices
    ]
    return sum(lengths) / len(lengths)


def matched_truncation_length(
    tokenized_documents: Sequence[Sequence[str]],
    fit_indices: Sequence[int],
    full_vocabulary: dict[str, int],
    reference_vocabulary: dict[str, int],
    reference_max_length: int,
) -> int:
    """Choose a training-only full-text cap matching a reference mean length."""
    if not fit_indices:
        raise ValueError("Cannot match length on an empty training scope")
    if reference_max_length < 1:
        raise ValueError("reference_max_length must be positive")
    target = mean_effective_length(
        tokenized_documents, fit_indices, reference_vocabulary, reference_max_length
    )
    raw_counts = [
        retained_token_count(tokenized_documents[index], full_vocabulary)
        for index in fit_indices
    ]
    return min(
        range(1, reference_max_length + 1),
        key=lambda cap: (
            abs(sum(max(1, min(count, cap)) for count in raw_counts) / len(raw_counts) - target),
            cap,
        ),
    )


class TextSequenceDataset(Dataset[EncodedDocument]):
    def __init__(
        self,
        tokenized_documents: Sequence[Sequence[str]],
        labels: Sequence[str],
        indices: Sequence[int],
        vocabulary: dict[str, int],
        label_to_index: dict[str, int],
        max_length: int,
    ) -> None:
        self.records: list[EncodedDocument] = []
        for record_index in indices:
            token_ids, length = encode_document(
                tokenized_documents[record_index], vocabulary, max_length
            )
            label = labels[record_index]
            if label not in label_to_index:
                raise ValueError(f"Held-out label is absent from training scope: {label}")
            self.records.append(
                EncodedDocument(
                    token_ids=token_ids,
                    length=length,
                    label_id=label_to_index[label],
                    record_index=record_index,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EncodedDocument:
        return self.records[index]


def collate_sequences(
    records: Sequence[EncodedDocument], minimum_padded_length: int = 1
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("Cannot collate an empty batch")
    padded_length = max(
        minimum_padded_length, max(record.length for record in records)
    )
    input_ids = torch.full(
        (len(records), padded_length), PAD_INDEX, dtype=torch.long
    )
    for row, record in enumerate(records):
        input_ids[row, : record.length] = torch.tensor(record.token_ids, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "lengths": torch.tensor([record.length for record in records], dtype=torch.long),
        "labels": torch.tensor([record.label_id for record in records], dtype=torch.long),
        "record_indices": torch.tensor(
            [record.record_index for record in records], dtype=torch.long
        ),
    }
