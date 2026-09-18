"""Fit-scope enforcement, diagnostic output, and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from fs_lrtc.features.selectors import FilterFeatureSelector


def subset_by_indices(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    if len(set(indices)) != len(indices):
        raise ValueError("Fit indices contain duplicates")
    if any(index < 0 or index >= len(values) for index in indices):
        raise IndexError("Fit index is outside the dataset")
    return [values[index] for index in indices]


def fit_selector_at_scope(
    method: str,
    tokenized_documents: Sequence[Sequence[str]],
    labels: Sequence[str],
    fit_indices: Sequence[int],
    forbidden_indices: Sequence[int],
    minimum_document_frequency: int = 1,
) -> FilterFeatureSelector:
    fit_set = set(fit_indices)
    forbidden_set = set(forbidden_indices)
    overlap = fit_set & forbidden_set
    if overlap:
        raise ValueError(f"Forbidden held-out indices entered selector fit: {sorted(overlap)[:5]}")
    documents = subset_by_indices(tokenized_documents, fit_indices)
    fit_labels = subset_by_indices(labels, fit_indices)
    return FilterFeatureSelector(method, minimum_document_frequency).fit(documents, fit_labels)


def sequence_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_ranked_features(
    selector: FilterFeatureSelector,
    feature_count: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "rank", "token", "score", "document_frequency",
                "strongest_class", "strongest_class_document_frequency",
            ),
        )
        writer.writeheader()
        for rank, item in enumerate(selector.top(feature_count), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "token": item.token,
                    "score": format(item.score, ".17g"),
                    "document_frequency": item.document_frequency,
                    "strongest_class": item.strongest_class,
                    "strongest_class_document_frequency": item.strongest_class_document_frequency,
                }
            )


def write_fit_manifest(
    path: Path,
    dataset_id: str,
    fold: int,
    stage: str,
    fit_indices: Sequence[int],
    held_out_indices: Sequence[int],
    labels: Sequence[str],
    tokenizer_config: dict[str, Any],
) -> None:
    fit_labels = subset_by_indices(labels, fit_indices)
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "fold": fold,
        "stage": stage,
        "fit_scope": "inner_train_only" if stage == "development" else "outer_train_only",
        "fit_records": len(fit_indices),
        "held_out_records": len(held_out_indices),
        "fit_indices_sha256": sequence_sha256(fit_indices),
        "fit_labels_sha256": sequence_sha256(fit_labels),
        "tokenizer_config": tokenizer_config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
