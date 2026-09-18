"""Deterministic outer folds and inner validation splits."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from fs_lrtc.data.datasets import DatasetSpec, read_text_lines
from fs_lrtc.data.preparation import canonical_json_sha256, load_source_records
from fs_lrtc.utils.hashing import file_sha256


def stable_key(seed: int, *parts: object) -> str:
    joined = "\x1f".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def stratified_test_folds(labels: list[str], n_splits: int, seed: int, dataset_id: str) -> list[list[int]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    for label, indices in by_label.items():
        if len(indices) < n_splits:
            raise ValueError(
                f"{dataset_id}: class {label!r} has {len(indices)} records, fewer than {n_splits} folds"
            )
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda i: stable_key(seed, dataset_id, label, i))
        offset = int(stable_key(seed, dataset_id, label, "offset")[:8], 16) % n_splits
        for position, index in enumerate(ordered):
            folds[(position + offset) % n_splits].append(index)
    return [sorted(fold) for fold in folds]


def inner_validation_split(
    train_indices: list[int],
    labels: list[str],
    fraction: float,
    seed: int,
    dataset_id: str,
    outer_fold: int,
) -> tuple[list[int], list[int]]:
    if not 0 < fraction < 1:
        raise ValueError("inner validation fraction must be between 0 and 1")
    by_label: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        by_label[labels[index]].append(index)
    inner_train: list[int] = []
    validation: list[int] = []
    for label in sorted(by_label):
        ordered = sorted(
            by_label[label],
            key=lambda i: stable_key(seed, dataset_id, outer_fold, "validation", label, i),
        )
        if len(ordered) < 2:
            inner_train.extend(ordered)
            continue
        n_validation = max(1, round(len(ordered) * fraction))
        n_validation = min(n_validation, len(ordered) - 1)
        validation.extend(ordered[:n_validation])
        inner_train.extend(ordered[n_validation:])
    return sorted(inner_train), sorted(validation)


def grouped_inner_validation_split(
    train_indices: list[int],
    labels: list[str],
    group_keys: list[str],
    fraction: float,
    seed: int,
    dataset_id: str,
    outer_fold: int,
) -> tuple[list[int], list[int]]:
    """Choose atomic validation groups while retaining each class in training."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        groups[group_keys[index]].append(index)
    total = Counter(labels[index] for index in train_indices)
    target = {label: count * fraction for label, count in total.items()}
    validation_counts: Counter[str] = Counter()
    validation: list[int] = []
    ordered = sorted(
        groups.items(),
        key=lambda pair: stable_key(seed, dataset_id, outer_fold, "inner_group", pair[0]),
    )
    for _, indices in ordered:
        group_counts = Counter(labels[index] for index in indices)
        if any(total[label] - validation_counts[label] - count < 1 for label, count in group_counts.items()):
            continue
        current_cost = sum((validation_counts[label] - target[label]) ** 2 for label in total)
        candidate_cost = sum(
            (validation_counts[label] + group_counts[label] - target[label]) ** 2
            for label in total
        )
        if candidate_cost < current_cost:
            validation.extend(indices)
            validation_counts.update(group_counts)
    validation_set = set(validation)
    inner_train = [index for index in train_indices if index not in validation_set]
    return sorted(inner_train), sorted(validation)


def grouped_test_folds(
    labels: list[str], group_keys: list[str], n_splits: int, seed: int, dataset_id: str
) -> list[list[int]]:
    if len(labels) != len(group_keys):
        raise ValueError("labels and group_keys must have equal length")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        groups[key].append(index)
    global_counts = Counter(labels)
    fold_counts = [Counter() for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    ordered_groups = sorted(
        groups.items(),
        key=lambda pair: (-len(pair[1]), stable_key(seed, dataset_id, "group", pair[0])),
    )
    for group_key, indices in ordered_groups:
        group_counts = Counter(labels[index] for index in indices)
        candidates: list[tuple[float, int, str, int]] = []
        for fold in range(n_splits):
            for label, count in group_counts.items():
                fold_counts[fold][label] += count
            # Minimize dispersion of every class proportion across folds. This
            # is the deterministic analogue of stratified group assignment.
            class_cost = 0.0
            for label, total in global_counts.items():
                proportions = [counts[label] / total for counts in fold_counts]
                mean = sum(proportions) / n_splits
                class_cost += sum((value - mean) ** 2 for value in proportions) / n_splits
            for label, count in group_counts.items():
                fold_counts[fold][label] -= count
            candidates.append(
                (class_cost, fold_sizes[fold], stable_key(seed, dataset_id, group_key, fold), fold)
            )
        chosen = min(candidates)[-1]
        folds[chosen].extend(indices)
        fold_counts[chosen].update(group_counts)
        fold_sizes[chosen] += len(indices)
    return [sorted(fold) for fold in folds]


def create_primary_splits(
    dataset_id: str,
    processed_dir: Path,
    destination: Path,
    config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    labels = read_text_lines(processed_dir / "labels.txt")
    texts = read_text_lines(processed_dir / "texts.txt")
    if len(labels) != len(texts):
        raise ValueError(f"Processed text/label mismatch: {dataset_id}")
    seed = int(config["seed"])
    n_splits = int(config["outer_folds"])
    fraction = float(config["inner_validation_fraction"])
    folds = stratified_test_folds(labels, n_splits, seed, dataset_id)
    return _write_split_family(
        dataset_id, labels, [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts],
        folds, destination, config, fraction, overwrite, index_basis="zero_based_processed_index"
    )


def create_grouped_raw_splits(
    spec: DatasetSpec,
    preparation_config: dict[str, Any],
    destination: Path,
    config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    records = load_source_records(spec, preparation_config)
    labels = [record.label for record in records]
    hashes = [record.processed_hash for record in records]
    seed = int(config["seed"])
    n_splits = int(config["outer_folds"])
    fraction = float(config["inner_validation_fraction"])
    folds = grouped_test_folds(labels, hashes, n_splits, seed, spec.dataset_id)
    return _write_split_family(
        spec.dataset_id, labels, hashes, folds, destination, config, fraction, overwrite,
        index_basis="one_based_original_row", index_offset=1, group_inner=True
    )


def _write_split_family(
    dataset_id: str,
    labels: list[str],
    text_hashes: list[str],
    test_folds: list[list[int]],
    destination: Path,
    config: dict[str, Any],
    fraction: float,
    overwrite: bool,
    index_basis: str,
    index_offset: int = 0,
    group_inner: bool = False,
) -> dict[str, Any]:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Split directory already exists (use --overwrite): {destination}")
    temp = destination.parent / f".{destination.name}.building"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    all_indices = set(range(len(labels)))
    seen_test: list[int] = []
    fold_reports: list[dict[str, Any]] = []
    for outer_fold, test in enumerate(test_folds):
        if not test:
            raise RuntimeError(f"Empty outer test fold detected: {dataset_id}, fold={outer_fold}")
        test_set = set(test)
        outer_train = sorted(all_indices - test_set)
        if group_inner:
            inner_train, validation = grouped_inner_validation_split(
                outer_train, labels, text_hashes, fraction, int(config["seed"]), dataset_id, outer_fold
            )
        else:
            inner_train, validation = inner_validation_split(
                outer_train, labels, fraction, int(config["seed"]), dataset_id, outer_fold
            )
        _validate_fold(inner_train, validation, test, text_hashes)
        payload = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "index_basis": index_basis,
            "inner_train": [i + index_offset for i in inner_train],
            "validation": [i + index_offset for i in validation],
            "outer_train": [i + index_offset for i in outer_train],
            "test": [i + index_offset for i in test],
            "class_counts": {
                "inner_train": dict(sorted(Counter(labels[i] for i in inner_train).items())),
                "validation": dict(sorted(Counter(labels[i] for i in validation).items())),
                "test": dict(sorted(Counter(labels[i] for i in test).items())),
            },
        }
        fold_path = temp / f"fold_{outer_fold}.json"
        fold_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        seen_test.extend(test)
        fold_reports.append(
            {"fold": outer_fold, "train": len(outer_train), "validation": len(validation), "test": len(test), "sha256": file_sha256(fold_path)}
        )
    if sorted(seen_test) != list(range(len(labels))):
        raise RuntimeError(f"Test folds do not partition dataset exactly once: {dataset_id}")
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "records": len(labels),
        "classes": dict(sorted(Counter(labels).items())),
        "split_config_sha256": canonical_json_sha256(config),
        "index_basis": index_basis,
        "folds": fold_reports,
    }
    (temp / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    temp.rename(destination)
    return manifest


def _validate_fold(
    inner_train: Iterable[int], validation: Iterable[int], test: Iterable[int], hashes: list[str]
) -> None:
    train_set, validation_set, test_set = set(inner_train), set(validation), set(test)
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise RuntimeError("Split index overlap detected")
    train_hashes = {hashes[i] for i in train_set}
    validation_hashes = {hashes[i] for i in validation_set}
    test_hashes = {hashes[i] for i in test_set}
    if train_hashes & validation_hashes or train_hashes & test_hashes or validation_hashes & test_hashes:
        raise RuntimeError("Exact processed text crosses a split boundary")
