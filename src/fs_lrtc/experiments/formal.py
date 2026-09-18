"""Validation and aggregation for the frozen classic-model main experiment."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from statistics import mean, stdev
from typing import Any, Iterable, Sequence


RunKey = tuple[str, int, str, str]


def build_shard_configs(master: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Expand one authoritative protocol into non-overlapping shard configs."""
    orchestration = master.get("formal_orchestration")
    if not isinstance(orchestration, dict):
        raise ValueError("Formal master config is missing formal_orchestration")
    shards = orchestration.get("shards")
    output_base = orchestration.get("output_base")
    if not isinstance(shards, dict) or not shards:
        raise ValueError("formal_orchestration.shards must be a non-empty mapping")
    if not isinstance(output_base, str) or not output_base:
        raise ValueError("formal_orchestration.output_base must be a non-empty string")
    master_datasets = [str(value) for value in master["datasets"]]
    seen: list[str] = []
    expanded: dict[int, dict[str, Any]] = {}
    for raw_number, datasets in sorted(shards.items(), key=lambda pair: int(pair[0])):
        number = int(raw_number)
        if number in expanded or not isinstance(datasets, list) or not datasets:
            raise ValueError(f"Invalid formal shard: {raw_number}")
        names = [str(value) for value in datasets]
        seen.extend(names)
        config = copy.deepcopy(master)
        config.pop("formal_orchestration", None)
        config["datasets"] = names
        config["experiment_id"] = f"{master['experiment_id']}_shard{number}"
        config["output"]["directory"] = f"{output_base}/shard_{number}"
        expanded[number] = config
    if len(seen) != len(set(seen)):
        raise ValueError("Formal shard dataset lists overlap")
    if set(seen) != set(master_datasets) or len(seen) != len(master_datasets):
        raise ValueError("Formal shards must partition the master dataset list exactly")
    return expanded


def expected_run_keys(configs: Sequence[dict[str, Any]]) -> set[RunKey]:
    keys: set[RunKey] = set()
    for config in configs:
        for dataset in config["datasets"]:
            for fold in config["folds"]:
                for selector in config["feature_selection"]["selectors"]:
                    for model in config["models"]:
                        key = (str(dataset), int(fold), str(selector), str(model))
                        if key in keys:
                            raise ValueError(f"Formal configurations overlap at {key}")
                        keys.add(key)
    return keys


def row_key(row: dict[str, Any]) -> RunKey:
    return (
        str(row["dataset_id"]),
        int(row["fold"]),
        str(row["selector"]),
        str(row["model"]),
    )


def validate_formal_rows(
    rows: Sequence[dict[str, Any]], configs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected = expected_run_keys(configs)
    actual_keys = [row_key(row) for row in rows]
    counts = Counter(actual_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    actual = set(actual_keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    split_hash_fields = (
        "inner_train_indices_sha256",
        "validation_indices_sha256",
        "outer_train_indices_sha256",
        "test_indices_sha256",
    )
    split_hash_conflicts: list[tuple[str, int, str]] = []
    for field in split_hash_fields:
        grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in rows:
            grouped[(str(row["dataset_id"]), int(row["fold"]))].add(str(row[field]))
        split_hash_conflicts.extend(
            (dataset, fold, field)
            for (dataset, fold), values in grouped.items()
            if len(values) != 1
        )

    vocabulary_hash_conflicts: list[tuple[str, int, str, str]] = []
    for field in ("development_vocabulary_sha256", "final_vocabulary_sha256"):
        grouped_vocab: dict[tuple[str, int, str], set[str]] = defaultdict(set)
        for row in rows:
            grouped_vocab[
                (str(row["dataset_id"]), int(row["fold"]), str(row["selector"]))
            ].add(str(row[field]))
        vocabulary_hash_conflicts.extend(
            (dataset, fold, selector, field)
            for (dataset, fold, selector), values in grouped_vocab.items()
            if len(values) != 1
        )

    status = "passed" if not (
        duplicates or missing or extra or split_hash_conflicts or vocabulary_hash_conflicts
    ) else "failed"
    return {
        "status": status,
        "expected_runs": len(expected),
        "actual_rows": len(rows),
        "unique_runs": len(actual),
        "duplicate_keys": [list(value) for value in duplicates],
        "missing_keys": [list(value) for value in missing],
        "extra_keys": [list(value) for value in extra],
        "split_hash_conflicts": [list(value) for value in split_hash_conflicts],
        "vocabulary_hash_conflicts": [list(value) for value in vocabulary_hash_conflicts],
    }


def _sample_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset_id"]), str(row["model"]), str(row["selector"]))].append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "test_accuracy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "validation_accuracy",
        "best_epoch",
        "development_feature_selection_seconds",
        "final_feature_selection_seconds",
        "development_training_seconds",
        "final_training_seconds",
        "test_inference_seconds",
    )
    for (dataset, model, selector), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "dataset_id": dataset,
            "model": model,
            "selector": selector,
            "folds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = _sample_std(values)
        output.append(result)
    return output


def paired_effects(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row_key(row): row for row in rows}
    datasets = sorted({str(row["dataset_id"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows})
    selectors = sorted({str(row["selector"]) for row in rows if row["selector"] != "none"})
    folds_by_dataset: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        fold = int(row["fold"])
        dataset = str(row["dataset_id"])
        if fold not in folds_by_dataset[dataset]:
            folds_by_dataset[dataset].append(fold)
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        folds = sorted(folds_by_dataset[dataset])
        for model in models:
            baseline_keys = [(dataset, fold, "none", model) for fold in folds]
            if not all(key in lookup for key in baseline_keys):
                continue
            for selector in selectors:
                selected_keys = [(dataset, fold, selector, model) for fold in folds]
                if not all(key in lookup for key in selected_keys):
                    continue
                deltas = [
                    float(lookup[selected]["test_accuracy"])
                    - float(lookup[baseline]["test_accuracy"])
                    for selected, baseline in zip(selected_keys, baseline_keys, strict=True)
                ]
                tolerance = 1e-12
                output.append(
                    {
                        "dataset_id": dataset,
                        "model": model,
                        "selector": selector,
                        "folds": len(deltas),
                        "accuracy_delta_mean": mean(deltas),
                        "accuracy_delta_std": _sample_std(deltas),
                        "wins": sum(value > tolerance for value in deltas),
                        "ties": sum(abs(value) <= tolerance for value in deltas),
                        "losses": sum(value < -tolerance for value in deltas),
                    }
                )
    return output


def overall_selector_summary(
    aggregate: Sequence[dict[str, Any]], effects: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    accuracy_by_selector: dict[str, list[float]] = defaultdict(list)
    for row in aggregate:
        accuracy_by_selector[str(row["selector"])].append(float(row["test_accuracy_mean"]))
    effects_by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        effects_by_selector[str(row["selector"])].append(row)
    output: list[dict[str, Any]] = []
    for selector in sorted(accuracy_by_selector, key=lambda value: (value != "none", value)):
        cells = accuracy_by_selector[selector]
        selector_effects = effects_by_selector.get(selector, [])
        output.append(
            {
                "selector": selector,
                "dataset_model_cells": len(cells),
                "mean_cell_accuracy": mean(cells),
                "mean_accuracy_delta_vs_none": (
                    mean(float(row["accuracy_delta_mean"]) for row in selector_effects)
                    if selector_effects else 0.0
                ),
                "cell_wins": sum(float(row["accuracy_delta_mean"]) > 0 for row in selector_effects),
                "cell_ties": sum(float(row["accuracy_delta_mean"]) == 0 for row in selector_effects),
                "cell_losses": sum(float(row["accuracy_delta_mean"]) < 0 for row in selector_effects),
            }
        )
    return output
