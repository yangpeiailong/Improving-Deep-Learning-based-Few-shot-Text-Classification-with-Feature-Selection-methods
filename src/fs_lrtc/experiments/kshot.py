"""Deterministic K-shot sampling, orchestration, and result validation."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from statistics import mean, stdev
from typing import Any, Sequence

from fs_lrtc.data.splitting import stable_key


KShotRunKey = tuple[str, int, int, str, str]


def kshot_split(
    frozen_split: dict[str, Any],
    labels: Sequence[str],
    shots_per_class: int,
    validation_fraction: float,
    seed: int,
    dataset_id: str,
    outer_fold: int,
) -> dict[str, Any]:
    """Select exactly K support records per class without touching the test fold.

    The final model is fitted on all K records per class.  Epoch selection uses
    a deterministic validation subset of that same K-shot support set, so no
    additional labelled examples are introduced by validation.
    """
    if shots_per_class < 2:
        raise ValueError("shots_per_class must be at least 2")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    outer_train = [int(index) for index in frozen_split["outer_train"]]
    test = [int(index) for index in frozen_split["test"]]
    by_label: dict[str, list[int]] = defaultdict(list)
    for index in outer_train:
        by_label[str(labels[index])].append(index)
    if len(by_label) < 2:
        raise ValueError("K-shot training pool must contain at least two classes")

    support: list[int] = []
    for label in sorted(by_label):
        available = by_label[label]
        if len(available) < shots_per_class:
            raise ValueError(
                f"{dataset_id} fold={outer_fold} class={label!r} has "
                f"{len(available)} outer-training records, fewer than K={shots_per_class}"
            )
        # K is deliberately absent from the ranking key, making smaller support
        # sets strict subsets of larger ones for the same dataset and fold.
        ordered = sorted(
            available,
            key=lambda index: stable_key(
                seed, dataset_id, outer_fold, "kshot_support", label, index
            ),
        )
        support.extend(ordered[:shots_per_class])

    support_by_label: dict[str, list[int]] = defaultdict(list)
    for index in support:
        support_by_label[str(labels[index])].append(index)
    inner_train: list[int] = []
    validation: list[int] = []
    for label in sorted(support_by_label):
        ordered = sorted(
            support_by_label[label],
            key=lambda index: stable_key(
                seed, dataset_id, outer_fold, shots_per_class,
                "kshot_validation", label, index,
            ),
        )
        validation_count = max(1, round(shots_per_class * validation_fraction))
        validation_count = min(validation_count, shots_per_class - 1)
        validation.extend(ordered[:validation_count])
        inner_train.extend(ordered[validation_count:])

    scoped = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "outer_fold": int(outer_fold),
        "scope": "k_shot",
        "shots_per_class": int(shots_per_class),
        "inner_train": sorted(inner_train),
        "validation": sorted(validation),
        "outer_train": sorted(support),
        "test": sorted(test),
    }
    validate_kshot_split(scoped, labels)
    if not set(scoped["outer_train"]).issubset(set(outer_train)):
        raise RuntimeError("K-shot support contains a record outside frozen outer training")
    if scoped["test"] != sorted(test):
        raise RuntimeError("K-shot transformation changed the frozen test fold")
    return scoped


def validate_kshot_split(split: dict[str, Any], labels: Sequence[str]) -> None:
    required = ("inner_train", "validation", "outer_train", "test")
    for name in required:
        values = split.get(name)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"Invalid K-shot index list: {name}")
        if any(not isinstance(index, int) or index < 0 or index >= len(labels) for index in values):
            raise IndexError(f"K-shot index is out of range: {name}")
    inner = set(split["inner_train"])
    validation = set(split["validation"])
    support = set(split["outer_train"])
    test = set(split["test"])
    if inner & validation or support & test:
        raise ValueError("K-shot training, validation, and test scopes overlap")
    if inner | validation != support:
        raise ValueError("K-shot inner_train + validation must equal support")
    shots = int(split["shots_per_class"])
    support_counts = Counter(str(labels[index]) for index in support)
    if not support_counts or any(count != shots for count in support_counts.values()):
        raise ValueError("Final K-shot support does not contain exactly K records per class")
    inner_counts = Counter(str(labels[index]) for index in inner)
    validation_counts = Counter(str(labels[index]) for index in validation)
    if set(inner_counts) != set(support_counts) or set(validation_counts) != set(support_counts):
        raise ValueError("Every class must occur in K-shot development train and validation")


def configured_shots(config: dict[str, Any]) -> list[int | None]:
    scope = config.get("training_scope")
    if scope is None:
        return [None]
    if not isinstance(scope, dict) or str(scope.get("mode")) != "k_shot":
        raise ValueError("Only training_scope.mode=k_shot is supported")
    raw = scope.get("shots_per_class")
    if raw is None or (isinstance(raw, list) and not raw):
        raise ValueError("training_scope.shots_per_class must not be empty")
    shots = [int(value) for value in raw] if isinstance(raw, list) else [int(raw)]
    if any(value < 2 for value in shots) or len(shots) != len(set(shots)):
        raise ValueError("K-shot values must be unique integers of at least 2")
    return shots


def apply_training_scope(
    frozen_split: dict[str, Any],
    labels: Sequence[str],
    config: dict[str, Any],
    dataset_id: str,
    outer_fold: int,
    shots_per_class: int | None,
) -> dict[str, Any]:
    if shots_per_class is None:
        return frozen_split
    scope = config["training_scope"]
    return kshot_split(
        frozen_split=frozen_split,
        labels=labels,
        shots_per_class=shots_per_class,
        validation_fraction=float(scope["validation_fraction"]),
        seed=int(scope.get("sampling_seed", config["seed"])),
        dataset_id=dataset_id,
        outer_fold=outer_fold,
    )


def build_kshot_shard_configs(master: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    orchestration = master.get("kshot_orchestration")
    if not isinstance(orchestration, dict):
        raise ValueError("K-shot master config is missing kshot_orchestration")
    shards = orchestration.get("shards")
    output_base = orchestration.get("output_base")
    if not isinstance(shards, dict) or not shards:
        raise ValueError("kshot_orchestration.shards must be a non-empty mapping")
    if not isinstance(output_base, str) or not output_base:
        raise ValueError("kshot_orchestration.output_base must be a non-empty string")
    master_datasets = [str(value) for value in master["datasets"]]
    seen: list[str] = []
    master_shots = configured_shots(master)
    if any(value is None for value in master_shots):
        raise ValueError("K-shot master config has no shot values")
    expanded: dict[int, list[dict[str, Any]]] = {}
    for raw_number, raw_datasets in sorted(shards.items(), key=lambda pair: int(pair[0])):
        number = int(raw_number)
        datasets = [str(value) for value in raw_datasets]
        if number in expanded or not datasets:
            raise ValueError(f"Invalid K-shot shard: {raw_number}")
        seen.extend(datasets)
        expanded[number] = []
        for shots in master_shots:
            config = copy.deepcopy(master)
            config.pop("kshot_orchestration", None)
            config["datasets"] = datasets
            config["training_scope"]["shots_per_class"] = int(shots)
            config["experiment_id"] = (
                f"{master['experiment_id']}_shard{number}_k{shots}"
            )
            config["output"]["directory"] = (
                f"{output_base}/shard_{number}/k_{shots}"
            )
            expanded[number].append(config)
    if len(seen) != len(set(seen)):
        raise ValueError("K-shot shard dataset lists overlap")
    if set(seen) != set(master_datasets) or len(seen) != len(master_datasets):
        raise ValueError("K-shot shards must partition master datasets exactly")
    return expanded


def expected_kshot_keys(configs: Sequence[dict[str, Any]]) -> set[KShotRunKey]:
    keys: set[KShotRunKey] = set()
    for config in configs:
        for dataset in config["datasets"]:
            for fold in config["folds"]:
                for shots in configured_shots(config):
                    if shots is None:
                        raise ValueError("K-shot formal config has no K-shot scope")
                    for selector in config["feature_selection"]["selectors"]:
                        for model in config["models"]:
                            key = (str(dataset), int(fold), shots, str(selector), str(model))
                            if key in keys:
                                raise ValueError(f"K-shot configurations overlap at {key}")
                            keys.add(key)
    return keys


def kshot_row_key(row: dict[str, Any]) -> KShotRunKey:
    return (
        str(row["dataset_id"]), int(row["fold"]), int(row["shots_per_class"]),
        str(row["selector"]), str(row["model"]),
    )


def validate_kshot_rows(
    rows: Sequence[dict[str, Any]], configs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected = expected_kshot_keys(configs)
    keys = [kshot_row_key(row) for row in rows]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    actual = set(keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    split_conflicts: list[tuple[Any, ...]] = []
    # Test folds must remain identical for every K, selector, and model.
    tests: dict[tuple[str, int], set[str]] = defaultdict(set)
    # Support/development scopes must be identical across selectors and models.
    scoped: dict[tuple[str, int, int], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    vocabularies: dict[tuple[str, int, int, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        dataset, fold, shots, selector, _ = kshot_row_key(row)
        tests[(dataset, fold)].add(str(row["test_indices_sha256"]))
        for field in (
            "inner_train_indices_sha256", "validation_indices_sha256",
            "outer_train_indices_sha256",
        ):
            scoped[(dataset, fold, shots)][field].add(str(row[field]))
        for field in ("development_vocabulary_sha256", "final_vocabulary_sha256"):
            vocabularies[(dataset, fold, shots, selector)][field].add(str(row[field]))
    split_conflicts.extend(
        (dataset, fold, "test_indices_sha256")
        for (dataset, fold), values in tests.items() if len(values) != 1
    )
    split_conflicts.extend(
        (*scope, field)
        for scope, fields in scoped.items() for field, values in fields.items()
        if len(values) != 1
    )
    vocabulary_conflicts = [
        (*scope, field)
        for scope, fields in vocabularies.items() for field, values in fields.items()
        if len(values) != 1
    ]
    invalid_sizes = [
        list(kshot_row_key(row))
        for row in rows
        if int(row["final_train_size"])
        != int(row["shots_per_class"]) * int(row["class_count"])
    ]
    status = "passed" if not (
        duplicates or missing or extra or split_conflicts
        or vocabulary_conflicts or invalid_sizes
    ) else "failed"
    return {
        "status": status,
        "expected_runs": len(expected),
        "actual_rows": len(rows),
        "unique_runs": len(actual),
        "duplicate_keys": [list(value) for value in duplicates],
        "missing_keys": [list(value) for value in missing],
        "extra_keys": [list(value) for value in extra],
        "split_hash_conflicts": [list(value) for value in split_conflicts],
        "vocabulary_hash_conflicts": [list(value) for value in vocabulary_conflicts],
        "invalid_final_training_sizes": invalid_sizes,
    }


def aggregate_kshot_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["dataset_id"]), int(row["shots_per_class"]),
            str(row["model"]), str(row["selector"]),
        )].append(row)
    metrics = (
        "test_accuracy", "test_macro_f1", "test_balanced_accuracy",
        "validation_accuracy", "best_epoch",
        "development_feature_selection_seconds", "final_feature_selection_seconds",
        "development_training_seconds", "final_training_seconds", "test_inference_seconds",
    )
    output: list[dict[str, Any]] = []
    for (dataset, shots, model, selector), group in sorted(grouped.items()):
        result: dict[str, Any] = {
            "dataset_id": dataset, "shots_per_class": shots,
            "model": model, "selector": selector, "folds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def paired_kshot_effects(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {kshot_row_key(row): row for row in rows}
    groups = sorted({(str(row["dataset_id"]), int(row["shots_per_class"]), str(row["model"])) for row in rows})
    selectors = sorted({str(row["selector"]) for row in rows if row["selector"] != "none"})
    folds_by_dataset = {
        dataset: sorted({int(row["fold"]) for row in rows if str(row["dataset_id"]) == dataset})
        for dataset in {str(row["dataset_id"]) for row in rows}
    }
    output: list[dict[str, Any]] = []
    for dataset, shots, model in groups:
        folds = folds_by_dataset[dataset]
        for selector in selectors:
            deltas = [
                float(lookup[(dataset, fold, shots, selector, model)]["test_accuracy"])
                - float(lookup[(dataset, fold, shots, "none", model)]["test_accuracy"])
                for fold in folds
            ]
            output.append({
                "dataset_id": dataset, "shots_per_class": shots,
                "model": model, "selector": selector, "folds": len(deltas),
                "accuracy_delta_mean": mean(deltas),
                "accuracy_delta_std": stdev(deltas) if len(deltas) > 1 else 0.0,
                "wins": sum(value > 1e-12 for value in deltas),
                "ties": sum(abs(value) <= 1e-12 for value in deltas),
                "losses": sum(value < -1e-12 for value in deltas),
            })
    return output


def kshot_trend_summary(effects: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        grouped[(int(row["shots_per_class"]), str(row["selector"]))].append(row)
    output: list[dict[str, Any]] = []
    for (shots, selector), group in sorted(grouped.items()):
        deltas = [float(row["accuracy_delta_mean"]) for row in group]
        output.append({
            "shots_per_class": shots,
            "selector": selector,
            "dataset_model_cells": len(group),
            "mean_accuracy_delta_vs_none": mean(deltas),
            "cell_wins": sum(value > 0 for value in deltas),
            "cell_ties": sum(value == 0 for value in deltas),
            "cell_losses": sum(value < 0 for value in deltas),
        })
    return output
