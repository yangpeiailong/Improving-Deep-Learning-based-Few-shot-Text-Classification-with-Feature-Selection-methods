"""Planning and summaries for the retained-feature-count sensitivity study."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


SensitivityKey = tuple[str, str, int, str, str, str]


def build_sensitivity_configs(
    sensitivity: Mapping[str, Any],
    base_configs: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Create one resumable runner config per model family and new feature count."""
    datasets = [str(value) for value in sensitivity["datasets"]]
    folds = [int(value) for value in sensitivity["folds"]]
    selectors = [str(value) for value in sensitivity["selectors"]]
    new_counts = [int(value) for value in sensitivity["new_feature_counts"]]
    analysis_counts = [int(value) for value in sensitivity["analysis_feature_counts"]]
    if len(datasets) != len(set(datasets)) or len(folds) != len(set(folds)):
        raise ValueError("Sensitivity datasets and folds must be unique")
    if selectors != ["df", "ig", "dfs"]:
        raise ValueError("Sensitivity selectors must be exactly [df, ig, dfs]")
    if 1000 not in analysis_counts or 1000 in new_counts:
        raise ValueError("The existing 1000-feature results must be reused, not rerun")
    if set(new_counts) != set(analysis_counts) - {1000}:
        raise ValueError("new_feature_counts must contain all non-1000 analysis counts")

    output = sensitivity["output"]
    base_directory = str(output["base_directory"]).strip("/\\")
    families = sensitivity["families"]
    expanded: dict[tuple[str, int], dict[str, Any]] = {}
    for family, family_spec in families.items():
        family = str(family)
        if family not in base_configs:
            raise ValueError(f"Missing base configuration for family: {family}")
        requested_models = [str(value) for value in family_spec["models"]]
        base = base_configs[family]
        unknown = sorted(set(requested_models) - {str(value) for value in base["models"]})
        if unknown:
            raise ValueError(f"Unknown {family} sensitivity models: {unknown}")
        for count in new_counts:
            config = copy.deepcopy(dict(base))
            config.pop("formal_orchestration", None)
            config["experiment_id"] = (
                f"{sensitivity['experiment_id']}_{family}_k{count}"
            )
            config["datasets"] = datasets
            config["folds"] = folds
            config["models"] = requested_models
            config["feature_selection"]["selectors"] = selectors
            config["feature_selection"]["feature_count"] = count
            config["output"].update(
                {
                    "directory": f"{base_directory}/{family}/k_{count}",
                    "save_predictions": bool(output["save_predictions"]),
                    "save_selected_features": bool(output["save_selected_features"]),
                    "save_training_history": bool(output["save_training_history"]),
                    "save_model_checkpoints": bool(output["save_model_checkpoints"]),
                }
            )
            expanded[(family, count)] = config
    return expanded


def expected_unit_runs(config: Mapping[str, Any]) -> int:
    return (
        len(config["datasets"])
        * len(config["folds"])
        * len(config["feature_selection"]["selectors"])
        * len(config["models"])
    )


def standardized_row(row: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Keep common, paper-relevant fields across classic, GCN, and BERT runs."""
    count = str(row["feature_count_requested"])
    return {
        "family": family,
        "dataset_id": str(row["dataset_id"]),
        "fold": int(row["fold"]),
        "selector": str(row["selector"]),
        "feature_count": count,
        "model": str(row["model"]),
        "test_accuracy": float(row["test_accuracy"]),
        "test_macro_f1": float(row["test_macro_f1"]),
        "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
        "best_epoch": float(row["best_epoch"]),
        "development_feature_selection_seconds": float(
            row["development_feature_selection_seconds"]
        ),
        "final_feature_selection_seconds": float(row["final_feature_selection_seconds"]),
        "development_training_seconds": float(row["development_training_seconds"]),
        "final_training_seconds": float(row["final_training_seconds"]),
        "test_inference_seconds": float(row["test_inference_seconds"]),
        "inner_train_indices_sha256": str(row["inner_train_indices_sha256"]),
        "validation_indices_sha256": str(row["validation_indices_sha256"]),
        "outer_train_indices_sha256": str(row["outer_train_indices_sha256"]),
        "test_indices_sha256": str(row["test_indices_sha256"]),
        "development_vocabulary_sha256": str(row["development_vocabulary_sha256"]),
        "final_vocabulary_sha256": str(row["final_vocabulary_sha256"]),
    }


def sensitivity_key(row: Mapping[str, Any]) -> SensitivityKey:
    return (
        str(row["family"]),
        str(row["dataset_id"]),
        int(row["fold"]),
        str(row["selector"]),
        str(row["feature_count"]),
        str(row["model"]),
    )


def expected_sensitivity_keys(sensitivity: Mapping[str, Any]) -> set[SensitivityKey]:
    keys: set[SensitivityKey] = set()
    for family, spec in sensitivity["families"].items():
        for dataset in sensitivity["datasets"]:
            for fold in sensitivity["folds"]:
                for model in spec["models"]:
                    keys.add((str(family), str(dataset), int(fold), "none", "all", str(model)))
                    for selector in sensitivity["selectors"]:
                        for count in sensitivity["analysis_feature_counts"]:
                            keys.add(
                                (
                                    str(family), str(dataset), int(fold), str(selector),
                                    str(int(count)), str(model),
                                )
                            )
    return keys


def validate_sensitivity_rows(
    rows: Sequence[Mapping[str, Any]], sensitivity: Mapping[str, Any]
) -> dict[str, Any]:
    expected = expected_sensitivity_keys(sensitivity)
    actual_keys = [sensitivity_key(row) for row in rows]
    counts = Counter(actual_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    actual = set(actual_keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    split_hash_conflicts = []
    for field in (
        "inner_train_indices_sha256", "validation_indices_sha256",
        "outer_train_indices_sha256", "test_indices_sha256",
    ):
        grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in rows:
            grouped[(str(row["dataset_id"]), int(row["fold"]))].add(str(row[field]))
        split_hash_conflicts.extend(
            (dataset, fold, field)
            for (dataset, fold), values in grouped.items() if len(values) != 1
        )
    vocabulary_hash_conflicts = []
    for field in ("development_vocabulary_sha256", "final_vocabulary_sha256"):
        grouped_vocab: dict[tuple[str, int, str, str], set[str]] = defaultdict(set)
        for row in rows:
            grouped_vocab[
                (
                    str(row["dataset_id"]), int(row["fold"]),
                    str(row["selector"]), str(row["feature_count"]),
                )
            ].add(str(row[field]))
        vocabulary_hash_conflicts.extend(
            (dataset, fold, selector, count, field)
            for (dataset, fold, selector, count), values in grouped_vocab.items()
            if len(values) != 1
        )
    return {
        "status": "passed" if not (
            duplicates or missing or extra or split_hash_conflicts or vocabulary_hash_conflicts
        ) else "failed",
        "expected_runs": len(expected),
        "actual_rows": len(rows),
        "unique_runs": len(actual),
        "duplicate_keys": [list(value) for value in duplicates],
        "missing_keys": [list(value) for value in missing],
        "extra_keys": [list(value) for value in extra],
        "split_hash_conflicts": [list(value) for value in split_hash_conflicts],
        "vocabulary_hash_conflicts": [list(value) for value in vocabulary_hash_conflicts],
    }


def _std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate_sensitivity(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["family"]), str(row["dataset_id"]), str(row["model"]),
                str(row["selector"]), str(row["feature_count"]),
            )
        ].append(row)
    metrics = ("test_accuracy", "test_macro_f1", "test_balanced_accuracy", "best_epoch")
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        family, dataset, model, selector, count = key
        record: dict[str, Any] = {
            "family": family, "dataset_id": dataset, "model": model,
            "selector": selector, "feature_count": count, "folds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = _std(values)
        output.append(record)
    return output


def paired_sensitivity_effects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {sensitivity_key(row): row for row in rows}
    output: list[dict[str, Any]] = []
    cells = sorted(
        {
            (str(row["family"]), str(row["dataset_id"]), str(row["model"]))
            for row in rows
        }
    )
    folds = sorted({int(row["fold"]) for row in rows})
    selectors = sorted({str(row["selector"]) for row in rows if row["selector"] != "none"})
    feature_counts = sorted(
        {int(row["feature_count"]) for row in rows if row["selector"] != "none"}
    )
    for family, dataset, model in cells:
        for selector in selectors:
            for count in feature_counts:
                deltas = []
                for fold in folds:
                    selected = lookup[(family, dataset, fold, selector, str(count), model)]
                    baseline = lookup[(family, dataset, fold, "none", "all", model)]
                    deltas.append(float(selected["test_accuracy"]) - float(baseline["test_accuracy"]))
                output.append(
                    {
                        "family": family, "dataset_id": dataset, "model": model,
                        "selector": selector, "feature_count": count, "folds": len(deltas),
                        "accuracy_delta_mean": mean(deltas),
                        "accuracy_delta_std": _std(deltas),
                        "wins": sum(value > 1e-12 for value in deltas),
                        "ties": sum(abs(value) <= 1e-12 for value in deltas),
                        "losses": sum(value < -1e-12 for value in deltas),
                    }
                )
    return output


def trend_summary(
    aggregate: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    accuracy: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    deltas: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in aggregate:
        if row["selector"] != "none":
            key = (str(row["family"]), str(row["selector"]), int(row["feature_count"]))
            accuracy[key].append(float(row["test_accuracy_mean"]))
    for row in effects:
        key = (str(row["family"]), str(row["selector"]), int(row["feature_count"]))
        deltas[key].append(float(row["accuracy_delta_mean"]))
    output = []
    for key in sorted(accuracy):
        family, selector, count = key
        values = deltas[key]
        output.append(
            {
                "family": family, "selector": selector, "feature_count": count,
                "dataset_model_cells": len(accuracy[key]),
                "mean_cell_accuracy": mean(accuracy[key]),
                "mean_accuracy_delta_vs_none": mean(values),
                "cell_wins": sum(value > 1e-12 for value in values),
                "cell_ties": sum(abs(value) <= 1e-12 for value in values),
                "cell_losses": sum(value < -1e-12 for value in values),
            }
        )
    return output
