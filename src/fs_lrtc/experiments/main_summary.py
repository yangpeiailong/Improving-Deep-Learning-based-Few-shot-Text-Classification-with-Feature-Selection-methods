"""Cross-family validation and summaries for the fixed-1000 main experiment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable, Sequence

from scipy.stats import rankdata, wilcoxon

from fs_lrtc.experiments.formal import aggregate_results, paired_effects, validate_formal_rows


FAMILY_MODELS: dict[str, frozenset[str]] = {
    "classic": frozenset(
        {
            "random_wa",
            "fasttext_wa",
            "random_cnn",
            "fasttext_cnn",
            "random_lstm",
            "fasttext_lstm",
        }
    ),
    "gcn": frozenset({"fasttext_residual_word_gcn"}),
    "bert": frozenset({"bert_cls", "bert_mean", "bert_cnn", "bert_bilstm_attention"}),
}

MODEL_FAMILY = {
    model: family for family, models in FAMILY_MODELS.items() for model in models
}


def family_for_model(model: str) -> str:
    """Return the frozen model family for a main-experiment model."""
    try:
        return MODEL_FAMILY[model]
    except KeyError as exc:
        raise ValueError(f"Unknown main-experiment model: {model}") from exc


def add_family(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy rows and add a stable family field."""
    return [{"family": family_for_model(str(row["model"])), **row} for row in rows]


def validate_family_rows(
    rows: Sequence[dict[str, Any]], configs: Sequence[dict[str, Any]], family: str
) -> dict[str, Any]:
    """Validate one complete family and its fixed feature-count protocol."""
    report = validate_formal_rows(rows, configs)
    observed_models = {str(row["model"]) for row in rows}
    model_mismatch = observed_models != set(FAMILY_MODELS[family])
    feature_count_errors: list[list[Any]] = []
    for row in rows:
        selector = str(row["selector"])
        requested = str(row.get("feature_count_requested", ""))
        final_count = int(float(row["final_feature_count"]))
        if selector == "none":
            invalid = requested.lower() not in {"", "all", "none"}
        else:
            invalid = requested != "1000" or final_count != 1000
        if invalid:
            feature_count_errors.append(
                [str(row["dataset_id"]), int(row["fold"]), selector, str(row["model"])]
            )
    report["family"] = family
    report["observed_models"] = sorted(observed_models)
    report["expected_models"] = sorted(FAMILY_MODELS[family])
    report["model_set_mismatch"] = model_mismatch
    report["feature_count_errors"] = feature_count_errors
    if model_mismatch or feature_count_errors:
        report["status"] = "failed"
    return report


def validate_cross_family_rows(
    rows: Sequence[dict[str, Any]], configs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Validate completeness and provenance consistency across all families."""
    report = validate_formal_rows(rows, configs)
    expected_models = set(MODEL_FAMILY)
    observed_models = {str(row["model"]) for row in rows}
    report.update(
        {
            "datasets": sorted({str(row["dataset_id"]) for row in rows}),
            "models": sorted(observed_models),
            "selectors": sorted({str(row["selector"]) for row in rows}),
            "folds": sorted({int(row["fold"]) for row in rows}),
            "expected_dataset_count": 16,
            "expected_model_count": 11,
            "expected_selector_count": 4,
            "expected_fold_count": 5,
            "model_set_mismatch": observed_models != expected_models,
        }
    )
    if (
        len(report["datasets"]) != 16
        or observed_models != expected_models
        or report["selectors"] != ["df", "dfs", "ig", "none"]
        or report["folds"] != [0, 1, 2, 3, 4]
    ):
        report["status"] = "failed"
    return report


def aggregate_with_family(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the five folds and add model-family labels."""
    return [
        {"family": family_for_model(str(row["model"])), **row}
        for row in aggregate_results(rows)
    ]


def effects_with_family(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate fold-paired effects against no feature selection."""
    return [
        {"family": family_for_model(str(row["model"])), **row}
        for row in paired_effects(rows)
    ]


def _sample_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def selector_summary(
    aggregate: Sequence[dict[str, Any]], effects: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize selectors over all 176 dataset-model cells."""
    accuracy: dict[str, list[float]] = defaultdict(list)
    delta: dict[str, list[float]] = defaultdict(list)
    for row in aggregate:
        accuracy[str(row["selector"])].append(float(row["test_accuracy_mean"]))
    for row in effects:
        delta[str(row["selector"])].append(float(row["accuracy_delta_mean"]))
    output: list[dict[str, Any]] = []
    for selector in sorted(accuracy, key=lambda value: (value != "none", value)):
        values = accuracy[selector]
        differences = delta.get(selector, [])
        output.append(
            {
                "selector": selector,
                "dataset_model_cells": len(values),
                "mean_test_accuracy": mean(values),
                "mean_accuracy_delta_vs_none": mean(differences) if differences else 0.0,
                "median_accuracy_delta_vs_none": median(differences) if differences else 0.0,
                "cell_wins": sum(value > 1e-12 for value in differences),
                "cell_ties": sum(abs(value) <= 1e-12 for value in differences),
                "cell_losses": sum(value < -1e-12 for value in differences),
            }
        )
    return output


def family_summary(
    aggregate: Sequence[dict[str, Any]], effects: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize accuracy and effect separately for classic, GCN, and BERT."""
    accuracy: dict[tuple[str, str], list[float]] = defaultdict(list)
    delta: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in aggregate:
        accuracy[(str(row["family"]), str(row["selector"]))].append(
            float(row["test_accuracy_mean"])
        )
    for row in effects:
        delta[(str(row["family"]), str(row["selector"]))].append(
            float(row["accuracy_delta_mean"])
        )
    output: list[dict[str, Any]] = []
    for family, selector in sorted(accuracy, key=lambda key: (key[0], key[1] != "none", key[1])):
        values = accuracy[(family, selector)]
        differences = delta.get((family, selector), [])
        output.append(
            {
                "family": family,
                "selector": selector,
                "dataset_model_cells": len(values),
                "mean_test_accuracy": mean(values),
                "mean_accuracy_delta_vs_none": mean(differences) if differences else 0.0,
                "median_accuracy_delta_vs_none": median(differences) if differences else 0.0,
                "cell_wins": sum(value > 1e-12 for value in differences),
                "cell_ties": sum(abs(value) <= 1e-12 for value in differences),
                "cell_losses": sum(value < -1e-12 for value in differences),
            }
        )
    return output


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _rank_biserial(differences: Sequence[float]) -> float:
    nonzero = [value for value in differences if abs(value) > 1e-12]
    if not nonzero:
        return 0.0
    ranks = rankdata([abs(value) for value in nonzero], method="average")
    positive = sum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero, strict=True) if value < 0)
    return float((positive - negative) / (positive + negative))


def significance_tests(effects: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run paired Wilcoxon tests over dataset-model mean effects."""
    scopes: dict[str, list[dict[str, Any]]] = {"overall": list(effects)}
    for family in sorted(FAMILY_MODELS):
        scopes[family] = [row for row in effects if row["family"] == family]
    output: list[dict[str, Any]] = []
    for scope, rows in scopes.items():
        scope_rows: list[dict[str, Any]] = []
        for selector in ("df", "ig", "dfs"):
            differences = [
                float(row["accuracy_delta_mean"])
                for row in rows
                if row["selector"] == selector
            ]
            if not differences:
                continue
            if all(abs(value) <= 1e-12 for value in differences):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            scope_rows.append(
                {
                    "scope": scope,
                    "selector": selector,
                    "paired_dataset_model_cells": len(differences),
                    "mean_accuracy_delta": mean(differences),
                    "median_accuracy_delta": median(differences),
                    "wilcoxon_statistic": statistic,
                    "p_value_raw": p_value,
                    "rank_biserial_correlation": _rank_biserial(differences),
                }
            )
        adjusted = _holm_adjust([float(row["p_value_raw"]) for row in scope_rows])
        for row, p_value in zip(scope_rows, adjusted, strict=True):
            row["p_value_holm"] = p_value
            row["significant_at_0_05"] = p_value < 0.05
            output.append(row)
    return output


def runtime_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize final-stage runtime components without counting development tuning."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(family_for_model(str(row["model"])), str(row["selector"]))].append(row)
    component_fields = {
        "feature_selection_seconds": "final_feature_selection_seconds",
        "graph_construction_seconds": "final_graph_seconds",
        "embedding_matrix_seconds": "embedding_matrix_seconds",
        "training_seconds": "final_training_seconds",
        "test_inference_seconds": "test_inference_seconds",
    }
    interim: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        result: dict[str, Any] = {"family": key[0], "selector": key[1], "fold_runs": len(group)}
        totals: list[float] = []
        for label, field in component_fields.items():
            values = [float(row.get(field) or 0.0) for row in group]
            result[f"mean_{label}"] = mean(values)
        for row in group:
            totals.append(sum(float(row.get(field) or 0.0) for field in component_fields.values()))
        result["mean_final_pipeline_seconds"] = mean(totals)
        result["std_final_pipeline_seconds"] = _sample_std(totals)
        peak_values = [float(row.get("peak_gpu_memory_mb") or 0.0) for row in group]
        result["mean_peak_gpu_memory_mb"] = mean(peak_values)
        interim[key] = result
    output: list[dict[str, Any]] = []
    for key in sorted(interim, key=lambda item: (item[0], item[1] != "none", item[1])):
        row = interim[key]
        baseline = float(interim[(key[0], "none")]["mean_final_pipeline_seconds"])
        total = float(row["mean_final_pipeline_seconds"])
        row["relative_final_pipeline_time_vs_none"] = total / baseline if baseline else 0.0
        output.append(row)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def union_fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return deterministic field order for heterogeneous family result rows."""
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields
