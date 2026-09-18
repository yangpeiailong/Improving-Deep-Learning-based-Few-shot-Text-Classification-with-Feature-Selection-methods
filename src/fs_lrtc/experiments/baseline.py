"""Leakage-safe TF-IDF logistic-regression sanity baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from fs_lrtc.features.pipeline import fit_selector_at_scope
from fs_lrtc.features.selectors import FilterFeatureSelector, RandomFeatureSelector


@dataclass(frozen=True)
class CandidateResult:
    c_value: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    training_seconds: float
    inference_seconds: float


def validate_partition(split: dict[str, Any], record_count: int) -> None:
    """Reject altered or ambiguous split files at experiment runtime."""
    required = ("inner_train", "validation", "outer_train", "test")
    for key in required:
        if key not in split or not isinstance(split[key], list):
            raise ValueError(f"Split is missing list `{key}`")
        if len(split[key]) != len(set(split[key])):
            raise ValueError(f"Split `{key}` contains duplicate indices")
        if any(not isinstance(index, int) or index < 0 or index >= record_count for index in split[key]):
            raise IndexError(f"Split `{key}` contains an out-of-range index")
    inner, validation = set(split["inner_train"]), set(split["validation"])
    outer, test = set(split["outer_train"]), set(split["test"])
    if inner & validation or outer & test:
        raise ValueError("Training and held-out indices overlap")
    if inner | validation != outer:
        raise ValueError("inner_train + validation must equal outer_train")
    if outer | test != set(range(record_count)):
        raise ValueError("outer_train + test must partition the processed dataset")


def subset(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def sequence_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def vocabulary_at_scope(
    method: str,
    tokenized_documents: Sequence[Sequence[str]],
    labels: Sequence[str],
    fit_indices: Sequence[int],
    forbidden_indices: Sequence[int],
    feature_count: int,
    minimum_document_frequency: int,
    random_selection_seed: int = 42,
) -> tuple[list[str], FilterFeatureSelector | RandomFeatureSelector | None]:
    """Create a vocabulary from fit records only, including the no-FS baseline."""
    fit_set, forbidden_set = set(fit_indices), set(forbidden_indices)
    if fit_set & forbidden_set:
        raise ValueError("Held-out records entered vocabulary/selector fit")
    if method == "none":
        vocabulary = sorted(
            {token for index in fit_indices for token in tokenized_documents[index]}
        )
        if not vocabulary:
            raise ValueError("No tokens are available in the training scope")
        return vocabulary, None
    if method == "random":
        selector = RandomFeatureSelector(
            seed=random_selection_seed,
            minimum_document_frequency=minimum_document_frequency,
        ).fit(subset(tokenized_documents, fit_indices))
        vocabulary = [item.token for item in selector.top(feature_count)]
        if not vocabulary:
            raise ValueError("Random control produced an empty vocabulary")
        return vocabulary, selector
    selector = fit_selector_at_scope(
        method,
        tokenized_documents,
        labels,
        fit_indices,
        forbidden_indices,
        minimum_document_frequency,
    )
    vocabulary = [item.token for item in selector.top(feature_count)]
    if not vocabulary:
        raise ValueError(f"Selector {method} produced an empty vocabulary")
    return vocabulary, selector


def build_vectorizer(vocabulary: Sequence[str], config: dict[str, Any]) -> TfidfVectorizer:
    mapping = {token: index for index, token in enumerate(vocabulary)}
    return TfidfVectorizer(
        analyzer=lambda document: document,
        vocabulary=mapping,
        lowercase=False,
        token_pattern=None,
        use_idf=bool(config.get("use_idf", True)),
        smooth_idf=bool(config.get("smooth_idf", True)),
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        norm=config.get("norm", "l2"),
    )


def make_classifier(c_value: float, config: dict[str, Any], seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        solver=str(config.get("solver", "lbfgs")),
        max_iter=int(config.get("max_iter", 2000)),
        tol=float(config.get("tolerance", 1e-4)),
        random_state=seed,
    )


def metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def select_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    """Select on validation metrics only; prefer smaller C for exact ties."""
    if not candidates:
        raise ValueError("No validation candidates were evaluated")
    return max(
        candidates,
        key=lambda item: (item.accuracy, item.macro_f1, item.balanced_accuracy, -item.c_value),
    )


def development_search(
    tokenized_documents: Sequence[Sequence[str]],
    labels: Sequence[str],
    split: dict[str, Any],
    method: str,
    feature_count: int,
    minimum_document_frequency: int,
    representation_config: dict[str, Any],
    classifier_config: dict[str, Any],
    seed: int,
) -> tuple[
    CandidateResult,
    list[CandidateResult],
    list[str],
    FilterFeatureSelector | RandomFeatureSelector | None,
    float,
]:
    fit_indices = split["inner_train"]
    forbidden = sorted(split["validation"] + split["test"])
    selection_start = time.perf_counter()
    vocabulary, selector = vocabulary_at_scope(
        method, tokenized_documents, labels, fit_indices, forbidden,
        feature_count, minimum_document_frequency,
    )
    selection_seconds = time.perf_counter() - selection_start
    vectorizer = build_vectorizer(vocabulary, representation_config)
    x_train = vectorizer.fit_transform(subset(tokenized_documents, fit_indices))
    x_validation = vectorizer.transform(subset(tokenized_documents, split["validation"]))
    y_train = subset(labels, fit_indices)
    y_validation = subset(labels, split["validation"])

    candidates: list[CandidateResult] = []
    for c_value in sorted(float(value) for value in classifier_config["c_values"]):
        classifier = make_classifier(c_value, classifier_config, seed)
        start = time.perf_counter()
        classifier.fit(x_train, y_train)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = classifier.predict(x_validation)
        inference_seconds = time.perf_counter() - start
        score = metrics(y_validation, prediction)
        candidates.append(
            CandidateResult(
                c_value=c_value,
                training_seconds=training_seconds,
                inference_seconds=inference_seconds,
                **score,
            )
        )
    return select_candidate(candidates), candidates, vocabulary, selector, selection_seconds


def final_evaluation(
    tokenized_documents: Sequence[Sequence[str]],
    labels: Sequence[str],
    split: dict[str, Any],
    method: str,
    feature_count: int,
    minimum_document_frequency: int,
    representation_config: dict[str, Any],
    classifier_config: dict[str, Any],
    selected_c: float,
    seed: int,
) -> tuple[
    dict[str, float],
    list[str],
    list[str],
    list[str],
    FilterFeatureSelector | RandomFeatureSelector | None,
    dict[str, float],
]:
    fit_indices, test_indices = split["outer_train"], split["test"]
    selection_start = time.perf_counter()
    vocabulary, selector = vocabulary_at_scope(
        method, tokenized_documents, labels, fit_indices, test_indices,
        feature_count, minimum_document_frequency,
    )
    selection_seconds = time.perf_counter() - selection_start
    vectorization_start = time.perf_counter()
    vectorizer = build_vectorizer(vocabulary, representation_config)
    x_train = vectorizer.fit_transform(subset(tokenized_documents, fit_indices))
    x_test = vectorizer.transform(subset(tokenized_documents, test_indices))
    vectorization_seconds = time.perf_counter() - vectorization_start
    classifier = make_classifier(selected_c, classifier_config, seed)
    start = time.perf_counter()
    classifier.fit(x_train, subset(labels, fit_indices))
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = classifier.predict(x_test).tolist()
    inference_seconds = time.perf_counter() - start
    truth = subset(labels, test_indices)
    timing = {
        "feature_selection_seconds": selection_seconds,
        "vectorization_seconds": vectorization_seconds,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": selection_seconds + vectorization_seconds + training_seconds + inference_seconds,
    }
    return metrics(truth, prediction), truth, prediction, vocabulary, selector, timing


def write_selected_features(
    path: Path,
    vocabulary: Sequence[str],
    selector: FilterFeatureSelector | RandomFeatureSelector | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    score_by_token = (
        {item.token: item for item in selector.ranked_features_} if selector else {}
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("rank", "token", "score", "document_frequency", "strongest_class")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, token in enumerate(vocabulary, start=1):
            item = score_by_token.get(token)
            writer.writerow(
                {
                    "rank": rank,
                    "token": token,
                    "score": "" if item is None else format(item.score, ".17g"),
                    "document_frequency": "" if item is None else item.document_frequency,
                    "strongest_class": "" if item is None else item.strongest_class,
                }
            )


def write_predictions(
    path: Path,
    indices: Sequence[int],
    truth: Sequence[str],
    prediction: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("processed_index", "true_label", "predicted_label", "correct")
        )
        writer.writeheader()
        for index, actual, predicted in zip(indices, truth, prediction, strict=True):
            writer.writerow(
                {
                    "processed_index": index,
                    "true_label": actual,
                    "predicted_label": predicted,
                    "correct": int(actual == predicted),
                }
            )


def result_to_dict(candidate: CandidateResult) -> dict[str, float]:
    return {
        "c_value": candidate.c_value,
        "accuracy": candidate.accuracy,
        "macro_f1": candidate.macro_f1,
        "balanced_accuracy": candidate.balanced_accuracy,
        "training_seconds": candidate.training_seconds,
        "inference_seconds": candidate.inference_seconds,
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
