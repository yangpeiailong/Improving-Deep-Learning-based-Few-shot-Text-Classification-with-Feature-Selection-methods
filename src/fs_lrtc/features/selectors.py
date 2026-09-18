"""DF, information gain, and distinguishing feature selector (DFS)."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FeatureScore:
    token: str
    score: float
    document_frequency: int
    strongest_class: str
    strongest_class_document_frequency: int


class FilterFeatureSelector:
    """Global filter selector using binary term presence per document."""

    METHODS = {"df", "ig", "dfs"}

    def __init__(self, method: str, minimum_document_frequency: int = 1):
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(f"Unsupported selector: {method}")
        if minimum_document_frequency < 1:
            raise ValueError("minimum_document_frequency must be at least 1")
        self.method = method
        self.minimum_document_frequency = minimum_document_frequency
        self._ranked: list[FeatureScore] | None = None
        self.n_documents_: int | None = None
        self.class_counts_: Counter[str] | None = None

    def fit(
        self, documents: Sequence[Sequence[str]], labels: Sequence[str] | None = None
    ) -> "FilterFeatureSelector":
        if not documents:
            raise ValueError("Cannot fit a feature selector on zero documents")
        if labels is None:
            if self.method != "df":
                raise ValueError(f"Labels are required for supervised selector {self.method}")
            labels = ["__unlabeled__"] * len(documents)
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have equal length")

        class_counts: Counter[str] = Counter(labels)
        if self.method in {"ig", "dfs"} and len(class_counts) < 2:
            raise ValueError(f"Selector {self.method} requires at least two training classes")
        document_frequency: Counter[str] = Counter()
        class_document_frequency: dict[str, Counter[str]] = defaultdict(Counter)
        for document, label in zip(documents, labels, strict=True):
            present = set(document)
            document_frequency.update(present)
            class_document_frequency[label].update(present)

        n_documents = len(documents)
        classes = sorted(class_counts)
        ranked: list[FeatureScore] = []
        for token, total_df in document_frequency.items():
            if total_df < self.minimum_document_frequency:
                continue
            per_class = {label: class_document_frequency[label][token] for label in classes}
            if self.method == "df":
                score = float(total_df)
            elif self.method == "ig":
                score = _information_gain(total_df, per_class, class_counts, n_documents)
            else:
                score = _dfs(total_df, per_class, class_counts, n_documents)
            strongest_class = min(
                classes,
                key=lambda label: (-per_class[label] / class_counts[label], label),
            )
            ranked.append(
                FeatureScore(
                    token=token,
                    score=score,
                    document_frequency=total_df,
                    strongest_class=strongest_class,
                    strongest_class_document_frequency=per_class[strongest_class],
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.token))
        self._ranked = ranked
        self.n_documents_ = n_documents
        self.class_counts_ = class_counts
        return self

    @property
    def ranked_features_(self) -> list[FeatureScore]:
        if self._ranked is None:
            raise RuntimeError("Selector has not been fitted")
        return list(self._ranked)

    def top(self, feature_count: int) -> list[FeatureScore]:
        if feature_count < 1:
            raise ValueError("feature_count must be at least 1")
        return self.ranked_features_[:feature_count]

    def transform(
        self, documents: Sequence[Sequence[str]], feature_count: int
    ) -> list[list[str]]:
        selected = {item.token for item in self.top(feature_count)}
        return [[token for token in document if token in selected] for document in documents]


class RandomFeatureSelector:
    """Deterministic random-vocabulary control fitted on training documents only."""

    def __init__(self, seed: int, minimum_document_frequency: int = 1):
        if minimum_document_frequency < 1:
            raise ValueError("minimum_document_frequency must be at least 1")
        self.seed = int(seed)
        self.minimum_document_frequency = minimum_document_frequency
        self._ranked: list[FeatureScore] | None = None
        self.n_documents_: int | None = None

    def fit(
        self, documents: Sequence[Sequence[str]], labels: Sequence[str] | None = None
    ) -> "RandomFeatureSelector":
        if not documents:
            raise ValueError("Cannot fit a feature selector on zero documents")
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        eligible = [
            token
            for token, frequency in document_frequency.items()
            if frequency >= self.minimum_document_frequency
        ]

        def random_key(token: str) -> tuple[bytes, str]:
            payload = f"{self.seed}\0{token}".encode("utf-8")
            return hashlib.sha256(payload).digest(), token

        eligible.sort(key=random_key)
        self._ranked = [
            FeatureScore(
                token=token,
                score=float(len(eligible) - rank),
                document_frequency=document_frequency[token],
                strongest_class="__random_control__",
                strongest_class_document_frequency=0,
            )
            for rank, token in enumerate(eligible)
        ]
        self.n_documents_ = len(documents)
        return self

    @property
    def ranked_features_(self) -> list[FeatureScore]:
        if self._ranked is None:
            raise RuntimeError("Selector has not been fitted")
        return list(self._ranked)

    def top(self, feature_count: int) -> list[FeatureScore]:
        if feature_count < 1:
            raise ValueError("feature_count must be at least 1")
        return self.ranked_features_[:feature_count]

    def transform(
        self, documents: Sequence[Sequence[str]], feature_count: int
    ) -> list[list[str]]:
        selected = {item.token for item in self.top(feature_count)}
        return [[token for token in document if token in selected] for document in documents]


def _entropy(counts: Iterable[int], total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count:
            probability = count / total
            value -= probability * math.log2(probability)
    return value


def _information_gain(
    total_df: int,
    per_class: dict[str, int],
    class_counts: Counter[str],
    n_documents: int,
) -> float:
    class_entropy = _entropy(class_counts.values(), n_documents)
    present_entropy = _entropy(per_class.values(), total_df)
    absent_total = n_documents - total_df
    absent_entropy = _entropy(
        (class_counts[label] - per_class[label] for label in class_counts), absent_total
    )
    value = class_entropy - (total_df / n_documents) * present_entropy - (
        absent_total / n_documents
    ) * absent_entropy
    return max(0.0, value)  # suppress harmless negative floating-point roundoff


def _dfs(
    total_df: int,
    per_class: dict[str, int],
    class_counts: Counter[str],
    n_documents: int,
) -> float:
    score = 0.0
    for label, class_size in class_counts.items():
        present_in_class = per_class[label]
        p_class_given_term = present_in_class / total_df
        p_not_term_given_class = (class_size - present_in_class) / class_size
        outside_class = n_documents - class_size
        p_term_given_not_class = (
            (total_df - present_in_class) / outside_class if outside_class else 0.0
        )
        score += p_class_given_term / (
            p_not_term_given_class + p_term_given_not_class + 1.0
        )
    return score
