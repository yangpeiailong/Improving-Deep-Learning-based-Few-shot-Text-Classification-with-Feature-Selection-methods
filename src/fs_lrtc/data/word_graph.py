"""Training-scope-only word graphs for inductive document classification."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence


@dataclass(frozen=True)
class SparseMatrixData:
    rows: list[int]
    columns: list[int]
    values: list[float]
    shape: tuple[int, int]


@dataclass(frozen=True)
class WordGraphData:
    adjacency: SparseMatrixData
    word_to_node: dict[str, int]
    unknown_node: int
    windows: int
    positive_pmi_pairs: int


def _document_windows(nodes: list[int], window_size: int) -> list[list[int]]:
    if not nodes:
        return []
    if len(nodes) <= window_size:
        return [nodes]
    return [nodes[start : start + window_size] for start in range(len(nodes) - window_size + 1)]


def build_word_graph(
    tokenized_documents: Sequence[Sequence[str]],
    fit_indices: Sequence[int],
    vocabulary: Sequence[str],
    window_size: int = 10,
) -> WordGraphData:
    """Build a symmetric positive-PMI graph using only ``fit_indices``."""
    if window_size < 2:
        raise ValueError("word graph window_size must be at least 2")
    if not fit_indices:
        raise ValueError("word graph fit scope is empty")
    if not vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("word graph vocabulary must be non-empty and unique")
    word_to_node = {word: index for index, word in enumerate(vocabulary)}
    unknown_node = len(word_to_node)
    occurrence: Counter[int] = Counter()
    pair_count: Counter[tuple[int, int]] = Counter()
    window_count = 0
    for record_index in fit_indices:
        nodes = [
            word_to_node[token]
            for token in tokenized_documents[record_index]
            if token in word_to_node
        ]
        for window in _document_windows(nodes, window_size):
            unique = sorted(set(window))
            if not unique:
                continue
            window_count += 1
            occurrence.update(unique)
            pair_count.update(combinations(unique, 2))
    if window_count == 0:
        raise ValueError("No selected vocabulary tokens occur in graph training documents")

    edges: dict[tuple[int, int], float] = {}
    for (left, right), count in pair_count.items():
        denominator = occurrence[left] * occurrence[right]
        if denominator <= 0:
            continue
        pmi = math.log((count * window_count) / denominator)
        if pmi > 0:
            edges[(left, right)] = pmi
            edges[(right, left)] = pmi
    node_count = len(vocabulary) + 1
    for node in range(node_count):
        edges[(node, node)] = 1.0

    degree = [0.0] * node_count
    for (row, _), weight in edges.items():
        degree[row] += weight
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (row, column), weight in sorted(edges.items()):
        normalized = weight / math.sqrt(degree[row] * degree[column])
        rows.append(row)
        columns.append(column)
        values.append(normalized)
    return WordGraphData(
        adjacency=SparseMatrixData(rows, columns, values, (node_count, node_count)),
        word_to_node=word_to_node,
        unknown_node=unknown_node,
        windows=window_count,
        positive_pmi_pairs=(len(edges) - node_count) // 2,
    )


def document_term_matrix(
    tokenized_documents: Sequence[Sequence[str]],
    indices: Sequence[int],
    graph: WordGraphData,
) -> SparseMatrixData:
    """Create row-normalized term-frequency document inputs for a frozen graph."""
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, record_index in enumerate(indices):
        counts = Counter(
            graph.word_to_node[token]
            for token in tokenized_documents[record_index]
            if token in graph.word_to_node
        )
        if not counts:
            counts[graph.unknown_node] = 1
        total = sum(counts.values())
        for column, count in sorted(counts.items()):
            rows.append(row)
            columns.append(column)
            values.append(count / total)
    return SparseMatrixData(
        rows=rows,
        columns=columns,
        values=values,
        shape=(len(indices), graph.adjacency.shape[0]),
    )
