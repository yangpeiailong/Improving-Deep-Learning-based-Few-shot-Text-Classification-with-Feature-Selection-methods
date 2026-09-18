"""End-to-end, read-only data audit and report generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

from fs_lrtc.config import ConfigError, load_paths
from fs_lrtc.data.datasets import DatasetSpec, iter_paired_records, load_dataset_specs, read_text_lines
from fs_lrtc.data.validation import compare_observed_metadata, verify_file
from fs_lrtc.utils.hashing import text_sha256
from fs_lrtc.utils.paths import ensure_directory, write_csv, write_json


@dataclass
class AuditTables:
    summaries: list[dict[str, Any]] = field(default_factory=list)
    class_distribution: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    lengths: list[dict[str, Any]] = field(default_factory=list)
    hashes: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    manifests: list[dict[str, Any]] = field(default_factory=list)
    exact_text_sets: dict[str, set[str]] = field(default_factory=dict)


def run_audit(
    paths_config: str | Path,
    datasets_config: str | Path,
    scope: str = "all",
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Audit registered datasets and write reproducible CSV/JSON reports."""
    paths = load_paths(paths_config)
    required = ("raw_data_root", "audit_root")
    missing = [key for key in required if key not in paths]
    if missing:
        raise ConfigError(f"Missing required configured paths: {', '.join(missing)}")

    specs = load_dataset_specs(datasets_config, paths["raw_data_root"], scope=scope)
    output = ensure_directory(output_directory or paths["audit_root"])
    tables = AuditTables()
    for spec in specs:
        _audit_one(spec, tables)

    cross_overlaps = _cross_dataset_overlaps(tables.exact_text_sets)
    generated_at = datetime.now(timezone.utc).isoformat()
    fatal_count = sum(issue["severity"] == "fatal" for issue in tables.issues)
    warning_count = sum(issue["severity"] == "warning" for issue in tables.issues)
    report = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "scope": scope,
        "dataset_count": len(specs),
        "fatal_issue_count": fatal_count,
        "warning_count": warning_count,
        "status": "failed" if fatal_count else "passed_with_warnings" if warning_count else "passed",
        "issues": tables.issues,
    }
    manifest = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "scope": scope,
        "datasets": tables.manifests,
    }

    _write_reports(output, tables, cross_overlaps, report, manifest)
    return {**report, "output_directory": str(output)}


def _audit_one(spec: DatasetSpec, tables: AuditTables) -> None:
    metadata: dict[str, Any] = {}
    if not spec.metadata_path.is_file():
        tables.issues.append(_issue(spec.dataset_id, "fatal", "missing_metadata_file", str(spec.metadata_path)))
    else:
        try:
            metadata = spec.read_metadata()
        except ConfigError as exc:
            tables.issues.append(_issue(spec.dataset_id, "fatal", "invalid_metadata", str(exc)))

    metadata_files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
    text_hash_row, text_hash_issues = verify_file(
        spec.dataset_id, "text", spec.texts_path, metadata_files.get("text_sha256")
    )
    label_hash_row, label_hash_issues = verify_file(
        spec.dataset_id, "label", spec.labels_path, metadata_files.get("label_sha256")
    )
    tables.hashes.extend((text_hash_row, label_hash_row))
    tables.issues.extend(text_hash_issues + label_hash_issues)

    if not spec.texts_path.is_file() or not spec.labels_path.is_file():
        tables.summaries.append(_empty_summary(spec))
        return
    decoding = metadata.get("decoding") if isinstance(metadata.get("decoding"), dict) else {}
    encoding = decoding.get("encoding", "utf-8")
    errors = decoding.get("errors", "strict")
    if not isinstance(encoding, str) or not encoding:
        tables.issues.append(_issue(spec.dataset_id, "fatal", "invalid_encoding_setting", repr(encoding)))
        tables.summaries.append(_empty_summary(spec))
        return
    if errors not in {"strict", "replace"}:
        tables.issues.append(_issue(spec.dataset_id, "fatal", "invalid_decode_error_policy", repr(errors)))
        tables.summaries.append(_empty_summary(spec))
        return
    try:
        texts = read_text_lines(spec.texts_path, encoding=encoding, errors=errors)
        labels = read_text_lines(spec.labels_path, encoding=encoding, errors=errors)
    except (UnicodeDecodeError, LookupError) as exc:
        tables.issues.append(_issue(spec.dataset_id, "fatal", "utf8_decode_error", str(exc)))
        tables.summaries.append(_empty_summary(spec))
        return

    replacement_characters = sum(text.count("\ufffd") for text in texts) + sum(
        label.count("\ufffd") for label in labels
    )
    if replacement_characters:
        severity = "warning" if errors == "replace" else "fatal"
        tables.issues.append(
            _issue(
                spec.dataset_id,
                severity,
                "decode_replacement_characters",
                f"encoding={encoding}; errors={errors}; count={replacement_characters}",
            )
        )

    if len(texts) != len(labels):
        tables.issues.append(
            _issue(
                spec.dataset_id,
                "fatal",
                "text_label_count_mismatch",
                f"texts={len(texts)}; labels={len(labels)}",
            )
        )

    paired_count = min(len(texts), len(labels))
    paired_texts = texts[:paired_count]
    paired_labels = labels[:paired_count]
    class_counts = Counter(paired_labels)
    empty_texts = sum(not text.strip() for text in paired_texts)
    exact_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_number, text, label in iter_paired_records(paired_texts, paired_labels):
        exact_groups[text].append((row_number, label))

    duplicate_groups = {text: records for text, records in exact_groups.items() if len(records) > 1}
    conflicting_groups = {
        text: records for text, records in duplicate_groups.items() if len({label for _, label in records}) > 1
    }
    duplicate_rows = sum(len(records) - 1 for records in duplicate_groups.values())

    for label, count in sorted(class_counts.items(), key=lambda item: str(item[0])):
        tables.class_distribution.append(
            {"dataset_id": spec.dataset_id, "role": spec.role, "label": label, "count": count}
        )
    for text, records in sorted(duplicate_groups.items(), key=lambda item: text_sha256(item[0])):
        row = _duplicate_row(spec.dataset_id, text, records)
        tables.duplicates.append(row)
        if row["is_conflicting"]:
            tables.conflicts.append(row.copy())

    token_lengths = [len(text.split()) for text in paired_texts]
    character_lengths = [len(text) for text in paired_texts]
    tables.lengths.append(_length_row(spec.dataset_id, "whitespace_tokens", token_lengths))
    tables.lengths.append(_length_row(spec.dataset_id, "characters", character_lengths))

    summary = {
        "dataset_id": spec.dataset_id,
        "role": spec.role,
        "encoding": encoding,
        "decode_errors": errors,
        "replacement_characters": replacement_characters,
        "texts": len(texts),
        "labels": len(labels),
        "paired_samples": paired_count,
        "classes": len(class_counts),
        "empty_texts": empty_texts,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_rows": duplicate_rows,
        "conflicting_duplicate_groups": len(conflicting_groups),
        "min_class_count": min(class_counts.values()) if class_counts else 0,
        "max_class_count": max(class_counts.values()) if class_counts else 0,
        "text_hash_matches": text_hash_row["hash_matches"],
        "label_hash_matches": label_hash_row["hash_matches"],
    }
    tables.summaries.append(summary)
    tables.issues.extend(compare_observed_metadata(spec.dataset_id, metadata, paired_count, len(class_counts)))
    if empty_texts:
        tables.issues.append(_issue(spec.dataset_id, "warning", "empty_texts", str(empty_texts)))
    if duplicate_groups:
        tables.issues.append(
            _issue(spec.dataset_id, "warning", "exact_duplicate_groups", str(len(duplicate_groups)))
        )
    if conflicting_groups:
        tables.issues.append(
            _issue(spec.dataset_id, "warning", "conflicting_duplicate_groups", str(len(conflicting_groups)))
        )
    if class_counts and min(class_counts.values()) < 10:
        tables.issues.append(
            _issue(
                spec.dataset_id,
                "warning",
                "very_rare_class",
                f"minimum class count={min(class_counts.values())}",
            )
        )

    tables.exact_text_sets[spec.dataset_id] = {text for text in paired_texts if text.strip()}
    tables.manifests.append(
        {
            "dataset_id": spec.dataset_id,
            "role": spec.role,
            "encoding": encoding,
            "decode_errors": errors,
            "replacement_characters": replacement_characters,
            "directory": str(spec.directory),
            "texts_path": str(spec.texts_path),
            "labels_path": str(spec.labels_path),
            "metadata_path": str(spec.metadata_path),
            "text_sha256": text_hash_row["actual_sha256"],
            "label_sha256": label_hash_row["actual_sha256"],
            "samples": paired_count,
            "classes": len(class_counts),
            "class_counts": dict(sorted(class_counts.items(), key=lambda item: str(item[0]))),
        }
    )


def _empty_summary(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "dataset_id": spec.dataset_id,
        "role": spec.role,
        "encoding": "",
        "decode_errors": "",
        "replacement_characters": 0,
        "texts": 0,
        "labels": 0,
        "paired_samples": 0,
        "classes": 0,
        "empty_texts": 0,
        "duplicate_groups": 0,
        "duplicate_rows": 0,
        "conflicting_duplicate_groups": 0,
        "min_class_count": 0,
        "max_class_count": 0,
        "text_hash_matches": False,
        "label_hash_matches": False,
    }


def _duplicate_row(dataset_id: str, text: str, records: list[tuple[int, str]]) -> dict[str, Any]:
    labels = [label for _, label in records]
    return {
        "dataset_id": dataset_id,
        "text_sha256": text_sha256(text),
        "occurrences": len(records),
        "row_numbers": ";".join(str(row_number) for row_number, _ in records),
        "labels": ";".join(labels),
        "unique_labels": len(set(labels)),
        "is_conflicting": len(set(labels)) > 1,
        "text_preview": text[:240].replace("\t", " "),
    }


def _length_row(dataset_id: str, unit: str, values: list[int]) -> dict[str, Any]:
    if not values:
        return {"dataset_id": dataset_id, "unit": unit, **{key: 0 for key in _LENGTH_KEYS}}
    ordered = sorted(values)
    return {
        "dataset_id": dataset_id,
        "unit": unit,
        "count": len(values),
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": median(ordered),
        "mean": round(mean(ordered), 6),
        "p75": _percentile(ordered, 0.75),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


_LENGTH_KEYS = ("count", "min", "p25", "median", "mean", "p75", "p95", "max")


def _percentile(ordered: list[int], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 6)


def _cross_dataset_overlaps(exact_text_sets: dict[str, set[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(exact_text_sets), 2):
        overlap = exact_text_sets[left] & exact_text_sets[right]
        if overlap:
            rows.append(
                {
                    "dataset_left": left,
                    "dataset_right": right,
                    "exact_overlap_count": len(overlap),
                    "example_text_sha256": text_sha256(min(overlap)),
                }
            )
    return rows


def _write_reports(
    output: Path,
    tables: AuditTables,
    cross_overlaps: list[dict[str, Any]],
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    write_csv(output / "data_summary.csv", tables.summaries, _SUMMARY_FIELDS)
    write_csv(output / "class_distribution.csv", tables.class_distribution, _CLASS_FIELDS)
    write_csv(output / "duplicate_samples.csv", tables.duplicates, _DUPLICATE_FIELDS)
    write_csv(output / "label_conflicts.csv", tables.conflicts, _DUPLICATE_FIELDS)
    write_csv(output / "length_statistics.csv", tables.lengths, _LENGTH_FIELDS)
    write_csv(output / "cross_dataset_overlap.csv", cross_overlaps, _OVERLAP_FIELDS)
    write_csv(output / "hash_verification.csv", tables.hashes, _HASH_FIELDS)
    write_csv(output / "audit_issues.csv", tables.issues, _ISSUE_FIELDS)
    write_json(output / "data_manifest.json", manifest)
    write_json(output / "audit_report.json", report)


def _issue(dataset_id: str, severity: str, code: str, detail: str) -> dict[str, Any]:
    return {"dataset_id": dataset_id, "severity": severity, "code": code, "detail": detail}


_SUMMARY_FIELDS = (
    "dataset_id", "role", "encoding", "decode_errors", "replacement_characters", "texts", "labels",
    "paired_samples", "classes", "empty_texts",
    "duplicate_groups", "duplicate_rows", "conflicting_duplicate_groups", "min_class_count",
    "max_class_count", "text_hash_matches", "label_hash_matches",
)
_CLASS_FIELDS = ("dataset_id", "role", "label", "count")
_DUPLICATE_FIELDS = (
    "dataset_id", "text_sha256", "occurrences", "row_numbers", "labels", "unique_labels",
    "is_conflicting", "text_preview",
)
_LENGTH_FIELDS = ("dataset_id", "unit", *_LENGTH_KEYS)
_OVERLAP_FIELDS = ("dataset_left", "dataset_right", "exact_overlap_count", "example_text_sha256")
_HASH_FIELDS = (
    "dataset_id", "file_kind", "path", "exists", "expected_sha256", "actual_sha256", "hash_matches",
)
_ISSUE_FIELDS = ("dataset_id", "severity", "code", "detail")
