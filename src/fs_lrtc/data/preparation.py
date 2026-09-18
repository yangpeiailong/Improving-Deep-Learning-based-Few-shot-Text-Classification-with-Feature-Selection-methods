"""Deterministic, provenance-preserving dataset preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fs_lrtc.config import ConfigError, load_yaml
from fs_lrtc.data.datasets import DatasetSpec, read_text_lines
from fs_lrtc.utils.hashing import file_sha256


_HEADER_NAME = (
    r"Path|From|Newsgroups|Subject|Message-ID|Date|Organization|Lines|"
    r"Distribution|NNTP-Posting-Host|Nntp-Posting-Host|Keywords|References|"
    r"Reply-To|Sender|Followup-To|Xref|Approved|Expires|Summary|Supersedes|"
    r"Control|News-Software|X-Newsreader|X-News-Server-Date"
)
_HEADER_MARKER = re.compile(rf"(?:^|\s)(?:{_HEADER_NAME}):", re.IGNORECASE)
_MULTISPACE = re.compile(r"\s{2,}")


@dataclass(frozen=True)
class SourceRecord:
    original_row: int
    label: str
    raw_text: str
    processed_text: str
    raw_hash: str
    processed_hash: str
    transform: str


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_flattened_20ng_headers(text: str, scan_max_characters: int = 2500) -> tuple[str, bool]:
    """Remove a flattened leading Usenet header using a documented heuristic.

    The archived documents contain no original newlines. We scan only the leading
    region, locate the final recognized header marker there, and use the first
    multi-space boundary following it as the header/body boundary.
    """
    scan_end = min(len(text), max(1, scan_max_characters))
    scan = text[:scan_end]
    markers = list(_HEADER_MARKER.finditer(scan))
    if len(markers) < 2:
        return text, False
    second_marker_end = markers[1].end()
    for boundary in _MULTISPACE.finditer(scan, second_marker_end):
        remainder = text[boundary.end() :].lstrip()
        # Some exports contain a second header block. Continue through it.
        if _HEADER_MARKER.match(remainder):
            continue
        body = remainder.strip()
        if body:
            return body, True
    return text, False


def transform_text(dataset_id: str, text: str, config: dict[str, Any]) -> tuple[str, str]:
    transforms = config.get("dataset_transforms", {})
    default = transforms.get("default", {}) if isinstance(transforms, dict) else {}
    specific = transforms.get(dataset_id, {}) if isinstance(transforms, dict) else {}
    policy = {**default, **specific}
    applied: list[str] = []
    current = text
    if policy.get("remove_flattened_leading_headers"):
        current, changed = strip_flattened_20ng_headers(
            current, int(policy.get("header_scan_max_characters", 2500))
        )
        if changed:
            applied.append("remove_flattened_leading_headers")
    if policy.get("strip_outer_whitespace", True):
        stripped = current.strip()
        if stripped != current:
            applied.append("strip_outer_whitespace")
        current = stripped
    return current, "+".join(applied) if applied else "none"


def load_source_records(spec: DatasetSpec, config: dict[str, Any]) -> list[SourceRecord]:
    metadata = spec.read_metadata()
    decoding = metadata.get("decoding", {})
    if not isinstance(decoding, dict):
        decoding = {}
    encoding = str(decoding.get("encoding", metadata.get("encoding", "utf-8")))
    errors = str(
        decoding.get(
            "errors", metadata.get("decode_errors", metadata.get("errors", "strict"))
        )
    )
    texts = read_text_lines(spec.texts_path, encoding=encoding, errors=errors)
    labels = read_text_lines(spec.labels_path, encoding=encoding, errors=errors)
    if len(texts) != len(labels):
        raise ConfigError(
            f"Text/label count mismatch for {spec.dataset_id}: {len(texts)} != {len(labels)}"
        )
    records: list[SourceRecord] = []
    for row, (raw_text, label) in enumerate(zip(texts, labels, strict=True), start=1):
        processed, transform = transform_text(spec.dataset_id, raw_text, config)
        records.append(
            SourceRecord(
                original_row=row,
                label=label,
                raw_text=raw_text,
                processed_text=processed,
                raw_hash=text_sha256(raw_text),
                processed_hash=text_sha256(processed),
                transform=transform,
            )
        )
    return records


def select_primary_records(
    records: Iterable[SourceRecord],
) -> tuple[list[SourceRecord], list[dict[str, Any]]]:
    groups: dict[str, list[SourceRecord]] = defaultdict(list)
    empty: list[SourceRecord] = []
    for record in records:
        if not record.processed_text:
            empty.append(record)
        else:
            groups[record.processed_hash].append(record)

    kept: list[SourceRecord] = []
    excluded: list[dict[str, Any]] = []
    for record in empty:
        excluded.append(_exclusion(record, "empty_after_transform", 1, (record.label,)))

    for group_hash in sorted(groups):
        group = sorted(groups[group_hash], key=lambda item: item.original_row)
        texts = {item.processed_text for item in group}
        if len(texts) != 1:
            raise RuntimeError(f"SHA-256 collision detected for processed text: {group_hash}")
        labels = tuple(sorted({item.label for item in group}))
        if len(labels) > 1:
            for record in group:
                excluded.append(_exclusion(record, "conflicting_exact_text_group", len(group), labels))
            continue
        kept.append(group[0])
        for record in group[1:]:
            excluded.append(_exclusion(record, "same_label_exact_duplicate", len(group), labels))

    kept.sort(key=lambda item: item.original_row)
    excluded.sort(key=lambda item: int(item["original_row"]))
    return kept, excluded


def prepare_dataset(
    spec: DatasetSpec,
    config: dict[str, Any],
    processed_root: Path,
    raw_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    version = str(config.get("version", "v1"))
    version_root = processed_root / version
    destination = version_root / spec.dataset_id
    _guard_destination(destination, raw_root)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Prepared dataset already exists (use --overwrite): {destination}")

    records = load_source_records(spec, config)
    kept, excluded = select_primary_records(records)
    temp = version_root / f".{spec.dataset_id}.building"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=False)
    outputs = config.get("outputs", {})
    texts_name = str(outputs.get("texts_filename", "texts.txt"))
    labels_name = str(outputs.get("labels_filename", "labels.txt"))
    records_name = str(outputs.get("records_filename", "records.csv"))
    exclusions_name = str(outputs.get("exclusions_filename", "exclusions.csv"))
    report_name = str(outputs.get("report_filename", "preparation_report.json"))

    _write_lines(temp / texts_name, (item.processed_text for item in kept))
    _write_lines(temp / labels_name, (item.label for item in kept))
    _write_csv(
        temp / records_name,
        [
            {
                "processed_index": index,
                "original_row": item.original_row,
                "label": item.label,
                "raw_text_sha256": item.raw_hash,
                "processed_text_sha256": item.processed_hash,
                "transform": item.transform,
            }
            for index, item in enumerate(kept)
        ],
        ("processed_index", "original_row", "label", "raw_text_sha256", "processed_text_sha256", "transform"),
    )
    _write_csv(
        temp / exclusions_name,
        excluded,
        ("original_row", "label", "reason", "group_size", "group_labels", "raw_text_sha256", "processed_text_sha256"),
    )
    report = {
        "schema_version": 1,
        "dataset_id": spec.dataset_id,
        "role": spec.role,
        "version": version,
        "policy_sha256": canonical_json_sha256(config),
        "source": {
            "texts_path": str(spec.texts_path),
            "labels_path": str(spec.labels_path),
            "texts_sha256": file_sha256(spec.texts_path),
            "labels_sha256": file_sha256(spec.labels_path),
        },
        "counts": {
            "input": len(records),
            "kept": len(kept),
            "excluded": len(excluded),
            "excluded_by_reason": dict(sorted(Counter(row["reason"] for row in excluded).items())),
            "classes": dict(sorted(Counter(item.label for item in kept).items())),
            "transformed_records": sum(item.transform != "none" for item in records),
        },
        "outputs": {
            texts_name: file_sha256(temp / texts_name),
            labels_name: file_sha256(temp / labels_name),
            records_name: file_sha256(temp / records_name),
            exclusions_name: file_sha256(temp / exclusions_name),
        },
    }
    (temp / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    temp.rename(destination)
    return report


def _exclusion(
    record: SourceRecord, reason: str, group_size: int, labels: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "original_row": record.original_row,
        "label": record.label,
        "reason": reason,
        "group_size": group_size,
        "group_labels": ";".join(labels),
        "raw_text_sha256": record.raw_hash,
        "processed_text_sha256": record.processed_hash,
    }


def _write_lines(path: Path, values: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(value + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _guard_destination(destination: Path, raw_root: Path) -> None:
    destination_resolved = destination.resolve()
    raw_resolved = raw_root.resolve()
    if destination_resolved == raw_resolved or destination_resolved.is_relative_to(raw_resolved):
        raise ValueError(f"Refusing to write inside frozen raw-data root: {destination}")
