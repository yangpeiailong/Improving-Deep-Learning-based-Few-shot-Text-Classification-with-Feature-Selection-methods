"""Metadata and file-integrity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fs_lrtc.utils.hashing import file_sha256


def verify_file(
    dataset_id: str,
    file_kind: str,
    path: Path,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify existence and optional SHA-256, returning a report row and issues."""
    issues: list[dict[str, Any]] = []
    exists = path.is_file()
    actual = file_sha256(path) if exists else None
    expected = expected_sha256 or None
    matches = bool(exists and expected and actual == expected)

    if not exists:
        issues.append(_issue(dataset_id, "fatal", f"missing_{file_kind}_file", str(path)))
    elif expected is None:
        issues.append(_issue(dataset_id, "warning", f"missing_{file_kind}_hash", str(path)))
    elif actual != expected:
        issues.append(
            _issue(
                dataset_id,
                "fatal",
                f"{file_kind}_hash_mismatch",
                f"expected={expected}; actual={actual}; path={path}",
            )
        )

    row = {
        "dataset_id": dataset_id,
        "file_kind": file_kind,
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected or "",
        "actual_sha256": actual or "",
        "hash_matches": matches,
    }
    return row, issues


def compare_observed_metadata(
    dataset_id: str,
    metadata: dict[str, Any],
    samples: int,
    classes: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if metadata.get("dataset_id") not in {None, dataset_id}:
        issues.append(
            _issue(
                dataset_id,
                "fatal",
                "metadata_dataset_id_mismatch",
                f"metadata={metadata.get('dataset_id')}; registry={dataset_id}",
            )
        )
    observed = metadata.get("observed")
    if not isinstance(observed, dict):
        return [_issue(dataset_id, "warning", "missing_observed_metadata", "observed mapping is absent")]
    for key, actual in (("samples", samples), ("classes", classes)):
        expected = observed.get(key)
        if expected is not None and expected != actual:
            issues.append(
                _issue(
                    dataset_id,
                    "warning",
                    f"metadata_{key}_mismatch",
                    f"metadata={expected}; actual={actual}",
                )
            )
    return issues


def _issue(dataset_id: str, severity: str, code: str, detail: str) -> dict[str, Any]:
    return {"dataset_id": dataset_id, "severity": severity, "code": code, "detail": detail}
