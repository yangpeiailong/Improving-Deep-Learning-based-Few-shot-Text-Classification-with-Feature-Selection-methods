"""Filesystem and portable result-path helpers."""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write an Excel-friendly UTF-8 CSV with deterministic columns."""
    target = Path(path)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def compact_artifact_filename(stem: str, suffix: str, max_stem_length: int = 36) -> str:
    """Return a deterministic short filename while preserving short readable stems."""
    stem = str(stem)
    if len(stem) <= max_stem_length:
        return f"{stem}{suffix}"
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]
    prefix_length = max_stem_length - len(digest) - 1
    if prefix_length < 1:
        raise ValueError("max_stem_length is too small for a collision-resistant name")
    prefix = stem[:prefix_length].rstrip("_.- ") or stem[:prefix_length]
    return f"{prefix}_{digest}{suffix}"
