"""Read-only access to registered datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fs_lrtc.config import ConfigError, load_yaml


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    role: str
    directory: Path
    texts_path: Path
    labels_path: Path
    metadata_path: Path

    def read_metadata(self) -> dict[str, Any]:
        return load_yaml(self.metadata_path)


def read_text_lines(
    path: str | Path,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> list[str]:
    """Read logical lines using an explicitly recorded decoding policy."""
    if errors not in {"strict", "replace"}:
        raise ValueError(f"Unsupported decoding error policy: {errors}")
    with Path(path).open("r", encoding=encoding, errors=errors, newline=None) as stream:
        return stream.read().splitlines()


def read_utf8_lines(path: str | Path) -> list[str]:
    """Backward-compatible strict UTF-8 reader."""
    return read_text_lines(path, encoding="utf-8", errors="strict")


def load_dataset_specs(
    datasets_config_path: str | Path,
    raw_data_root: str | Path,
    scope: str = "all",
) -> list[DatasetSpec]:
    """Resolve registry entries without reading or modifying dataset bytes."""
    config = load_yaml(datasets_config_path)
    registry = config.get("registry")
    if not isinstance(registry, dict) or not registry:
        raise ConfigError("`registry` must be a non-empty mapping in datasets.yaml")

    main = _as_id_set(config.get("main_datasets"), "main_datasets")
    reserve = _as_id_set(config.get("reserve_datasets"), "reserve_datasets")
    if main & reserve:
        overlap = ", ".join(sorted(main & reserve))
        raise ConfigError(f"Datasets cannot be both main and reserve: {overlap}")
    if scope not in {"all", "main", "reserve"}:
        raise ConfigError(f"Unsupported dataset scope: {scope}")

    text_filename = _required_string(config, "text_filename")
    label_filename = _required_string(config, "label_filename")
    metadata_filename = _required_string(config, "metadata_filename")
    root = Path(raw_data_root)
    specs: list[DatasetSpec] = []

    for dataset_id, entry in registry.items():
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ConfigError("Every registry key must be a non-empty dataset id")
        role = "main" if dataset_id in main else "reserve" if dataset_id in reserve else "unassigned"
        if scope != "all" and role != scope:
            continue
        if not isinstance(entry, dict):
            raise ConfigError(f"Registry entry must be a mapping: {dataset_id}")
        directory_name = entry.get("directory", dataset_id)
        if not isinstance(directory_name, str) or not directory_name:
            raise ConfigError(f"Invalid directory for dataset: {dataset_id}")
        directory = root / directory_name
        metadata_relative = entry.get("metadata")
        if isinstance(metadata_relative, str) and metadata_relative:
            metadata_path = root / metadata_relative
        else:
            metadata_path = directory / metadata_filename
        specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                role=role,
                directory=directory,
                texts_path=directory / text_filename,
                labels_path=directory / label_filename,
                metadata_path=metadata_path,
            )
        )
    return specs


def _as_id_set(value: Any, key: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"`{key}` must be a list of dataset ids")
    if len(value) != len(set(value)):
        raise ConfigError(f"`{key}` contains duplicate dataset ids")
    return set(value)


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"Missing non-empty `{key}` in datasets.yaml")
    return value


def iter_paired_records(texts: Iterable[str], labels: Iterable[str]):
    """Yield one-based row numbers with aligned text-label records."""
    for row_number, (text, label) in enumerate(zip(texts, labels, strict=False), start=1):
        yield row_number, text, label
