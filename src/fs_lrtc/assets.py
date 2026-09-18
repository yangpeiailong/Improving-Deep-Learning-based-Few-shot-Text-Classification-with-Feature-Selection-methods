"""Configuration and integrity helpers for local pretrained assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fs_lrtc.config import ConfigError, _expand_tree, load_yaml
from fs_lrtc.utils.hashing import file_sha256


@dataclass(frozen=True)
class FastTextAsset:
    source_path: Path
    reduced_path: Path
    expected_source_dimension: int
    target_dimension: int
    overwrite_reduced: bool


@dataclass(frozen=True)
class BertAsset:
    path: Path
    identity: str
    local_files_only: bool


@dataclass(frozen=True)
class AssetConfig:
    fasttext: FastTextAsset
    bert: BertAsset
    manifest_path: Path


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Missing `{key}` mapping in asset configuration")
    return value


def _positive_integer(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"`{key}` must be a positive integer")
    return value


def _path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"`{key}` must be a non-empty path string")
    return Path(value)


def load_asset_config(path: str | Path, project_paths: Mapping[str, Path]) -> AssetConfig:
    """Load machine-local assets, expanding placeholders from project paths."""
    raw = load_yaml(path)
    variables = {key: str(value) for key, value in project_paths.items()}
    expanded = _expand_tree(raw, variables)
    fasttext = _mapping(expanded, "fasttext")
    bert = _mapping(expanded, "bert")

    source_path = _path(fasttext, "source_path")
    reduced_path = _path(fasttext, "reduced_path")
    if source_path.resolve(strict=False) == reduced_path.resolve(strict=False):
        raise ConfigError("FastText source and reduced paths must be different")

    identity = bert.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ConfigError("`bert.identity` must be a non-empty string")
    local_only = bert.get("local_files_only", True)
    if local_only is not True:
        raise ConfigError("`bert.local_files_only` must be true for reproducible offline loading")

    overwrite = fasttext.get("overwrite_reduced", False)
    if not isinstance(overwrite, bool):
        raise ConfigError("`fasttext.overwrite_reduced` must be true or false")

    return AssetConfig(
        fasttext=FastTextAsset(
            source_path=source_path,
            reduced_path=reduced_path,
            expected_source_dimension=_positive_integer(
                fasttext, "expected_source_dimension"
            ),
            target_dimension=_positive_integer(fasttext, "target_dimension"),
            overwrite_reduced=overwrite,
        ),
        bert=BertAsset(
            path=_path(bert, "path"),
            identity=identity,
            local_files_only=True,
        ),
        manifest_path=_path(expanded, "manifest_path"),
    )


def validate_bert_directory(path: str | Path) -> dict[str, list[Path]]:
    """Return core BERT files or raise before Transformers is imported."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"BERT directory does not exist: {root}")

    required = [root / "config.json", root / "vocab.txt"]
    missing = [item for item in required if not item.is_file()]
    weight_files = sorted(root.glob("model*.safetensors")) + sorted(
        root.glob("pytorch_model*.bin")
    )
    weight_indexes = [
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    ]
    present_indexes = [item for item in weight_indexes if item.is_file()]
    if missing:
        raise FileNotFoundError(
            "BERT directory is missing required files: "
            + ", ".join(item.name for item in missing)
        )
    if not weight_files and not present_indexes:
        raise FileNotFoundError(
            "BERT directory contains no PyTorch/safetensors model weights"
        )

    optional_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    optional = [root / name for name in optional_names if (root / name).is_file()]
    return {
        "required": required,
        "weights": weight_files + present_indexes,
        "optional": optional,
    }


def hash_inventory(files: list[Path], root: str | Path) -> list[dict[str, Any]]:
    """Hash a deterministic list of model files for the provenance manifest."""
    base = Path(root)
    inventory: list[dict[str, Any]] = []
    for path in sorted(set(files), key=lambda item: item.as_posix()):
        stat = path.stat()
        inventory.append(
            {
                "relative_path": path.relative_to(base).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": file_sha256(path),
            }
        )
    return inventory
