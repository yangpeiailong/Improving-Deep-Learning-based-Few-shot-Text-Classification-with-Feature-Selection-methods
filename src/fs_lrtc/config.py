"""YAML configuration loading and path interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is missing or invalid."""


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {file_path}")
    return data


def _expand_string(value: str, variables: Mapping[str, str]) -> str:
    """Expand ${name} placeholders from config roots or the environment."""
    current = value
    for _ in range(10):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            name = match.group(1)
            if name in variables:
                changed = True
                return variables[name]
            if name in os.environ:
                changed = True
                return os.environ[name]
            raise ConfigError(f"Unknown path placeholder: ${{{name}}}")

        expanded = _PLACEHOLDER.sub(replace, current)
        current = expanded
        if not changed:
            return current
    raise ConfigError(f"Circular or excessively nested placeholder: {value}")


def _expand_tree(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, variables)
    if isinstance(value, list):
        return [_expand_tree(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_tree(item, variables) for key, item in value.items()}
    return value


def load_paths(path: str | Path) -> dict[str, Path]:
    """Load and expand `paths.local.yaml` into absolute/usable Paths."""
    config = load_yaml(path)
    root_keys = ("paper_root", "code_root")
    variables: dict[str, str] = {}
    for key in root_keys:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Missing non-empty `{key}` in {path}")
        variables[key] = value

    expanded = _expand_tree(config, variables)
    paths = expanded.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError(f"Missing `paths` mapping in {path}")

    resolved: dict[str, Path] = {
        "paper_root": Path(expanded["paper_root"]),
        "code_root": Path(expanded["code_root"]),
    }
    for key, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Path `{key}` must be a non-empty string")
        resolved[key] = Path(value)
    return resolved
