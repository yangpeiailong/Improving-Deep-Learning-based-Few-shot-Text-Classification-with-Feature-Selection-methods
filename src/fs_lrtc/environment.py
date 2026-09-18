"""Environment inspection helpers without eager scientific imports."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Iterable


def package_versions(distributions: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def base_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def compare_versions(
    observed: dict[str, str | None], expected: dict[str, str]
) -> list[str]:
    return [
        f"{name}: expected {expected_version}, observed {observed.get(name)}"
        for name, expected_version in expected.items()
        if observed.get(name) != expected_version
    ]
