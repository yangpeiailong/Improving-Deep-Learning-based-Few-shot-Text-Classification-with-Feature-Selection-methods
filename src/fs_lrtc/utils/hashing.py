"""Deterministic hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    """Return a stable digest for an exact Unicode text value."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
