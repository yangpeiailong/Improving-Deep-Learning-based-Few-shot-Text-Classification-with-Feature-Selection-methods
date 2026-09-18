#!/usr/bin/env python
"""Audit local BERT/FastText resources and optionally reduce FastText once."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from fs_lrtc.assets import (
    AssetConfig,
    hash_inventory,
    load_asset_config,
    validate_bert_directory,
)
from fs_lrtc.config import load_paths
from fs_lrtc.utils.hashing import file_sha256


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.building")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_fasttext_model(path: Path):
    try:
        import fasttext
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Cannot import fasttext. Install requirements-assets.txt in the active "
            "virtual environment."
        ) from exc
    return fasttext.load_model(str(path))


def inspect_fasttext(path: Path, expected_dimension: int) -> tuple[Any, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"FastText model does not exist: {path}")
    print(f"Hashing FastText model (this may take several minutes): {path}", flush=True)
    digest = file_sha256(path)
    print(f"Loading FastText model: {path}", flush=True)
    model = load_fasttext_model(path)
    dimension = int(model.get_dimension())
    if dimension != expected_dimension:
        raise RuntimeError(
            f"Unexpected FastText dimension at {path}: {dimension} != {expected_dimension}"
        )
    vector = np.asarray(model.get_word_vector("hello"))
    if vector.shape != (expected_dimension,) or not np.isfinite(vector).all():
        raise RuntimeError("FastText smoke-test vector is invalid")
    stat = path.stat()
    return model, {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": digest,
        "dimension": dimension,
        "smoke_test_word": "hello",
        "smoke_test_vector_finite": True,
    }


def reduce_fasttext(config: AssetConfig, source_model: Any) -> tuple[Any, str]:
    target = config.fasttext.reduced_path
    expected = config.fasttext.target_dimension
    if target.is_file() and not config.fasttext.overwrite_reduced:
        print(f"Reusing existing reduced FastText model: {target}", flush=True)
        return load_fasttext_model(target), "reused"
    if target.exists() and not target.is_file():
        raise RuntimeError(f"Reduced FastText destination is not a file: {target}")

    try:
        import fasttext.util
    except (ImportError, OSError) as exc:
        raise RuntimeError("fasttext.util is unavailable") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.building.bin")
    print(
        f"Reducing FastText {config.fasttext.expected_source_dimension}d -> {expected}d. "
        "This is CPU/RAM intensive and can take a long time.",
        flush=True,
    )
    try:
        fasttext.util.reduce_model(source_model, expected)
        source_model.save_model(str(temporary))
        reduced = load_fasttext_model(temporary)
        if int(reduced.get_dimension()) != expected:
            raise RuntimeError("Temporary reduced FastText model has the wrong dimension")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return load_fasttext_model(target), "created"


def inspect_reduced_fasttext(config: AssetConfig, model: Any, state: str) -> dict[str, Any]:
    path = config.fasttext.reduced_path
    dimension = int(model.get_dimension())
    if dimension != config.fasttext.target_dimension:
        raise RuntimeError(
            f"Reduced FastText dimension is {dimension}, expected "
            f"{config.fasttext.target_dimension}"
        )
    print(f"Hashing reduced FastText model: {path}", flush=True)
    return {
        "state": state,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "dimension": dimension,
    }


def inspect_bert(config: AssetConfig, requested_device: str) -> dict[str, Any]:
    import torch
    from transformers import BertModel, BertTokenizer

    groups = validate_bert_directory(config.bert.path)
    files = groups["required"] + groups["weights"] + groups["optional"]
    print("Hashing BERT configuration, vocabulary, and weight files...", flush=True)
    inventory = hash_inventory(files, config.bert.path)

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    print(f"Loading BERT strictly offline on {device}: {config.bert.path}", flush=True)
    tokenizer = BertTokenizer.from_pretrained(
        str(config.bert.path), local_files_only=config.bert.local_files_only
    )
    model = BertModel.from_pretrained(
        str(config.bert.path), local_files_only=config.bert.local_files_only
    ).to(device)
    model.eval()
    encoded = tokenizer(
        ["offline pretrained asset smoke test"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded).last_hidden_state
    if not torch.isfinite(output).all():
        raise RuntimeError("BERT offline forward pass produced non-finite values")
    result = {
        "path": str(config.bert.path),
        "declared_identity": config.bert.identity,
        "local_files_only": True,
        "device": device,
        "model_type": model.config.model_type,
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "vocab_size": int(model.config.vocab_size),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "smoke_test_output_shape": list(output.shape),
        "files": inventory,
    }
    del output, encoded, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, help="Path to paths.local.yaml")
    parser.add_argument("--assets", required=True, help="Path to assets.local.yaml")
    parser.add_argument(
        "--reduce-fasttext",
        action="store_true",
        help="Create/reuse and verify the configured reduced FastText model",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_paths(args.paths)
        config = load_asset_config(args.assets, paths)
        source_model, fasttext_source = inspect_fasttext(
            config.fasttext.source_path,
            config.fasttext.expected_source_dimension,
        )
        fasttext_reduced = None
        if args.reduce_fasttext:
            reduced_model, state = reduce_fasttext(config, source_model)
            fasttext_reduced = inspect_reduced_fasttext(config, reduced_model, state)
            del reduced_model
        bert = inspect_bert(config, args.device)
        payload = {
            "schema_version": 1,
            "status": "passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "audit_and_reduce" if args.reduce_fasttext else "audit_only",
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "executable": sys.executable,
            },
            "packages": {
                "numpy": package_version("numpy"),
                "fasttext-wheel": package_version("fasttext-wheel"),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
            },
            "fasttext": {
                "source": fasttext_source,
                "reduced": fasttext_reduced,
            },
            "bert": bert,
        }
        atomic_json_dump(config.manifest_path, payload)
        print("Asset status: passed")
        print(f"Mode: {payload['mode']}")
        print(f"Manifest: {config.manifest_path}")
        return 0
    except Exception as exc:
        print(f"Asset status: failed\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
