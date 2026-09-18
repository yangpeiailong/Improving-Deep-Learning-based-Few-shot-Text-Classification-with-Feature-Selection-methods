#!/usr/bin/env python
"""Preflight and run the four immutable shards of the classic main experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.experiments.formal import build_shard_configs, expected_run_keys


MASTER_CONFIG = "formal_classic_main.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--shards", nargs="*", type=int, choices=range(1, 5))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def completed_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def completed_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    paths = load_paths(args.paths)
    master = load_yaml(project_root / "configs" / MASTER_CONFIG)
    available = build_shard_configs(master)
    selected = args.shards or sorted(available)
    loaded = [available[number] for number in selected]
    print(f"Selected shards: {selected}")
    print(f"Expected selected runs: {len(expected_run_keys(loaded))}")

    statuses: list[tuple[int, str, int, int]] = []
    for number, config in zip(selected, loaded, strict=True):
        expected = len(expected_run_keys([config]))
        for dataset in config["datasets"]:
            data_dir = paths["processed_data_root"] / str(config["data_version"]) / str(dataset)
            for name in ("texts.txt", "labels.txt"):
                if not (data_dir / name).is_file():
                    raise FileNotFoundError(f"Missing processed input: {data_dir / name}")
            for fold in config["folds"]:
                split = (
                    paths["split_root"] / str(config["split_version"])
                    / str(config["split_family"]) / str(dataset) / f"fold_{fold}.json"
                )
                if not split.is_file():
                    raise FileNotFoundError(f"Missing frozen split: {split}")
        relative = Path(str(config["output"]["directory"]))
        destination = paths["results_root"] / relative
        staging = destination.parent / f".{destination.name}.inprogress"
        final_rows = completed_rows(destination / "results.csv")
        partial_rows = completed_jsonl_rows(staging / "run_records.jsonl")
        manifest_path = destination / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else None
        )
        if (
            isinstance(manifest, dict)
            and manifest.get("status") == "passed"
            and int(manifest.get("completed_runs", -1)) == expected
            and final_rows == expected
        ):
            state, count = "completed", final_rows
        elif staging.is_dir():
            state, count = "resumable", partial_rows
        elif destination.exists():
            raise RuntimeError(f"Unexpected non-complete result directory: {destination}")
        else:
            state, count = "pending", 0
        statuses.append((number, state, count, expected))
        print(f"Shard {number}: {state} ({count}/{expected}) datasets={config['datasets']}")

    if args.preflight_only:
        print("Preflight status: passed")
        return 0

    runner = project_root / "scripts" / "run_neural_smoke.py"
    for (number, state, _, _), config in zip(statuses, loaded, strict=True):
        if state == "completed":
            print(f"Skipping completed shard {number}")
            continue
        with tempfile.TemporaryDirectory(prefix=f"fs_lrtc_shard_{number}_") as temporary:
            effective_path = Path(temporary) / f"formal_classic_main_shard{number}.json"
            effective_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            command = [
                sys.executable,
                str(runner),
                "--paths", str(Path(args.paths).resolve()),
                "--assets", str(Path(args.assets).resolve()),
                "--config", str(effective_path),
            ]
            print(f"Starting shard {number}", flush=True)
            subprocess.run(command, check=True, cwd=project_root)
    print("All selected shards completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
