#!/usr/bin/env python
"""Run only the missing 500/1500/2000 retained-feature sensitivity units."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.experiments.feature_count import build_sensitivity_configs, expected_unit_runs


MASTER_CONFIG = "feature_count_sensitivity.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--families", nargs="*", choices=("classic", "gcn", "bert"))
    parser.add_argument("--feature-counts", nargs="*", type=int, choices=(500, 1500, 2000))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    paths = load_paths(args.paths)
    sensitivity = load_yaml(project_root / "configs" / MASTER_CONFIG)
    family_specs = sensitivity["families"]
    base_configs = {
        family: load_yaml(project_root / "configs" / str(spec["base_config"]))
        for family, spec in family_specs.items()
    }
    available = build_sensitivity_configs(sensitivity, base_configs)
    families = set(args.families or family_specs)
    counts = set(args.feature_counts or sensitivity["new_feature_counts"])
    selected = [key for key in available if key[0] in families and key[1] in counts]
    if not selected:
        raise ValueError("No sensitivity units were selected")
    print(f"Selected units: {selected}")
    print(f"Expected selected new runs: {sum(expected_unit_runs(available[key]) for key in selected)}")

    statuses = []
    for key in selected:
        family, count = key
        config = available[key]
        expected = expected_unit_runs(config)
        for dataset in config["datasets"]:
            data_dir = paths["processed_data_root"] / str(config["data_version"]) / dataset
            for name in ("texts.txt", "labels.txt"):
                if not (data_dir / name).is_file():
                    raise FileNotFoundError(f"Missing processed input: {data_dir / name}")
            for fold in config["folds"]:
                split = (
                    paths["split_root"] / str(config["split_version"])
                    / str(config["split_family"]) / dataset / f"fold_{fold}.json"
                )
                if not split.is_file():
                    raise FileNotFoundError(f"Missing frozen split: {split}")
        destination = paths["results_root"] / Path(config["output"]["directory"])
        staging = destination.parent / f".{destination.name}.inprogress"
        manifest_path = destination / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        final_count = csv_rows(destination / "results.csv")
        partial_count = jsonl_rows(staging / "run_records.jsonl")
        if (
            isinstance(manifest, dict) and manifest.get("status") == "passed"
            and int(manifest.get("completed_runs", -1)) == expected and final_count == expected
        ):
            state, completed = "completed", final_count
        elif staging.is_dir():
            state, completed = "resumable", partial_count
        elif destination.exists():
            raise RuntimeError(f"Unexpected non-complete directory: {destination}")
        else:
            state, completed = "pending", 0
        statuses.append((key, state, completed, expected, config))
        print(f"{family} k={count}: {state} ({completed}/{expected})")

    if args.preflight_only:
        print("Feature-count sensitivity preflight status: passed")
        return 0

    for (family, count), state, _, _, config in statuses:
        if state == "completed":
            print(f"Skipping completed {family} k={count}")
            continue
        runner = project_root / "scripts" / str(family_specs[family]["runner"])
        with tempfile.TemporaryDirectory(prefix=f"fs_lrtc_sensitivity_{family}_{count}_") as temp:
            config_path = Path(temp) / f"{family}_k{count}.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            command = [
                sys.executable, str(runner),
                "--paths", str(Path(args.paths).resolve()),
                "--assets", str(Path(args.assets).resolve()),
                "--config", str(config_path),
            ]
            print(f"Starting {family} k={count}", flush=True)
            subprocess.run(command, check=True, cwd=project_root)
    print("All selected feature-count sensitivity units completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
