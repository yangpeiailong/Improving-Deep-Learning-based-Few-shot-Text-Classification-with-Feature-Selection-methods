#!/usr/bin/env python
"""Validate and aggregate all four classic-main result shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.experiments.formal import (
    aggregate_results,
    build_shard_configs,
    overall_selector_summary,
    paired_effects,
    validate_formal_rows,
)


MASTER_CONFIG = "formal_classic_main.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Formal result file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty summary: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    paths = load_paths(args.paths)
    master = load_yaml(project_root / "configs" / MASTER_CONFIG)
    configs = list(build_shard_configs(master).values())
    rows: list[dict[str, str]] = []
    manifests: list[dict[str, object]] = []
    for config in configs:
        destination = paths["results_root"] / Path(str(config["output"]["directory"]))
        rows.extend(read_rows(destination / "results.csv"))
        manifest_path = destination / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Formal manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "passed":
            raise RuntimeError(f"Formal shard did not pass: {manifest_path}")
        manifests.append(manifest)

    report = validate_formal_rows(rows, configs)
    if report["status"] != "passed":
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    aggregate = aggregate_results(rows)
    effects = paired_effects(rows)
    overall = overall_selector_summary(aggregate, effects)

    destination = paths["results_root"] / "derived" / "classic_main_v1"
    if destination.exists():
        raise FileExistsError(f"Immutable derived directory already exists: {destination}")
    temporary = destination.parent / ".classic_main_v1.inprogress"
    if temporary.exists():
        raise FileExistsError(f"Incomplete derived directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    write_csv(temporary / "combined_results.csv", rows)
    write_csv(temporary / "aggregate_mean_std.csv", aggregate)
    write_csv(temporary / "paired_effects_vs_none.csv", effects)
    write_csv(temporary / "overall_selector_summary.csv", overall)
    report.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_experiment_ids": [manifest.get("experiment_id") for manifest in manifests],
            "aggregate_rows": len(aggregate),
            "paired_effect_rows": len(effects),
        }
    )
    (temporary / "completeness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    print("Formal summary status: passed")
    print(f"Runs: {report['actual_rows']}/{report['expected_runs']}")
    print(f"Summary: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
