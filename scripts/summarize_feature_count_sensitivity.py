#!/usr/bin/env python
"""Combine reused formal results with new feature-count sensitivity results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.experiments.feature_count import (
    aggregate_sensitivity,
    build_sensitivity_configs,
    paired_sensitivity_effects,
    standardized_row,
    trend_summary,
    validate_sensitivity_rows,
)
from fs_lrtc.experiments.formal import build_shard_configs


MASTER_CONFIG = "feature_count_sensitivity.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Sensitivity source is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty summary: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    paths = load_paths(args.paths)
    sensitivity = load_yaml(project_root / "configs" / MASTER_CONFIG)
    base_configs = {
        family: load_yaml(project_root / "configs" / str(spec["base_config"]))
        for family, spec in sensitivity["families"].items()
    }
    units = build_sensitivity_configs(sensitivity, base_configs)
    datasets = set(map(str, sensitivity["datasets"]))
    combined: list[dict[str, object]] = []
    source_experiments: list[str] = []

    # Reuse the already frozen none and 1000-feature formal runs directly from
    # their immutable shards; older derived directories need not contain a
    # combined_results.csv file.
    for family, spec in sensitivity["families"].items():
        models = set(map(str, spec["models"]))
        for formal_config in build_shard_configs(base_configs[family]).values():
            if not datasets.intersection(map(str, formal_config["datasets"])):
                continue
            source = paths["results_root"] / Path(str(formal_config["output"]["directory"]))
            manifest_path = source / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Formal manifest is missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "passed":
                raise RuntimeError(f"Formal source did not pass: {manifest_path}")
            source_experiments.append(str(manifest.get("experiment_id")))
            for row in read_csv(source / "results.csv"):
                requested = str(row["feature_count_requested"])
                if (
                    str(row["dataset_id"]) in datasets
                    and str(row["model"]) in models
                    and (
                        str(row["selector"]) == "none"
                        or (str(row["selector"]) != "none" and requested == "1000")
                    )
                ):
                    combined.append(standardized_row(row, str(family)))

    # Add only the new 500/1500/2000 units.
    for (family, _), config in units.items():
        source = paths["results_root"] / Path(str(config["output"]["directory"]))
        manifest_path = source / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Sensitivity manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "passed":
            raise RuntimeError(f"Sensitivity unit did not pass: {manifest_path}")
        source_experiments.append(str(manifest.get("experiment_id")))
        combined.extend(standardized_row(row, family) for row in read_csv(source / "results.csv"))

    report = validate_sensitivity_rows(combined, sensitivity)
    if report["status"] != "passed":
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    aggregate = aggregate_sensitivity(combined)
    effects = paired_sensitivity_effects(combined)
    trends = trend_summary(aggregate, effects)

    destination = paths["results_root"] / Path(sensitivity["output"]["derived_directory"])
    if destination.exists():
        raise FileExistsError(f"Immutable derived directory already exists: {destination}")
    temporary = destination.parent / f".{destination.name}.inprogress"
    if temporary.exists():
        raise FileExistsError(f"Incomplete derived directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    write_csv(temporary / "combined_results.csv", combined)
    write_csv(temporary / "aggregate_mean_std.csv", aggregate)
    write_csv(temporary / "paired_effects_vs_none.csv", effects)
    write_csv(temporary / "feature_count_trend_summary.csv", trends)
    report.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_experiment_ids": source_experiments,
            "aggregate_rows": len(aggregate),
            "paired_effect_rows": len(effects),
            "trend_rows": len(trends),
        }
    )
    (temporary / "completeness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    print("Feature-count sensitivity summary status: passed")
    print(f"Runs: {report['actual_rows']}/{report['expected_runs']}")
    print(f"Summary: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
