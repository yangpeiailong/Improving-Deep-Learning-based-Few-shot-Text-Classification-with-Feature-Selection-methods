#!/usr/bin/env python
"""Create frozen primary and group-aware raw sensitivity folds."""

from __future__ import annotations

import argparse
import json

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.data.datasets import load_dataset_specs
from fs_lrtc.data.splitting import create_grouped_raw_splits, create_primary_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--preparation-config", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--scope", choices=("all", "main", "reserve"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = load_paths(args.paths)
    prep_config = load_yaml(args.preparation_config)
    split_config = load_yaml(args.split_config)
    specs = load_dataset_specs(args.datasets, paths["raw_data_root"], args.scope)
    version = str(split_config.get("version", "v1"))
    processed_version = str(prep_config.get("version", "v1"))
    reports = []
    for spec in specs:
        primary = create_primary_splits(
            spec.dataset_id,
            paths["processed_data_root"] / processed_version / spec.dataset_id,
            paths["split_root"] / version / "primary" / spec.dataset_id,
            split_config,
            args.overwrite,
        )
        grouped_raw = create_grouped_raw_splits(
            spec,
            prep_config,
            paths["split_root"] / version / "grouped_raw" / spec.dataset_id,
            split_config,
            args.overwrite,
        )
        reports.append({"dataset_id": spec.dataset_id, "primary": primary, "grouped_raw": grouped_raw})
        print(f"{spec.dataset_id}: primary={primary['records']}; grouped_raw={grouped_raw['records']}")

    summary_path = paths["audit_root"] / "splits_v1_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Split datasets: {len(reports)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
