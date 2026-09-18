#!/usr/bin/env python
"""Build immutable processed-v1 data with provenance reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.data.datasets import load_dataset_specs
from fs_lrtc.data.preparation import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", choices=("all", "main", "reserve"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    specs = load_dataset_specs(args.datasets, paths["raw_data_root"], args.scope)
    reports = []
    for spec in specs:
        report = prepare_dataset(
            spec, config, paths["processed_data_root"], paths["raw_data_root"], args.overwrite
        )
        reports.append(report)
        counts = report["counts"]
        print(f"{spec.dataset_id}: {counts['input']} -> {counts['kept']} (excluded {counts['excluded']})")

    summary_path = paths["audit_root"] / "preparation_v1_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared datasets: {len(reports)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
