#!/usr/bin/env python
"""Validate the factorial coverage of the result files shipped with the repository."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    main_dir = ROOT / "results" / "main"
    main_rows = read_csv(main_dir / "main_fold_results.csv")
    main_report = json.loads(
        (main_dir / "main_completeness_report.json").read_text(encoding="utf-8")
    )
    main_keys = {
        (row["dataset_id"], row["model"], row["selector"], row["fold"])
        for row in main_rows
    }
    require(main_report.get("status") == "passed", "Main completeness report did not pass")
    require(len(main_rows) == 3520, f"Expected 3520 main rows, found {len(main_rows)}")
    require(len(main_keys) == 3520, "Duplicate main dataset/model/selector/fold key")
    require(len({row["dataset_id"] for row in main_rows}) == 16, "Expected 16 datasets")
    require(len({row["model"] for row in main_rows}) == 11, "Expected 11 models")
    require({row["selector"] for row in main_rows} == {"none", "df", "ig", "dfs"}, "Selector mismatch")
    require({row["fold"] for row in main_rows} == {"0", "1", "2", "3", "4"}, "Fold mismatch")

    feature_dir = ROOT / "results" / "feature_count"
    feature_report = json.loads(
        (feature_dir / "completeness_report.json").read_text(encoding="utf-8")
    )
    feature_rows = read_csv(feature_dir / "aggregate_mean_std.csv")
    require(feature_report.get("status") == "passed", "Feature-count completeness report did not pass")
    require(feature_report.get("actual_rows") == 2080, "Expected 2080 feature-count source rows")
    require(len(feature_rows) == 416, f"Expected 416 feature-count aggregates, found {len(feature_rows)}")
    require(
        {row["feature_count"] for row in feature_rows} == {"all", "500", "1000", "1500", "2000"},
        "Feature-count levels do not match the released protocol",
    )

    print("Published-result verification: passed")
    print("Main experiment: 3520/3520 unique fold-level evaluations")
    print("Feature-count analysis: 2080/2080 source rows; 416 aggregate cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
