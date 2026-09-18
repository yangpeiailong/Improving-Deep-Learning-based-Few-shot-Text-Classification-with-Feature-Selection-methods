#!/usr/bin/env python
"""Validate and combine classic, GCN, and BERT fixed-1000 main results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.experiments.formal import build_shard_configs
from fs_lrtc.experiments.main_summary import (
    FAMILY_MODELS,
    add_family,
    aggregate_with_family,
    effects_with_family,
    family_summary,
    selector_summary,
    sha256_file,
    significance_tests,
    runtime_summary,
    union_fieldnames,
    validate_cross_family_rows,
    validate_family_rows,
)


FAMILY_CONFIGS = {
    "classic": "formal_classic_main.yaml",
    "gcn": "formal_gcn.yaml",
    "bert": "formal_bert.yaml",
}
CANONICAL_RELATIVE_PATHS = {
    "classic": Path("derived/classic_main_v1/combined_results.csv"),
    "gcn": Path("derived/gcn_v2/combined_results.csv"),
    "bert": Path("derived/bert_v3_full16/combined_results.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, help="Local path configuration YAML")
    parser.add_argument("--classic", help="Override classic combined_results.csv")
    parser.add_argument("--gcn", help="Override GCN combined_results.csv")
    parser.add_argument("--bert", help="Override BERT combined_results.csv")
    parser.add_argument(
        "--output-name",
        default="main_fixed1000_v1",
        help="Directory name created below results_root/derived",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty summary: {path.name}")
    names = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_family_configs(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        family: list(
            build_shard_configs(load_yaml(project_root / "configs" / filename)).values()
        )
        for family, filename in FAMILY_CONFIGS.items()
    }


def _valid_candidate(
    path: Path, family: str, configs: Sequence[dict[str, Any]]
) -> tuple[bool, dict[str, Any] | None]:
    try:
        rows = read_rows(path)
        report = validate_family_rows(rows, configs, family)
    except (KeyError, TypeError, ValueError):
        return False, None
    return report["status"] == "passed", report


def discover_source(
    results_root: Path,
    family: str,
    configs: Sequence[dict[str, Any]],
    override: str | None,
) -> Path:
    if override:
        candidate = Path(override).expanduser()
        valid, report = _valid_candidate(candidate, family, configs)
        if not valid:
            raise RuntimeError(
                f"The --{family} file is not a complete valid {family} main result: "
                f"{candidate}\n{json.dumps(report, ensure_ascii=False, indent=2)}"
            )
        return candidate

    canonical = results_root / CANONICAL_RELATIVE_PATHS[family]
    if canonical.is_file() and _valid_candidate(canonical, family, configs)[0]:
        return canonical

    matches: list[Path] = []
    for candidate in sorted(results_root.rglob("combined_results.csv")):
        if "main_fixed1000" in candidate.parts:
            continue
        if _valid_candidate(candidate, family, configs)[0]:
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"No complete {family} combined_results.csv was found below {results_root}. "
            f"Run its family summarizer first or pass --{family} with the exact file path."
        )
    if len(matches) > 1:
        choices = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"More than one complete {family} result was found. Choose one explicitly with "
            f"--{family}:\n{choices}"
        )
    return matches[0]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    paths = load_paths(args.paths)
    results_root = paths["results_root"]
    family_configs = load_family_configs(project_root)

    sources: dict[str, Path] = {}
    family_rows: dict[str, list[dict[str, str]]] = {}
    family_reports: dict[str, dict[str, Any]] = {}
    for family in ("classic", "gcn", "bert"):
        source = discover_source(
            results_root, family, family_configs[family], getattr(args, family)
        )
        rows = read_rows(source)
        report = validate_family_rows(rows, family_configs[family], family)
        if report["status"] != "passed":
            raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
        sources[family] = source
        family_rows[family] = rows
        family_reports[family] = report
        print(f"Found {family}: {source} ({len(rows)} rows)")

    rows = [row for family in ("classic", "gcn", "bert") for row in family_rows[family]]
    all_configs = [
        config
        for family in ("classic", "gcn", "bert")
        for config in family_configs[family]
    ]
    report = validate_cross_family_rows(rows, all_configs)
    if report["status"] != "passed":
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))

    combined = add_family(rows)
    aggregate = aggregate_with_family(rows)
    effects = effects_with_family(rows)
    overall = selector_summary(aggregate, effects)
    by_family = family_summary(aggregate, effects)
    tests = significance_tests(effects)
    runtimes = runtime_summary(rows)

    destination = results_root / "derived" / args.output_name
    temporary = destination.parent / f".{args.output_name}.inprogress"
    if destination.exists():
        raise FileExistsError(
            f"Immutable output directory already exists: {destination}. "
            "Keep it for provenance, or use --output-name main_fixed1000_v2."
        )
    if temporary.exists():
        raise FileExistsError(
            f"Incomplete output directory exists: {temporary}. Inspect it before removing it."
        )
    temporary.mkdir(parents=True)
    try:
        write_csv(temporary / "main_fold_results.csv", combined, union_fieldnames(combined))
        write_csv(temporary / "main_mean_std.csv", aggregate)
        write_csv(temporary / "main_paired_effects_vs_none.csv", effects)
        write_csv(temporary / "main_selector_summary.csv", overall)
        write_csv(temporary / "main_family_summary.csv", by_family)
        write_csv(temporary / "main_significance_tests.csv", tests)
        write_csv(temporary / "main_runtime_summary.csv", runtimes)
        report.update(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": {
                    "datasets": 16,
                    "models": 11,
                    "folds": 5,
                    "selectors": ["none", "df", "ig", "dfs"],
                    "selected_feature_count": 1000,
                    "expected_runs": 3520,
                },
                "family_reports": family_reports,
                "sources": {
                    family: {
                        "file_name": path.name,
                        "parent_name": path.parent.name,
                        "sha256": sha256_file(path),
                        "rows": len(family_rows[family]),
                    }
                    for family, path in sources.items()
                },
                "output_rows": {
                    "main_fold_results.csv": len(combined),
                    "main_mean_std.csv": len(aggregate),
                    "main_paired_effects_vs_none.csv": len(effects),
                    "main_selector_summary.csv": len(overall),
                    "main_family_summary.csv": len(by_family),
                    "main_significance_tests.csv": len(tests),
                    "main_runtime_summary.csv": len(runtimes),
                },
            }
        )
        (temporary / "main_completeness_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except Exception:
        print(f"Summary failed; partial files retained in: {temporary}")
        raise

    print("Main fixed-1000 summary status: passed")
    print(f"Runs: {report['actual_rows']}/{report['expected_runs']}")
    print(f"Datasets: {len(report['datasets'])}; models: {len(report['models'])}")
    print(f"Output: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
