"""Command-line entry point for the frozen-data audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fs_lrtc.config import ConfigError
from fs_lrtc.data.audit import run_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit frozen text-classification datasets without modifying them.")
    parser.add_argument("--paths", required=True, type=Path, help="Path to configs/paths.local.yaml")
    parser.add_argument("--datasets", required=True, type=Path, help="Path to configs/datasets.yaml")
    parser.add_argument("--scope", choices=("all", "main", "reserve"), default="all")
    parser.add_argument("--output", type=Path, default=None, help="Optional output-directory override")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 if fatal issues are found")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_audit(args.paths, args.datasets, scope=args.scope, output_directory=args.output)
    except (ConfigError, OSError) as exc:
        print(f"Audit could not start: {exc}", file=sys.stderr)
        return 1

    print(f"Audit status: {report['status']}")
    print(f"Datasets audited: {report['dataset_count']}")
    print(f"Fatal issues: {report['fatal_issue_count']}")
    print(f"Warnings: {report['warning_count']}")
    print(f"Reports: {report['output_directory']}")
    if args.strict and report["fatal_issue_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
