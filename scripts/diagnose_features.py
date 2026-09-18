#!/usr/bin/env python
"""Fit selectors at leakage-safe scopes and save ranked feature diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.data.datasets import read_text_lines
from fs_lrtc.features.pipeline import fit_selector_at_scope, write_fit_manifest, write_ranked_features
from fs_lrtc.text import TextTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--feature-config", required=True)
    parser.add_argument("--diagnostic-config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = load_paths(args.paths)
    feature_config = load_yaml(args.feature_config)
    diagnostic = load_yaml(args.diagnostic_config)
    version = str(feature_config.get("version", "v1"))
    tokenizer_config = feature_config.get("tokenizer", {})
    tokenizer = TextTokenizer.from_config(tokenizer_config)
    selection = feature_config.get("selection", {})
    min_df = int(selection.get("minimum_document_frequency", 1))
    output_root = paths["data_root"] / "features" / version
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Feature output version already exists (use --overwrite or change version): {output_root}"
            )
        shutil.rmtree(output_root)
    summaries = []

    for dataset_id in diagnostic["datasets"]:
        data_dir = paths["processed_data_root"] / "v1" / dataset_id
        texts = read_text_lines(data_dir / "texts.txt")
        labels = read_text_lines(data_dir / "labels.txt")
        tokenized = tokenizer.tokenize_many(texts)
        for fold in diagnostic["folds"]:
            split_path = paths["split_root"] / "v1" / "primary" / dataset_id / f"fold_{fold}.json"
            split = json.loads(split_path.read_text(encoding="utf-8"))
            for stage in diagnostic["stages"]:
                if stage == "development":
                    fit_indices = split["inner_train"]
                    held_out = sorted(split["validation"] + split["test"])
                elif stage == "final":
                    fit_indices = split["outer_train"]
                    held_out = split["test"]
                else:
                    raise ValueError(f"Unsupported diagnostic stage: {stage}")
                stage_dir = output_root / dataset_id / f"fold_{fold}" / stage
                write_fit_manifest(
                    stage_dir / "fit_manifest.json", dataset_id, fold, stage,
                    fit_indices, held_out, labels, tokenizer_config,
                )
                for method in diagnostic["selectors"]:
                    selector = fit_selector_at_scope(
                        method, tokenized, labels, fit_indices, held_out, min_df
                    )
                    for feature_count in diagnostic["feature_counts"]:
                        output = stage_dir / method / f"top_{feature_count}.csv"
                        write_ranked_features(selector, int(feature_count), output)
                    top_preview = selector.top(int(diagnostic.get("top_features_to_preview", 20)))
                    summaries.append(
                        {
                            "dataset_id": dataset_id,
                            "fold": fold,
                            "stage": stage,
                            "selector": method,
                            "fit_records": len(fit_indices),
                            "available_features": len(selector.ranked_features_),
                            "top_features": [item.token for item in top_preview],
                        }
                    )
                    print(
                        f"{dataset_id} fold={fold} stage={stage} selector={method} "
                        f"fit={len(fit_indices)} vocab={len(selector.ranked_features_)}"
                    )
    summary_path = paths["audit_root"] / "feature_diagnostic_v1_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Diagnostic fits: {len(summaries)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
