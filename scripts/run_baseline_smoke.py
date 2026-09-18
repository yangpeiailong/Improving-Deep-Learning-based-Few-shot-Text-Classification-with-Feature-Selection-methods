#!/usr/bin/env python
"""Run the leakage-safe TF-IDF logistic-regression end-to-end smoke experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from threadpoolctl import threadpool_limits

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.data.datasets import read_text_lines
from fs_lrtc.experiments.baseline import (
    canonical_json_sha256,
    development_search,
    final_evaluation,
    result_to_dict,
    sequence_sha256,
    validate_partition,
    write_predictions,
    write_selected_features,
)
from fs_lrtc.text import TextTokenizer
from fs_lrtc.utils.hashing import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    output_relative = Path(str(config["output"]["directory"]))
    if output_relative.is_absolute() or ".." in output_relative.parts:
        raise ValueError("Output directory must be a safe path relative to results_root")
    destination = paths["results_root"] / output_relative
    if destination.exists():
        raise FileExistsError(
            f"Result directory already exists; results are immutable: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.building"
    temporary.mkdir(parents=False, exist_ok=False)

    tokenizer = TextTokenizer.from_config(config["tokenizer"])
    selection_config = config["feature_selection"]
    representation_config = config["representation"]
    classifier_config = config["classifier"]
    seed = int(config["seed"])
    rows: list[dict[str, object]] = []
    try:
        write_json(temporary / "config_snapshot.json", config)
        with threadpool_limits(limits=int(config["runtime"].get("thread_limit", 1))):
            for dataset_id in config["datasets"]:
                data_dir = paths["processed_data_root"] / str(config["data_version"]) / dataset_id
                texts_path, labels_path = data_dir / "texts.txt", data_dir / "labels.txt"
                texts = read_text_lines(texts_path)
                labels = read_text_lines(labels_path)
                if len(texts) != len(labels):
                    raise ValueError(f"Text/label count mismatch: {dataset_id}")
                tokenized = tokenizer.tokenize_many(texts)
                for fold in config["folds"]:
                    split_path = (
                        paths["split_root"] / str(config["split_version"])
                        / str(config["split_family"]) / dataset_id / f"fold_{fold}.json"
                    )
                    split = json.loads(split_path.read_text(encoding="utf-8"))
                    validate_partition(split, len(texts))
                    for method in selection_config["selectors"]:
                        print(f"Running dataset={dataset_id} fold={fold} selector={method}", flush=True)
                        best, candidates, development_vocabulary, development_selector, fs_seconds = development_search(
                            tokenized, labels, split, str(method),
                            int(selection_config["feature_count"]),
                            int(selection_config.get("minimum_document_frequency", 1)),
                            representation_config, classifier_config, seed,
                        )
                        final_scores, truth, prediction, final_vocabulary, final_selector, timing = final_evaluation(
                            tokenized, labels, split, str(method),
                            int(selection_config["feature_count"]),
                            int(selection_config.get("minimum_document_frequency", 1)),
                            representation_config, classifier_config, best.c_value, seed,
                        )
                        run_id = f"{dataset_id}_fold{fold}_{method}"
                        if config["output"].get("save_selected_features", True):
                            write_selected_features(
                                temporary / "selected_features" / f"{run_id}_development.csv",
                                development_vocabulary, development_selector,
                            )
                            write_selected_features(
                                temporary / "selected_features" / f"{run_id}_final.csv",
                                final_vocabulary, final_selector,
                            )
                        if config["output"].get("save_predictions", True):
                            write_predictions(
                                temporary / "predictions" / f"{run_id}.csv",
                                split["test"], truth, prediction,
                            )
                        row: dict[str, object] = {
                            "experiment_id": config["experiment_id"],
                            "dataset_id": dataset_id,
                            "fold": int(fold),
                            "selector": str(method),
                            "feature_count_requested": "all" if method == "none" else int(selection_config["feature_count"]),
                            "development_feature_count": len(development_vocabulary),
                            "final_feature_count": len(final_vocabulary),
                            "selected_c": best.c_value,
                            "validation_accuracy": best.accuracy,
                            "validation_macro_f1": best.macro_f1,
                            "test_accuracy": final_scores["accuracy"],
                            "test_macro_f1": final_scores["macro_f1"],
                            "test_balanced_accuracy": final_scores["balanced_accuracy"],
                            "development_feature_selection_seconds": fs_seconds,
                            **timing,
                            "inner_train_indices_sha256": sequence_sha256(split["inner_train"]),
                            "validation_indices_sha256": sequence_sha256(split["validation"]),
                            "outer_train_indices_sha256": sequence_sha256(split["outer_train"]),
                            "test_indices_sha256": sequence_sha256(split["test"]),
                            "development_vocabulary_sha256": sequence_sha256(development_vocabulary),
                            "final_vocabulary_sha256": sequence_sha256(final_vocabulary),
                        }
                        rows.append(row)
                        write_json(
                            temporary / "validation" / f"{run_id}.json",
                            {
                                "selection_rule": "accuracy, macro_f1, balanced_accuracy, then smaller_C",
                                "selected": result_to_dict(best),
                                "candidates": [result_to_dict(item) for item in candidates],
                                "fit_scope": "inner_train_only",
                                "held_out_scope": "validation_only; test untouched",
                            },
                        )
                        print(
                            f"  C={best.c_value:g} val_acc={best.accuracy:.4f} "
                            f"test_acc={final_scores['accuracy']:.4f}",
                            flush=True,
                        )

        fieldnames = list(rows[0])
        with (temporary / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema_version": 1,
            "status": "passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": config["experiment_id"],
            "config_sha256": canonical_json_sha256(config),
            "host": {"platform": platform.platform(), "python": platform.python_version(), "executable": sys.executable},
            "protocol": {
                "development_selector_fit": "inner_train_only",
                "validation_use": "classifier_hyperparameter_selection_only",
                "final_selector_fit": "outer_train_only",
                "test_use": "one_final_evaluation_per_configuration",
                "stateless_tokenization_before_split": True,
            },
            "inputs": {
                "processed_texts_sha256": {
                    dataset_id: file_sha256(paths["processed_data_root"] / str(config["data_version"]) / dataset_id / "texts.txt")
                    for dataset_id in config["datasets"]
                },
                "processed_labels_sha256": {
                    dataset_id: file_sha256(paths["processed_data_root"] / str(config["data_version"]) / dataset_id / "labels.txt")
                    for dataset_id in config["datasets"]
                },
            },
            "completed_runs": len(rows),
        }
        write_json(temporary / "run_manifest.json", manifest)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"Completed runs: {len(rows)}")
    print(f"Results: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
