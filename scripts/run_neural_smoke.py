#!/usr/bin/env python
"""Run leakage-safe Random/FastText WA, CNN, and BiLSTM experiments."""

from __future__ import annotations

import os

# Required by deterministic CUDA matrix operations; must precede torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from fs_lrtc.assets import load_asset_config
from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.data.datasets import read_text_lines
from fs_lrtc.data.sequences import (
    TextSequenceDataset,
    build_vocabulary,
    matched_truncation_length,
    mean_effective_length,
)
from fs_lrtc.experiments.baseline import (
    canonical_json_sha256,
    sequence_sha256,
    validate_partition,
    vocabulary_at_scope,
    write_selected_features,
)
from fs_lrtc.experiments.neural import (
    build_model,
    evaluate,
    fasttext_embedding_matrix,
    label_mapping,
    make_loader,
    model_family,
    resolve_device,
    seed_everything,
    train_fixed_epochs,
    train_with_validation,
    training_result_dict,
    uses_fasttext,
)
from fs_lrtc.experiments.kshot import apply_training_scope, configured_shots
from fs_lrtc.text import TextTokenizer
from fs_lrtc.utils.hashing import file_sha256
from fs_lrtc.utils.paths import compact_artifact_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_verified_fasttext(asset_config, asset_manifest_path: Path):
    try:
        import fasttext
    except (ImportError, OSError) as exc:
        raise RuntimeError("Cannot import fasttext from the active environment") from exc
    if not asset_manifest_path.is_file():
        raise FileNotFoundError(f"Pretrained asset manifest is missing: {asset_manifest_path}")
    manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    reduced = manifest.get("fasttext", {}).get("reduced")
    if manifest.get("status") != "passed" or not isinstance(reduced, dict):
        raise RuntimeError("Asset manifest does not contain a passed reduced FastText audit")
    path = asset_config.fasttext.reduced_path
    if Path(str(reduced.get("path"))).resolve(strict=False) != path.resolve(strict=False):
        raise RuntimeError("Configured reduced FastText path differs from the audited path")
    if int(reduced.get("dimension", -1)) != asset_config.fasttext.target_dimension:
        raise RuntimeError("Audited FastText dimension differs from the configured dimension")
    if int(reduced.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError("Reduced FastText size changed after its asset audit")
    print(f"Verifying reduced FastText SHA-256: {path}", flush=True)
    digest = file_sha256(path)
    if digest != reduced.get("sha256"):
        raise RuntimeError("Reduced FastText SHA-256 changed after its asset audit")
    print(f"Loading verified reduced FastText: {path}", flush=True)
    model = fasttext.load_model(str(path))
    if int(model.get_dimension()) != asset_config.fasttext.target_dimension:
        raise RuntimeError("Loaded FastText dimension is incorrect")
    return model, digest


def prediction_rows(indices, truth, prediction, confidences, index_to_label):
    rows = []
    for record_index, actual, predicted, confidence in zip(
        indices, truth, prediction, confidences, strict=True
    ):
        rows.append(
            {
                "processed_index": record_index,
                "true_label": index_to_label[actual],
                "predicted_label": index_to_label[predicted],
                "confidence": confidence,
                "correct": int(actual == predicted),
            }
        )
    return rows


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths)
    assets = load_asset_config(args.assets, paths)
    config = load_yaml(args.config)
    relative_output = Path(str(config["output"]["directory"]))
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise ValueError("Output directory must be relative to results_root")
    destination = paths["results_root"] / relative_output
    if destination.exists():
        raise FileExistsError(f"Immutable result directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.inprogress"
    config_digest = canonical_json_sha256(config)
    partial_results_path = temporary / "run_records.jsonl"
    if temporary.exists():
        snapshot_path = temporary / "config_snapshot.json"
        if not snapshot_path.is_file():
            raise RuntimeError(f"Incomplete staging directory has no config snapshot: {temporary}")
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if canonical_json_sha256(previous) != config_digest:
            raise RuntimeError(
                f"Staging directory belongs to a different configuration: {temporary}"
            )
    else:
        temporary.mkdir(exist_ok=False)
        write_json(temporary / "config_snapshot.json", config)

    device = resolve_device(str(config.get("device", "auto")))
    deterministic = bool(config.get("deterministic", True))
    seed = int(config["seed"])
    tokenizer = TextTokenizer.from_config(config["tokenizer"])
    selection_config = config["feature_selection"]
    architecture = config["architecture"]
    sequence_config = config["sequence"]
    training_config = config["training"]
    model_names = [str(name) for name in config["models"]]
    shot_values = configured_shots(config)
    if len(shot_values) != 1:
        raise ValueError(
            "The neural runner accepts one K value; use the K-shot orchestrator for a grid"
        )
    shots_per_class = shot_values[0]
    if str(training_config.get("optimizer", "adam")).lower() != "adam":
        raise ValueError("The classic neural runner supports the fixed Adam protocol only")
    if str(sequence_config.get("empty_document_policy", "unk")) != "unk":
        raise ValueError("The v7 runner requires empty_document_policy=unk")

    fasttext_model = None
    fasttext_digest = None
    if any(uses_fasttext(name) for name in model_names):
        fasttext_model, fasttext_digest = load_verified_fasttext(
            assets, assets.manifest_path
        )

    results: list[dict[str, object]] = []
    if partial_results_path.is_file():
        for line in partial_results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
    completed_run_ids = {str(row["run_id"]) for row in results}
    try:
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
                frozen_split = json.loads(split_path.read_text(encoding="utf-8"))
                validate_partition(frozen_split, len(texts))
                split = apply_training_scope(
                    frozen_split, labels, config, dataset_id, int(fold), shots_per_class
                )
                development_labels = label_mapping(labels, split["inner_train"])
                final_labels = label_mapping(labels, split["outer_train"])
                if set(development_labels) != set(final_labels):
                    raise RuntimeError("Development and final training class sets differ")

                for selector_name in selection_config["selectors"]:
                    method = str(selector_name)
                    vocabulary_method = "none" if method == "none_short" else method
                    random_selection_seed = int(
                        selection_config.get("random_selection_seed", 42)
                    )
                    selection_start = time.perf_counter()
                    development_vocabulary, development_selector = vocabulary_at_scope(
                        vocabulary_method, tokenized, labels, split["inner_train"],
                        sorted(split["validation"] + split["test"]),
                        int(selection_config["feature_count"]),
                        int(selection_config.get("minimum_document_frequency", 1)),
                        random_selection_seed,
                    )
                    development_selection_seconds = time.perf_counter() - selection_start
                    selection_start = time.perf_counter()
                    final_vocabulary, final_selector = vocabulary_at_scope(
                        vocabulary_method, tokenized, labels, split["outer_train"], split["test"],
                        int(selection_config["feature_count"]),
                        int(selection_config.get("minimum_document_frequency", 1)),
                        random_selection_seed,
                    )
                    final_selection_seconds = time.perf_counter() - selection_start
                    if config["output"].get("save_selected_features", True):
                        feature_root = temporary / "selected_features" / dataset_id / f"fold_{fold}"
                        if shots_per_class is not None:
                            feature_root = feature_root / f"k_{shots_per_class}"
                        write_selected_features(
                            feature_root / f"{method}_development.csv",
                            development_vocabulary, development_selector,
                        )
                        write_selected_features(
                            feature_root / f"{method}_final.csv",
                            final_vocabulary, final_selector,
                        )

                    development_word_to_index = build_vocabulary(development_vocabulary)
                    final_word_to_index = build_vocabulary(final_vocabulary)
                    development_max_length = int(sequence_config["max_length"])
                    final_max_length = int(sequence_config["max_length"])
                    if method == "none_short":
                        development_reference, _ = vocabulary_at_scope(
                            "dfs", tokenized, labels, split["inner_train"],
                            sorted(split["validation"] + split["test"]),
                            int(selection_config["feature_count"]),
                            int(selection_config.get("minimum_document_frequency", 1)),
                        )
                        final_reference, _ = vocabulary_at_scope(
                            "dfs", tokenized, labels, split["outer_train"], split["test"],
                            int(selection_config["feature_count"]),
                            int(selection_config.get("minimum_document_frequency", 1)),
                        )
                        development_max_length = matched_truncation_length(
                            tokenized, split["inner_train"], development_word_to_index,
                            build_vocabulary(development_reference),
                            int(sequence_config["max_length"]),
                        )
                        final_max_length = matched_truncation_length(
                            tokenized, split["outer_train"], final_word_to_index,
                            build_vocabulary(final_reference),
                            int(sequence_config["max_length"]),
                        )
                    for model_name in model_names:
                        scope_id = (
                            f"_k{shots_per_class}" if shots_per_class is not None else ""
                        )
                        run_id = f"{dataset_id}_fold{fold}{scope_id}_{method}_{model_name}"
                        if run_id in completed_run_ids:
                            print(f"Skipping completed run={run_id}", flush=True)
                            continue
                        print(
                            f"Running dataset={dataset_id} fold={fold} selector={method} "
                            f"model={model_name} "
                            f"scope={'K=' + str(shots_per_class) if shots_per_class is not None else 'full'} "
                            f"device={device}",
                            flush=True,
                        )
                        family = model_family(model_name)
                        minimum_length = (
                            max(int(value) for value in architecture["cnn_kernel_sizes"])
                            if family == "cnn" else 1
                        )
                        development_weights = None
                        final_weights = None
                        embedding_start = time.perf_counter()
                        if uses_fasttext(model_name):
                            development_weights = fasttext_embedding_matrix(
                                development_word_to_index,
                                fasttext_model,
                                int(architecture["embedding_dimension"]),
                            )
                            final_weights = fasttext_embedding_matrix(
                                final_word_to_index,
                                fasttext_model,
                                int(architecture["embedding_dimension"]),
                            )
                        embedding_seconds = time.perf_counter() - embedding_start

                        development_train = TextSequenceDataset(
                            tokenized, labels, split["inner_train"],
                            development_word_to_index, development_labels,
                            development_max_length,
                        )
                        development_validation = TextSequenceDataset(
                            tokenized, labels, split["validation"],
                            development_word_to_index, development_labels,
                            development_max_length,
                        )
                        seed_everything(seed, deterministic)
                        development_model = build_model(
                            model_name, len(development_word_to_index),
                            len(development_labels), architecture, development_weights,
                        )
                        parameter_count = sum(
                            parameter.numel() for parameter in development_model.parameters()
                        )
                        trainable_parameter_count = sum(
                            parameter.numel()
                            for parameter in development_model.parameters()
                            if parameter.requires_grad
                        )
                        training_loader = make_loader(
                            development_train, int(training_config["batch_size"]), True,
                            seed, minimum_length,
                        )
                        validation_loader = make_loader(
                            development_validation,
                            int(training_config["evaluation_batch_size"]), False,
                            seed, minimum_length,
                        )
                        development_model, development_result = train_with_validation(
                            development_model, training_loader, validation_loader,
                            training_config, device,
                        )
                        del development_model, training_loader, validation_loader
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        final_train = TextSequenceDataset(
                            tokenized, labels, split["outer_train"],
                            final_word_to_index, final_labels,
                            final_max_length,
                        )
                        final_test = TextSequenceDataset(
                            tokenized, labels, split["test"],
                            final_word_to_index, final_labels,
                            final_max_length,
                        )
                        seed_everything(seed, deterministic)
                        final_model = build_model(
                            model_name, len(final_word_to_index), len(final_labels),
                            architecture, final_weights,
                        )
                        final_training_loader = make_loader(
                            final_train, int(training_config["batch_size"]), True,
                            seed, minimum_length,
                        )
                        test_loader = make_loader(
                            final_test, int(training_config["evaluation_batch_size"]), False,
                            seed, minimum_length,
                        )
                        final_model, final_training_seconds, final_history = train_fixed_epochs(
                            final_model, final_training_loader,
                            development_result.best_epoch, training_config, device,
                        )
                        scores, truth, prediction, indices, confidences, inference_seconds = evaluate(
                            final_model, test_loader, device
                        )
                        index_to_label = {value: key for key, value in final_labels.items()}
                        if config["output"].get("save_predictions", True):
                            write_predictions(
                                temporary / "predictions"
                                / compact_artifact_filename(run_id, ".csv"),
                                prediction_rows(
                                    indices, truth, prediction, confidences, index_to_label
                                ),
                            )
                        if config["output"].get("save_training_history", True):
                            write_json(
                                temporary / "training_history"
                                / compact_artifact_filename(run_id, ".json"),
                                {
                                    "development": training_result_dict(development_result),
                                    "final": {
                                        "epochs": development_result.best_epoch,
                                        "training_seconds": final_training_seconds,
                                        "history": final_history,
                                    },
                                },
                            )
                        result_row = {
                                "run_id": run_id,
                                "experiment_id": config["experiment_id"],
                                "dataset_id": dataset_id,
                                "fold": int(fold),
                                "training_scope": (
                                    "k_shot" if shots_per_class is not None else "full_outer_train"
                                ),
                                "shots_per_class": (
                                    int(shots_per_class) if shots_per_class is not None else ""
                                ),
                                "class_count": len(final_labels),
                                "development_train_size": len(split["inner_train"]),
                                "validation_size": len(split["validation"]),
                                "final_train_size": len(split["outer_train"]),
                                "test_size": len(split["test"]),
                                "selector": method,
                                "feature_count_requested": "all" if method in {"none", "none_short"} else int(selection_config["feature_count"]),
                                "model": model_name,
                                "embedding_source": "fasttext_cc_en_100" if uses_fasttext(model_name) else "random",
                                "development_feature_count": len(development_vocabulary),
                                "final_feature_count": len(final_vocabulary),
                                "development_max_length": development_max_length,
                                "final_max_length": final_max_length,
                                "development_train_mean_effective_length": mean_effective_length(
                                    tokenized, split["inner_train"], development_word_to_index,
                                    development_max_length,
                                ),
                                "development_validation_mean_effective_length": mean_effective_length(
                                    tokenized, split["validation"], development_word_to_index,
                                    development_max_length,
                                ),
                                "final_train_mean_effective_length": mean_effective_length(
                                    tokenized, split["outer_train"], final_word_to_index,
                                    final_max_length,
                                ),
                                "test_mean_effective_length": mean_effective_length(
                                    tokenized, split["test"], final_word_to_index,
                                    final_max_length,
                                ),
                                "parameter_count_development": parameter_count,
                                "trainable_parameter_count_development": trainable_parameter_count,
                                "best_epoch": development_result.best_epoch,
                                "epochs_completed": development_result.epochs_completed,
                                "validation_accuracy": development_result.best_accuracy,
                                "validation_macro_f1": development_result.best_macro_f1,
                                "validation_balanced_accuracy": development_result.best_balanced_accuracy,
                                "test_accuracy": scores["accuracy"],
                                "test_macro_f1": scores["macro_f1"],
                                "test_balanced_accuracy": scores["balanced_accuracy"],
                                "development_feature_selection_seconds": development_selection_seconds,
                                "final_feature_selection_seconds": final_selection_seconds,
                                "embedding_matrix_seconds": embedding_seconds,
                                "development_training_seconds": development_result.training_seconds,
                                "final_training_seconds": final_training_seconds,
                                "test_inference_seconds": inference_seconds,
                                "inner_train_indices_sha256": sequence_sha256(split["inner_train"]),
                                "validation_indices_sha256": sequence_sha256(split["validation"]),
                                "outer_train_indices_sha256": sequence_sha256(split["outer_train"]),
                                "test_indices_sha256": sequence_sha256(split["test"]),
                                "development_vocabulary_sha256": sequence_sha256(development_vocabulary),
                                "final_vocabulary_sha256": sequence_sha256(final_vocabulary),
                            }
                        results.append(result_row)
                        with partial_results_path.open("a", encoding="utf-8", newline="\n") as stream:
                            stream.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        completed_run_ids.add(run_id)
                        print(
                            f"  best_epoch={development_result.best_epoch} "
                            f"val_acc={development_result.best_accuracy:.4f} "
                            f"test_acc={scores['accuracy']:.4f}",
                            flush=True,
                        )
                        del final_model, final_training_loader, test_loader
                        del development_weights, final_weights
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

        if not results:
            raise RuntimeError("Configuration produced no neural runs")
        with (temporary / "results.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": 2,
                "status": "passed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": config["experiment_id"],
                "config_sha256": config_digest,
                "host": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "executable": sys.executable,
                    "torch": torch.__version__,
                    "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                },
                "protocol": {
                    "training_scope": (
                        "exact_k_per_class_from_frozen_outer_train"
                        if shots_per_class is not None else "full_frozen_outer_train"
                    ),
                    "shots_per_class": shots_per_class,
                    "kshot_validation": (
                        "partitioned_from_the_same_k_shot_support_then_refit_on_all_k"
                        if shots_per_class is not None else None
                    ),
                    "frozen_outer_test_fold_unchanged": True,
                    "development_vocabulary_and_selector_fit": "inner_train_only",
                    "epoch_selection": "inner_validation_only",
                    "final_vocabulary_and_selector_refit": "outer_train_only",
                    "final_training_epochs": "development_best_epoch",
                    "test_use": "one_final_evaluation_per_configuration",
                    "network_output": "raw_logits_for_cross_entropy",
                    "lstm_sequence_handling": "packed_true_lengths_and_final_bidirectional_hidden_states",
                    "initial_hidden_state": "framework_zero_state",
                    "padding_aware_pooling": True,
                    "mechanism_controls": {
                        "random": "deterministic salted-hash ranking fitted on training scope only",
                        "none_short": "full training vocabulary with training-only cap matched to DFS mean effective length",
                    },
                    "diagnostics": [
                        "training_loss",
                        "training_accuracy",
                        "validation_loss",
                        "validation_accuracy",
                        "mean_preclip_gradient_norm",
                    ],
                },
                "pretrained_assets_manifest_sha256": file_sha256(assets.manifest_path),
                "fasttext_reduced_sha256": fasttext_digest,
                "processed_inputs": {
                    dataset_id: {
                        "texts_sha256": file_sha256(paths["processed_data_root"] / str(config["data_version"]) / dataset_id / "texts.txt"),
                        "labels_sha256": file_sha256(paths["processed_data_root"] / str(config["data_version"]) / dataset_id / "labels.txt"),
                    }
                    for dataset_id in config["datasets"]
                },
                "split_files": {
                    f"{dataset_id}/fold_{fold}": file_sha256(
                        paths["split_root"] / str(config["split_version"])
                        / str(config["split_family"]) / dataset_id / f"fold_{fold}.json"
                    )
                    for dataset_id in config["datasets"]
                    for fold in config["folds"]
                },
                "completed_runs": len(results),
            },
        )
        os.replace(temporary, destination)
    except Exception as exc:
        print(
            f"Run failed; completed records were retained for resume in: {temporary}\n"
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    print(f"Completed runs: {len(results)}")
    print(f"Results: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
