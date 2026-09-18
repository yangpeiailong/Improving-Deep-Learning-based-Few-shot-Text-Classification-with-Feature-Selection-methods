#!/usr/bin/env python
"""Run strictly offline, leakage-safe BERT head experiments."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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

from fs_lrtc.assets import load_asset_config, validate_bert_directory
from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.data.bert import BertTextDataset, encode_texts, filtered_bert_texts
from fs_lrtc.data.datasets import read_text_lines
from fs_lrtc.experiments.baseline import (
    canonical_json_sha256,
    sequence_sha256,
    validate_partition,
    vocabulary_at_scope,
    write_selected_features,
)
from fs_lrtc.experiments.bert import (
    bert_training_result_dict,
    build_bert_model,
    evaluate_bert,
    make_bert_loader,
    seed_bert,
    train_bert_fixed_epochs,
    train_bert_with_validation,
)
from fs_lrtc.experiments.neural import label_mapping, resolve_device
from fs_lrtc.text import TextTokenizer
from fs_lrtc.utils.hashing import file_sha256
from fs_lrtc.utils.paths import compact_artifact_filename


ALLOWED_MODELS = ["bert_cls", "bert_mean", "bert_cnn", "bert_bilstm_attention"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_bert_asset(assets):
    manifest_path = assets.manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pretrained asset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bert = manifest.get("bert")
    if manifest.get("status") != "passed" or not isinstance(bert, dict):
        raise RuntimeError("Asset manifest does not contain a passed BERT audit")
    configured = assets.bert.path.resolve(strict=False)
    audited = Path(str(bert.get("path"))).resolve(strict=False)
    if configured != audited or bert.get("local_files_only") is not True:
        raise RuntimeError("Configured BERT directory differs from the offline audit")
    groups = validate_bert_directory(configured)
    inventory = bert.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("BERT audit has no file inventory")
    for record in inventory:
        path = configured / str(record["relative_path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"Audited BERT file changed or is missing: {path}")
    return bert, file_sha256(manifest_path)


def encoded_dataset(tokenizer, tokenized, indices, vocabulary, labels, mapping, max_length):
    texts = filtered_bert_texts(
        tokenized,
        indices,
        vocabulary,
        tokenizer.unk_token or "[UNK]",
    )
    encodings = encode_texts(tokenizer, texts, max_length)
    targets = [mapping[str(labels[index])] for index in indices]
    lengths = [len(value) for value in encodings["input_ids"]]
    return BertTextDataset(encodings, targets), lengths


def prediction_rows(indices, truth, prediction, confidence, index_to_label):
    return [
        {
            "processed_index": index,
            "true_label": index_to_label[actual],
            "predicted_label": index_to_label[predicted],
            "confidence": probability,
            "correct": int(actual == predicted),
        }
        for index, actual, predicted, probability in zip(
            indices, truth, prediction, confidence, strict=True
        )
    ]


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths)
    assets = load_asset_config(args.assets, paths)
    audited_bert, asset_manifest_digest = verify_bert_asset(assets)
    config = load_yaml(args.config)
    evaluation_scope = str(config.get("evaluation_scope", "full"))
    if evaluation_scope not in {"full", "development_only"}:
        raise ValueError(f"Unsupported BERT evaluation_scope: {evaluation_scope}")
    if list(config["models"]) != ALLOWED_MODELS and not set(config["models"]).issubset(ALLOWED_MODELS):
        raise ValueError(f"Unsupported BERT model list: {config['models']}")
    relative_output = Path(str(config["output"]["directory"]))
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise ValueError("Output directory must be relative to results_root")
    destination = paths["results_root"] / relative_output
    if destination.exists():
        raise FileExistsError(f"Immutable result directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.inprogress"
    config_digest = canonical_json_sha256(config)
    partial_path = temporary / "run_records.jsonl"
    if temporary.exists():
        snapshot = temporary / "config_snapshot.json"
        if not snapshot.is_file() or canonical_json_sha256(
            json.loads(snapshot.read_text(encoding="utf-8"))
        ) != config_digest:
            raise RuntimeError("BERT staging directory belongs to another configuration")
    else:
        temporary.mkdir(exist_ok=False)
        write_json(temporary / "config_snapshot.json", config)

    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(
        str(assets.bert.path), local_files_only=True
    )
    word_tokenizer = TextTokenizer.from_config(config["tokenizer"])
    device = resolve_device(str(config.get("device", "auto")))
    seed = int(config["seed"])
    deterministic = bool(config.get("deterministic", True))
    selection = config["feature_selection"]
    training = config["training"]
    sequence = config["sequence"]
    architecture = config["architecture"]
    batch_size = int(training["batch_size"])
    evaluation_batch_size = int(training["evaluation_batch_size"])
    pad_multiple = sequence.get("pad_to_multiple_of", 8)
    max_length = int(sequence["max_length"])

    results: list[dict[str, object]] = []
    if partial_path.is_file():
        results = [
            json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {str(row["run_id"]) for row in results}
    try:
        for dataset_id in config["datasets"]:
            data_dir = paths["processed_data_root"] / str(config["data_version"]) / dataset_id
            texts_path, labels_path = data_dir / "texts.txt", data_dir / "labels.txt"
            texts, labels = read_text_lines(texts_path), read_text_lines(labels_path)
            if len(texts) != len(labels):
                raise ValueError(f"Text/label count mismatch: {dataset_id}")
            tokenized = word_tokenizer.tokenize_many(texts)
            for fold in config["folds"]:
                split_path = (
                    paths["split_root"] / str(config["split_version"])
                    / str(config["split_family"]) / dataset_id / f"fold_{fold}.json"
                )
                split = json.loads(split_path.read_text(encoding="utf-8"))
                validate_partition(split, len(texts))
                development_mapping = label_mapping(labels, split["inner_train"])
                final_mapping = None
                if evaluation_scope == "full":
                    final_mapping = label_mapping(labels, split["outer_train"])
                    if set(development_mapping) != set(final_mapping):
                        raise RuntimeError("Development and final BERT class sets differ")

                for selector_name in selection["selectors"]:
                    method = str(selector_name)
                    selection_start = time.perf_counter()
                    development_vocabulary, development_selector = vocabulary_at_scope(
                        method, tokenized, labels, split["inner_train"],
                        sorted(split["validation"] + split["test"]),
                        int(selection["feature_count"]),
                        int(selection.get("minimum_document_frequency", 1)),
                    )
                    development_selection_seconds = time.perf_counter() - selection_start
                    final_vocabulary: list[str] = []
                    final_selector = None
                    final_selection_seconds = 0.0
                    if evaluation_scope == "full":
                        selection_start = time.perf_counter()
                        final_vocabulary, final_selector = vocabulary_at_scope(
                            method, tokenized, labels, split["outer_train"], split["test"],
                            int(selection["feature_count"]),
                            int(selection.get("minimum_document_frequency", 1)),
                        )
                        final_selection_seconds = time.perf_counter() - selection_start
                    if config["output"].get("save_selected_features", True):
                        root = temporary / "selected_features" / dataset_id / f"fold_{fold}"
                        write_selected_features(
                            root / f"{method}_development.csv",
                            development_vocabulary, development_selector,
                        )
                        if evaluation_scope == "full":
                            write_selected_features(
                                root / f"{method}_final.csv", final_vocabulary, final_selector,
                            )
                    development_filter = None if method == "none" else development_vocabulary
                    inner_dataset, inner_lengths = encoded_dataset(
                        tokenizer, tokenized, split["inner_train"], development_filter,
                        labels, development_mapping, max_length,
                    )
                    validation_dataset, validation_lengths = encoded_dataset(
                        tokenizer, tokenized, split["validation"], development_filter,
                        labels, development_mapping, max_length,
                    )
                    outer_dataset = test_dataset = None
                    outer_lengths: list[int] = []
                    test_lengths: list[int] = []
                    if evaluation_scope == "full":
                        final_filter = None if method == "none" else final_vocabulary
                        outer_dataset, outer_lengths = encoded_dataset(
                            tokenizer, tokenized, split["outer_train"], final_filter,
                            labels, final_mapping, max_length,
                        )
                        test_dataset, test_lengths = encoded_dataset(
                            tokenizer, tokenized, split["test"], final_filter,
                            labels, final_mapping, max_length,
                        )

                    for model_name in config["models"]:
                        run_id = f"{dataset_id}_fold{fold}_{method}_{model_name}"
                        if run_id in completed:
                            print(f"Skipping completed run={run_id}", flush=True)
                            continue
                        print(
                            f"Running dataset={dataset_id} fold={fold} selector={method} "
                            f"model={model_name} device={device}", flush=True,
                        )
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        seed_bert(seed, deterministic)
                        train_loader = make_bert_loader(
                            inner_dataset, tokenizer, batch_size, True, seed, pad_multiple
                        )
                        validation_loader = make_bert_loader(
                            validation_dataset, tokenizer, evaluation_batch_size,
                            False, seed, pad_multiple,
                        )
                        development_model = build_bert_model(
                            str(assets.bert.path), model_name,
                            len(development_mapping), architecture,
                        )
                        parameter_count = sum(p.numel() for p in development_model.parameters())
                        trainable_parameter_count = sum(
                            p.numel() for p in development_model.parameters() if p.requires_grad
                        )
                        development_model, development_result = train_bert_with_validation(
                            development_model, train_loader, validation_loader, training, device
                        )
                        if evaluation_scope == "development_only":
                            peak_memory = (
                                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                if device.type == "cuda" else 0.0
                            )
                            if config["output"].get("save_training_history", True):
                                write_json(
                                    temporary / "training_history"
                                    / compact_artifact_filename(run_id, ".json"),
                                    {"development": bert_training_result_dict(development_result)},
                                )
                            row = {
                                "run_id": run_id,
                                "experiment_id": config["experiment_id"],
                                "evaluation_scope": evaluation_scope,
                                "dataset_id": dataset_id,
                                "fold": int(fold),
                                "selector": method,
                                "feature_count_requested": (
                                    "all" if method == "none" else int(selection["feature_count"])
                                ),
                                "model": model_name,
                                "development_feature_count": len(development_vocabulary),
                                "parameter_count_development": parameter_count,
                                "trainable_parameter_count_development": trainable_parameter_count,
                                "best_epoch": development_result.best_epoch,
                                "epochs_completed": development_result.epochs_completed,
                                "validation_accuracy": development_result.best_accuracy,
                                "validation_macro_f1": development_result.best_macro_f1,
                                "validation_balanced_accuracy": development_result.best_balanced_accuracy,
                                "development_feature_selection_seconds": development_selection_seconds,
                                "development_training_seconds": development_result.training_seconds,
                                "peak_gpu_memory_mb": peak_memory,
                                "development_mean_wordpiece_length": sum(inner_lengths) / len(inner_lengths),
                                "validation_mean_wordpiece_length": sum(validation_lengths) / len(validation_lengths),
                                "inner_train_indices_sha256": sequence_sha256(split["inner_train"]),
                                "validation_indices_sha256": sequence_sha256(split["validation"]),
                                "development_vocabulary_sha256": sequence_sha256(development_vocabulary),
                            }
                            results.append(row)
                            with partial_path.open("a", encoding="utf-8", newline="\n") as stream:
                                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                                stream.flush()
                                os.fsync(stream.fileno())
                            completed.add(run_id)
                            print(
                                f"  best_epoch={development_result.best_epoch} "
                                f"val_acc={development_result.best_accuracy:.4f} "
                                f"peak_mb={peak_memory:.1f} development_only=true",
                                flush=True,
                            )
                            del development_model, train_loader, validation_loader
                            gc.collect()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            continue
                        del development_model, train_loader, validation_loader
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        seed_bert(seed, deterministic)
                        outer_loader = make_bert_loader(
                            outer_dataset, tokenizer, batch_size, True, seed, pad_multiple
                        )
                        test_loader = make_bert_loader(
                            test_dataset, tokenizer, evaluation_batch_size,
                            False, seed, pad_multiple,
                        )
                        final_model = build_bert_model(
                            str(assets.bert.path), model_name, len(final_mapping), architecture
                        )
                        final_model, final_training_seconds, final_history = train_bert_fixed_epochs(
                            final_model, outer_loader, development_result.best_epoch,
                            training, device,
                        )
                        scores, truth, prediction, confidence, inference_seconds = evaluate_bert(
                            final_model, test_loader, device
                        )
                        peak_memory = (
                            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                            if device.type == "cuda" else 0.0
                        )
                        index_to_label = {value: key for key, value in final_mapping.items()}
                        if config["output"].get("save_predictions", True):
                            write_csv(
                                temporary / "predictions"
                                / compact_artifact_filename(run_id, ".csv"),
                                prediction_rows(
                                    split["test"], truth, prediction, confidence, index_to_label
                                ),
                            )
                        if config["output"].get("save_training_history", True):
                            write_json(
                                temporary / "training_history"
                                / compact_artifact_filename(run_id, ".json"),
                                {
                                    "development": bert_training_result_dict(development_result),
                                    "final": {
                                        "epochs": development_result.best_epoch,
                                        "training_seconds": final_training_seconds,
                                        "history": final_history,
                                    },
                                },
                            )
                        row = {
                            "run_id": run_id,
                            "experiment_id": config["experiment_id"],
                            "dataset_id": dataset_id,
                            "fold": int(fold),
                            "selector": method,
                            "feature_count_requested": (
                                "all" if method == "none" else int(selection["feature_count"])
                            ),
                            "model": model_name,
                            "embedding_source": "local_bert_base_uncased_full_finetuning",
                            "development_feature_count": len(development_vocabulary),
                            "final_feature_count": len(final_vocabulary),
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
                            "development_training_seconds": development_result.training_seconds,
                            "final_training_seconds": final_training_seconds,
                            "test_inference_seconds": inference_seconds,
                            "peak_gpu_memory_mb": peak_memory,
                            "development_mean_wordpiece_length": sum(inner_lengths) / len(inner_lengths),
                            "validation_mean_wordpiece_length": sum(validation_lengths) / len(validation_lengths),
                            "final_mean_wordpiece_length": sum(outer_lengths) / len(outer_lengths),
                            "test_mean_wordpiece_length": sum(test_lengths) / len(test_lengths),
                            "inner_train_indices_sha256": sequence_sha256(split["inner_train"]),
                            "validation_indices_sha256": sequence_sha256(split["validation"]),
                            "outer_train_indices_sha256": sequence_sha256(split["outer_train"]),
                            "test_indices_sha256": sequence_sha256(split["test"]),
                            "development_vocabulary_sha256": sequence_sha256(development_vocabulary),
                            "final_vocabulary_sha256": sequence_sha256(final_vocabulary),
                        }
                        results.append(row)
                        with partial_path.open("a", encoding="utf-8", newline="\n") as stream:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        completed.add(run_id)
                        print(
                            f"  best_epoch={development_result.best_epoch} "
                            f"val_acc={development_result.best_accuracy:.4f} "
                            f"test_acc={scores['accuracy']:.4f} peak_mb={peak_memory:.1f}",
                            flush=True,
                        )
                        del final_model, outer_loader, test_loader
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

        if not results:
            raise RuntimeError("Configuration produced no BERT runs")
        write_csv(temporary / "results.csv", results)
        write_json(temporary / "run_manifest.json", {
            "schema_version": 1,
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
                "evaluation_scope": evaluation_scope,
                "pretrained_model": audited_bert.get("declared_identity"),
                "loading": "strictly_offline_local_files_only",
                "encoder_training": "full_finetuning",
                "development_selector_fit": "inner_train_only",
                "final_selector_fit": (
                    "outer_train_only" if evaluation_scope == "full" else "not_run"
                ),
                "none_input": "all_normalized_document_tokens",
                "selected_input": "only_training_selected_word_tokens",
                "epoch_selection": "inner_validation_only",
                "final_training_epochs": (
                    "development_best_epoch" if evaluation_scope == "full" else "not_run"
                ),
                "test_use": (
                    "one_final_evaluation_per_configuration"
                    if evaluation_scope == "full" else "not_evaluated_development_only"
                ),
            },
            "pretrained_assets": {
                "asset_manifest_sha256": asset_manifest_digest,
                "bert_path": str(assets.bert.path),
                "bert_inventory": audited_bert.get("files"),
            },
            "processed_inputs": {
                dataset_id: {
                    "texts_sha256": file_sha256(
                        paths["processed_data_root"] / str(config["data_version"])
                        / dataset_id / "texts.txt"
                    ),
                    "labels_sha256": file_sha256(
                        paths["processed_data_root"] / str(config["data_version"])
                        / dataset_id / "labels.txt"
                    ),
                }
                for dataset_id in config["datasets"]
            },
            "split_files": {
                f"{dataset_id}/fold_{fold}": file_sha256(
                    paths["split_root"] / str(config["split_version"])
                    / str(config["split_family"]) / dataset_id / f"fold_{fold}.json"
                )
                for dataset_id in config["datasets"] for fold in config["folds"]
            },
            "completed_runs": len(results),
        })
        os.replace(temporary, destination)
    except Exception as exc:
        print(
            f"Run failed; completed records retained in: {temporary}\n"
            f"{type(exc).__name__}: {exc}", file=sys.stderr,
        )
        raise
    print(f"Completed runs: {len(results)}")
    print(f"Results: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
