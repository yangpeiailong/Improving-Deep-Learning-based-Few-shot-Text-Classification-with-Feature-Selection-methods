#!/usr/bin/env python
"""Run leakage-safe inductive word-graph GCN experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Required by deterministic CUDA matrix operations; must precede torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from fs_lrtc.assets import load_asset_config
from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.data.datasets import read_text_lines
from fs_lrtc.data.word_graph import build_word_graph, document_term_matrix
from fs_lrtc.experiments.baseline import (
    canonical_json_sha256,
    sequence_sha256,
    validate_partition,
    vocabulary_at_scope,
    write_selected_features,
)
from fs_lrtc.experiments.gcn import (
    build_gcn,
    encoded_labels,
    evaluate_gcn,
    fasttext_node_matrix,
    gcn_training_result_dict,
    seed_gcn,
    sparse_tensor,
    train_gcn_fixed_epochs,
    train_gcn_with_validation,
)
from fs_lrtc.experiments.neural import label_mapping, resolve_device
from fs_lrtc.text import TextTokenizer
from fs_lrtc.utils.hashing import file_sha256
from fs_lrtc.utils.paths import compact_artifact_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_verified_fasttext(asset_config):
    try:
        import fasttext
    except (ImportError, OSError) as exc:
        raise RuntimeError("Cannot import fasttext from the active environment") from exc
    manifest_path = asset_config.manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pretrained asset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reduced = manifest.get("fasttext", {}).get("reduced")
    if manifest.get("status") != "passed" or not isinstance(reduced, dict):
        raise RuntimeError("Asset manifest has no passed reduced FastText audit")
    path = asset_config.fasttext.reduced_path
    if Path(str(reduced.get("path"))).resolve(strict=False) != path.resolve(strict=False):
        raise RuntimeError("Configured FastText path differs from the audited path")
    if int(reduced.get("dimension", -1)) != asset_config.fasttext.target_dimension:
        raise RuntimeError("Audited FastText dimension differs from GCN configuration")
    if int(reduced.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError("Reduced FastText file changed after its asset audit")
    print(f"Verifying reduced FastText SHA-256: {path}", flush=True)
    digest = file_sha256(path)
    if digest != reduced.get("sha256"):
        raise RuntimeError("Reduced FastText SHA-256 changed after its asset audit")
    print(f"Loading verified reduced FastText: {path}", flush=True)
    model = fasttext.load_model(str(path))
    if int(model.get_dimension()) != asset_config.fasttext.target_dimension:
        raise RuntimeError("Loaded FastText dimension is incorrect")
    return model, digest


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


def graph_sha256(graph) -> str:
    values = [
        f"{row}:{column}:{weight:.12g}"
        for row, column, weight in zip(
            graph.adjacency.rows,
            graph.adjacency.columns,
            graph.adjacency.values,
            strict=True,
        )
    ]
    return sequence_sha256(values)


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
    partial_path = temporary / "run_records.jsonl"
    if temporary.exists():
        snapshot = temporary / "config_snapshot.json"
        if not snapshot.is_file():
            raise RuntimeError(f"Incomplete staging directory has no snapshot: {temporary}")
        if canonical_json_sha256(json.loads(snapshot.read_text(encoding="utf-8"))) != config_digest:
            raise RuntimeError("Staging directory belongs to a different GCN configuration")
    else:
        temporary.mkdir(exist_ok=False)
        write_json(temporary / "config_snapshot.json", config)

    device = resolve_device(str(config.get("device", "auto")))
    deterministic = bool(config.get("deterministic", True))
    seed = int(config["seed"])
    tokenizer = TextTokenizer.from_config(config["tokenizer"])
    selection = config["feature_selection"]
    graph_config = config["graph"]
    training = config["training"]
    architecture = config["architecture"]
    model_name = str(config["models"][0])
    if config["models"] != ["fasttext_residual_word_gcn"]:
        raise ValueError("GCN runner requires models=[fasttext_residual_word_gcn]")
    if str(training.get("optimizer", "adam")).lower() != "adam":
        raise ValueError("GCN runner supports Adam only")
    if int(architecture["embedding_dimension"]) != assets.fasttext.target_dimension:
        raise ValueError("GCN embedding dimension must match reduced FastText")
    fasttext_model, fasttext_digest = load_verified_fasttext(assets)

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
                development_labels = label_mapping(labels, split["inner_train"])
                final_labels = label_mapping(labels, split["outer_train"])
                if set(development_labels) != set(final_labels):
                    raise RuntimeError("Development and final class sets differ")

                for selector_name in selection["selectors"]:
                    method = str(selector_name)
                    run_id = f"{dataset_id}_fold{fold}_{method}_{model_name}"
                    if run_id in completed:
                        print(f"Skipping completed run={run_id}", flush=True)
                        continue
                    print(
                        f"Running dataset={dataset_id} fold={fold} selector={method} "
                        f"model={model_name} device={device}", flush=True,
                    )
                    selection_start = time.perf_counter()
                    development_vocabulary, development_selector = vocabulary_at_scope(
                        method, tokenized, labels, split["inner_train"],
                        sorted(split["validation"] + split["test"]),
                        int(selection["feature_count"]),
                        int(selection.get("minimum_document_frequency", 1)),
                    )
                    development_selection_seconds = time.perf_counter() - selection_start
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
                        write_selected_features(
                            root / f"{method}_final.csv",
                            final_vocabulary, final_selector,
                        )

                    graph_start = time.perf_counter()
                    development_graph = build_word_graph(
                        tokenized, split["inner_train"], development_vocabulary,
                        int(graph_config["window_size"]),
                    )
                    development_graph_seconds = time.perf_counter() - graph_start
                    graph_start = time.perf_counter()
                    final_graph = build_word_graph(
                        tokenized, split["outer_train"], final_vocabulary,
                        int(graph_config["window_size"]),
                    )
                    final_graph_seconds = time.perf_counter() - graph_start

                    embedding_start = time.perf_counter()
                    development_weights = fasttext_node_matrix(
                        development_vocabulary,
                        fasttext_model,
                        int(architecture["embedding_dimension"]),
                    )
                    final_weights = fasttext_node_matrix(
                        final_vocabulary,
                        fasttext_model,
                        int(architecture["embedding_dimension"]),
                    )
                    embedding_matrix_seconds = time.perf_counter() - embedding_start

                    development_adjacency = sparse_tensor(
                        development_graph.adjacency, device
                    )
                    development_train_documents = sparse_tensor(
                        document_term_matrix(tokenized, split["inner_train"], development_graph),
                        device,
                    )
                    development_validation_documents = sparse_tensor(
                        document_term_matrix(tokenized, split["validation"], development_graph),
                        device,
                    )
                    development_targets = encoded_labels(
                        labels, split["inner_train"], development_labels, device
                    )
                    validation_targets = encoded_labels(
                        labels, split["validation"], development_labels, device
                    )
                    seed_gcn(seed, deterministic)
                    development_model = build_gcn(
                        development_graph.adjacency.shape[0],
                        len(development_labels), architecture, development_weights,
                    )
                    parameter_count = sum(p.numel() for p in development_model.parameters())
                    trainable_parameter_count = sum(
                        p.numel() for p in development_model.parameters() if p.requires_grad
                    )
                    development_model, development_result = train_gcn_with_validation(
                        development_model, development_adjacency,
                        development_train_documents, development_targets,
                        development_validation_documents, validation_targets,
                        training, device,
                    )
                    del development_model, development_adjacency
                    del development_train_documents, development_validation_documents
                    del development_targets, validation_targets
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    final_adjacency = sparse_tensor(final_graph.adjacency, device)
                    final_train_documents = sparse_tensor(
                        document_term_matrix(tokenized, split["outer_train"], final_graph), device
                    )
                    test_documents = sparse_tensor(
                        document_term_matrix(tokenized, split["test"], final_graph), device
                    )
                    final_targets = encoded_labels(
                        labels, split["outer_train"], final_labels, device
                    )
                    test_targets = encoded_labels(labels, split["test"], final_labels, device)
                    seed_gcn(seed, deterministic)
                    final_model = build_gcn(
                        final_graph.adjacency.shape[0], len(final_labels), architecture,
                        final_weights,
                    )
                    final_model, final_training_seconds, final_history = train_gcn_fixed_epochs(
                        final_model, final_adjacency, final_train_documents, final_targets,
                        development_result.best_epoch, training, device,
                    )
                    scores, truth, prediction, confidence, inference_seconds = evaluate_gcn(
                        final_model, final_adjacency, test_documents, test_targets, device
                    )
                    index_to_label = {value: key for key, value in final_labels.items()}
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
                                "development": gcn_training_result_dict(development_result),
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
                        "embedding_source": "fasttext_cc_en_100_frozen_with_graph_residual",
                        "development_feature_count": len(development_vocabulary),
                        "final_feature_count": len(final_vocabulary),
                        "development_graph_nodes": development_graph.adjacency.shape[0],
                        "final_graph_nodes": final_graph.adjacency.shape[0],
                        "development_positive_pmi_pairs": development_graph.positive_pmi_pairs,
                        "final_positive_pmi_pairs": final_graph.positive_pmi_pairs,
                        "development_windows": development_graph.windows,
                        "final_windows": final_graph.windows,
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
                        "development_graph_seconds": development_graph_seconds,
                        "final_graph_seconds": final_graph_seconds,
                        "embedding_matrix_seconds": embedding_matrix_seconds,
                        "development_training_seconds": development_result.training_seconds,
                        "final_training_seconds": final_training_seconds,
                        "test_inference_seconds": inference_seconds,
                        "inner_train_indices_sha256": sequence_sha256(split["inner_train"]),
                        "validation_indices_sha256": sequence_sha256(split["validation"]),
                        "outer_train_indices_sha256": sequence_sha256(split["outer_train"]),
                        "test_indices_sha256": sequence_sha256(split["test"]),
                        "development_vocabulary_sha256": sequence_sha256(development_vocabulary),
                        "final_vocabulary_sha256": sequence_sha256(final_vocabulary),
                        "development_graph_sha256": graph_sha256(development_graph),
                        "final_graph_sha256": graph_sha256(final_graph),
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
                        f"test_acc={scores['accuracy']:.4f}", flush=True,
                    )
                    del final_model, final_adjacency, final_train_documents, test_documents
                    del final_targets, test_targets, development_graph, final_graph
                    del development_weights, final_weights
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        if not results:
            raise RuntimeError("Configuration produced no GCN runs")
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
                "device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
                ),
            },
            "protocol": {
                "graph_type": "inductive_training_only_word_graph",
                "word_word_edges": "positive_PMI_from_training_windows_only",
                "development_graph_fit": "inner_train_only",
                "final_graph_fit": "outer_train_only",
                "test_documents_in_training_graph": False,
                "development_selector_fit": "inner_train_only",
                "final_selector_fit": "outer_train_only",
                "epoch_selection": "inner_validation_only",
                "final_training_epochs": "development_best_epoch",
                "test_use": "one_final_evaluation_per_configuration",
                "cuda_sparse_determinism": "fixed_seed_with_warn_only_for_unsupported_ops",
                "node_features": "audited_FastText_100d_frozen",
                "graph_encoder": "two_layer_positive_PMI_GCN_residual",
            },
            "pretrained_assets": {
                "fasttext_reduced_path": str(assets.fasttext.reduced_path),
                "fasttext_reduced_sha256": fasttext_digest,
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
