"""Training utilities for deterministic classic neural smoke experiments."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from fs_lrtc.data.sequences import (
    PAD_INDEX,
    UNK_INDEX,
    TextSequenceDataset,
    collate_sequences,
)
from fs_lrtc.models.classic import (
    BiLSTMClassifier,
    TextCNNClassifier,
    WordAveragingClassifier,
)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    training_loss: float
    training_accuracy: float
    mean_gradient_norm: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    validation_balanced_accuracy: float
    epoch_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_accuracy: float
    best_macro_f1: float
    best_balanced_accuracy: float
    epochs_completed: int
    training_seconds: float
    history: list[EpochResult]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def label_mapping(labels: Sequence[str], fit_indices: Sequence[int]) -> dict[str, int]:
    classes = sorted({labels[index] for index in fit_indices})
    if len(classes) < 2:
        raise ValueError("Training scope must contain at least two classes")
    return {label: index for index, label in enumerate(classes)}


def fasttext_embedding_matrix(
    vocabulary: dict[str, int], fasttext_model: Any, dimension: int
) -> torch.Tensor:
    if int(fasttext_model.get_dimension()) != dimension:
        raise ValueError("FastText model dimension does not match experiment configuration")
    matrix = np.zeros((len(vocabulary), dimension), dtype=np.float32)
    for token, index in vocabulary.items():
        if index in (PAD_INDEX, UNK_INDEX):
            continue
        vector = np.asarray(fasttext_model.get_word_vector(token), dtype=np.float32)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid FastText vector for token: {token}")
        matrix[index] = vector
    return torch.from_numpy(matrix)


def model_family(model_name: str) -> str:
    for family in ("wa", "cnn", "lstm"):
        if model_name.endswith(f"_{family}"):
            return family
    raise ValueError(f"Unsupported model name: {model_name}")


def uses_fasttext(model_name: str) -> bool:
    if model_name.startswith("fasttext_"):
        return True
    if model_name.startswith("random_"):
        return False
    raise ValueError(f"Unsupported embedding prefix: {model_name}")


def build_model(
    model_name: str,
    vocabulary_size: int,
    class_count: int,
    architecture: dict[str, Any],
    pretrained_weights: torch.Tensor | None,
) -> nn.Module:
    common = {
        "vocabulary_size": vocabulary_size,
        "embedding_dimension": int(architecture["embedding_dimension"]),
        "class_count": class_count,
        "dropout": float(architecture["dropout"]),
        "pretrained_weights": pretrained_weights,
        "embeddings_trainable": bool(architecture.get("embeddings_trainable", True)),
    }
    family = model_family(model_name)
    if family == "wa":
        return WordAveragingClassifier(**common)
    if family == "cnn":
        return TextCNNClassifier(
            **common,
            filter_count=int(architecture["cnn_filters"]),
            kernel_sizes=architecture["cnn_kernel_sizes"],
        )
    return BiLSTMClassifier(
        **common,
        hidden_size=int(architecture["lstm_hidden_size"]),
        classifier_hidden_size=int(architecture["lstm_classifier_hidden_size"]),
    )


def make_loader(
    dataset: TextSequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    minimum_padded_length: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        collate_fn=partial(
            collate_sequences, minimum_padded_length=minimum_padded_length
        ),
        pin_memory=torch.cuda.is_available(),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], list[int], list[int], list[int], list[float], float]:
    model.eval()
    truth: list[int] = []
    predictions: list[int] = []
    indices: list[int] = []
    confidences: list[float] = []
    total_loss = 0.0
    sample_count = 0
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            logits = model(input_ids, lengths)
            labels = batch["labels"].to(device, non_blocking=True)
            total_loss += float(
                nn.functional.cross_entropy(logits, labels, reduction="sum").detach()
            )
            sample_count += labels.size(0)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            truth.extend(batch["labels"].tolist())
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            indices.extend(batch["record_indices"].tolist())
    synchronize(device)
    seconds = time.perf_counter() - start
    scores = {
        "loss": total_loss / sample_count,
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
    }
    return scores, truth, predictions, indices, confidences, seconds


def train_with_validation(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, TrainingResult]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    maximum_epochs = int(config["maximum_epochs"])
    minimum_epochs = int(config["minimum_epochs"])
    patience = int(config["early_stopping_patience"])
    min_delta = float(config["early_stopping_min_delta"])
    clip = float(config["gradient_clip_norm"])
    best_key = (-1.0, -1.0, -1.0)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[EpochResult] = []
    synchronize(device)
    total_start = time.perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        sample_count = 0
        correct_count = 0
        gradient_norm_total = 0.0
        update_count = 0
        for batch in training_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += float(loss.detach()) * labels.size(0)
            sample_count += labels.size(0)
            correct_count += int(
                (logits.argmax(dim=1) == labels).sum().detach().item()
            )
            gradient_norm_total += float(gradient_norm.detach().item())
            update_count += 1
        scores, _, _, _, _, _ = evaluate(model, validation_loader, device)
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            EpochResult(
                epoch=epoch,
                training_loss=total_loss / sample_count,
                training_accuracy=correct_count / sample_count,
                mean_gradient_norm=gradient_norm_total / update_count,
                validation_loss=scores["loss"],
                validation_accuracy=scores["accuracy"],
                validation_macro_f1=scores["macro_f1"],
                validation_balanced_accuracy=scores["balanced_accuracy"],
                epoch_seconds=epoch_seconds,
            )
        )
        print(
            f"    development epoch={epoch} train_loss={history[-1].training_loss:.4f} "
            f"train_acc={history[-1].training_accuracy:.4f} "
            f"val_loss={scores['loss']:.4f} val_acc={scores['accuracy']:.4f} "
            f"grad_norm={history[-1].mean_gradient_norm:.4f}",
            flush=True,
        )
        key = (scores["accuracy"], scores["macro_f1"], scores["balanced_accuracy"])
        accuracy_improved = key[0] > best_key[0] + min_delta
        tie_improved = abs(key[0] - best_key[0]) <= min_delta and key[1:] > best_key[1:]
        if accuracy_improved or tie_improved:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= minimum_epochs and stale_epochs >= patience:
            break

    synchronize(device)
    training_seconds = time.perf_counter() - total_start
    if best_state is None:
        raise RuntimeError("Training completed without a selectable validation epoch")
    model.load_state_dict(best_state)
    return model, TrainingResult(
        best_epoch=best_epoch,
        best_accuracy=best_key[0],
        best_macro_f1=best_key[1],
        best_balanced_accuracy=best_key[2],
        epochs_completed=len(history),
        training_seconds=training_seconds,
        history=history,
    )


def train_fixed_epochs(
    model: nn.Module,
    training_loader: DataLoader,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, float, list[dict[str, float]]]:
    if epochs < 1:
        raise ValueError("Final epoch count must be positive")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    clip = float(config["gradient_clip_norm"])
    history: list[dict[str, float]] = []
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        sample_count = 0
        correct_count = 0
        gradient_norm_total = 0.0
        update_count = 0
        for batch in training_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += float(loss.detach()) * labels.size(0)
            sample_count += labels.size(0)
            correct_count += int(
                (logits.argmax(dim=1) == labels).sum().detach().item()
            )
            gradient_norm_total += float(gradient_norm.detach().item())
            update_count += 1
        synchronize(device)
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / sample_count,
                "training_accuracy": correct_count / sample_count,
                "mean_gradient_norm": gradient_norm_total / update_count,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        print(
            f"    final epoch={epoch}/{epochs} "
            f"train_loss={history[-1]['training_loss']:.4f} "
            f"train_acc={history[-1]['training_accuracy']:.4f} "
            f"grad_norm={history[-1]['mean_gradient_norm']:.4f}",
            flush=True,
        )
    synchronize(device)
    return model, time.perf_counter() - start, history


def training_result_dict(result: TrainingResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["history"] = [asdict(item) for item in result.history]
    return payload
