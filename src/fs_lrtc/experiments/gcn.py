"""Training utilities for the inductive word-graph GCN experiment."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

from fs_lrtc.data.word_graph import SparseMatrixData
from fs_lrtc.models.gcn import InductiveWordGCN


@dataclass(frozen=True)
class GCNEpochResult:
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
class GCNTrainingResult:
    best_epoch: int
    best_accuracy: float
    best_macro_f1: float
    best_balanced_accuracy: float
    epochs_completed: int
    training_seconds: float
    history: list[GCNEpochResult]


def seed_gcn(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # CUDA sparse reductions may not be bitwise deterministic on every driver.
    # warn_only preserves a runnable experiment while recording this limitation.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def sparse_tensor(data: SparseMatrixData, device: torch.device) -> torch.Tensor:
    indices = torch.tensor([data.rows, data.columns], dtype=torch.long)
    values = torch.tensor(data.values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, data.shape).coalesce().to(device)


def encoded_labels(
    labels: Sequence[str], indices: Sequence[int], mapping: dict[str, int], device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [mapping[str(labels[index])] for index in indices], dtype=torch.long, device=device
    )


def build_gcn(
    node_count: int,
    class_count: int,
    architecture: dict[str, Any],
    pretrained_weights: torch.Tensor | None = None,
) -> InductiveWordGCN:
    return InductiveWordGCN(
        node_count=node_count,
        class_count=class_count,
        embedding_dimension=int(architecture["embedding_dimension"]),
        hidden_dimension=int(architecture["hidden_dimension"]),
        output_dimension=int(architecture["output_dimension"]),
        dropout=float(architecture["dropout"]),
        pretrained_weights=pretrained_weights,
        embeddings_trainable=bool(architecture.get("embeddings_trainable", True)),
        residual_scale_initial=float(architecture.get("residual_scale_initial", 0.1)),
    )


def fasttext_node_matrix(
    vocabulary: Sequence[str],
    fasttext_model: Any,
    dimension: int,
) -> torch.Tensor:
    """Align audited FastText vectors with graph nodes; keep unknown node zero."""
    if int(fasttext_model.get_dimension()) != dimension:
        raise ValueError("FastText dimension does not match GCN architecture")
    matrix = np.zeros((len(vocabulary) + 1, dimension), dtype=np.float32)
    for index, token in enumerate(vocabulary):
        vector = np.asarray(fasttext_model.get_word_vector(token), dtype=np.float32)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid FastText vector for graph token: {token}")
        matrix[index] = vector
    return torch.from_numpy(matrix)


def _scores(truth: list[int], prediction: list[int], loss: float) -> dict[str, float]:
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }


def evaluate_gcn(
    model: InductiveWordGCN,
    adjacency: torch.Tensor,
    document_terms: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int], list[float], float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(adjacency, document_terms)
        loss = float(nn.functional.cross_entropy(logits, targets).detach().item())
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    truth_values = targets.detach().cpu().tolist()
    prediction_values = prediction.detach().cpu().tolist()
    scores = _scores(truth_values, prediction_values, loss)
    return scores, truth_values, prediction_values, confidence.cpu().tolist(), seconds


def train_gcn_with_validation(
    model: InductiveWordGCN,
    adjacency: torch.Tensor,
    train_documents: torch.Tensor,
    train_targets: torch.Tensor,
    validation_documents: torch.Tensor,
    validation_targets: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[InductiveWordGCN, GCNTrainingResult]:
    model = model.to(device)
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
    stale = 0
    history: list[GCNEpochResult] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(adjacency, train_documents)
        loss = nn.functional.cross_entropy(logits, train_targets)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_accuracy = float(
            (logits.argmax(dim=1) == train_targets).float().mean().detach().item()
        )
        validation, _, _, _, _ = evaluate_gcn(
            model, adjacency, validation_documents, validation_targets, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(GCNEpochResult(
            epoch=epoch,
            training_loss=float(loss.detach().item()),
            training_accuracy=train_accuracy,
            mean_gradient_norm=float(gradient_norm.detach().item()),
            validation_loss=validation["loss"],
            validation_accuracy=validation["accuracy"],
            validation_macro_f1=validation["macro_f1"],
            validation_balanced_accuracy=validation["balanced_accuracy"],
            epoch_seconds=epoch_seconds,
        ))
        key = (
            validation["accuracy"], validation["macro_f1"],
            validation["balanced_accuracy"],
        )
        improved = key[0] > best_key[0] + min_delta or (
            abs(key[0] - best_key[0]) <= min_delta and key[1:] > best_key[1:]
        )
        if improved:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"    development epoch={epoch} train_loss={loss.item():.4f} "
            f"train_acc={train_accuracy:.4f} val_acc={validation['accuracy']:.4f}",
            flush=True,
        )
        if epoch >= minimum_epochs and stale >= patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - total_start
    if best_state is None:
        raise RuntimeError("GCN training did not produce a best state")
    model.load_state_dict(best_state)
    return model, GCNTrainingResult(
        best_epoch=best_epoch,
        best_accuracy=best_key[0],
        best_macro_f1=best_key[1],
        best_balanced_accuracy=best_key[2],
        epochs_completed=len(history),
        training_seconds=total_seconds,
        history=history,
    )


def train_gcn_fixed_epochs(
    model: InductiveWordGCN,
    adjacency: torch.Tensor,
    documents: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[InductiveWordGCN, float, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    clip = float(config["gradient_clip_norm"])
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(adjacency, documents)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        accuracy = float(
            (logits.argmax(dim=1) == targets).float().mean().detach().item()
        )
        history.append({
            "epoch": epoch,
            "training_loss": float(loss.detach().item()),
            "training_accuracy": accuracy,
            "mean_gradient_norm": float(gradient_norm.detach().item()),
            "epoch_seconds": time.perf_counter() - epoch_start,
        })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, time.perf_counter() - start, history


def gcn_training_result_dict(result: GCNTrainingResult) -> dict[str, Any]:
    value = asdict(result)
    value["history"] = [asdict(item) for item in result.history]
    return value
