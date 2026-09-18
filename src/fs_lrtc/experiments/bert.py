"""Training utilities for leakage-safe offline BERT experiments."""

from __future__ import annotations

import math
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

from fs_lrtc.data.bert import BertBatchCollator, BertTextDataset
from fs_lrtc.models.bert_heads import BertTextClassifier


@dataclass(frozen=True)
class BertEpochResult:
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
class BertTrainingResult:
    best_epoch: int
    best_accuracy: float
    best_macro_f1: float
    best_balanced_accuracy: float
    epochs_completed: int
    training_seconds: float
    history: list[BertEpochResult]


def seed_bert(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def make_bert_loader(
    dataset: BertTextDataset,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pad_to_multiple_of: int | None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=BertBatchCollator(tokenizer, pad_to_multiple_of),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_bert_model(
    local_model_path: str,
    model_name: str,
    class_count: int,
    architecture: dict[str, Any],
) -> BertTextClassifier:
    from transformers import BertModel

    encoder = BertModel.from_pretrained(
        local_model_path,
        local_files_only=True,
        attn_implementation="eager",
    )
    if bool(architecture.get("gradient_checkpointing", True)):
        encoder.gradient_checkpointing_enable()
    return BertTextClassifier(
        encoder=encoder,
        class_count=class_count,
        head=model_name,
        dropout=float(architecture["dropout"]),
        cnn_filter_count=int(architecture["cnn_filter_count"]),
        cnn_kernel_sizes=architecture["cnn_kernel_sizes"],
        lstm_hidden_size=int(architecture["lstm_hidden_size"]),
        attention_hidden_size=int(architecture["attention_hidden_size"]),
    )


def _optimizer_and_scheduler(
    model: BertTextClassifier,
    config: dict[str, Any],
    optimizer_steps: int,
):
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder_parameters = [p for p in model.parameters() if id(p) in encoder_ids and p.requires_grad]
    head_parameters = [p for p in model.parameters() if id(p) not in encoder_ids and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(config["encoder_learning_rate"])},
            {"params": head_parameters, "lr": float(config["head_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )
    warmup = int(round(optimizer_steps * float(config["warmup_ratio"])))

    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / warmup
        remaining = max(optimizer_steps - warmup, 1)
        return max(float(optimizer_steps - step) / remaining, 0.0)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    labels = batch["labels"].to(device, non_blocking=True)
    inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items() if key != "labels"
    }
    return inputs, labels


def _scores(truth: list[int], predictions: list[int], loss: float) -> dict[str, float]:
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
    }


def evaluate_bert(model, loader, device):
    model.eval()
    total_loss = 0.0
    truth: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs, labels = _move_batch(batch, device)
            logits = model(**inputs)
            total_loss += float(nn.functional.cross_entropy(logits, labels, reduction="sum").item())
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            truth.extend(labels.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    return _scores(truth, predictions, total_loss / len(truth)), truth, predictions, confidences, seconds


def _train_epoch(model, loader, optimizer, scheduler, scaler, config, device):
    model.train()
    accumulation = int(config["gradient_accumulation_steps"])
    clip = float(config["gradient_clip_norm"])
    mixed = bool(config["mixed_precision"]) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct = 0
    count = 0
    gradient_norms: list[float] = []
    for step, batch in enumerate(loader):
        inputs, labels = _move_batch(batch, device)
        remainder = len(loader) % accumulation
        divisor = (
            remainder
            if remainder and step >= len(loader) - remainder
            else accumulation
        )
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed):
            logits = model(**inputs)
            raw_loss = nn.functional.cross_entropy(logits, labels)
            loss = raw_loss / divisor
        scaler.scale(loss).backward()
        total_loss += float(raw_loss.detach().item()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum().detach().item())
        count += len(labels)
        should_step = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), clip)
            norm_value = float(norm.detach().item())
            if not math.isfinite(norm_value) and not mixed:
                raise FloatingPointError(
                    "BERT produced a non-finite gradient norm during FP32 training"
                )
            if math.isfinite(norm_value):
                gradient_norms.append(norm_value)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    mean_gradient_norm = (
        float(np.mean(gradient_norms)) if gradient_norms else float("nan")
    )
    return total_loss / count, correct / count, mean_gradient_norm


def train_bert_with_validation(model, train_loader, validation_loader, config, device):
    model = model.to(device)
    maximum_epochs = int(config["maximum_epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / int(config["gradient_accumulation_steps"]))
    optimizer, scheduler = _optimizer_and_scheduler(model, config, maximum_epochs * steps_per_epoch)
    mixed = bool(config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    best_key = (-1.0, -1.0, -1.0)
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[BertEpochResult] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        epoch_start = time.perf_counter()
        training_loss, training_accuracy, gradient_norm = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, device
        )
        validation, _, _, _, _ = evaluate_bert(model, validation_loader, device)
        history.append(BertEpochResult(
            epoch, training_loss, training_accuracy, gradient_norm,
            validation["loss"], validation["accuracy"], validation["macro_f1"],
            validation["balanced_accuracy"], time.perf_counter() - epoch_start,
        ))
        key = (validation["accuracy"], validation["macro_f1"], validation["balanced_accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"    development epoch={epoch} train_loss={training_loss:.4f} "
            f"train_acc={training_accuracy:.4f} val_acc={validation['accuracy']:.4f}",
            flush=True,
        )
        if epoch >= int(config["minimum_epochs"]) and stale >= int(config["early_stopping_patience"]):
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("BERT training produced no best state")
    model.load_state_dict(best_state)
    return model, BertTrainingResult(
        best_epoch, best_key[0], best_key[1], best_key[2], len(history), seconds, history
    )


def train_bert_fixed_epochs(model, loader, epochs, config, device):
    model = model.to(device)
    steps_per_epoch = math.ceil(len(loader) / int(config["gradient_accumulation_steps"]))
    # Keep the same learning-rate trajectory used during development.  The
    # final refit executes only the selected prefix of that fixed schedule.
    schedule_epochs = int(config["maximum_epochs"])
    optimizer, scheduler = _optimizer_and_scheduler(
        model, config, schedule_epochs * steps_per_epoch
    )
    mixed = bool(config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        loss, accuracy, gradient_norm = _train_epoch(
            model, loader, optimizer, scheduler, scaler, config, device
        )
        history.append({
            "epoch": epoch,
            "training_loss": loss,
            "training_accuracy": accuracy,
            "mean_gradient_norm": gradient_norm,
            "epoch_seconds": time.perf_counter() - epoch_start,
        })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, time.perf_counter() - start, history


def bert_training_result_dict(result: BertTrainingResult) -> dict[str, Any]:
    value = asdict(result)
    value["history"] = [asdict(item) for item in result.history]
    return value
