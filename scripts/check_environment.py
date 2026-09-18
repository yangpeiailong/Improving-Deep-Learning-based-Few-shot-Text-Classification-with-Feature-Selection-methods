#!/usr/bin/env python
"""Audit the pinned scientific, CUDA, cuDNN, LSTM, and BERT stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from fs_lrtc.config import load_paths, load_yaml
from fs_lrtc.environment import base_environment, compare_versions, package_versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    expected = config["expected"]
    expected_packages = {str(k): str(v) for k, v in config["packages"].items()}
    report: dict[str, Any] = {
        "schema_version": 1,
        "environment_id": config["environment_id"],
        "status": "running",
        "base": base_environment(),
        "packages": package_versions(expected_packages),
        "checks": [],
        "errors": [],
    }

    _check(report, "python_version", lambda: _require(
        report["base"]["python"].startswith(expected["python_major_minor"] + "."),
        f"expected Python {expected['python_major_minor']}.x, observed {report['base']['python']}",
    ))
    _check(report, "package_versions", lambda: _require_no_items(
        compare_versions(report["packages"], expected_packages)
    ))
    _check(report, "pip_check", lambda: _pip_check(report))
    _check(report, "numpy_sklearn", lambda: _numpy_sklearn_test(report))
    _check(report, "cuda_and_cudnn", lambda: _cuda_test(report, expected, config["tests"]))
    _check(report, "tiny_bert_backward", lambda: _bert_test(report, config["tests"]))
    report["nvidia_smi"] = _nvidia_smi()
    report["status"] = "passed" if not report["errors"] else "failed"

    output = paths["audit_root"] / "environment_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for check in report["checks"]:
        print(f"{check['name']}: {check['status']}")
    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"- {error['check']}: {error['message']}")
    print(f"Environment status: {report['status']}")
    print(f"Report: {output}")
    return 0 if report["status"] == "passed" else 1


def _check(report: dict[str, Any], name: str, operation: Callable[[], None]) -> None:
    try:
        operation()
    except Exception as exc:  # report every independent environment failure
        report["checks"].append({"name": name, "status": "failed"})
        report["errors"].append(
            {"check": name, "message": str(exc), "traceback": traceback.format_exc()}
        )
    else:
        report["checks"].append({"name": name, "status": "passed"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_no_items(items: list[str]) -> None:
    if items:
        raise RuntimeError("; ".join(items))


def _pip_check(report: dict[str, Any]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    report["pip_check_output"] = (result.stdout + result.stderr).strip()
    _require(result.returncode == 0, report["pip_check_output"] or "pip check failed")


def _numpy_sklearn_test(report: dict[str, Any]) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0).fit(features, labels)
    accuracy = float(model.score(features, labels))
    _require(accuracy == 1.0, f"unexpected scikit-learn smoke accuracy: {accuracy}")
    report["numpy_sklearn"] = {"smoke_accuracy": accuracy}


def _cuda_test(report: dict[str, Any], expected: dict[str, Any], tests: dict[str, Any]) -> None:
    import torch

    _require(torch.__version__ == expected["torch"], f"unexpected torch: {torch.__version__}")
    _require(torch.version.cuda == expected["torch_cuda"], f"unexpected CUDA runtime: {torch.version.cuda}")
    _require(torch.cuda.is_available() is bool(expected["cuda_available"]), "CUDA availability mismatch")
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    architectures = torch.cuda.get_arch_list()
    _require(expected["device_name_contains"] in name, f"unexpected CUDA device: {name}")
    _require(capability == expected["device_capability"], f"unexpected capability: {capability}")
    _require(expected["required_cuda_architecture"] in architectures, "required sm_61 is absent")

    size = int(tests["matrix_size"])
    matrix = torch.randn(size, size, device="cuda", requires_grad=True)
    (matrix @ matrix).mean().backward()
    lstm = torch.nn.LSTM(64, 128, batch_first=True).cuda()
    values = torch.randn(
        int(tests["lstm_batch_size"]), int(tests["lstm_sequence_length"]), 64,
        device="cuda", requires_grad=True,
    )
    output, _ = lstm(values)
    output.mean().backward()
    free, total = torch.cuda.mem_get_info()
    report["cuda"] = {
        "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": name,
        "capability": capability,
        "architectures": architectures,
        "memory_free_mib": round(free / 1024**2, 2),
        "memory_total_mib": round(total / 1024**2, 2),
    }
    del matrix, lstm, values, output
    torch.cuda.empty_cache()


def _bert_test(report: dict[str, Any], tests: dict[str, Any]) -> None:
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    configuration = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=3,
    )
    model = BertForSequenceClassification(configuration).cuda().train()
    sequence_length = int(tests["bert_sequence_length"])
    input_ids = torch.randint(0, 128, (2, sequence_length), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([0, 2], device="cuda")
    result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    result.loss.backward()
    _require(torch.isfinite(result.loss).item(), "tiny BERT produced non-finite loss")
    report["tiny_bert"] = {"loss": float(result.loss.detach().cpu())}
    del model, input_ids, attention_mask, labels, result
    torch.cuda.empty_cache()


def _nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
