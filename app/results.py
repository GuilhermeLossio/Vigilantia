"""Load and validate the published benchmark results file (schema version 1)."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

SCHEMA_VERSION = 1
RUN_STATUSES = {"complete", "partial"}
KERNEL_SOURCES = {
    "classical": "Classical model",
    "simulation": "Quantum kernel: exact simulation",
    "qpu": "Quantum kernel: IBM Quantum hardware, uncorrected",
    "qpu_calibrated": "Quantum kernel: IBM Quantum hardware, calibrated",
}
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision")
RUNTIME_KEYS = ("fit", "test_kernel", "predict")
QPU_USAGE_KEYS = ("execution_seconds", "charged_seconds", "resource_units")


class ResultsUnavailable(Exception):
    """No results file has been published yet."""


class ResultsInvalid(Exception):
    """A results file exists but does not satisfy the published contract."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ResultsInvalid(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_duration(value: object) -> bool:
    return (
        value is None
        or isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )


def _validate_runtime(results: dict) -> None:
    """Runtimes are optional; rows that report them need the top-level method."""
    timed = [model for model in results["models"] if "runtime_seconds" in model]
    for model in timed:
        seconds = model["runtime_seconds"]
        _require(
            isinstance(seconds, dict)
            and set(seconds) == set(RUNTIME_KEYS)
            and all(_is_duration(value) for value in seconds.values()),
            f"runtime_seconds of {model['name']} must hold {list(RUNTIME_KEYS)} as null or "
            "non-negative numbers.",
        )
    runtime = results.get("runtime")
    if runtime is None:
        _require(not timed, "Models with runtime_seconds require a runtime object.")
        return
    _require(isinstance(runtime, dict), "runtime must be an object.")
    for key in ("method", "environment"):
        _require(_is_text(runtime.get(key)), f"runtime.{key} must be a non-empty string.")
    qpu = runtime.get("qpu")
    _require(
        qpu is None
        or isinstance(qpu, dict)
        and set(qpu) == set(QPU_USAGE_KEYS)
        and all(_is_duration(value) for value in qpu.values()),
        f"runtime.qpu must be null or hold {list(QPU_USAGE_KEYS)} as null or non-negative numbers.",
    )


def _validate_model(index: int, model: object, test_size: int) -> None:
    where = f"models[{index}]"
    _require(isinstance(model, dict), f"{where} must be an object.")
    _require(_is_text(model.get("name")), f"{where}.name must be a non-empty string.")
    _require(
        model.get("kernel_source") in KERNEL_SOURCES,
        f"{where}.kernel_source must be one of {sorted(KERNEL_SOURCES)}.",
    )
    metrics = model.get("metrics")
    _require(isinstance(metrics, dict), f"{where}.metrics must be an object.")
    _require(set(metrics) == set(METRICS), f"{where}.metrics must contain exactly {list(METRICS)}.")
    for name, value in metrics.items():
        if value is None:
            continue
        _require(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isfinite(value)
            and 0 <= value <= 1,
            f"{where}.metrics.{name} must be null or a number in [0, 1].",
        )
    matrix = model.get("confusion_matrix")
    _require(
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        and all(_is_int(cell) and cell >= 0 for row in matrix for cell in row),
        f"{where}.confusion_matrix must be [[TN, FP], [FN, TP]] with non-negative integers.",
    )
    _require(
        sum(sum(row) for row in matrix) == test_size,
        f"{where}.confusion_matrix must sum to split.test ({test_size}).",
    )


def validate(results: object) -> dict:
    """Return the results unchanged when they satisfy schema version 1."""
    _require(isinstance(results, dict), "Results must be a JSON object.")
    _require(
        results.get("schema_version") == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION}.",
    )
    for key in ("run_id", "generated_at_utc"):
        _require(_is_text(results.get(key)), f"{key} must be a non-empty string.")
    _require(
        results.get("status") in RUN_STATUSES,
        f"status must be one of {sorted(RUN_STATUSES)}.",
    )

    split = results.get("split")
    _require(isinstance(split, dict), "split must be an object.")
    for key in ("train", "test", "seed"):
        _require(_is_int(split.get(key)), f"split.{key} must be an integer.")
    _require(split["train"] > 0 and split["test"] > 0, "split sizes must be positive.")

    hardware = results.get("hardware")
    _require(isinstance(hardware, dict), "hardware must be an object.")
    backend = hardware.get("backend")
    _require(backend is None or _is_text(backend), "hardware.backend must be a string or null.")
    shots = hardware.get("shots")
    _require(
        shots is None or (_is_int(shots) and shots > 0),
        "hardware.shots must be a positive integer or null.",
    )
    job_ids = hardware.get("job_ids")
    _require(
        isinstance(job_ids, list) and all(_is_text(job) for job in job_ids),
        "hardware.job_ids must be a list of non-empty strings.",
    )

    models = results.get("models")
    _require(isinstance(models, list) and models, "models must be a non-empty list.")
    for index, model in enumerate(models):
        _validate_model(index, model, split["test"])
    uses_qpu = any(model["kernel_source"] in {"qpu", "qpu_calibrated"} for model in models)
    _require(
        not uses_qpu or (backend and shots and job_ids),
        "QPU results require hardware.backend, hardware.shots, and at least one job ID.",
    )

    _validate_runtime(results)

    limitations = results.get("limitations")
    _require(
        isinstance(limitations, list) and all(_is_text(item) for item in limitations),
        "limitations must be a list of non-empty strings.",
    )
    return results


def load(path: Path) -> dict:
    """Read and validate a results file; distinguish absent from invalid results."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ResultsUnavailable("Benchmark results have not been published yet.") from error
    try:
        results = json.loads(text)
    except json.JSONDecodeError as error:
        raise ResultsInvalid(f"Results file is not valid JSON: {error.msg}.") from error
    return validate(results)
