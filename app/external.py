"""Load and validate the published external-validation report (schema version 1).

The report is produced by scripts/evaluate_external.py, whose validator lives in
src/evaluation/external.py. That module needs pandas and the training code, which the
service image does not ship, so the service checks the same contract here with the
standard library only.
"""

from __future__ import annotations

import json
import re
from math import isfinite
from pathlib import Path

from app.results import ResultsInvalid, ResultsUnavailable

SCHEMA_VERSION = 1
STEPS = {
    "invalid_host": "No valid host",
    "conflicting_labels": "Domain with conflicting labels",
    "phiusiil_overlap": "Domain also in PhiUSIIL",
    "duplicate_group": "Duplicate row of a kept domain",
}
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision")
MODELS = {
    "majority": ("Majority baseline", "classical"),
    "logistic_regression": ("Logistic regression", "classical"),
    "rbf_svm": ("RBF SVM", "classical"),
    "quantum_kernel_svm": ("Quantum-kernel SVM", "simulation"),
}
SOURCES = {"classical": "Classical model", "simulation": "Quantum kernel: exact simulation"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_TEXT = re.compile(r"://|[A-Za-z]:[\\/]|\\Users\\|/Users/|/home/|AppData", re.IGNORECASE)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ResultsInvalid(message)


def _is_int(value: object, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_unit(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
        and 0 <= value <= 1
    )


def _strings(value: object):
    """Yield every string in a JSON value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _validate_sources(report: dict) -> None:
    protocol = report.get("protocol")
    _require(
        isinstance(protocol, dict)
        and _is_text(protocol.get("path"))
        and isinstance(protocol.get("sha256"), str)
        and _SHA256.fullmatch(protocol["sha256"])
        and _is_text(protocol.get("frozen_at_utc")),
        "protocol must hold path, a SHA-256 hex digest, and frozen_at_utc.",
    )
    dataset = report.get("dataset")
    _require(
        isinstance(dataset, dict)
        and all(_is_text(dataset.get(key)) for key in ("title", "doi", "file", "sha256", "license"))
        and _is_int(dataset.get("rows"), 1),
        "dataset must hold title, doi, file, sha256, license, and a positive row count.",
    )
    benchmark = report.get("benchmark")
    _require(
        isinstance(benchmark, dict)
        and _is_text(benchmark.get("model_version"))
        and _is_int(benchmark.get("training_examples"), 1),
        "benchmark must hold model_version and a positive training_examples count.",
    )


def _validate_preparation(preparation: object) -> int:
    """Check the exclusion funnel and return the number of primary rows."""
    _require(isinstance(preparation, dict), "preparation must be an object.")
    exclusions = preparation.get("exclusions")
    _require(
        isinstance(exclusions, list)
        and [entry.get("step") if isinstance(entry, dict) else None for entry in exclusions]
        == list(STEPS),
        f"preparation.exclusions must list the steps {list(STEPS)} in order.",
    )
    _require(
        all(_is_int(entry.get("rows")) for entry in exclusions),
        "Each exclusion must count rows as a non-negative integer.",
    )
    primary = preparation.get("primary_rows")
    _require(_is_int(primary, 1), "preparation.primary_rows must be a positive integer.")
    _require(
        _is_int(preparation.get("input_rows"))
        and preparation["input_rows"] == primary + sum(entry["rows"] for entry in exclusions),
        "preparation.input_rows must equal the exclusions plus the primary rows.",
    )
    counts = preparation.get("class_counts")
    _require(
        isinstance(counts, dict)
        and set(counts) == {"legitimate", "phishing"}
        and all(_is_int(value) for value in counts.values())
        and sum(counts.values()) == primary,
        "preparation.class_counts must split primary_rows into legitimate and phishing.",
    )
    return primary


def _validate_model(index: int, model: object, primary: int) -> None:
    where = f"models[{index}]"
    _require(isinstance(model, dict), f"{where} must be an object.")
    _require(model.get("model") in MODELS, f"{where}.model must be one of {list(MODELS)}.")
    name, source = MODELS[model["model"]]
    _require(
        model.get("name") == name and model.get("kernel_source") == source,
        f"{where} must be named {name!r} with kernel_source {source!r}.",
    )
    metrics = model.get("metrics")
    intervals = model.get("confidence_intervals")
    _require(
        isinstance(metrics, dict) and set(metrics) == set(METRICS),
        f"{where}.metrics must contain exactly {list(METRICS)}.",
    )
    _require(
        isinstance(intervals, dict) and set(intervals) == set(METRICS),
        f"{where}.confidence_intervals must contain exactly {list(METRICS)}.",
    )
    for metric in METRICS:
        value, interval = metrics[metric], intervals[metric]
        _require(value is None or _is_unit(value), f"{where}.metrics.{metric} must be in [0, 1].")
        _require(
            (value is None) == (interval is None),
            f"{where}.confidence_intervals.{metric} must be null exactly when the metric is.",
        )
        _require(
            interval is None
            or isinstance(interval, list)
            and len(interval) == 2
            and all(_is_unit(bound) for bound in interval)
            and interval[0] <= interval[1],
            f"{where}.confidence_intervals.{metric} must be [low, high] within [0, 1].",
        )
    matrix = model.get("confusion_matrix")
    _require(
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        and all(_is_int(cell) for row in matrix for cell in row)
        and sum(sum(row) for row in matrix) == primary,
        f"{where}.confusion_matrix must be [[TN, FP], [FN, TP]] summing to primary_rows.",
    )


def validate(report: object) -> dict:
    """Return the report unchanged when it satisfies schema version 1."""
    _require(isinstance(report, dict), "Report must be a JSON object.")
    _require(
        report.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}."
    )
    _require(_is_text(report.get("generated_at_utc")), "generated_at_utc must be a string.")
    _validate_sources(report)
    primary = _validate_preparation(report.get("preparation"))

    clipped = report.get("clipped_values")
    _require(
        isinstance(clipped, dict)
        and isinstance(clipped.get("per_feature"), dict)
        and clipped["per_feature"]
        and all(_is_int(v) and v <= primary for v in clipped["per_feature"].values())
        and _is_int(clipped.get("rows_with_any_clipped"))
        and clipped["rows_with_any_clipped"] <= primary,
        "clipped_values must count clipped values per feature and rows with any clipping.",
    )
    bootstrap = report.get("bootstrap")
    _require(
        isinstance(bootstrap, dict)
        and _is_int(bootstrap.get("resamples"), 1)
        and _is_int(bootstrap.get("resamples_used"))
        and bootstrap["resamples_used"] <= bootstrap["resamples"]
        and _is_int(bootstrap.get("seed"))
        and _is_unit(bootstrap.get("level"))
        and _is_text(bootstrap.get("method")),
        "bootstrap must hold method, level, resamples, resamples_used, and seed.",
    )

    models = report.get("models")
    _require(isinstance(models, list), "models must be a list.")
    for index, model in enumerate(models):
        _validate_model(index, model, primary)
    _require(
        sorted(model["model"] for model in models) == sorted(MODELS),
        f"models must report each of {list(MODELS)} exactly once.",
    )
    limitations = report.get("limitations")
    _require(
        isinstance(limitations, list) and limitations and all(_is_text(i) for i in limitations),
        "limitations must be a non-empty list of strings.",
    )
    _require(
        not any(_FORBIDDEN_TEXT.search(text) for text in _strings(report)),
        "The report must not contain URLs or local paths.",
    )
    return report


def load(path: Path) -> dict:
    """Read and validate the report; distinguish absent from invalid."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ResultsUnavailable("External validation has not been published yet.") from error
    try:
        report = json.loads(text)
    except json.JSONDecodeError as error:
        raise ResultsInvalid(f"External validation file is not valid JSON: {error.msg}.") from error
    return validate(report)
