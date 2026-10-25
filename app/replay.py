"""Load and validate the replay file of recorded test predictions (schema version 1)."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

from app.results import ResultsInvalid, ResultsUnavailable

SCHEMA_VERSION = 1
SOURCES = ("simulation", "qpu", "qpu_calibrated")
LABELS = {0: "legitimate", 1: "phishing"}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ResultsInvalid(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate(data: object) -> dict:
    _require(isinstance(data, dict), "Replay must be a JSON object.")
    _require(data.get("schema_version") == SCHEMA_VERSION, "Replay schema_version must be 1.")
    _require(
        isinstance(data.get("label"), str) and "replay" in data["label"], "Replay label missing."
    )
    runs = data.get("source_runs")
    _require(
        isinstance(runs, dict) and isinstance(runs.get("simulation"), str), "source_runs missing."
    )
    samples = data.get("samples")
    _require(isinstance(samples, list) and samples, "samples must be a non-empty list.")
    for index, sample in enumerate(samples):
        where = f"samples[{index}]"
        _require(isinstance(sample, dict), f"{where} must be an object.")
        for key in ("position", "row_index"):
            _require(_is_int(sample.get(key)), f"{where}.{key} must be an integer.")
        _require(isinstance(sample.get("group_hash"), str), f"{where}.group_hash must be a string.")
        _require(sample.get("is_phishing") in LABELS, f"{where}.is_phishing must be 0 or 1.")
        predictions = sample.get("predictions")
        _require(
            isinstance(predictions, dict) and predictions and set(predictions) <= set(SOURCES),
            f"{where}.predictions must use sources {list(SOURCES)}.",
        )
        _require(
            ("qpu" in predictions or "qpu_calibrated" in predictions) == ("qpu" in runs),
            f"{where} QPU predictions require a recorded QPU run, and vice versa.",
        )
        for source, prediction in predictions.items():
            label = prediction.get("is_phishing") if isinstance(prediction, dict) else None
            _require(label in LABELS, f"{where}.{source}.is_phishing must be 0 or 1.")
            _require(
                prediction.get("prediction") == LABELS[label], f"{where}.{source} label mismatch."
            )
            score = prediction.get("decision_score")
            _require(
                score is None
                or (
                    isinstance(score, int | float)
                    and not isinstance(score, bool)
                    and isfinite(score)
                ),
                f"{where}.{source}.decision_score must be a finite number or null.",
            )
    return data


def load(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ResultsUnavailable("Replay predictions have not been published yet.") from error
    try:
        return validate(json.loads(text))
    except json.JSONDecodeError as error:
        raise ResultsInvalid(f"Replay file is not valid JSON: {error.msg}.") from error
