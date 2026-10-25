"""Test-set metrics defined in docs/benchmark-protocol.md (phishing is the positive class)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision")


def evaluate(model, inputs: np.ndarray, labels: np.ndarray) -> dict:
    """Predict with the model's default decision rule and compute every reported metric.

    Average precision uses continuous decision scores. A model without them, such as the
    majority baseline, reports it as null rather than as a ranking of constant scores.
    """
    started = time.perf_counter()
    predictions = model.predict(inputs)
    scores = model.decision_function(inputs) if hasattr(model, "decision_function") else None
    elapsed = time.perf_counter() - started
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": None
        if scores is None
        else float(average_precision_score(labels, scores)),
    }
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "predictions": [int(value) for value in predictions],
        "decision_scores": None if scores is None else [float(value) for value in scores],
        "predict_seconds": elapsed,
    }
