"""Deterministic partitions, operating points, metrics and exact four-qubit states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from src.data.benchmark_split import group_hash, prepare
from src.features.host_features import CANDIDATE_FEATURES

# PhiUSIIL page-content columns; URL-derived columns are excluded because they encode the
# collection process (scheme, path, `www.` prefix) or, like URLSimilarityIndex, the label.
PAGE_CANDIDATES = (
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "DomainTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
)
# Top four of the development-only permutation ranking recorded by the prepare stage.
PAGE_FEATURES = ("LineOfCode", "NoOfSelfRef", "LargestLineLength", "NoOfImage")
FEATURE_SETS = {"host": CANDIDATE_FEATURES, "page": PAGE_FEATURES}


def matrix(rows: pd.DataFrame, features: str) -> np.ndarray:
    """Page counts are heavy-tailed, so they enter every model on a log1p scale."""
    values = rows[list(FEATURE_SETS[features])].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"Rows lack finite {features} features.")
    return np.log1p(values) if features == "page" else values


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomic checkpoints; non-finite JSON numbers are always a bug."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def partitions(raw: pd.DataFrame, excluded: set[str], seed: int) -> tuple[dict, dict]:
    prepared = prepare(raw)
    rows = prepared.rows.sort_values("row_index").drop_duplicates("group").copy()
    rows["group_hash"] = rows["group"].map(group_hash)
    # Remove historical test groups before drawing any v2 partition, so moving them
    # cannot silently change the declared 60/20/20 proportions.
    removed = int(rows["group_hash"].isin(excluded).sum())
    rows = rows[~rows["group_hash"].isin(excluded)].reset_index(drop=True)
    dev, held = train_test_split(rows, test_size=0.4, stratify=rows.is_phishing, random_state=seed)
    calibration, test = train_test_split(
        held, test_size=0.5, stratify=held.is_phishing, random_state=seed
    )
    result = {
        name: table.sort_values("row_index").reset_index(drop=True)
        for name, table in (("development", dev), ("calibration", calibration), ("test", test))
    }
    return result, {**prepared.exclusions, "historical_test_groups": removed}


def nested_samples(development: pd.DataFrame, sizes: list[int], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    classes = [rng.permutation(development.index[development.is_phishing == c]) for c in (0, 1)]
    if any(n < 6 or n % 2 for n in sizes) or max(sizes) // 2 > min(map(len, classes)):
        raise ValueError("Training sizes must be even, >= 6, and fit both development classes.")
    return {n: np.ravel(np.column_stack([c[: n // 2] for c in classes])) for n in sizes}


def operating_point(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> dict:
    """Scores >= threshold are positive; null threshold means predict no positives."""
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Calibration needs both classes and finite scores.")
    fpr, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    feasible = np.flatnonzero(fpr <= max_fpr)
    best = min(feasible, key=lambda i: (-recall[i], fpr[i], -thresholds[i]))
    return {
        "threshold": None if np.isinf(thresholds[best]) else float(thresholds[best]),
        "recall": float(recall[best]),
        "false_positive_rate": float(fpr[best]),
    }


def predict(scores: np.ndarray, threshold: float | None) -> np.ndarray:
    return (
        np.zeros(len(scores), dtype=int) if threshold is None else (scores >= threshold).astype(int)
    )


def metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "accuracy": (tn + tp) / len(labels),
        "balanced_accuracy": (recall + 1 - fpr) / 2,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "average_precision": float(average_precision_score(labels, scores))
        if (labels == 1).any()
        else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "rows": len(labels),
    }


def angles(values: np.ndarray, low: np.ndarray, high: np.ndarray, maximum: float) -> tuple:
    span = high - low
    scaled = np.zeros_like(values, dtype=float)
    varying = span != 0
    scaled[:, varying] = (values[:, varying] - low[varying]) / span[varying]
    outside = (scaled < 0) | (scaled > 1)
    return np.clip(scaled, 0, 1) * maximum, {
        "per_feature": outside.sum(axis=0).astype(int).tolist(),
        "rows_with_any_clipped": int(outside.any(axis=1).sum()),
        "constant_features": np.flatnonzero(~varying).tolist(),
        "constant_feature_shift_rows": int(((values != low) & ~varying).any(axis=1).sum()),
    }


def states(encoded: np.ndarray, encoding: str) -> np.ndarray:
    """Exact vectorization of Qiskit Z/ZZ maps, reps=1, alpha=2, little endian.

    H gates create uniform amplitudes; P gates add 2*x_j on bit j, and
    CX-P-CX adds 2*(pi-x_j)*(pi-x_k) on the XOR of neighboring bits.
    Tests compare these states and cross-kernels with Qiskit Statevector.
    """
    if encoding not in {"z", "zz"} or encoded.ndim != 2 or encoded.shape[1] != 4:
        raise ValueError("Expected a z/zz map and four encoded columns.")
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
    phases = 2 * encoded @ bits.T
    if encoding == "zz":
        for j in range(3):
            parity = np.abs(bits[:, j] - bits[:, j + 1])
            phases += 2 * ((np.pi - encoded[:, j]) * (np.pi - encoded[:, j + 1]))[:, None] * parity
    return np.exp(1j * phases) / 4


def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left.conj() @ right.T) ** 2


def kernel_diagnostics(matrix: np.ndarray, labels: np.ndarray) -> dict:
    off = ~np.eye(len(matrix), dtype=bool)
    same = labels[:, None] == labels[None, :]
    return {
        "off_diagonal_mean": float(matrix[off].mean()),
        "off_diagonal_std": float(matrix[off].std()),
        "same_class_mean": float(matrix[off & same].mean()),
        "different_class_mean": float(matrix[~same].mean()),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
    }
