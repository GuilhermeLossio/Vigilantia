"""External validation of the frozen benchmark models (configs/external-validation.json).

- `prepare` applies the four frozen preparation steps in order and counts each exclusion.
- `featurize` computes the host features in the scaler's order and scales them unchanged.
- `bootstrap_intervals` gives percentile confidence intervals over rows.
- `build_report` and `validate` produce and check reports/external-validation.json, which
  holds aggregates only: no host, URL, or local path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.benchmark_split import group_key
from src.evaluation.publish import DISPLAY as MODELS
from src.features.host_features import host_features, normalize_host
from src.features.scaling import AngleScaler
from src.inference.predictor import PredictionError, extract_host

SCHEMA_VERSION = 1
STEPS = ("invalid_host", "conflicting_labels", "phiusiil_overlap", "duplicate_group")
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision")
BOOTSTRAP_METHOD = (
    "Percentile bootstrap over rows of the primary set; the same resampled row indices are "
    "used for every model. A resample containing only one class leaves balanced accuracy, "
    "recall, and average precision undefined, so such resamples are skipped for every metric "
    "and counted in resamples_used. Intervals are null when the point metric is null."
)
LIMITATIONS = [
    "The models were trained on only 24 PhiUSIIL examples (12 per class) and were not "
    "retrained, recalibrated, or re-thresholded for this dataset.",
    "Features use only the host of each URL (entropy, label count, digit ratio, hyphen "
    "count); paths, page content, and the dataset's 87 published features are ignored.",
    "The external dataset was collected by different authors with a different method and "
    "period (file dated May 2020), so this is a test under distribution shift; host values "
    "outside the training range are clipped and counted.",
    "One row per registrable domain is kept and domains seen anywhere in PhiUSIIL are "
    "removed, so the primary set is smaller than the published 11,430 rows.",
    "The quantum-kernel SVM uses an exact statevector simulation of the kernel between each "
    "external example and the 24 training examples; no IBM Quantum hardware job was run.",
    "No claim of quantum advantage is made.",
]
_FORBIDDEN_TEXT = re.compile(r"://|[A-Za-z]:[\\/]|\\Users\\|/Users/|/home/|AppData", re.IGNORECASE)


class ExternalError(Exception):
    """External inputs are missing, inconsistent, or the report breaks its schema."""


def protocol_sha256(path: Path) -> str:
    """Hash of the protocol with LF line endings, stable across Git checkouts."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass
class PreparedExternal:
    rows: pd.DataFrame
    input_rows: int
    exclusions: list[dict] = field(default_factory=list)

    @property
    def class_counts(self) -> dict[str, int]:
        phishing = int(self.rows["is_phishing"].sum())
        return {"legitimate": int(len(self.rows)) - phishing, "phishing": phishing}


def _host_or_none(url: str) -> str | None:
    try:
        return extract_host(url)
    except PredictionError:
        return None


def phiusiil_groups(domains: pd.Series) -> set[str]:
    """Registrable-domain groups of every PhiUSIIL row, built as the benchmark builds them."""
    groups = (group_key(normalize_host(domain)) for domain in domains.astype(str))
    return {group for group in groups if group is not None}


def prepare(
    raw: pd.DataFrame, label_map: dict[str, int], benchmark_groups: set[str]
) -> PreparedExternal:
    """Apply the frozen steps in order; `raw` needs url and status columns, in file order."""
    unknown = set(raw["status"]) - set(label_map)
    if unknown:
        raise ExternalError(f"Unknown status values: {len(unknown)}.")
    data = pd.DataFrame(
        {
            "row_index": np.arange(len(raw)),
            "is_phishing": raw["status"].map(label_map).astype(int).to_numpy(),
            "host": pd.Series([_host_or_none(url) for url in raw["url"].astype(str)], dtype=object),
        }
    )
    data["group"] = pd.Series(
        [None if host is None else group_key(host) for host in data["host"]], dtype=object
    )
    prepared = PreparedExternal(rows=data, input_rows=len(data))

    # A valid host with no registrable domain cannot be grouped, so it fails this step too.
    unusable = data["host"].isna() | data["group"].isna()
    data = _exclude(prepared, "invalid_host", data, unusable)
    conflicting = data.groupby("group")["is_phishing"].transform("nunique") > 1
    data = _exclude(prepared, "conflicting_labels", data, conflicting)
    data = _exclude(prepared, "phiusiil_overlap", data, data["group"].isin(benchmark_groups))
    data = _exclude(prepared, "duplicate_group", data, data["group"].duplicated(keep="first"))
    prepared.rows = data.reset_index(drop=True)
    return prepared


def _exclude(
    prepared: PreparedExternal, step: str, data: pd.DataFrame, mask: pd.Series
) -> pd.DataFrame:
    entry = {"step": step, "rows": int(mask.sum())}
    if step in {"conflicting_labels", "phiusiil_overlap"}:
        entry["groups"] = int(data.loc[mask, "group"].nunique())
    if step == "invalid_host":
        entry["valid_host_without_registrable_domain"] = int(
            (data["host"].notna() & data["group"].isna()).sum()
        )
    prepared.exclusions.append(entry)
    return data[~mask]


def featurize(hosts: pd.Series, scaler: AngleScaler) -> tuple[np.ndarray, np.ndarray, dict]:
    """Raw features in the scaler's order, their angles, and clipping counts."""
    names = list(scaler.features)
    raw = np.array([[host_features(host)[name] for name in names] for host in hosts], dtype=float)
    angles, clipped = scaler.transform(raw)
    outside = (raw < np.array(scaler.minima)) | (raw > np.array(scaler.maxima))
    outside[:, np.array(scaler.constant)] = False  # transform maps these to 0, never clips
    summary = {
        "per_feature": dict(zip(names, clipped, strict=True)),
        "rows_with_any_clipped": int(outside.any(axis=1).sum()),
    }
    return raw, angles, summary


def point_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray | None) -> dict:
    """The metrics of src.evaluation.metrics.evaluate, with zero_division=0."""
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "average_precision": None
        if scores is None
        else float(average_precision_score(labels, scores)),
    }


def resample_indices(size: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, size, size=(resamples, size))


def bootstrap_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    indices: np.ndarray,
    level: float,
) -> tuple[dict[str, list[float] | None], int]:
    """Percentile intervals per metric and the number of two-class resamples used."""
    values: dict[str, list[float]] = {name: [] for name in METRICS}
    used = 0
    for sample in indices:
        resampled = labels[sample]
        if resampled.min() == resampled.max():
            continue
        used += 1
        metrics = point_metrics(
            resampled, predictions[sample], None if scores is None else scores[sample]
        )
        for name, value in metrics.items():
            if value is not None:
                values[name].append(value)
    tail = (1 - level) / 2 * 100
    intervals = {
        name: None
        if not collected
        else [float(v) for v in np.percentile(collected, [tail, 100 - tail])]
        for name, collected in values.items()
    }
    return intervals, used


def model_row(key: str, evaluation: dict, intervals: dict) -> dict:
    name, source = MODELS[key]
    return {
        "name": name,
        "model": key,
        "kernel_source": source,
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "confidence_intervals": intervals,
    }


def build_report(
    *,
    protocol: dict,
    dataset: dict,
    benchmark: dict,
    prepared: PreparedExternal,
    clipped: dict,
    bootstrap: dict,
    models: list[dict],
) -> dict:
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "dataset": dataset,
        "benchmark": benchmark,
        "preparation": {
            "input_rows": prepared.input_rows,
            "exclusions": prepared.exclusions,
            "primary_rows": int(len(prepared.rows)),
            "class_counts": prepared.class_counts,
        },
        "clipped_values": clipped,
        "bootstrap": bootstrap,
        "models": models,
        "limitations": list(LIMITATIONS),
    }
    return validate(report)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ExternalError(message)


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
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _validate_preparation(preparation: object) -> int:
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

    protocol = report.get("protocol")
    _require(
        isinstance(protocol, dict)
        and _is_text(protocol.get("path"))
        and isinstance(protocol.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", protocol["sha256"])
        and _is_text(protocol.get("frozen_at_utc")),
        "protocol must hold path, a SHA-256 hex digest, and frozen_at_utc.",
    )
    dataset = report.get("dataset")
    _require(
        isinstance(dataset, dict)
        and all(_is_text(dataset.get(key)) for key in ("doi", "file", "sha256", "license"))
        and _is_int(dataset.get("rows"), 1),
        "dataset must hold doi, file, sha256, license, and a positive row count.",
    )
    benchmark = report.get("benchmark")
    _require(
        isinstance(benchmark, dict)
        and _is_text(benchmark.get("model_version"))
        and _is_int(benchmark.get("training_examples"), 1),
        "benchmark must hold model_version and a positive training_examples count.",
    )

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


def write(report: dict, path: Path) -> None:
    path.write_text(json.dumps(validate(report), indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as error:
        raise ExternalError("External validation results have not been published yet.") from error
    except json.JSONDecodeError as error:
        raise ExternalError(f"Report is not valid JSON: {error.msg}.") from error
