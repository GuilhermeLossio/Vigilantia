"""Schema of reports/external-validation.json: a valid report and the ways it can break."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import external

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def evaluation(predictions: list[int], scores: list[float] | None) -> dict:
    labels = np.array([0, 0, 1, 1])
    metrics = external.point_metrics(
        labels, np.array(predictions), None if scores is None else np.array(scores)
    )
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels, strict=True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels, strict=True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels, strict=True))
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels, strict=True))
    return {"metrics": metrics, "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def intervals(metrics: dict) -> dict:
    return {name: None if value is None else [value, value] for name, value in metrics.items()}


def valid_report() -> dict:
    prepared = external.PreparedExternal(
        rows=pd.DataFrame({"is_phishing": [0, 0, 1, 1]}),
        input_rows=10,
        exclusions=[
            {"step": "invalid_host", "rows": 1, "valid_host_without_registrable_domain": 0},
            {"step": "conflicting_labels", "rows": 2, "groups": 1},
            {"step": "phiusiil_overlap", "rows": 1, "groups": 1},
            {"step": "duplicate_group", "rows": 2},
        ],
    )
    evaluations = {
        "majority": evaluation([0, 0, 0, 0], None),
        "logistic_regression": evaluation([0, 1, 1, 1], [-1.0, 0.2, 0.5, 1.0]),
        "rbf_svm": evaluation([0, 0, 1, 1], [-1.0, -0.5, 0.5, 1.0]),
        "quantum_kernel_svm": evaluation([1, 0, 0, 1], [0.3, -0.2, -0.1, 0.4]),
    }
    return external.build_report(
        protocol={
            "path": "configs/external-validation.json",
            "sha256": "a" * 64,
            "frozen_at_utc": "2026-10-25T11:10:26+00:00",
        },
        dataset={
            "doi": "10.17632/c2gw7fy2j4.2",
            "file": "dataset_B_05_2020.csv",
            "sha256": "b" * 64,
            "rows": 10,
            "license": "CC BY 4.0",
        },
        benchmark={"model_version": "a2e084f5eb36", "training_examples": 24},
        prepared=prepared,
        clipped={"per_feature": {"host_entropy": 1}, "rows_with_any_clipped": 1},
        bootstrap={
            "method": external.BOOTSTRAP_METHOD,
            "level": 0.95,
            "resamples": 2000,
            "resamples_used": 2000,
            "seed": 42,
        },
        models=[
            external.model_row(name, value, intervals(value["metrics"]))
            for name, value in evaluations.items()
        ],
    )


def test_valid_report_round_trips(tmp_path: Path) -> None:
    report = valid_report()
    path = tmp_path / "external.json"
    external.write(report, path)
    assert external.load(path) == json.loads(path.read_text(encoding="utf-8"))
    sources = {row["model"]: row["kernel_source"] for row in report["models"]}
    assert sources["quantum_kernel_svm"] == "simulation"
    assert {sources[n] for n in ("majority", "logistic_regression", "rbf_svm")} == {"classical"}
    assert report["models"][0]["confidence_intervals"]["average_precision"] is None
    assert any("advantage" in item for item in report["limitations"])
    assert any("hardware" in item for item in report["limitations"])


def mutate(path: str, value: object) -> dict:
    report = copy.deepcopy(valid_report())
    *parents, last = path.split(".")
    target = report
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if value is KeyError:
        del target[int(last) if isinstance(target, list) else last]
    elif isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return report


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("protocol.sha256", "not-a-hash", "protocol"),
        ("dataset.rows", 0, "dataset"),
        ("benchmark.model_version", "", "benchmark"),
        ("preparation.exclusions.0.step", "other", "in order"),
        ("preparation.exclusions.1.rows", -1, "non-negative"),
        ("preparation.input_rows", 11, "input_rows"),
        ("preparation.class_counts.phishing", 3, "class_counts"),
        ("clipped_values.rows_with_any_clipped", 5, "clipped_values"),
        ("bootstrap.resamples_used", 2001, "bootstrap"),
        ("models.1.kernel_source", "qpu", "kernel_source"),
        ("models.3.kernel_source", "classical", "kernel_source"),
        ("models.1.metrics.accuracy", 1.5, "metrics.accuracy"),
        ("models.1.metrics.f1", KeyError, "metrics must contain"),
        ("models.1.confidence_intervals.f1", [0.9, 0.1], "confidence_intervals.f1"),
        ("models.1.confidence_intervals.f1", None, "null exactly when"),
        ("models.0.confidence_intervals.average_precision", [0.1, 0.2], "null exactly when"),
        ("models.1.confusion_matrix", [[1, 1], [1, 2]], "summing to primary_rows"),
        ("models.2", KeyError, "exactly once"),
        ("limitations", [], "limitations"),
        ("dataset.file", "https://example.com/data.csv", "URLs or local paths"),
        ("protocol.path", "C:\\Users\\someone\\protocol.json", "URLs or local paths"),
        ("dataset.license", "see /Users/someone/notes", "URLs or local paths"),
        ("dataset.license", "D:/data/x.csv", "URLs or local paths"),
        ("dataset.license", "/home/someone/x", "URLs or local paths"),
    ],
)
def test_invalid_reports_are_rejected(path: str, value: object, message: str) -> None:
    with pytest.raises(external.ExternalError, match=message):
        external.validate(mutate(path, value))


def test_non_object_and_missing_file_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(external.ExternalError, match="JSON object"):
        external.validate([])
    with pytest.raises(external.ExternalError, match="not been published"):
        external.load(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(external.ExternalError, match="not valid JSON"):
        external.load(broken)


def test_published_report_satisfies_the_schema_and_protocol() -> None:
    path = ROOT / "reports" / "external-validation.json"
    if not path.exists():
        pytest.skip("External validation has not been published.")
    report = external.load(path)
    protocol = ROOT / "configs" / "external-validation.json"
    assert report["protocol"]["sha256"] == external.protocol_sha256(protocol)
    config = json.loads(protocol.read_text(encoding="utf-8"))
    assert report["dataset"]["sha256"] == config["dataset"]["sha256"]
    assert (
        report["bootstrap"]["resamples"] == config["metrics"]["confidence_intervals"]["resamples"]
    )
    assert report["benchmark"]["training_examples"] == 24
