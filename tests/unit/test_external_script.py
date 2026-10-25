"""Orchestration helpers of scripts/evaluate_external.py on synthetic inputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_external as script
from src.evaluation import external

pytestmark = pytest.mark.unit


def fake_evaluations() -> dict:
    scores = [-1.0, 0.5, 0.2, 1.0]
    rows = {}
    for name in external.MODELS:
        majority = name == "majority"
        rows[name] = {
            "metrics": external.point_metrics(
                np.array([0, 0, 1, 1]),
                np.array([0, 0, 0, 0] if majority else [0, 1, 1, 1]),
                None if majority else np.array(scores),
            ),
            "confusion_matrix": [[2, 0], [2, 0]] if majority else [[1, 1], [0, 2]],
            "predictions": [0, 0, 0, 0] if majority else [0, 1, 1, 1],
            "decision_scores": None if majority else scores,
        }
    return rows


def test_intervals_cover_every_model_with_shared_resamples() -> None:
    labels = np.array([0, 0, 1, 1])
    rows, used = script.with_intervals(
        fake_evaluations(), labels, {"resamples": 200, "seed": 42, "level": 0.95}
    )
    assert [row["model"] for row in rows] == list(external.MODELS)
    assert 0 < used <= 200
    assert rows[0]["confidence_intervals"]["average_precision"] is None
    assert rows[1]["confidence_intervals"]["average_precision"] is not None
    assert rows[1]["confidence_intervals"] == rows[2]["confidence_intervals"]


def test_saved_predictions_hold_no_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "ARTIFACT_DIR", tmp_path / "external")
    prepared = external.PreparedExternal(
        rows=pd.DataFrame(
            {
                "row_index": [3, 5, 8, 9],
                "is_phishing": [0, 0, 1, 1],
                "host": ["a.example.com", "b.example.org", "c.example.net", "d.example.io"],
                "group": ["dns:a", "dns:b", "dns:c", "dns:d"],
            }
        ),
        input_rows=4,
    )
    script.save_predictions(prepared, fake_evaluations())
    text = (tmp_path / "external" / "predictions.csv").read_text(encoding="utf-8")
    assert "example" not in text and "dns:" not in text
    table = pd.read_csv(tmp_path / "external" / "predictions.csv")
    assert table["row_index"].tolist() == [3, 5, 8, 9]
    assert "majority_score" not in table and "quantum_kernel_svm_score" in table


class ShapeRecorder:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        self.shapes.append(np.asarray(inputs).shape)
        return np.array([0, 1, 0])


def test_only_the_quantum_model_receives_the_simulated_kernel() -> None:
    models = {name: ShapeRecorder() for name in external.MODELS}
    rng = np.random.default_rng(3)
    data = SimpleNamespace(models=models, train_angles=rng.uniform(0, np.pi / 2, (5, 4)))
    angles = rng.uniform(0, np.pi / 2, (3, 4))
    _, kernel, _ = script.classify(data, angles, np.array([0, 1, 1]))
    assert kernel.shape == (3, 5)
    for name, model in models.items():
        assert model.shapes == [(3, 5) if name == "quantum_kernel_svm" else (3, 4)]


def test_phiusiil_checksum_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "phiusiil.csv"
    csv.write_text("Domain\nwww.example.com\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dataset": {"csv_sha256": "0" * 64}}), encoding="utf-8")
    monkeypatch.setattr(script, "PHIUSIIL_CSV", csv)
    monkeypatch.setattr(script, "SPLIT_PATH", split)
    with pytest.raises(external.ExternalError, match="differs from the benchmark"):
        script.load_phiusiil_groups()
    monkeypatch.setattr(script, "PHIUSIIL_CSV", tmp_path / "absent.csv")
    with pytest.raises(external.ExternalError, match="missing"):
        script.load_phiusiil_groups()


def test_missing_snapshot_publishes_nothing_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script.download_external, "DEFAULT_DESTINATION", tmp_path / "absent")
    monkeypatch.setattr(
        script.download_external, "urlopen", lambda *a, **kw: pytest.fail("Downloaded")
    )
    monkeypatch.setattr(script, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(script, "ARTIFACT_DIR", tmp_path / "artifacts")
    assert script.main() == 1
    assert list(tmp_path.iterdir()) == []


def test_unusable_inputs_return_an_error_without_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script, "verified_snapshot", lambda dataset: {})

    def unavailable(dataset: dict):
        raise external.ExternalError("Preparation left no rows to evaluate.")

    monkeypatch.setattr(script, "load_inputs", unavailable)
    monkeypatch.setattr(script, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(script, "ARTIFACT_DIR", tmp_path / "artifacts")
    assert script.main() == 1
    assert list(tmp_path.iterdir()) == []
