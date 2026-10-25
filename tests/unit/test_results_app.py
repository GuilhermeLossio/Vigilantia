"""Results service behavior with synthetic results files; no network or real results."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app import create_app
from app import results as results_module

pytestmark = pytest.mark.unit

METRICS = dict.fromkeys(results_module.METRICS, 0.5)
VALID = {
    "schema_version": 1,
    "run_id": "test-run",
    "generated_at_utc": "2026-01-01T00:00:00+00:00",
    "status": "complete",
    "split": {"train": 24, "test": 12, "seed": 42},
    "hardware": {"backend": "test_backend", "shots": 1024, "job_ids": ["job-1"]},
    "models": [
        {
            "name": "Logistic regression",
            "kernel_source": "classical",
            "metrics": METRICS,
            "confusion_matrix": [[3, 3], [3, 3]],
        },
        {
            "name": "Quantum-kernel SVM",
            "kernel_source": "qpu_calibrated",
            "metrics": {**METRICS, "average_precision": None},
            "confusion_matrix": [[6, 0], [1, 5]],
        },
    ],
    "limitations": ["Synthetic test fixture <not a real result>."],
}


def write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "benchmark-results.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def variant(**changes: object) -> dict:
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


def client_for(path: Path):
    return create_app(path).test_client()


def test_health_is_independent_of_results(tmp_path: Path) -> None:
    response = client_for(tmp_path / "missing.json").get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_valid_results_are_served_unchanged(tmp_path: Path) -> None:
    response = client_for(write(tmp_path, VALID)).get("/v1/benchmarks")
    assert response.status_code == 200
    assert response.get_json() == VALID


def test_page_labels_sources_and_escapes_text(tmp_path: Path) -> None:
    response = client_for(write(tmp_path, VALID)).get("/")
    page = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "IBM Quantum hardware, calibrated" in page
    assert "Classical model" in page
    assert "n/a" in page
    assert "&lt;not a real result&gt;" in page
    assert "<not a real result>" not in page


def test_missing_results_are_pending_not_an_error_page(tmp_path: Path) -> None:
    client = client_for(tmp_path / "missing.json")
    api = client.get("/v1/benchmarks")
    assert api.status_code == 503
    assert api.get_json()["error"] == "results_unavailable"
    page = client.get("/")
    assert page.status_code == 503
    assert "Results pending" in page.get_data(as_text=True)


def test_results_path_can_come_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIGILANTIA_RESULTS_PATH", str(write(tmp_path, VALID)))
    assert create_app().test_client().get("/v1/benchmarks").status_code == 200


def test_security_headers_are_set(tmp_path: Path) -> None:
    headers = client_for(write(tmp_path, VALID)).get("/").headers
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("{not json", "not valid JSON"),
        ([], "JSON object"),
        (variant(schema_version=2), "schema_version"),
        (variant(run_id=" "), "run_id"),
        (variant(status="done"), "status"),
        (variant(split={"train": 24, "test": "12", "seed": 42}), "split.test"),
        (variant(models=[]), "models"),
        (variant(limitations="none"), "limitations"),
        (variant(hardware={"backend": None, "shots": None, "job_ids": []}), "QPU results"),
    ],
)
def test_invalid_results_files_are_rejected(tmp_path: Path, data: object, message: str) -> None:
    response = client_for(write(tmp_path, data)).get("/v1/benchmarks")
    assert response.status_code == 500
    body = response.get_json()
    assert body["error"] == "results_invalid"
    assert message in body["detail"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kernel_source": "quantum"}, "kernel_source"),
        ({"metrics": {**METRICS, "accuracy": 1.5}}, "metrics.accuracy"),
        ({"metrics": {**METRICS, "accuracy": True}}, "metrics.accuracy"),
        ({"metrics": {**METRICS, "accuracy": float("nan")}}, "metrics.accuracy"),
        ({"metrics": {"accuracy": 0.5}}, "must contain exactly"),
        ({"confusion_matrix": [[3, 3], [3, 2]]}, "sum to split.test"),
        ({"confusion_matrix": [[3, -3], [3, 9]]}, "non-negative integers"),
        ({"name": ""}, "name"),
    ],
)
def test_invalid_model_entries_are_rejected(change: dict, message: str) -> None:
    data = copy.deepcopy(VALID)
    data["models"][0].update(change)
    with pytest.raises(results_module.ResultsInvalid, match=message.replace(".", r"\.")):
        results_module.validate(data)


def test_classical_only_results_do_not_require_hardware() -> None:
    data = copy.deepcopy(VALID)
    data["models"] = data["models"][:1]
    data["hardware"] = {"backend": None, "shots": None, "job_ids": []}
    assert results_module.validate(data) is data
