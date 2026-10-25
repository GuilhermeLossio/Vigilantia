"""Flask prediction, replay, and chart routes with stubs; nothing is fetched or trained."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app import create_app
from app import replay as replay_module
from app.results import ResultsInvalid
from src.inference.predictor import PredictionError

pytestmark = pytest.mark.unit


def prediction(label: int, score: float | None = 0.4) -> dict:
    return {
        "prediction": "phishing" if label else "legitimate",
        "is_phishing": label,
        "decision_score": score,
    }


REPLAY = {
    "schema_version": 1,
    "label": "replay of recorded predictions; not live inference",
    "model": "quantum_kernel_svm",
    "generated_at_utc": "2026-01-01T00:00:00+00:00",
    "source_runs": {"simulation": "sim-x", "qpu": "full-fake-1", "backend": "fake"},
    "samples": [
        {
            "position": 0,
            "row_index": 10,
            "group_hash": "abc",
            "is_phishing": 1,
            "predictions": {
                "simulation": prediction(1),
                "qpu": prediction(0, -0.2),
                "qpu_calibrated": prediction(1),
            },
        }
    ],
}


class StubPredictor:
    def predict(self, value: str) -> dict:
        if value == "bad":
            raise PredictionError("The host is invalid.")
        return {
            "host": value,
            "prediction": "phishing",
            "is_phishing": 1,
            "decision_score": 0.5,
            "score_semantics": "not a probability",
            "features": {"host_entropy": 1.0},
            "outside_training_range": [],
            "model": "rbf_svm",
            "model_version": "v1",
            "source": "classical",
        }


def client(tmp_path: Path, replay: object = REPLAY, factory=StubPredictor, chart: bool = True):
    replay_path = tmp_path / "replay.json"
    if replay is not None:
        replay_path.write_text(json.dumps(replay), encoding="utf-8")
    chart_path = tmp_path / "chart.svg"
    if chart:
        chart_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    app = create_app(tmp_path / "missing-results.json", replay_path, chart_path, factory)
    return app.test_client()


def test_predict_returns_the_classical_prediction(tmp_path: Path) -> None:
    response = client(tmp_path).post("/v1/phishing/predict", json={"host": "x.web.app"})
    assert response.status_code == 200
    assert response.get_json()["source"] == "classical"


@pytest.mark.parametrize("body", [{"host": "bad"}, {"host": 5}, {"url": "x.com"}, None])
def test_predict_rejects_invalid_input(tmp_path: Path, body: object) -> None:
    c = client(tmp_path)
    response = (
        c.post("/v1/phishing/predict", json=body)
        if body is not None
        else c.post("/v1/phishing/predict", data="not json")
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_input"


def test_predict_rejects_large_bodies(tmp_path: Path) -> None:
    response = client(tmp_path).post("/v1/phishing/predict", json={"host": "a" * 5000 + ".com"})
    assert response.status_code == 413


def test_missing_model_is_unavailable_not_a_crash(tmp_path: Path) -> None:
    def broken():
        raise FileNotFoundError("models.pkl")

    c = client(tmp_path, factory=broken)
    response = c.post("/v1/phishing/predict", json={"host": "x.com"})
    assert response.status_code == 503
    assert response.get_json()["error"] == "model_unavailable"
    page = c.get("/?host=x.com").get_data(as_text=True)
    assert "Not classified." in page


def test_page_form_shows_prediction_and_margin_semantics(tmp_path: Path) -> None:
    page = client(tmp_path).get("/?host=login.example.web.app").get_data(as_text=True)
    assert "login.example.web.app" in page
    assert "not a probability" in page
    assert 'action="/"' in page


def test_replay_route_and_page_label_replay(tmp_path: Path) -> None:
    c = client(tmp_path)
    assert c.get("/v1/benchmarks/replay").get_json()["source_runs"]["qpu"] == "full-fake-1"
    page = c.get("/").get_data(as_text=True)
    assert "Replay of recorded predictions, not live inference" in page
    assert "QPU, calibrated" in page


def test_missing_and_invalid_replay(tmp_path: Path) -> None:
    assert client(tmp_path, replay=None).get("/v1/benchmarks/replay").status_code == 503
    assert (
        client(tmp_path, replay={"schema_version": 2}).get("/v1/benchmarks/replay").status_code
        == 500
    )


def test_chart_route(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    response = client(tmp_path).get("/benchmark.svg")
    assert response.status_code == 200 and response.mimetype == "image/svg+xml"
    assert client(empty, chart=False).get("/benchmark.svg").status_code == 404


def test_csp_allows_only_same_origin_images_and_forms(tmp_path: Path) -> None:
    policy = client(tmp_path).get("/").headers["Content-Security-Policy"]
    assert "img-src 'self'" in policy and "form-action 'self'" in policy
    assert "script-src" not in policy


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(label="live results"),
        lambda d: d["samples"][0].update(is_phishing=2),
        lambda d: d["samples"][0]["predictions"]["qpu"].update(
            prediction="legitimate", is_phishing=1
        ),
        lambda d: d["samples"][0]["predictions"]["qpu"].update(decision_score=float("nan")),
        lambda d: d["samples"][0]["predictions"].update(other=prediction(1)),
        lambda d: d["source_runs"].pop("qpu"),
        lambda d: d.update(samples=[]),
    ],
)
def test_replay_validation_rejects_inconsistent_files(mutate) -> None:
    data = copy.deepcopy(REPLAY)
    mutate(data)
    with pytest.raises(ResultsInvalid):
        replay_module.validate(data)


def test_simulation_only_replay_is_valid() -> None:
    data = copy.deepcopy(REPLAY)
    data["source_runs"] = {"simulation": "sim-x"}
    data["samples"][0]["predictions"] = {"simulation": prediction(1)}
    assert replay_module.validate(data) is data
