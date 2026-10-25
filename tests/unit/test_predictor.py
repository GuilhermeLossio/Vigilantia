"""Live classical inference with synthetic artifacts; nothing is fetched."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.svm import SVC

from src.features.host_features import host_features
from src.features.scaling import AngleScaler
from src.inference.predictor import PredictionError, Predictor, extract_host

pytestmark = pytest.mark.unit
FEATURES = ["host_entropy", "host_label_count", "host_digit_ratio", "host_hyphen_count"]
LEGIT = ["example.com", "library.org", "weather.gov", "museum.net", "garden.co.uk", "books.com"]
PHISH = [
    "secure-login-1234.web.app",
    "verify-acc0unt-77.firebaseapp.com",
    "paypa1-update-9.xyz",
    "bank-alert-2024.top",
    "x9-k2-reset.webflow.io",
    "auth-0-help.repl.co",
]


def raw(host: str) -> list[float]:
    values = host_features(host)
    return [values[name] for name in FEATURES]


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    train = np.array([raw(h) for h in LEGIT + PHISH])
    labels = np.array([0] * len(LEGIT) + [1] * len(PHISH))
    scaler = AngleScaler.fit(FEATURES, train)
    angles, _ = scaler.transform(train)
    models = {"rbf_svm": SVC(kernel="rbf", C=1.0, gamma="scale").fit(angles, labels)}
    scaler_path, models_path = tmp_path / "scaler.json", tmp_path / "models.pkl"
    scaler_path.write_text(json.dumps(scaler.to_dict()), encoding="utf-8")
    models_path.write_bytes(pickle.dumps(models))
    return scaler_path, models_path


@pytest.mark.parametrize(
    ("value", "host"),
    [
        ("Example.COM", "example.com"),
        ("www.example.com", "example.com"),
        ("https://user@www.Login.example.com:8443/path?q=1", "login.example.com"),
        ("example.com/some/path", "example.com"),
        ("10.0.0.1", "10.0.0.1"),
    ],
)
def test_extract_host_uses_only_the_host(value: str, host: str) -> None:
    assert extract_host(value) == host


@pytest.mark.parametrize(
    "value", ["", "   ", "localhost", "a..b.com", "bad host.com", "x" * 300 + ".com", 5]
)
def test_invalid_hosts_are_rejected(value: object) -> None:
    with pytest.raises(PredictionError):
        extract_host(value)  # type: ignore[arg-type]


def test_prediction_matches_the_model_on_scaled_features(artifacts: tuple[Path, Path]) -> None:
    scaler_path, models_path = artifacts
    predictor = Predictor.from_artifacts("rbf_svm", scaler_path, models_path)
    for host in LEGIT + PHISH:
        result = predictor.predict(host)
        angles, _ = predictor.scaler.transform([raw(host)])
        assert result["is_phishing"] == int(predictor.model.predict(angles)[0])
        assert result["decision_score"] == pytest.approx(
            predictor.model.decision_function(angles)[0]
        )
        assert result["prediction"] == ("phishing" if result["is_phishing"] else "legitimate")
        assert result["source"] == "classical"
        assert "not a probability" in result["score_semantics"]


def test_out_of_range_features_are_reported(artifacts: tuple[Path, Path]) -> None:
    predictor = Predictor.from_artifacts("rbf_svm", *artifacts)
    result = predictor.predict("a-b-c-d-e-f-g-h-i-j-k-l-m-n.very.deep.sub.domain.example.com")
    assert "host_hyphen_count" in result["outside_training_range"]
    assert "host_label_count" in result["outside_training_range"]


def test_model_version_changes_with_the_artifacts(artifacts: tuple[Path, Path]) -> None:
    scaler_path, models_path = artifacts
    first = Predictor.from_artifacts("rbf_svm", scaler_path, models_path).model_version
    assert first == Predictor.from_artifacts("rbf_svm", scaler_path, models_path).model_version
    models_path.write_bytes(models_path.read_bytes() + b" ")
    assert Predictor.from_artifacts("rbf_svm", scaler_path, models_path).model_version != first


def test_quantum_model_is_not_served_live(artifacts: tuple[Path, Path]) -> None:
    with pytest.raises(PredictionError, match="not served live"):
        Predictor.from_artifacts("quantum_kernel_svm", *artifacts)


def test_missing_artifacts_and_models_fail_clearly(
    artifacts: tuple[Path, Path], tmp_path: Path
) -> None:
    scaler_path, models_path = artifacts
    with pytest.raises(PredictionError, match="Missing artifact"):
        Predictor.from_artifacts("rbf_svm", scaler_path, tmp_path / "absent.pkl")
    with pytest.raises(PredictionError, match="do not contain"):
        Predictor.from_artifacts("logistic_regression", scaler_path, models_path)
