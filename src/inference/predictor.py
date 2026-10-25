"""Live classical inference for new hosts. No page or URL is ever fetched.

The served model is the RBF SVM, named the primary nonlinear baseline in the benchmark
protocol before any test result existed; test results were not used to choose it. The
quantum-kernel SVM is not served live: each prediction would need QPU kernel evaluations
against the training examples, so its saved test predictions are shown only as replay.

Input is a host name or a URL; only the host part is used. The four frozen host features are
computed locally, scaled with the training-fitted scaler, and classified. The score is the
SVM decision margin (positive means phishing), not a probability.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from src.features.host_features import host_features, normalize_host
from src.features.scaling import AngleScaler

ROOT = Path(__file__).resolve().parents[2]
SCALER_PATH = ROOT / "configs" / "feature-scaler.json"
MODELS_PATH = ROOT / "artifacts" / "benchmark" / "models.pkl"
SERVED_MODEL = "rbf_svm"
LIVE_MODELS = ("rbf_svm", "logistic_regression")
MAX_HOST_LENGTH = 253
_HOST = re.compile(r"^[a-z0-9.\-_:\[\]]+$")
SCORE_SEMANTICS = (
    "Decision margin of the classifier: positive means phishing, larger means further from "
    "the boundary. It is not a probability."
)


class PredictionError(ValueError):
    """The input cannot be classified (invalid host or unusable artifacts)."""


def extract_host(value: str) -> str:
    """Host name from a bare host or a URL, normalized like the training data."""
    if not isinstance(value, str) or not value.strip():
        raise PredictionError("Provide a non-empty host or URL.")
    text = value.strip()
    if "://" in text:
        try:
            netloc = urlsplit(text).netloc
        except ValueError as error:
            raise PredictionError("The URL could not be parsed.") from error
        text = netloc.rsplit("@", 1)[-1]
    else:
        text = text.split("/", 1)[0]
    host = normalize_host(text)
    if not host or len(host) > MAX_HOST_LENGTH or not _HOST.match(host) or ".." in host:
        raise PredictionError("The host is empty, too long, or contains invalid characters.")
    if "." not in host and ":" not in host:
        raise PredictionError("The host must contain a domain suffix or be an IP address.")
    return host


def _version(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()[:12]


@dataclass
class Predictor:
    scaler: AngleScaler
    model: object
    model_name: str
    model_version: str

    @classmethod
    def from_artifacts(
        cls,
        model_name: str = SERVED_MODEL,
        scaler_path: Path = SCALER_PATH,
        models_path: Path = MODELS_PATH,
    ) -> Predictor:
        if model_name not in LIVE_MODELS:
            raise PredictionError(
                f"{model_name} is not served live; choose one of {', '.join(LIVE_MODELS)}."
            )
        try:
            scaler_bytes = scaler_path.read_bytes()
            models_bytes = models_path.read_bytes()
        except FileNotFoundError as error:
            raise PredictionError(
                f"Missing artifact {error.filename}; run the benchmark scripts first."
            ) from error
        scaler = AngleScaler.from_dict(json.loads(scaler_bytes))
        models = pickle.loads(models_bytes)  # local artifact written by train_benchmark.py
        if model_name not in models:
            raise PredictionError(f"The trained artifacts do not contain {model_name}.")
        return cls(scaler, models[model_name], model_name, _version(scaler_bytes, models_bytes))

    def predict(self, value: str) -> dict:
        host = extract_host(value)
        raw = host_features(host)
        features = [raw[name] for name in self.scaler.features]
        angles, clipped = self.scaler.transform([features])
        is_phishing = int(self.model.predict(angles)[0])
        return {
            "host": host,
            "prediction": "phishing" if is_phishing else "legitimate",
            "is_phishing": is_phishing,
            "decision_score": float(self.model.decision_function(angles)[0]),
            "score_semantics": SCORE_SEMANTICS,
            "features": dict(zip(self.scaler.features, features, strict=True)),
            "outside_training_range": [
                name for name, count in zip(self.scaler.features, clipped, strict=True) if count
            ],
            "model": self.model_name,
            "model_version": self.model_version,
            "source": "classical",
        }
