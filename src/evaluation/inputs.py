"""Load the frozen benchmark inputs and fitted models, refusing any inconsistency."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.scaling import AngleScaler

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs" / "benchmark-split.json"
SCALER_PATH = ROOT / "configs" / "feature-scaler.json"
SAMPLE_PATH = ROOT / "data" / "processed" / "benchmark-sample.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "benchmark"


class InputError(Exception):
    """Benchmark inputs are missing or do not match each other."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class BenchmarkInputs:
    features: list[str]
    train_angles: np.ndarray
    test_angles: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    test_clipped: dict[str, int]
    models: dict
    input_hashes: dict[str, str]


def _partition(sample: pd.DataFrame, manifest: dict, name: str) -> pd.DataFrame:
    rows = sample[sample["partition"] == name].reset_index(drop=True)
    expected = [(r["row_index"], r["is_phishing"]) for r in manifest[name]]
    if list(zip(rows["row_index"], rows["is_phishing"], strict=True)) != expected:
        raise InputError(f"{name} rows differ from the manifest; re-run prepare_benchmark.py.")
    return rows


def load() -> BenchmarkInputs:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        scaler = AngleScaler.from_dict(json.loads(SCALER_PATH.read_text(encoding="utf-8")))
        sample = pd.read_csv(SAMPLE_PATH)
        summary = json.loads((ARTIFACT_DIR / "training-summary.json").read_text(encoding="utf-8"))
        saved_train_angles = np.load(ARTIFACT_DIR / "training-angles.npy")
        with (ARTIFACT_DIR / "models.pkl").open("rb") as stream:
            models = pickle.load(stream)  # local artifact written by train_benchmark.py
    except FileNotFoundError as error:
        raise InputError(f"Missing input: {error.filename}. Run the earlier scripts.") from error

    features = list(scaler.features)
    if features != manifest["features"]:
        raise InputError("Scaler features differ from the manifest.")
    hashes = {"split_manifest_sha256": sha256(MANIFEST_PATH), "scaler_sha256": sha256(SCALER_PATH)}
    if summary["inputs"] != hashes:
        raise InputError("Models were trained on other inputs; re-run train_benchmark.py.")

    train = _partition(sample, manifest, "train")
    test = _partition(sample, manifest, "test")
    train_angles, _ = scaler.transform(train[features])
    if not np.allclose(train_angles, saved_train_angles, atol=0, rtol=0):
        raise InputError("Training angles differ from the trained artifacts.")
    test_angles, clipped = scaler.transform(test[features])
    return BenchmarkInputs(
        features=features,
        train_angles=train_angles,
        test_angles=test_angles,
        train_labels=train["is_phishing"].to_numpy(),
        test_labels=test["is_phishing"].to_numpy(),
        test_clipped=dict(zip(features, clipped, strict=True)),
        models=models,
        input_hashes=hashes,
    )
