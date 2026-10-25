"""Build the exact training kernel and fit the four benchmark models. No test data is used.

Inputs: configs/benchmark-split.json, configs/feature-scaler.json, and the local
data/processed/benchmark-sample.csv. Outputs go to the Git-ignored artifacts/benchmark/:
- models.pkl: the four fitted models.
- training-angles.npy and training-kernel.npy: reference examples and their kernel.
- training-summary.json: settings, kernel diagnostics, versions, and input checksums.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.scaling import AngleScaler  # noqa: E402
from src.models import benchmark_models  # noqa: E402
from src.quantum import kernel as qkernel  # noqa: E402

MANIFEST_PATH = ROOT / "configs" / "benchmark-split.json"
SCALER_PATH = ROOT / "configs" / "feature-scaler.json"
SAMPLE_PATH = ROOT / "data" / "processed" / "benchmark-sample.csv"
OUTPUT_DIR = ROOT / "artifacts" / "benchmark"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_rows() -> tuple[pd.DataFrame, AngleScaler, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    scaler = AngleScaler.from_dict(json.loads(SCALER_PATH.read_text(encoding="utf-8")))
    if list(scaler.features) != manifest["features"]:
        raise SystemExit("Scaler features differ from the manifest; re-run scale_benchmark.py.")
    sample = pd.read_csv(SAMPLE_PATH)
    train = sample[sample["partition"] == "train"].reset_index(drop=True)
    expected = [(r["row_index"], r["is_phishing"]) for r in manifest["train"]]
    if list(zip(train["row_index"], train["is_phishing"], strict=True)) != expected:
        raise SystemExit("Training rows differ from the manifest; re-run prepare_benchmark.py.")
    return train, scaler, manifest


def main() -> int:
    train, scaler, manifest = load_training_rows()
    angles, clipped = scaler.transform(train[list(scaler.features)])
    if any(clipped):
        raise SystemExit("Training values were clipped; the scaler does not match the sample.")
    labels = train["is_phishing"].to_numpy()

    kernel = qkernel.statevector_kernel(angles)
    diagnostics = qkernel.training_kernel_diagnostics(kernel)
    if not diagnostics["eigenvalue_check_passed"]:
        raise SystemExit(f"Training kernel failed the eigenvalue check: {diagnostics}")

    models = benchmark_models.fit_all(angles, kernel, labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "training-angles.npy", angles)
    np.save(OUTPUT_DIR / "training-kernel.npy", kernel)
    with (OUTPUT_DIR / "models.pkl").open("wb") as stream:
        pickle.dump(models, stream)

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "uses_test_data": False,
        "inputs": {
            "split_manifest_sha256": sha256(MANIFEST_PATH),
            "scaler_sha256": sha256(SCALER_PATH),
        },
        "training_rows": [int(i) for i in train["row_index"]],
        "training_labels": {"legitimate": int((labels == 0).sum()), "phishing": int(labels.sum())},
        "features": list(scaler.features),
        "feature_map": qkernel.FEATURE_MAP_SETTINGS,
        "training_kernel": "exact statevector simulation",
        "kernel_diagnostics": diagnostics,
        "models": benchmark_models.settings(models),
        "fitted": {
            "majority_class": int(
                models["majority"].classes_[np.argmax(models["majority"].class_prior_)]
            ),
            "logistic_regression_coefficients": dict(
                zip(
                    scaler.features,
                    models["logistic_regression"].coef_[0].round(6).tolist(),
                    strict=True,
                )
            ),
            "rbf_svm_support_vectors": int(models["rbf_svm"].n_support_.sum()),
            "quantum_kernel_svm_support_vectors": int(
                models["quantum_kernel_svm"].n_support_.sum()
            ),
        },
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("qiskit", "numpy", "scikit-learn")},
        },
    }
    (OUTPUT_DIR / "training-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("Training kernel (exact simulation), 24 x 24:")
    print(f"  identical_example_pairs: {diagnostics['identical_example_pairs']}")
    for key in ("min_eigenvalue", "max_asymmetry", "max_diagonal_error"):
        print(f"  {key}: {diagnostics[key]:.3e}")
    for key in ("off_diagonal_mean", "off_diagonal_std", "off_diagonal_min", "off_diagonal_max"):
        print(f"  {key}: {diagnostics[key]:.4f}")
    print(f"Fitted: {json.dumps(summary['fitted'], indent=2)}")
    print(f"Artifacts: {OUTPUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
