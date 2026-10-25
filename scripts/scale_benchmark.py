"""Fit the angle scaler on the 24 training rows and scale the frozen benchmark sample.

Reads data/processed/benchmark-sample.csv (from prepare_benchmark.py) and checks it against
configs/benchmark-split.json before fitting. Writes:
- configs/feature-scaler.json: versioned scaler parameters and clipping counts.
- data/processed/benchmark-scaled.csv: local, Git-ignored angles for every sample row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.features.scaling import AngleScaler  # noqa: E402

MANIFEST_PATH = ROOT / "configs" / "benchmark-split.json"
SAMPLE_PATH = ROOT / "data" / "processed" / "benchmark-sample.csv"
SCALER_PATH = ROOT / "configs" / "feature-scaler.json"
SCALED_PATH = ROOT / "data" / "processed" / "benchmark-scaled.csv"


def check_sample(sample: pd.DataFrame, manifest: dict) -> None:
    """Refuse a sample whose rows, order, or labels differ from the frozen manifest."""
    for partition in ("train", "test"):
        rows = sample[sample["partition"] == partition]
        expected = [(r["row_index"], r["is_phishing"]) for r in manifest[partition]]
        actual = list(zip(rows["row_index"], rows["is_phishing"], strict=True))
        if actual != expected:
            raise SystemExit(
                f"{partition} rows differ from the manifest; re-run prepare_benchmark.py."
            )


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    features = manifest["features"]
    sample = pd.read_csv(SAMPLE_PATH)
    check_sample(sample, manifest)

    train = sample[sample["partition"] == "train"]
    test = sample[sample["partition"] == "test"]
    scaler = AngleScaler.fit(features, train[features])
    train_angles, train_clipped = scaler.transform(train[features])
    test_angles, test_clipped = scaler.transform(test[features])
    if any(train_clipped):
        raise SystemExit("Training values were clipped; the scaler fit is inconsistent.")

    record = {
        **scaler.to_dict(),
        "fitted_on": "train partition of configs/benchmark-split.json (24 rows)",
        "clipped_values": {
            "test": dict(zip(features, test_clipped, strict=True)),
        },
    }
    SCALER_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    scaled = pd.concat(
        [
            train[["row_index", "partition", "is_phishing"]].assign(
                **dict(zip(features, train_angles.T, strict=True))
            ),
            test[["row_index", "partition", "is_phishing"]].assign(
                **dict(zip(features, test_angles.T, strict=True))
            ),
        ]
    )
    scaled.to_csv(SCALED_PATH, index=False)

    print("Training range per feature:")
    for name, low, high, flat in zip(
        features, scaler.minima, scaler.maxima, scaler.constant, strict=True
    ):
        print(f"  {name}: {low:.4f} to {high:.4f}{'  (constant -> 0)' if flat else ''}")
    print(f"Test values clipped: {record['clipped_values']['test']}")
    print(f"Scaler: {SCALER_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
