"""Build the frozen 24/12 benchmark sample and feature selection from the verified snapshot.

Writes:
- configs/benchmark-split.json: versioned manifest with row indices, group hashes, labels,
  exclusion counts, and the feature-selection record. It contains no URLs or hosts.
- data/processed/benchmark-sample.csv: local, Git-ignored rows with hosts and features.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data import benchmark_split as split  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
ACQUISITION_PATH = RAW_DIR / "acquisition.json"
MANIFEST_PATH = ROOT / "configs" / "benchmark-split.json"
SAMPLE_PATH = ROOT / "data" / "processed" / "benchmark-sample.csv"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_csv_hash() -> str:
    acquisition = json.loads(ACQUISITION_PATH.read_text(encoding="utf-8"))
    expected = acquisition["files"]["csv"]["sha256"]
    actual = sha256(CSV_PATH)
    if actual != expected:
        raise SystemExit("CSV checksum differs from acquisition.json; re-run the downloader.")
    return actual


def row_records(rows: pd.DataFrame) -> list[dict]:
    return [
        {
            "position": position,
            "row_index": int(row.row_index),
            "group_hash": split.group_hash(row.group),
            "is_phishing": int(row.is_phishing),
        }
        for position, row in enumerate(rows.itertuples())
    ]


def main() -> int:
    csv_hash = verified_csv_hash()
    raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig", keep_default_na=False)
    result = split.build(raw)
    features = result["selection"]["selected"]

    manifest = {
        "schema_version": 1,
        "dataset": {"csv_sha256": csv_hash, "csv_rows": int(len(raw))},
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")},
            "tldextract": version("tldextract"),
            "public_suffix_list": "tldextract bundled snapshot; private suffixes included",
        },
        "rules": {
            "seed": split.SEED,
            "target": "is_phishing = 1 - label",
            "deduplication": "exact URL, first occurrence kept",
            "grouping": "registrable domain of the host without www.; IP hosts by address",
            "conflicting_label_groups": "excluded",
            "rows_per_group": 1,
            "test_per_class": split.TEST_PER_CLASS,
            "train_per_class": split.TRAIN_PER_CLASS,
            "order": "test drawn first; features selected on the pool without test groups; "
            "training drawn from that pool",
        },
        "counts": {
            "prepared_rows": result["prepared_rows"],
            "training_pool_rows": result["pool_rows"],
            "exclusions": result["exclusions"],
        },
        "feature_selection": {
            "candidates": list(split.CANDIDATE_FEATURES),
            "method": "mutual information on a class-balanced pool sample, greedy order, "
            f"skip candidates with |Spearman| >= {split.MAX_ABS_SPEARMAN} to a selected one",
            **result["selection"],
        },
        "features": features,
        "train": row_records(result["train"]),
        "test": row_records(result["test"]),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    columns = ["row_index", "host", "is_phishing", *features]
    sample = pd.concat(
        [
            result["train"][columns].assign(partition="train"),
            result["test"][columns].assign(partition="test"),
        ]
    )
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)

    print(f"Features: {', '.join(features)}")
    print(f"Train: {len(result['train'])} rows; test: {len(result['test'])} rows.")
    print(f"Exclusions: {result['exclusions']}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
