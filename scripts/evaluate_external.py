"""Evaluate the frozen benchmark models on the external dataset (configs/external-validation.json).

Steps: verify the downloaded snapshot (no network access here), apply the frozen preparation, classify the primary
set with all four models (the quantum kernel by exact simulation; no QPU), add bootstrap
intervals, and publish aggregates only.

Writes:
- reports/external-validation.json: aggregates, validated by src.evaluation.external.
- artifacts/external/predictions.csv and external-kernel.npy (Git-ignored): per-row
  predictions keyed by file row and group hash, never by host.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import download_external  # noqa: E402
from src.data.benchmark_split import group_hash  # noqa: E402
from src.evaluation import external  # noqa: E402
from src.evaluation import inputs as benchmark_inputs  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.features.scaling import AngleScaler, ScalingError  # noqa: E402
from src.inference.predictor import MODELS_PATH, _version  # noqa: E402
from src.quantum import kernel as qkernel  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "external-validation.json"
SPLIT_PATH = ROOT / "configs" / "benchmark-split.json"
PHIUSIIL_CSV = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
REPORT_PATH = ROOT / "reports" / "external-validation.json"
ARTIFACT_DIR = ROOT / "artifacts" / "external"


def load_phiusiil_groups() -> set[str]:
    if not PHIUSIIL_CSV.exists():
        raise external.ExternalError("PhiUSIIL CSV missing; run scripts/download_dataset.py.")
    expected = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["dataset"]["csv_sha256"]
    if benchmark_inputs.sha256(PHIUSIIL_CSV) != expected:
        raise external.ExternalError("The PhiUSIIL CSV differs from the benchmark snapshot.")
    domains = pd.read_csv(PHIUSIIL_CSV, usecols=["Domain"], dtype=str, keep_default_na=False)
    return external.phiusiil_groups(domains["Domain"])


def classify(data, angles: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray, float]:
    started = time.perf_counter()
    kernel = qkernel.statevector_kernel(angles, data.train_angles)
    kernel_seconds = time.perf_counter() - started
    evaluations = {}
    for name in external.MODELS:
        inputs = kernel if name == "quantum_kernel_svm" else angles
        evaluations[name] = evaluate(data.models[name], inputs, labels)
    return evaluations, kernel, kernel_seconds


def with_intervals(evaluations: dict, labels: np.ndarray, settings: dict) -> tuple[list, int]:
    indices = external.resample_indices(len(labels), settings["resamples"], settings["seed"])
    rows, used = [], 0
    for name, evaluation in evaluations.items():
        scores = evaluation["decision_scores"]
        intervals, used = external.bootstrap_intervals(
            labels,
            np.asarray(evaluation["predictions"]),
            None if scores is None else np.asarray(scores),
            indices,
            settings["level"],
        )
        rows.append(external.model_row(name, evaluation, intervals))
    return rows, used


def save_predictions(prepared: external.PreparedExternal, evaluations: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "row_index": prepared.rows["row_index"],
            "group_hash": prepared.rows["group"].map(group_hash),
            "is_phishing": prepared.rows["is_phishing"],
        }
    )
    for name, evaluation in evaluations.items():
        table[f"{name}_prediction"] = evaluation["predictions"]
        if evaluation["decision_scores"] is not None:
            table[f"{name}_score"] = evaluation["decision_scores"]
    table.to_csv(ARTIFACT_DIR / "predictions.csv", index=False)


def print_table(rows: list[dict], preparation: dict) -> None:
    counts = preparation["class_counts"]
    print(f"Primary set: {preparation['primary_rows']} rows {counts}")
    for entry in preparation["exclusions"]:
        print(f"  excluded by {entry['step']:<20}{entry['rows']:>6}")
    print(f"{'model':<22}{'source':<12}" + "".join(f"{m[:9]:>11}" for m in external.METRICS))
    for row in rows:
        values = "".join(
            f"{'n/a' if v is None else f'{v:.3f}':>11}" for v in row["metrics"].values()
        )
        print(
            f"{row['model']:<22}{row['kernel_source']:<12}{values}   CM {row['confusion_matrix']}"
        )
    print("95% bootstrap intervals:")
    for row in rows:
        spans = "".join(
            f"{'n/a' if ci is None else f'{ci[0]:.2f}-{ci[1]:.2f}':>11}"
            for ci in row["confidence_intervals"].values()
        )
        print(f"{row['model']:<34}{spans}")


def verified_snapshot(dataset: dict) -> dict:
    """Verify the existing snapshot without downloading; see scripts/download_external.py."""
    destination = download_external.DEFAULT_DESTINATION
    if not destination.exists():
        raise download_external.AcquisitionError("no snapshot in data/external/")
    return download_external.reuse_existing(destination, dataset)


def load_inputs(dataset: dict) -> tuple:
    """Benchmark inputs, the prepared primary set, and the frozen scaler."""
    data = benchmark_inputs.load()
    raw = pd.read_csv(
        download_external.DEFAULT_DESTINATION / dataset["file"],
        usecols=dataset["columns_used"],
        dtype=str,
        keep_default_na=False,
    )
    prepared = external.prepare(raw, dataset["label_map"], load_phiusiil_groups())
    if prepared.rows.empty:
        raise external.ExternalError("Preparation left no rows to evaluate.")
    scaler = AngleScaler.from_dict(
        json.loads(benchmark_inputs.SCALER_PATH.read_text(encoding="utf-8"))
    )
    if list(scaler.features) != data.features:
        raise external.ExternalError("Scaler features differ from the benchmark inputs.")
    return data, prepared, scaler


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    try:
        manifest = verified_snapshot(dataset)
    except download_external.AcquisitionError as error:
        print(f"External dataset unavailable: {error}", file=sys.stderr)
        print("Run scripts/download_external.py first.", file=sys.stderr)
        return 1
    try:
        data, prepared, scaler = load_inputs(dataset)
    except (benchmark_inputs.InputError, OSError, ScalingError, external.ExternalError) as error:
        print(f"Cannot evaluate: {error}", file=sys.stderr)
        return 1
    _, angles, clipped = external.featurize(prepared.rows["host"], scaler)
    labels = prepared.rows["is_phishing"].to_numpy()

    evaluations, kernel, kernel_seconds = classify(data, angles, labels)
    settings = config["metrics"]["confidence_intervals"]
    rows, used = with_intervals(evaluations, labels, settings)

    report = external.build_report(
        protocol={
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": external.protocol_sha256(CONFIG_PATH),
            "frozen_at_utc": config["frozen_at_utc"],
        },
        dataset={
            "title": dataset["title"],
            "doi": dataset["doi"],
            "file": dataset["file"],
            "sha256": manifest["file"]["sha256"],
            "rows": manifest["inspection"]["rows"],
            "license": dataset["license"],
            "attribution": download_external.ATTRIBUTION,
        },
        benchmark={
            "model_version": _version(
                benchmark_inputs.SCALER_PATH.read_bytes(), MODELS_PATH.read_bytes()
            ),
            "training_examples": int(len(data.train_labels)),
            "benchmark_test_examples": int(len(data.test_labels)),
            "input_hashes": data.input_hashes,
        },
        prepared=prepared,
        clipped=clipped,
        bootstrap={
            "method": external.BOOTSTRAP_METHOD,
            "level": settings["level"],
            "resamples": settings["resamples"],
            "resamples_used": used,
            "seed": settings["seed"],
        },
        models=rows,
    )
    external.write(report, REPORT_PATH)
    save_predictions(prepared, evaluations)
    np.save(ARTIFACT_DIR / "external-kernel.npy", kernel)
    print_table(rows, report["preparation"])
    print(f"Clipped values: {clipped}")
    print(f"Simulated external kernel {kernel.shape} in {kernel_seconds:.1f} s; no QPU used.")
    print(f"Report: {REPORT_PATH.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
