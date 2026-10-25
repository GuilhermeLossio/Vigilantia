"""Build the published results file (reports/benchmark-results.json, schema version 1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app import results as contract

DISPLAY = {
    "majority": ("Majority baseline", "classical"),
    "logistic_regression": ("Logistic regression", "classical"),
    "rbf_svm": ("RBF SVM", "classical"),
    "quantum_kernel_svm": ("Quantum-kernel SVM", "simulation"),
}
BASE_LIMITATIONS = [
    "Exploratory benchmark: 24 training and 12 test examples, one split, one seed. "
    "Each test example changes accuracy by about 8.3 percentage points.",
    "Features use only the host of each URL, because PhiUSIIL's legitimate URLs are all bare "
    "https://www. homepages. Host features still reflect how the dataset was collected.",
    "The majority baseline breaks the 12/12 training tie toward legitimate, so it never "
    "predicts phishing. It has no decision scores, so average precision is not available.",
    "The quantum-kernel SVM is trained on an exact simulated kernel; hardware is used only to "
    "classify the test set. No model trained on hardware measurements is evaluated.",
    "No claim of quantum advantage is made.",
]
HARDWARE_PENDING = (
    "Quantum-kernel SVM rows here use an exact statevector simulation. IBM Quantum hardware "
    "results are pending and are not replaced by simulation."
)


def row(name: str, source: str, evaluation: dict) -> dict:
    return {
        "name": name,
        "kernel_source": source,
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
    }


def simulation_rows(evaluations: dict) -> list[dict]:
    return [row(*DISPLAY[name], evaluations[name]) for name in DISPLAY]


def build(
    run_id: str,
    rows: list[dict],
    split: dict,
    hardware: dict | None = None,
    extra_limitations: list[str] | None = None,
) -> dict:
    """Assemble and validate a results document against the service's contract."""
    has_qpu = any(r["kernel_source"] in {"qpu", "qpu_calibrated"} for r in rows)
    limitations = list(BASE_LIMITATIONS)
    if not has_qpu:
        limitations.insert(1, HARDWARE_PENDING)
    limitations += extra_limitations or []
    results = {
        "schema_version": contract.SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if has_qpu else "partial",
        "split": split,
        "hardware": hardware or {"backend": None, "shots": None, "job_ids": []},
        "models": rows,
        "limitations": limitations,
    }
    return contract.validate(results)


def write(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
