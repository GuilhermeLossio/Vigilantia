"""Build the replay file: recorded per-sample test predictions, by source.

The demo shows quantum-kernel SVM predictions as replay because live quantum inference would
need new QPU jobs. Each sample is identified by its test position, dataset row index, and
hashed group, never by host or URL, matching configs/benchmark-split.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
SOURCES = ("simulation", "qpu", "qpu_calibrated")


def _prediction(evaluation: dict, index: int) -> dict:
    label = evaluation["predictions"][index]
    scores = evaluation.get("decision_scores")
    return {
        "prediction": "phishing" if label else "legitimate",
        "is_phishing": int(label),
        "decision_score": None if scores is None else float(scores[index]),
    }


def build(manifest: dict, simulation: dict, qpu: dict | None) -> dict:
    """Combine the frozen test rows with the simulated and (optional) QPU predictions.

    `qpu` holds run_id, backend, inputs, and evaluations of a full QPU run.
    """
    if qpu is not None and qpu["inputs"] != simulation["inputs"]:
        raise ValueError("Simulation and QPU evaluations used different inputs.")
    if len(manifest["test"]) != len(simulation["evaluations"]["quantum_kernel_svm"]["predictions"]):
        raise ValueError("The manifest and the evaluations describe different test sets.")
    evaluations = {"simulation": simulation["evaluations"]["quantum_kernel_svm"]}
    if qpu is not None:
        evaluations.update(qpu["evaluations"])
    samples = []
    for index, row in enumerate(manifest["test"]):
        samples.append(
            {
                "position": row["position"],
                "row_index": row["row_index"],
                "group_hash": row["group_hash"],
                "is_phishing": row["is_phishing"],
                "predictions": {
                    source: _prediction(evaluations[source], index)
                    for source in SOURCES
                    if source in evaluations
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "label": "replay of recorded predictions; not live inference",
        "model": "quantum_kernel_svm",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_runs": {
            "simulation": simulation["run_id"],
            **({"qpu": qpu["run_id"], "backend": qpu["backend"]} if qpu else {}),
        },
        "samples": samples,
    }


def write(replay: dict, path: Path) -> None:
    path.write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
