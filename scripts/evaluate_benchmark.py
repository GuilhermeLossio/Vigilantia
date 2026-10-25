"""Classify the 12 test examples with all four models, using the exact simulated kernel.

Writes:
- reports/benchmark-results.json: the published results file read by the Flask service,
  with status "partial" until QPU results exist.
- artifacts/benchmark/evaluation-simulation.json and test-kernel-ideal.npy (Git-ignored).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.evaluation import inputs as benchmark_inputs  # noqa: E402
from src.evaluation import publish  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.quantum import kernel as qkernel  # noqa: E402

RESULTS_PATH = ROOT / "reports" / "benchmark-results.json"


def main() -> int:
    data = benchmark_inputs.load()
    started = time.perf_counter()
    test_kernel = qkernel.statevector_kernel(data.test_angles, data.train_angles)
    kernel_seconds = time.perf_counter() - started

    evaluations = {}
    for name, model in data.models.items():
        inputs = test_kernel if name == "quantum_kernel_svm" else data.test_angles
        evaluations[name] = evaluate(model, inputs, data.test_labels)

    run_id = "sim-" + data.input_hashes["split_manifest_sha256"][:8]
    split = {"train": len(data.train_labels), "test": len(data.test_labels), "seed": 42}
    results = publish.build(run_id, publish.simulation_rows(evaluations), split)
    generated = results["generated_at_utc"]
    if (
        RESULTS_PATH.exists()
        and json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get("status") == "complete"
    ):
        print("Kept the published file: it already contains QPU results (status complete).")
    else:
        publish.write(results, RESULTS_PATH)

    np.save(benchmark_inputs.ARTIFACT_DIR / "test-kernel-ideal.npy", test_kernel)
    detail = {
        "run_id": run_id,
        "generated_at_utc": generated,
        "inputs": data.input_hashes,
        "test_values_clipped": data.test_clipped,
        "ideal_test_kernel_seconds": kernel_seconds,
        "evaluations": evaluations,
    }
    (benchmark_inputs.ARTIFACT_DIR / "evaluation-simulation.json").write_text(
        json.dumps(detail, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"{'model':<22}{'source':<12}"
        + "".join(f"{m[:9]:>11}" for m in evaluations["rbf_svm"]["metrics"])
    )
    for name, evaluation in evaluations.items():
        values = "".join(
            f"{'n/a' if v is None else f'{v:.3f}':>11}" for v in evaluation["metrics"].values()
        )
        print(
            f"{name:<22}{publish.DISPLAY[name][1]:<12}{values}   CM {evaluation['confusion_matrix']}"
        )
    print(f"Results file: {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
