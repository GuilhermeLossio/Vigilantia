"""Add measured runtimes to reports/benchmark-results.json.

Times fit, test-kernel computation, and prediction for every published row on this machine
(median of repeats). Hardware rows reuse the IBM execution time recorded by the QPU run
named in the results file. Local only; no IBM call. Run it after evaluating, because a new
evaluation rewrites the results file without runtimes.

Also writes artifacts/benchmark/runtime-local.json (Git-ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import results as contract  # noqa: E402
from src.evaluation import inputs as benchmark_inputs  # noqa: E402
from src.evaluation import publish, runtime  # noqa: E402
from src.quantum.qpu_run import RUNS_DIR  # noqa: E402

RESULTS_PATH = ROOT / "reports" / "benchmark-results.json"
KERNEL_FILES = {"qpu": "test-kernel-raw.npy", "qpu_calibrated": "test-kernel-calibrated.npy"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=runtime.REPEATS)
    args = parser.parse_args()

    results = contract.load(RESULTS_PATH)
    data = benchmark_inputs.load()
    sources = {model["kernel_source"] for model in results["models"]}
    qpu_kernels, usage = {}, None
    if sources & set(KERNEL_FILES):
        run_dir = RUNS_DIR / results["run_id"]
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if record["inputs"] != data.input_hashes:
            print("The published QPU run used other inputs than the current artifacts.")
            return 1
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        usage = evaluation["usage"]
        qpu_kernels = {s: np.load(run_dir / KERNEL_FILES[s]) for s in sources & set(KERNEL_FILES)}

    measured = runtime.measure(data, qpu_kernels, args.repeats)
    environment = runtime.environment()
    timed = publish.with_runtime(results, measured, args.repeats, environment, usage)
    publish.write(timed, RESULTS_PATH)
    (benchmark_inputs.ARTIFACT_DIR / "runtime-local.json").write_text(
        json.dumps(
            {
                "measured_at_utc": datetime.now(UTC).isoformat(),
                "results_run_id": results["run_id"],
                "repeats": args.repeats,
                "environment": environment,
                "seconds": measured,
                "qpu_usage": usage,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"{'model':<22}{'source':<16}{'fit':>12}{'test kernel':>14}{'predict':>12}")
    for model in timed["models"]:
        values = "".join(
            f"{'n/a' if v is None else f'{v:.6f}':>{w}}"
            for v, w in zip(model["runtime_seconds"].values(), (12, 14, 12), strict=True)
        )
        print(f"{model['name']:<22}{model['kernel_source']:<16}{values}")
    print(f"Environment: {environment}")
    print(f"Results file: {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
