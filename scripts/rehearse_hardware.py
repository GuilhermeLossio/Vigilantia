"""Noisy rehearsal of the hardware path on a local fake IBM backend. No account or QPU is used.

Runs the exact hardware workload (288 test pairs, 24 calibration pairs, 24 diagonal controls,
1,024 shots) through ISA transpilation and SamplerV2 on a fake backend that carries a real
device's noise model and connectivity. Results are a diagnostic labeled "noisy simulation";
they are never published as hardware results.

Usage: python scripts/rehearse_hardware.py [--backend FakeFez] [--sampler v2|executor]
Outputs go to the Git-ignored artifacts/benchmark/rehearsal-<backend>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

from src.evaluation import inputs as benchmark_inputs  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.quantum import hardware  # noqa: E402
from src.quantum import kernel as qkernel  # noqa: E402

FAKE_BACKENDS = ("FakeFez", "FakeTorino", "FakeMarrakesh", "FakeKingston", "FakeBrisbane")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=FAKE_BACKENDS, default="FakeFez")
    # SamplerV2 is the default here: locally the client-side sampler took about 600 s for
    # the 336 circuits versus about 11 s. Its deprecation warning is expected.
    parser.add_argument("--sampler", choices=hardware.SAMPLERS, default="v2")
    args = parser.parse_args()
    backend = getattr(fake_provider, args.backend)()

    data = benchmark_inputs.load()
    n_train, n_test = len(data.train_angles), len(data.test_angles)
    plan = hardware.plan_pairs(n_train, n_test)
    circuits = hardware.build_circuits(plan, data.train_angles, data.test_angles)
    isa, transpile_stats = hardware.transpile_for(backend, circuits)
    counts, job = hardware.run_sampler(backend, isa, implementation=args.sampler)
    measured = hardware.assemble(plan, counts, n_train, n_test)

    ideal_train = qkernel.statevector_kernel(data.train_angles)
    ideal_test = qkernel.statevector_kernel(data.test_angles, data.train_angles)
    pairs = measured["calibration"]
    calibration = hardware.fit_calibration(
        np.array([ideal_train[i, j] for i, j, _ in pairs]), np.array([v for _, _, v in pairs])
    )
    raw_test = measured["test_kernel"]
    variants = {"noisy_raw": raw_test}
    clipped = None
    if calibration["valid"]:
        variants["noisy_calibrated"], clipped = hardware.apply_calibration(raw_test, calibration)

    model = data.models["quantum_kernel_svm"]
    evaluations = {
        "ideal": evaluate(model, ideal_test, data.test_labels),
        **{name: evaluate(model, k, data.test_labels) for name, k in variants.items()},
    }
    kernel_error = {name: float(np.abs(k - ideal_test).mean()) for name, k in variants.items()}
    diagonal = [v for _, v in measured["diagonal"]]

    out = benchmark_inputs.ARTIFACT_DIR / f"rehearsal-{backend.name}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test-kernel-raw.npy", raw_test)
    if "noisy_calibrated" in variants:
        np.save(out / "test-kernel-calibrated.npy", variants["noisy_calibrated"])
    (out / "counts.json").write_text(
        json.dumps(
            [
                {"kind": p.kind, "left": p.left, "right": p.right, "counts": c}
                for p, c in zip(plan, counts, strict=True)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "label": "noisy simulation (fake backend); not a hardware result",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": backend.name,
        "inputs": data.input_hashes,
        "circuits": len(plan),
        "plan": {
            kind: sum(p.kind == kind for p in plan) for kind in ("test", "calibration", "diagonal")
        },
        "transpilation": transpile_stats,
        "job": job,
        "calibration": {**calibration, "clipped_test_values": clipped},
        "diagonal_controls": {"mean": float(np.mean(diagonal)), "min": float(np.min(diagonal))},
        "mean_abs_kernel_error_vs_ideal": kernel_error,
        "evaluations": evaluations,
        "environment": {
            name: version(name) for name in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    t = transpile_stats
    print(f"Backend {backend.name}: {len(plan)} circuits, {job['shots_requested']} shots each.")
    print(
        f"Transpiled depth {t['depth']['min']}-{t['depth']['max']}, two-qubit gates {t['two_qubit_gates']['min']}-{t['two_qubit_gates']['max']}."
    )
    print(f"Diagonal controls: mean {np.mean(diagonal):.3f}, min {np.min(diagonal):.3f} (ideal 1).")
    print(f"Calibration: {calibration}")
    print(f"Mean |kernel - ideal|: {kernel_error}")
    for name, e in evaluations.items():
        m = e["metrics"]
        print(
            f"  {name:<17} acc {m['accuracy']:.3f}  bal {m['balanced_accuracy']:.3f}  CM {e['confusion_matrix']}"
        )
    print(
        f"Elapsed: transpile {t['transpile_seconds']:.1f}s, sampling {job['client_seconds']:.1f}s. Artifacts: {out.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
