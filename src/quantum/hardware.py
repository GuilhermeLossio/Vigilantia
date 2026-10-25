"""Hardware kernel acquisition: pair plan, ISA transpilation, sampling, and calibration.

The same code runs against a local fake backend (noisy rehearsal) and, later, against an
IBM Quantum backend. Workload per docs/benchmark-protocol.md, with 24 training and 12 test
examples: 288 test-to-training pairs, 24 calibration pairs drawn from the 276 training
pairs, and 24 training diagonal controls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from src.quantum.kernel import kernel_from_counts, pair_circuit

SEED = 42
SHOTS = 1024
CALIBRATION_PAIRS = 24
OPTIMIZATION_LEVEL = 1


@dataclass(frozen=True)
class PairSpec:
    """`kind` is test, calibration, or diagonal. `left` indexes the test set for test pairs
    and the training set otherwise; `right` always indexes the training set."""

    kind: str
    left: int
    right: int


def plan_pairs(
    n_train: int, n_test: int, n_calibration: int = CALIBRATION_PAIRS, seed: int = SEED
) -> list[PairSpec]:
    training_pairs = [(i, j) for i in range(n_train) for j in range(i + 1, n_train)]
    if n_calibration > len(training_pairs):
        raise ValueError("More calibration pairs requested than training pairs exist.")
    chosen = np.random.default_rng(seed).choice(len(training_pairs), n_calibration, replace=False)
    return (
        [PairSpec("test", i, j) for i in range(n_test) for j in range(n_train)]
        + [PairSpec("calibration", *training_pairs[k]) for k in sorted(chosen)]
        + [PairSpec("diagonal", i, i) for i in range(n_train)]
    )


def build_circuits(
    plan: list[PairSpec], train_angles: np.ndarray, test_angles: np.ndarray
) -> list[QuantumCircuit]:
    return [
        pair_circuit(
            test_angles[p.left] if p.kind == "test" else train_angles[p.left], train_angles[p.right]
        )
        for p in plan
    ]


def transpile_for(backend, circuits: list[QuantumCircuit], seed: int = SEED) -> tuple[list, dict]:
    """Bind-then-transpile each circuit for the backend; return ISA circuits and statistics."""
    manager = generate_preset_pass_manager(
        optimization_level=OPTIMIZATION_LEVEL, backend=backend, seed_transpiler=seed
    )
    started = time.perf_counter()
    isa = manager.run(circuits)
    elapsed = time.perf_counter() - started
    depths = [c.depth() for c in isa]
    two_qubit = [sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in isa]
    layouts = sorted({tuple(c.layout.final_index_layout()) for c in isa if c.layout is not None})
    return isa, {
        "optimization_level": OPTIMIZATION_LEVEL,
        "seed_transpiler": seed,
        "depth": {"min": min(depths), "mean": float(np.mean(depths)), "max": max(depths)},
        "two_qubit_gates": {
            "min": min(two_qubit),
            "mean": float(np.mean(two_qubit)),
            "max": max(two_qubit),
        },
        "distinct_physical_layouts": [list(layout) for layout in layouts],
        "transpile_seconds": elapsed,
    }


SAMPLERS = ("executor", "v2")


def sampler_class(implementation: str):
    """ "executor" is the client-side Sampler that replaces SamplerV2, which qiskit-ibm-runtime
    0.50 deprecated. "v2" is SamplerV2, the established path on the Open Plan."""
    if implementation == "executor":
        from qiskit_ibm_runtime.executor_sampler import Sampler
    elif implementation == "v2":
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    else:
        raise ValueError(f"Unknown sampler implementation; use one of {SAMPLERS}.")
    return Sampler


def submit(backend, isa_circuits: list, shots: int = SHOTS, implementation: str = "v2"):
    """Submit one job in job mode and return it without waiting for the result."""
    return sampler_class(implementation)(mode=backend).run(isa_circuits, shots=shots)


def counts_from_result(result) -> list[dict]:
    """Counts of the `meas` register (from measure_all) for every pub, in submission order."""
    return [pub.data.meas.get_counts() for pub in result]


def run_sampler(
    backend, isa_circuits: list, shots: int = SHOTS, implementation: str = "executor"
) -> tuple[list[dict], dict]:
    """Submit and wait for one job; used by the local rehearsal on fake backends."""
    started = time.perf_counter()
    job = submit(backend, isa_circuits, shots, implementation)
    job_id = job.job_id()
    counts = counts_from_result(job.result())
    elapsed = time.perf_counter() - started
    return counts, {
        "sampler": implementation,
        "job_ids": [job_id],
        "shots_requested": shots,
        "shots_returned": sorted({sum(c.values()) for c in counts}),
        "client_seconds": elapsed,
    }


def batches(count: int, size: int | None) -> list[tuple[int, int]]:
    """Half-open [start, stop) ranges covering `count` circuits in jobs of at most `size`."""
    if count <= 0:
        raise ValueError("Nothing to submit.")
    step = count if not size else size
    if step <= 0:
        raise ValueError("Batch size must be positive.")
    return [(start, min(start + step, count)) for start in range(0, count, step)]


def assemble(plan: list[PairSpec], counts: list[dict], n_train: int, n_test: int) -> dict:
    """Turn per-circuit counts into the test kernel and the calibration/diagonal records."""
    if len(plan) != len(counts):
        raise ValueError("Every planned pair needs exactly one counts record.")
    test_kernel = np.full((n_test, n_train), np.nan)
    calibration, diagonal = [], []
    for spec, record in zip(plan, counts, strict=True):
        value = kernel_from_counts(record)
        if spec.kind == "test":
            test_kernel[spec.left, spec.right] = value
        elif spec.kind == "calibration":
            calibration.append((spec.left, spec.right, value))
        else:
            diagonal.append((spec.left, value))
    if np.isnan(test_kernel).any():
        raise ValueError("The measured test kernel is incomplete.")
    return {"test_kernel": test_kernel, "calibration": calibration, "diagonal": diagonal}


def fit_calibration(ideal: np.ndarray, measured: np.ndarray) -> dict:
    """Least-squares fit of measured = a * ideal + c on training pairs only."""
    ideal, measured = np.asarray(ideal, dtype=float), np.asarray(measured, dtype=float)
    if len(ideal) < 2 or np.ptp(ideal) == 0:
        return {"valid": False, "reason": "calibration pairs do not span distinct ideal values"}
    a, c = np.polyfit(ideal, measured, 1)
    residual = measured - (a * ideal + c)
    fit = {"a": float(a), "c": float(c), "rmse": float(np.sqrt(np.mean(residual**2)))}
    if a <= 0:
        return {"valid": False, "reason": "non-positive slope", **fit}
    return {"valid": True, **fit}


def apply_calibration(kernel: np.ndarray, calibration: dict) -> tuple[np.ndarray, int]:
    """Map measured values back to the ideal scale; return the kernel and clipped count."""
    if not calibration["valid"]:
        raise ValueError("Cannot apply an invalid calibration.")
    mapped = (kernel - calibration["c"]) / calibration["a"]
    clipped = int(((mapped < 0) | (mapped > 1)).sum())
    return np.clip(mapped, 0.0, 1.0), clipped
