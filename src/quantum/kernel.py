"""Fidelity quantum kernel for the four-qubit ZZ feature map (docs/quantum-circuit.md).

K(x, y) = |<0000| U(y)^dagger U(x) |0000>|^2.

- `statevector_kernel` computes K exactly from simulated states. It produces the training
  kernel and the ideal test kernel.
- `pair_circuit` builds the compute-uncompute circuit whose all-zero probability estimates
  K(x, y) on hardware. `kernel_from_counts` turns measured counts into that estimate.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

NUM_QUBITS = 4
FEATURE_MAP_SETTINGS = {
    "feature_dimension": NUM_QUBITS,
    "reps": 1,
    "entanglement": "linear",
    "alpha": 2.0,
}
EIGENVALUE_TOLERANCE = 1e-10


@cache
def feature_map() -> QuantumCircuit:
    """The fixed encoding U(x); parameters are x[0]..x[3] in feature order."""
    return zz_feature_map(**FEATURE_MAP_SETTINGS)


def _check_inputs(values: object) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != NUM_QUBITS:
        raise ValueError(f"Expected rows of {NUM_QUBITS} encoded features.")
    if not np.isfinite(matrix).all():
        raise ValueError("Encoded features must be finite.")
    return matrix


def encode(row: np.ndarray) -> QuantumCircuit:
    circuit = feature_map()
    return circuit.assign_parameters(dict(zip(circuit.parameters, row, strict=True)))


def statevectors(values: object) -> np.ndarray:
    """One row per example: the simulated state U(x)|0000>."""
    return np.array([Statevector(encode(row)).data for row in _check_inputs(values)])


def statevector_kernel(left: object, right: object | None = None) -> np.ndarray:
    """Exact kernel matrix between two sets of examples (or one set with itself)."""
    left_states = statevectors(left)
    right_states = left_states if right is None else statevectors(right)
    return np.abs(left_states.conj() @ right_states.T) ** 2


def pair_circuit(x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
    """Prepare U(x)|0000>, apply U(y)^dagger, and measure all qubits."""
    circuit = encode(np.asarray(x, dtype=float)).compose(
        encode(np.asarray(y, dtype=float)).inverse()
    )
    circuit.measure_all()
    return circuit


def kernel_from_counts(counts: dict[str, int]) -> float:
    """Estimate K as the fraction of shots that returned all zeros."""
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("Counts contain no shots.")
    return counts.get("0" * NUM_QUBITS, 0) / shots


def training_kernel_diagnostics(kernel: np.ndarray) -> dict:
    """Validity checks and the concentration diagnostic required by the protocol."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Training kernel must be square.")
    off_diagonal = kernel[~np.eye(len(kernel), dtype=bool)]
    identical_pairs = int(np.triu(np.isclose(kernel, 1.0, atol=1e-12), k=1).sum())
    min_eigenvalue = float(np.linalg.eigvalsh((kernel + kernel.T) / 2).min())
    return {
        "shape": list(kernel.shape),
        "max_asymmetry": float(np.abs(kernel - kernel.T).max()),
        "max_diagonal_error": float(np.abs(np.diag(kernel) - 1).max()),
        "min_eigenvalue": min_eigenvalue,
        "eigenvalue_check_passed": min_eigenvalue >= -EIGENVALUE_TOLERANCE,
        "off_diagonal_mean": float(off_diagonal.mean()),
        "off_diagonal_std": float(off_diagonal.std()),
        "off_diagonal_min": float(off_diagonal.min()),
        "off_diagonal_max": float(off_diagonal.max()),
        "identical_example_pairs": identical_pairs,
    }
