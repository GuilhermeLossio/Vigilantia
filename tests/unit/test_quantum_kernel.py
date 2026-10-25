"""Quantum kernel and benchmark models on synthetic angles; local simulation only."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.models import benchmark_models
from src.quantum import kernel as qkernel

pytestmark = pytest.mark.unit
RNG = np.random.default_rng(0)
ANGLES = RNG.uniform(0, math.pi / 2, size=(8, 4))


def test_feature_map_shape() -> None:
    circuit = qkernel.feature_map()
    assert circuit.num_qubits == 4
    assert [p.name for p in circuit.parameters] == ["x[0]", "x[1]", "x[2]", "x[3]"]
    assert circuit.decompose().count_ops()["cx"] == 6


def test_exact_kernel_is_a_valid_training_kernel() -> None:
    kernel = qkernel.statevector_kernel(ANGLES)
    diagnostics = qkernel.training_kernel_diagnostics(kernel)
    assert kernel.shape == (8, 8)
    assert diagnostics["max_asymmetry"] < 1e-12
    assert diagnostics["max_diagonal_error"] < 1e-12
    assert diagnostics["eigenvalue_check_passed"]
    assert ((kernel >= -1e-12) & (kernel <= 1 + 1e-12)).all()


def test_rectangular_kernel_matches_square_kernel_rows() -> None:
    square = qkernel.statevector_kernel(ANGLES)
    rectangular = qkernel.statevector_kernel(ANGLES[:3], ANGLES)
    assert rectangular == pytest.approx(square[:3])


def test_endpoints_of_the_angle_range_stay_distinguishable() -> None:
    low = np.zeros((1, 4))
    for qubit in range(4):
        high = low.copy()
        high[0, qubit] = math.pi / 2
        assert qkernel.statevector_kernel(low, high)[0, 0] < 0.999


def test_pair_circuit_all_zero_probability_equals_exact_kernel() -> None:
    x, y = ANGLES[0], ANGLES[1]
    circuit = qkernel.pair_circuit(x, y)
    circuit.remove_final_measurements()
    probability = Statevector(circuit).probabilities_dict().get("0000", 0.0)
    assert probability == pytest.approx(qkernel.statevector_kernel([x], [y])[0, 0], abs=1e-12)


def test_sampled_pair_circuit_estimates_the_kernel() -> None:
    x, y = ANGLES[2], ANGLES[3]
    shots = 4096
    counts = (
        AerSimulator(seed_simulator=42)
        .run(qkernel.pair_circuit(x, y), shots=shots)
        .result()
        .get_counts()
    )
    exact = qkernel.statevector_kernel([x], [y])[0, 0]
    tolerance = 5 * math.sqrt(max(exact * (1 - exact), 1e-3) / shots)
    assert qkernel.kernel_from_counts(counts) == pytest.approx(exact, abs=tolerance)


def test_kernel_from_counts() -> None:
    assert qkernel.kernel_from_counts({"0000": 3, "0101": 1}) == 0.75
    assert qkernel.kernel_from_counts({"1111": 5}) == 0
    with pytest.raises(ValueError):
        qkernel.kernel_from_counts({})


@pytest.mark.parametrize("values", [np.zeros((2, 3)), np.array([[0, 0, 0, np.nan]]), np.zeros(4)])
def test_invalid_encoded_inputs_are_rejected(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        qkernel.statevector_kernel(values)


def test_diagnostics_count_identical_examples_and_flag_invalid_matrices() -> None:
    repeated = np.vstack([ANGLES[:3], ANGLES[:1]])
    assert (
        qkernel.training_kernel_diagnostics(qkernel.statevector_kernel(repeated))[
            "identical_example_pairs"
        ]
        == 1
    )
    invalid = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert not qkernel.training_kernel_diagnostics(invalid)["eigenvalue_check_passed"]
    with pytest.raises(ValueError):
        qkernel.training_kernel_diagnostics(np.ones((2, 3)))


def test_models_fit_with_fixed_settings() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    angles = ANGLES.copy()
    angles[labels == 1] += 0.3
    models = benchmark_models.fit_all(angles, qkernel.statevector_kernel(angles), labels)
    assert set(models) == set(benchmark_models.MODEL_NAMES)
    settings = benchmark_models.settings(models)
    assert settings["quantum_kernel_svm"] == {
        "C": 1.0,
        "class_weight": None,
        "kernel": "precomputed",
        "gamma": "scale",
        "max_iter": -1,
    }
    assert settings["logistic_regression"]["max_iter"] == 1000
    with pytest.raises(ValueError):
        benchmark_models.fit_all(angles[:3], qkernel.statevector_kernel(angles), labels)


def test_convergence_warning_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = np.array([0, 1] * 4)
    original = benchmark_models.build_models

    def slow_models() -> dict:
        models = original()
        models["logistic_regression"] = LogisticRegression(max_iter=1)
        return models

    monkeypatch.setattr(benchmark_models, "build_models", slow_models)
    with pytest.raises(ConvergenceWarning):
        benchmark_models.fit_all(ANGLES * 50, qkernel.statevector_kernel(ANGLES), labels)
