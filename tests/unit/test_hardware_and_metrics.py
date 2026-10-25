"""Hardware pair plan, calibration, fake-backend sampling, and metrics; local only."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from src.evaluation.metrics import METRIC_NAMES, evaluate
from src.quantum import hardware
from src.quantum import kernel as qkernel

pytestmark = pytest.mark.unit
RNG = np.random.default_rng(1)
TRAIN = RNG.uniform(0, math.pi / 2, size=(6, 4))
TEST = RNG.uniform(0, math.pi / 2, size=(3, 4))


def test_plan_matches_the_protocol_workload() -> None:
    plan = hardware.plan_pairs(24, 12)
    kinds = [p.kind for p in plan]
    assert (kinds.count("test"), kinds.count("calibration"), kinds.count("diagonal")) == (
        288,
        24,
        24,
    )
    assert len(plan) == 336
    calibration = [(p.left, p.right) for p in plan if p.kind == "calibration"]
    assert all(i < j < 24 for i, j in calibration)
    assert len(set(calibration)) == 24
    assert plan == hardware.plan_pairs(24, 12)


def test_plan_rejects_impossible_calibration() -> None:
    with pytest.raises(ValueError):
        hardware.plan_pairs(3, 1, n_calibration=4)


def test_build_circuits_uses_test_rows_only_for_test_pairs() -> None:
    plan = [hardware.PairSpec("test", 0, 1), hardware.PairSpec("calibration", 0, 1)]
    circuits = hardware.build_circuits(plan, TRAIN, TEST)
    for circuit, left in zip(circuits, (TEST[0], TRAIN[0]), strict=True):
        circuit.remove_final_measurements()
        expected = qkernel.statevector_kernel([left], [TRAIN[1]])[0, 0]
        assert Statevector(circuit).probabilities_dict().get("0000", 0) == pytest.approx(expected)


def test_assemble_places_every_value() -> None:
    plan = hardware.plan_pairs(3, 2, n_calibration=2)
    counts = [{"0000": k + 1, "1111": 10 - (k + 1)} for k in range(len(plan))]
    result = hardware.assemble(plan, counts, 3, 2)
    assert result["test_kernel"].shape == (2, 3)
    assert result["test_kernel"][0, 0] == pytest.approx(0.1)
    assert len(result["calibration"]) == 2 and len(result["diagonal"]) == 3
    with pytest.raises(ValueError):
        hardware.assemble(plan, counts[:-1], 3, 2)


def test_calibration_recovers_a_linear_distortion() -> None:
    ideal = np.linspace(0, 1, 10)
    fit = hardware.fit_calibration(ideal, 0.8 * ideal + 0.05)
    assert fit["valid"]
    assert fit["a"] == pytest.approx(0.8) and fit["c"] == pytest.approx(0.05)
    mapped, clipped = hardware.apply_calibration(np.array([[0.05, 0.45, 0.0]]), fit)
    assert mapped[0, :2] == pytest.approx([0, 0.5])
    assert mapped[0, 2] == 0 and clipped == 1


def test_invalid_calibrations_are_reported_not_applied() -> None:
    assert not hardware.fit_calibration(np.ones(4), np.ones(4))["valid"]
    flipped = hardware.fit_calibration(np.linspace(0, 1, 5), 1 - np.linspace(0, 1, 5))
    assert not flipped["valid"] and "slope" in flipped["reason"]
    with pytest.raises(ValueError):
        hardware.apply_calibration(np.zeros((1, 1)), flipped)


@pytest.mark.filterwarnings("ignore:The SamplerV2 class is deprecated:DeprecationWarning")
@pytest.mark.parametrize("implementation", hardware.SAMPLERS)
def test_fake_backend_sampling_round_trip(implementation: str) -> None:
    backend = FakeManilaV2()
    plan = hardware.plan_pairs(2, 1, n_calibration=1)
    isa, stats = hardware.transpile_for(backend, hardware.build_circuits(plan, TRAIN[:2], TEST[:1]))
    counts, job = hardware.run_sampler(backend, isa, shots=256, implementation=implementation)
    assert len(counts) == len(plan) == 5
    assert job["shots_returned"] == [256] and job["job_ids"]
    assert job["sampler"] == implementation
    assert stats["depth"]["max"] >= stats["depth"]["min"] >= 1
    diagonal = [
        qkernel.kernel_from_counts(c)
        for p, c in zip(plan, counts, strict=True)
        if p.kind == "diagonal"
    ]
    assert all(value > 0.7 for value in diagonal)


def test_unknown_sampler_is_rejected() -> None:
    with pytest.raises(ValueError):
        hardware.run_sampler(FakeManilaV2(), [], implementation="other")


def test_metrics_for_a_scored_and_an_unscored_model() -> None:
    x = np.array([[0.0], [0.1], [0.9], [1.0]])
    y = np.array([0, 0, 1, 1])
    scored = evaluate(SVC(kernel="linear").fit(x, y), x, y)
    assert set(scored["metrics"]) == set(METRIC_NAMES)
    assert scored["metrics"]["accuracy"] == 1.0
    assert scored["confusion_matrix"] == [[2, 0], [0, 2]]
    majority = evaluate(DummyClassifier(strategy="most_frequent").fit(x, y), x, y)
    assert majority["metrics"]["average_precision"] is None
    assert majority["metrics"]["precision"] == 0.0
    assert majority["confusion_matrix"] == [[2, 0], [2, 0]]
