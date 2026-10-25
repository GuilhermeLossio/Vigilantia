"""Measure local runtimes for the published benchmark: wall-clock medians over repeats.

Times are measured on the machine running the script, not on the QPU. The quantum-kernel
SVM's fit includes simulating its training kernel; its test kernel is simulated here, while
the hardware rows take their test-kernel time from IBM's recorded execution time.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from importlib import metadata

import numpy as np
from sklearn.base import clone

from src.quantum import kernel as qkernel

REPEATS = 25
PACKAGES = ("numpy", "scikit-learn", "qiskit")
MODEL_ROWS = {
    "majority": "majority",
    "logistic_regression": "logistic_regression",
    "rbf_svm": "rbf_svm",
    "quantum_kernel_svm": "simulation",
}


def median_seconds(task: Callable[[], object], repeats: int = REPEATS) -> float:
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        task()
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def _predict(model, inputs: np.ndarray) -> None:
    model.predict(inputs)
    if hasattr(model, "decision_function"):
        model.decision_function(inputs)


def measure(
    data,
    qpu_kernels: dict | None = None,
    repeats: int = REPEATS,
    test_angles: np.ndarray | None = None,
) -> dict:
    """Return {row: {"fit", "test_kernel", "predict"}} in seconds for this machine.

    Rows are the classical model names, "simulation", and each key of `qpu_kernels`
    (for example "qpu" and "qpu_calibrated"), whose test kernels were measured on hardware.
    `test_angles` replaces the benchmark test set, for example with an external subset.
    """
    measured = {}
    test = data.test_angles if test_angles is None else test_angles
    quantum = data.models["quantum_kernel_svm"]
    test_kernel = qkernel.statevector_kernel(test, data.train_angles)
    quantum_fit = median_seconds(
        lambda: clone(quantum).fit(
            qkernel.statevector_kernel(data.train_angles), data.train_labels
        ),
        repeats,
    )
    for name, row in MODEL_ROWS.items():
        model = data.models[name]
        if name == "quantum_kernel_svm":
            measured[row] = {
                "fit": quantum_fit,
                "test_kernel": median_seconds(
                    lambda: qkernel.statevector_kernel(test, data.train_angles),
                    repeats,
                ),
                "predict": median_seconds(lambda: _predict(quantum, test_kernel), repeats),
            }
            continue
        measured[row] = {
            "fit": median_seconds(
                lambda m=model: clone(m).fit(data.train_angles, data.train_labels), repeats
            ),
            "test_kernel": None,
            "predict": median_seconds(lambda m=model: _predict(m, test), repeats),
        }
    for source, kernel in (qpu_kernels or {}).items():
        measured[source] = {
            "fit": quantum_fit,
            "test_kernel": None,
            "predict": median_seconds(lambda k=kernel: _predict(quantum, k), repeats),
        }
    return measured


def environment() -> str:
    versions = ", ".join(f"{name} {metadata.version(name)}" for name in PACKAGES)
    return (
        f"{platform.system()} {platform.release()}, {platform.machine()}, "
        f"Python {platform.python_version()}, {versions}"
    )
