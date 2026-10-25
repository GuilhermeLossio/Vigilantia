"""Runtime measurement, publication, and contract checks with synthetic inputs."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from app import create_app
from app import results as results_module
from src.evaluation import publish, runtime
from src.models.benchmark_models import fit_all
from src.quantum import kernel as qkernel

pytestmark = pytest.mark.unit

METRICS = dict.fromkeys(results_module.METRICS, 0.5)
USAGE = {"execution_seconds": 80.5, "charged_seconds": 90.0, "resource_units": 1.5}


def row(name: str, source: str) -> dict:
    return {
        "name": name,
        "kernel_source": source,
        "metrics": METRICS,
        "confusion_matrix": [[1, 1], [1, 1]],
    }


RESULTS = {
    "schema_version": 1,
    "run_id": "full-fake",
    "generated_at_utc": "2026-01-01T00:00:00+00:00",
    "status": "complete",
    "split": {"train": 6, "test": 4, "seed": 42},
    "hardware": {"backend": "fake", "shots": 1024, "job_ids": ["job-1"]},
    "models": [
        row("Majority baseline", "classical"),
        row("Logistic regression", "classical"),
        row("RBF SVM", "classical"),
        row("Quantum-kernel SVM", "simulation"),
        row("Quantum-kernel SVM", "qpu"),
    ],
    "limitations": ["Synthetic fixture."],
}


@pytest.fixture(scope="module")
def data() -> SimpleNamespace:
    rng = np.random.default_rng(0)
    train = rng.uniform(0, np.pi / 2, (6, 4))
    test = rng.uniform(0, np.pi / 2, (4, 4))
    labels = np.array([0, 1, 0, 1, 0, 1])
    models = fit_all(train, qkernel.statevector_kernel(train), labels)
    return SimpleNamespace(train_angles=train, test_angles=test, train_labels=labels, models=models)


@pytest.fixture(scope="module")
def measured(data) -> dict:
    kernel = qkernel.statevector_kernel(data.test_angles, data.train_angles)
    return runtime.measure(data, {"qpu": kernel}, repeats=2)


def test_measure_times_every_row(measured) -> None:
    assert set(measured) == {"majority", "logistic_regression", "rbf_svm", "simulation", "qpu"}
    for row_name, seconds in measured.items():
        assert set(seconds) == set(results_module.RUNTIME_KEYS)
        assert seconds["fit"] > 0 and seconds["predict"] > 0
        assert (seconds["test_kernel"] is None) == (row_name != "simulation")
    assert measured["qpu"]["fit"] == measured["simulation"]["fit"]


def test_with_runtime_uses_ibm_execution_time_for_hardware(measured) -> None:
    timed = publish.with_runtime(RESULTS, measured, 2, "test machine", USAGE)
    by_source = {m["kernel_source"]: m["runtime_seconds"] for m in timed["models"][3:]}
    assert by_source["qpu"]["test_kernel"] == 80.5
    assert by_source["simulation"]["test_kernel"] == measured["simulation"]["test_kernel"]
    assert timed["runtime"]["qpu"] == USAGE
    assert "median of 2 repeats" in timed["runtime"]["method"]
    assert "runtime" not in RESULTS


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("runtime"),
        lambda d: d["runtime"].update(method=""),
        lambda d: d["runtime"]["qpu"].update(charged_seconds=-1),
        lambda d: d["runtime"]["qpu"].pop("resource_units"),
        lambda d: d["models"][0]["runtime_seconds"].update(fit=float("inf")),
        lambda d: d["models"][0]["runtime_seconds"].pop("predict"),
        lambda d: d["models"][0].update(runtime_seconds=0.1),
    ],
)
def test_contract_rejects_inconsistent_runtime(measured, mutate) -> None:
    timed = copy.deepcopy(publish.with_runtime(RESULTS, measured, 2, "test machine", USAGE))
    mutate(timed)
    with pytest.raises(results_module.ResultsInvalid):
        results_module.validate(timed)


def test_results_without_runtime_stay_valid() -> None:
    assert results_module.validate(copy.deepcopy(RESULTS))


def test_page_shows_runtime_table(tmp_path, measured) -> None:
    timed = publish.with_runtime(RESULTS, measured, 2, "test machine", USAGE)
    path = tmp_path / "results.json"
    publish.write(timed, path)
    page = create_app(path, tmp_path / "none.json", tmp_path / "none.svg").test_client()
    text = page.get("/").get_data(as_text=True)
    assert "<h3>Runtime</h3>" in text
    assert "80.5 s execution, 90.0 s billed, 1.50 resource units" in text
    assert "test machine" in text


def test_measure_accepts_other_test_rows(data, monkeypatch: pytest.MonkeyPatch) -> None:
    subset = np.vstack([data.test_angles, data.test_angles])
    kernel = qkernel.statevector_kernel(subset, data.train_angles)
    sizes: list[int] = []
    original = runtime._predict

    def recording(model, inputs) -> None:
        sizes.append(len(inputs))
        original(model, inputs)

    monkeypatch.setattr(runtime, "_predict", recording)
    measured = runtime.measure(data, {"qpu": kernel}, repeats=1, test_angles=subset)
    assert set(measured) == {"majority", "logistic_regression", "rbf_svm", "simulation", "qpu"}
    assert sizes and set(sizes) == {len(subset)}
