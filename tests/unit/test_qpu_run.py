"""QPU run lifecycle with stub jobs and services; no IBM account, network, or QPU."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import results as contract
from src.evaluation.inputs import BenchmarkInputs
from src.models import benchmark_models
from src.quantum import hardware, qpu_run
from src.quantum import kernel as qkernel

pytestmark = pytest.mark.unit
SHOTS = hardware.SHOTS


def synthetic_inputs() -> BenchmarkInputs:
    rng = np.random.default_rng(3)
    train = rng.uniform(0, math.pi / 2, size=(6, 4))
    test = rng.uniform(0, math.pi / 2, size=(4, 4))
    train_labels = np.array([0, 0, 0, 1, 1, 1])
    train[train_labels == 1] = np.clip(train[train_labels == 1] + 0.4, 0, math.pi / 2)
    models = benchmark_models.fit_all(train, qkernel.statevector_kernel(train), train_labels)
    return BenchmarkInputs(
        features=["a", "b", "c", "d"],
        train_angles=train,
        test_angles=test,
        train_labels=train_labels,
        test_labels=np.array([0, 0, 1, 1]),
        test_clipped={},
        models=models,
        input_hashes={"split_manifest_sha256": "s", "scaler_sha256": "k"},
    )


class StubJob:
    def __init__(self, job_id: str, counts: list[dict], status: str = "DONE") -> None:
        self._id, self._counts, self._status = job_id, counts, status

    def job_id(self) -> str:
        return self._id

    def status(self) -> str:
        return self._status

    def result(self) -> list:
        return [
            SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda c=c: c)))
            for c in self._counts
        ]

    def metrics(self) -> dict:
        n = len(self._counts)
        return {
            "usage": {"qpu_charge_time_seconds": 5, "value": 0.01 * n, "unit": "resource_units"},
            "circuits_execution_time_ns": int(0.25e9 * n),
        }


def ideal_counts(
    data: BenchmarkInputs, plan: list[hardware.PairSpec], scale: float = 0.9
) -> list[dict]:
    """Counts whose all-zero fraction is a scaled ideal kernel value, like uniform noise."""
    train = qkernel.statevector_kernel(data.train_angles)
    test = qkernel.statevector_kernel(data.test_angles, data.train_angles)
    counts = []
    for spec in plan:
        value = (
            test[spec.left, spec.right]
            if spec.kind == "test"
            else 1.0
            if spec.kind == "diagonal"
            else train[spec.left, spec.right]
        )
        zeros = round((scale * value + 0.02) * SHOTS)
        counts.append({"0000": zeros, "1111": SHOTS - zeros})
    return counts


def submitted_run(
    tmp_path: Path, data: BenchmarkInputs, pilot: int | None = None, batch: int | None = None
):
    plan = qpu_run.plan_for(data, pilot)
    run_id, record = qpu_run.new_record(
        data, "fake_backend", plan, bool(pilot), {}, len(qpu_run.full_plan(data))
    )
    counts = ideal_counts(data, plan)
    jobs: dict[str, StubJob] = {}

    def submit(backend, circuits, shots, sampler):
        assert sampler == "v2" and shots == SHOTS
        start = sum(len(job._counts) for job in jobs.values())
        job = StubJob(f"job-{len(jobs)}", counts[start : start + len(circuits)])
        jobs[job.job_id()] = job
        return job

    run_dir = tmp_path / run_id
    qpu_run.submit_run(run_dir, record, None, list(range(len(plan))), batch, submit=submit)
    service = SimpleNamespace(job=lambda job_id: jobs[job_id])
    return run_dir, service, jobs


def write_simulation(tmp_path: Path, data: BenchmarkInputs) -> Path:
    ideal = qkernel.statevector_kernel(data.test_angles, data.train_angles)
    from src.evaluation.metrics import evaluate

    evaluations = {
        name: evaluate(
            model, ideal if name == "quantum_kernel_svm" else data.test_angles, data.test_labels
        )
        for name, model in data.models.items()
    }
    path = tmp_path / "evaluation-simulation.json"
    path.write_text(
        json.dumps({"inputs": data.input_hashes, "evaluations": evaluations}), encoding="utf-8"
    )
    return path


def test_full_run_is_saved_retrieved_evaluated_and_published(tmp_path: Path) -> None:
    data = synthetic_inputs()
    run_dir, service, jobs = submitted_run(tmp_path, data, batch=20)
    record = qpu_run.load(run_dir)
    assert record["status"] == "submitted"
    assert [job["job_id"] for job in record["jobs"]] == list(jobs)
    assert record["jobs"][-1]["stop"] == len(record["plan"])

    assert qpu_run.retrieve_run(run_dir, service)["status"] == "retrieved"
    results_path = tmp_path / "benchmark-results.json"
    summary = qpu_run.evaluate_run(run_dir, data, results_path, write_simulation(tmp_path, data))

    assert summary["calibration"]["valid"]
    assert summary["calibration"]["a"] == pytest.approx(0.9, abs=0.01)
    assert (
        summary["test_kernel_error_vs_ideal"]["qpu_calibrated"]
        < summary["test_kernel_error_vs_ideal"]["qpu"]
    )
    published = contract.validate(json.loads(results_path.read_text(encoding="utf-8")))
    assert published["status"] == "complete"
    assert published["hardware"] == {
        "backend": "fake_backend",
        "shots": SHOTS,
        "job_ids": list(jobs),
    }
    assert [m["kernel_source"] for m in published["models"]][-3:] == [
        "simulation",
        "qpu",
        "qpu_calibrated",
    ]
    assert (run_dir / "test-kernel-raw.npy").exists()


def test_pilot_projects_quota_and_is_never_published(tmp_path: Path) -> None:
    data = synthetic_inputs()
    run_dir, service, _ = submitted_run(tmp_path, data, pilot=5)
    qpu_run.retrieve_run(run_dir, service)
    results_path = tmp_path / "benchmark-results.json"
    summary = qpu_run.evaluate_run(run_dir, data, results_path)
    full = len(qpu_run.full_plan(data))
    assert summary["usage"] == pytest.approx(
        {"charged_seconds": 5.0, "resource_units": 0.05, "execution_seconds": 1.25}
    )
    assert summary["projected_full_run"]["execution_seconds"] == pytest.approx(1.25 / 5 * full)
    assert summary["projected_full_run"]["resource_units"] == pytest.approx(0.05 / 5 * full)
    assert not results_path.exists()


def test_usage_reads_legacy_and_missing_metrics() -> None:
    record = {"jobs": [{"metrics": {"usage": {"quantum_seconds": 3}}}, {"metrics": {}}]}
    assert qpu_run.usage_totals(record)["charged_seconds"] is None
    record["jobs"].pop()
    assert qpu_run.usage_totals(record) == {
        "charged_seconds": 3.0,
        "resource_units": None,
        "execution_seconds": None,
    }


def test_runs_are_never_resubmitted(tmp_path: Path) -> None:
    data = synthetic_inputs()
    run_dir, _, _ = submitted_run(tmp_path, data, pilot=3)
    record = qpu_run.load(run_dir)
    with pytest.raises(qpu_run.RunError, match="already has jobs"):
        qpu_run.submit_run(
            run_dir, record, None, [0, 1, 2], submit=lambda *a: pytest.fail("resubmitted")
        )


@pytest.mark.parametrize(
    ("status", "expected"), [("RUNNING", "waiting"), ("QUEUED", "waiting"), ("ERROR", "failed")]
)
def test_unfinished_or_failed_jobs_block_evaluation(
    tmp_path: Path, status: str, expected: str
) -> None:
    data = synthetic_inputs()
    run_dir, service, jobs = submitted_run(tmp_path, data, pilot=4)
    for job in jobs.values():
        job._status = status
    assert qpu_run.retrieve_run(run_dir, service)["status"] == expected
    with pytest.raises(qpu_run.RunError, match="retrieve all jobs"):
        qpu_run.evaluate_run(run_dir, data)


def test_retrieval_resumes_after_jobs_finish(tmp_path: Path) -> None:
    data = synthetic_inputs()
    run_dir, service, jobs = submitted_run(tmp_path, data, pilot=4)
    for job in jobs.values():
        job._status = "RUNNING"
    qpu_run.retrieve_run(run_dir, service)
    for job in jobs.values():
        job._status = "DONE"
    assert qpu_run.retrieve_run(run_dir, service)["status"] == "retrieved"


def test_runs_with_other_inputs_are_rejected(tmp_path: Path) -> None:
    data = synthetic_inputs()
    run_dir, service, _ = submitted_run(tmp_path, data, pilot=3)
    qpu_run.retrieve_run(run_dir, service)
    data.input_hashes = {"split_manifest_sha256": "other", "scaler_sha256": "k"}
    with pytest.raises(qpu_run.RunError, match="other inputs"):
        qpu_run.evaluate_run(run_dir, data)


def test_pilot_size_and_batches_are_validated() -> None:
    data = synthetic_inputs()
    with pytest.raises(qpu_run.RunError):
        qpu_run.plan_for(data, len(qpu_run.full_plan(data)))
    assert hardware.batches(5, None) == [(0, 5)]
    assert hardware.batches(5, 2) == [(0, 2), (2, 4), (4, 5)]
    with pytest.raises(ValueError):
        hardware.batches(0, None)


def test_submit_command_requires_a_fixed_backend_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_qpu", Path(__file__).resolve().parents[2] / "scripts" / "run_qpu.py"
    )
    run_qpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_qpu)
    monkeypatch.setattr(run_qpu.ibm, "settings", lambda: {})
    monkeypatch.setattr(run_qpu.ibm, "connect", lambda *_: pytest.fail("contacted IBM"))
    args = SimpleNamespace(pilot=None, batch_size=None, yes=True)
    assert run_qpu.command_submit(args) == 1


def load_run_qpu():
    spec = importlib.util.spec_from_file_location(
        "run_qpu", Path(__file__).resolve().parents[2] / "scripts" / "run_qpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plans_accept_other_test_angles() -> None:
    data = synthetic_inputs()
    subset = np.vstack([data.test_angles, data.test_angles[:1]])
    kinds = [spec.kind for spec in qpu_run.full_plan(data, subset)]
    assert (kinds.count("test"), kinds.count("calibration"), kinds.count("diagonal")) == (30, 15, 6)
    assert qpu_run.full_plan(data) == qpu_run.full_plan(data, data.test_angles)
    assert len(qpu_run.full_plan(data)) == 24 + 15 + 6
    assert qpu_run.plan_for(data, 7, subset) == qpu_run.full_plan(data, subset)[:7]


def test_new_record_takes_inputs_prefix_and_extra_fields() -> None:
    data = synthetic_inputs()
    plan = qpu_run.full_plan(data)
    run_id, record = qpu_run.new_record(data, "fake_backend", plan, False, {}, len(plan))
    assert run_id.startswith("full-fake_backend-") and record["inputs"] == data.input_hashes
    pilot_id, _ = qpu_run.new_record(data, "fake_backend", plan[:2], True, {}, len(plan))
    assert pilot_id.startswith("pilot-fake_backend-")
    run_id, record = qpu_run.new_record(
        data,
        "fake_backend",
        plan,
        False,
        {},
        len(plan),
        inputs={"subset_row_index": [1, 2]},
        prefix="external",
        extra={"protocol": {"path": "p"}},
    )
    assert run_id.startswith("external-fake_backend-") and record["run_id"] == run_id
    assert record["inputs"] == {"subset_row_index": [1, 2]}
    assert record["protocol"] == {"path": "p"} and record["jobs"] == []


def test_measure_kernel_matches_the_published_evaluation(tmp_path: Path) -> None:
    data = synthetic_inputs()
    run_dir, service, _ = submitted_run(tmp_path, data, batch=20)
    qpu_run.retrieve_run(run_dir, service)
    summary = qpu_run.evaluate_run(run_dir, data)
    record = qpu_run.load(run_dir)
    measured = qpu_run.measure_kernel(
        run_dir,
        record,
        data.train_angles,
        data.test_angles,
        data.test_labels,
        data.models["quantum_kernel_svm"],
    )
    for key in (
        "calibration",
        "diagonal_controls_mean",
        "test_kernel_error_vs_ideal",
        "mean_abs_kernel_error_vs_ideal",
        "shots_returned",
        "circuits",
    ):
        assert measured[key] == summary[key]
    for source, evaluation in summary["evaluations"].items():
        timed = {key: value for key, value in evaluation.items() if key != "predict_seconds"}
        assert {k: v for k, v in measured["evaluations"][source].items() if k in timed} == timed
    assert set(measured["kernels"]) == {"qpu", "qpu_calibrated"}
    np.testing.assert_array_equal(
        measured["kernels"]["qpu"], np.load(run_dir / "test-kernel-raw.npy")
    )
    np.testing.assert_allclose(
        measured["ideal_test_kernel"],
        qkernel.statevector_kernel(data.test_angles, data.train_angles),
    )
    assert measured["limitations"] == []


def test_measure_kernel_uses_the_given_test_rows(tmp_path: Path) -> None:
    data = synthetic_inputs()
    subset = data.test_angles[[0, 2, 3]]
    plan = qpu_run.full_plan(data, subset)
    run_id, record = qpu_run.new_record(
        data, "fake_backend", plan, False, {}, len(plan), inputs={"x": 1}, prefix="external"
    )
    shifted = SimpleNamespace(train_angles=data.train_angles, test_angles=subset)
    jobs: dict[str, StubJob] = {}

    def submit(backend, circuits, shots, sampler):
        jobs["job-0"] = StubJob("job-0", ideal_counts(shifted, plan))
        return jobs["job-0"]

    run_dir = tmp_path / run_id
    qpu_run.submit_run(run_dir, record, None, list(range(len(plan))), submit=submit)
    record = qpu_run.retrieve_run(run_dir, SimpleNamespace(job=jobs.__getitem__))
    labels = data.test_labels[[0, 2, 3]]
    measured = qpu_run.measure_kernel(
        run_dir, record, data.train_angles, subset, labels, data.models["quantum_kernel_svm"]
    )
    assert measured["kernels"]["qpu"].shape == (3, 6)
    assert sum(map(sum, measured["evaluations"]["qpu"]["confusion_matrix"])) == 3


def test_measure_kernel_rejects_pilots_and_unretrieved_runs(tmp_path: Path) -> None:
    data = synthetic_inputs()
    args = (data.train_angles, data.test_angles, data.test_labels, data.models["rbf_svm"])
    run_dir, service, _ = submitted_run(tmp_path, data, pilot=4)
    with pytest.raises(qpu_run.RunError, match="retrieve all jobs"):
        qpu_run.measure_kernel(run_dir, qpu_run.load(run_dir), *args)
    record = qpu_run.retrieve_run(run_dir, service)
    with pytest.raises(qpu_run.RunError, match="pilot"):
        qpu_run.measure_kernel(run_dir, record, *args)


def test_pilot_projection_ignores_external_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_qpu = load_run_qpu()
    monkeypatch.setattr(qpu_run, "RUNS_DIR", tmp_path)
    for name, seconds in (
        ("pilot-ibm_fez-20260101T000000Z", 1.0),
        ("external-ibm_fez-20260102T000000Z", 99.0),
        ("pilot-ibm_fez-external-20260103T000000Z", None),
    ):
        (tmp_path / name).mkdir()
        projection = None if seconds is None else {"execution_seconds": seconds}
        (tmp_path / name / "evaluation.json").write_text(
            json.dumps({"projected_full_run": projection}), encoding="utf-8"
        )
    assert run_qpu.latest_pilot_projection("ibm_fez") == {"execution_seconds": 1.0}


def test_submit_preflight_is_shared_and_needs_typed_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    run_qpu = load_run_qpu()
    assert run_qpu.required_backend({"IBM_QUANTUM_BACKEND": " ibm_fez "}) == "ibm_fez"
    assert run_qpu.required_backend({"IBM_QUANTUM_BACKEND": " "}) is None
    service = SimpleNamespace(usage=lambda: {"usage_limit_reached": True})
    monkeypatch.setattr(run_qpu.ibm, "connect", lambda *_: service)
    assert run_qpu.open_instance({}) is None
    service.usage = lambda: {"usage_limit_reached": False, "usage_remaining_seconds": 5}
    assert run_qpu.open_instance({}) == (service, service.usage())
    monkeypatch.setattr("builtins.input", lambda _: "submit")
    assert not run_qpu.confirmed(False)
    monkeypatch.setattr("builtins.input", lambda _: " SUBMIT ")
    assert run_qpu.confirmed(False) and run_qpu.confirmed(True)
    assert "usage limit" in capsys.readouterr().out
