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
