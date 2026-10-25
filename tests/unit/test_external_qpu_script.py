"""scripts/run_qpu_external.py and the FakeFez rehearsal with stub IBM services; no network."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from scripts import rehearse_external_qpu as rehearsal
from scripts import run_qpu_external as script
from src.data.benchmark_split import group_hash
from src.evaluation import external_qpu
from src.evaluation.inputs import BenchmarkInputs
from src.models import benchmark_models
from src.quantum import hardware, qpu_run
from src.quantum import kernel as qkernel

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
PROTECTED = [
    ROOT / "reports" / "benchmark-results.json",
    ROOT / "reports" / "benchmark-replay.json",
]


def synthetic_loaded() -> script.SubsetInputs:
    rng = np.random.default_rng(5)
    train = rng.uniform(0, math.pi / 2, size=(6, 4))
    train_labels = np.array([0, 0, 0, 1, 1, 1])
    train[train_labels == 1] = np.clip(train[train_labels == 1] + 0.4, 0, math.pi / 2)
    angles = rng.uniform(0, math.pi / 2, size=(4, 4))
    data = BenchmarkInputs(
        features=["a", "b", "c", "d"],
        train_angles=train,
        test_angles=rng.uniform(0, math.pi / 2, size=(2, 4)),
        train_labels=train_labels,
        test_labels=np.array([0, 1]),
        test_clipped={},
        models=benchmark_models.fit_all(train, qkernel.statevector_kernel(train), train_labels),
        input_hashes={"split_manifest_sha256": "s", "scaler_sha256": "k"},
    )
    protocol = copy.deepcopy(external_qpu.load_protocol())
    protocol["metrics"]["confidence_intervals"]["resamples"] = 100
    return script.SubsetInputs(
        protocol=protocol,
        references={
            "protocol": external_qpu.protocol_reference(script.PROTOCOL_PATH, protocol, ROOT),
            "parent_protocol": {"path": "configs/external-validation.json", "sha256": "b" * 64},
        },
        data=data,
        subset=external_qpu.Subset(
            positions=np.arange(4),
            row_index=[3, 5, 8, 9],
            group_hash=[group_hash(f"dns:site{i}.example") for i in range(4)],
            labels=np.array([0, 0, 1, 1]),
        ),
        angles=angles,
        clipped={"per_feature": {"a": 0, "b": 1, "c": 0, "d": 0}, "rows_with_any_clipped": 1},
        primary_rows=10,
        plan=qpu_run.full_plan(data, angles),
        model_version="v-test",
    )


class StubJob:
    def __init__(self, job_id: str, counts: list[dict]) -> None:
        self._id, self._counts, self.state = job_id, counts, "DONE"

    def job_id(self) -> str:
        return self._id

    def status(self) -> str:
        return self.state

    def result(self) -> list:
        return [
            SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda c=c: c)))
            for c in self._counts
        ]

    def metrics(self) -> dict:
        n = len(self._counts)
        return {
            "usage": {"qpu_charge_time_seconds": 4, "value": 0.01 * n, "unit": "resource_units"},
            "circuits_execution_time_ns": int(0.25e9 * n),
        }


def scaled_counts(loaded: script.SubsetInputs, scale: float = 0.9) -> list[dict]:
    train = qkernel.statevector_kernel(loaded.data.train_angles)
    test = qkernel.statevector_kernel(loaded.angles, loaded.data.train_angles)
    counts = []
    for spec in loaded.plan:
        value = (
            test[spec.left, spec.right]
            if spec.kind == "test"
            else 1.0
            if spec.kind == "diagonal"
            else train[spec.left, spec.right]
        )
        zeros = round((scale * value + 0.02) * hardware.SHOTS)
        counts.append({"0000": zeros, "1111": hardware.SHOTS - zeros})
    return counts


class StubService:
    def __init__(self, remaining: float = 600, limit_reached: bool = False) -> None:
        self.remaining, self.limit_reached = remaining, limit_reached
        self.jobs: dict[str, StubJob] = {}
        self.backends: list[str] = []

    def usage(self) -> dict:
        return {
            "usage_limit_reached": self.limit_reached,
            "usage_remaining_seconds": self.remaining,
        }

    def backend(self, name: str):
        self.backends.append(name)
        return FakeManilaV2()

    def job(self, job_id: str) -> StubJob:
        return self.jobs[job_id]


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Synthetic inputs, a stub IBM service, and every output path under tmp_path."""
    loaded = synthetic_loaded()
    service = StubService()
    counts = scaled_counts(loaded)

    def stub_submit(backend, circuits, shots, sampler):
        assert sampler == "v2" and shots == hardware.SHOTS
        start = sum(len(job._counts) for job in service.jobs.values())
        job = StubJob(f"job-{len(service.jobs)}", counts[start : start + len(circuits)])
        service.jobs[job.job_id()] = job
        return job

    monkeypatch.setattr(script.ibm, "settings", lambda: {"IBM_QUANTUM_BACKEND": "ibm_fez"})
    monkeypatch.setattr(script.ibm, "connect", lambda *_: service)
    monkeypatch.setattr(script, "load_subset_inputs", lambda: loaded)
    monkeypatch.setattr(
        script.qpu_run, "submit_run", functools.partial(qpu_run.submit_run, submit=stub_submit)
    )
    monkeypatch.setattr(qpu_run, "RUNS_DIR", tmp_path / "qpu-runs")
    monkeypatch.setattr(script, "REPORT_PATH", tmp_path / "reports" / "external-qpu-subset.json")
    monkeypatch.setattr(script, "EVIDENCE_DIR", tmp_path / "reports" / "qpu-evidence-external")
    return SimpleNamespace(loaded=loaded, service=service, root=tmp_path)


def digests() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED if p.exists()}


def submit_args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{"yes": True, **overrides})


def only_run(root: Path) -> str:
    (run_dir,) = (root / "qpu-runs").iterdir()
    return run_dir.name


def test_plan_prints_counts_and_never_connects(isolated, monkeypatch, capsys) -> None:
    monkeypatch.setattr(script.ibm, "connect", lambda *_: pytest.fail("contacted IBM"))
    assert script.command_plan(SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "legitimate 2, phishing 2" in out
    assert "Circuits: 45 (24 test, 15 calibration, 6 diagonal); 1 job(s) of up to 252" in out
    assert "267 s execution" in out and f"{45 * 0.274:.1f} s billed" in out
    assert "No IBM connection was made." in out


def test_submit_requires_a_fixed_backend_before_connecting(isolated, monkeypatch) -> None:
    monkeypatch.setattr(script.ibm, "settings", lambda: {})
    monkeypatch.setattr(script.ibm, "connect", lambda *_: pytest.fail("contacted IBM"))
    assert script.command_submit(submit_args()) == 1


def test_submit_refuses_another_backend_than_the_protocol(isolated, monkeypatch) -> None:
    monkeypatch.setattr(script.ibm, "settings", lambda: {"IBM_QUANTUM_BACKEND": "ibm_kingston"})
    monkeypatch.setattr(script.ibm, "connect", lambda *_: pytest.fail("contacted IBM"))
    assert script.command_submit(submit_args()) == 1


@pytest.mark.parametrize(
    ("service", "message"),
    [
        (StubService(limit_reached=True), "usage limit"),
        (StubService(remaining=60), "budget guard"),
        (StubService(remaining=None), "did not report"),
    ],
)
def test_submit_refuses_without_quota(isolated, monkeypatch, capsys, service, message) -> None:
    monkeypatch.setattr(script.ibm, "connect", lambda *_: service)
    assert script.command_submit(submit_args()) == 1
    assert message in capsys.readouterr().out
    assert service.backends == [] and not (isolated.root / "qpu-runs").exists()


def test_submit_without_typed_confirmation_sends_nothing(isolated, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert script.command_submit(submit_args(yes=False)) == 1
    assert isolated.service.jobs == {} and not (isolated.root / "qpu-runs").exists()


def test_submit_retrieve_evaluate_publishes_only_the_subset_report(isolated, capsys) -> None:
    before = digests()
    assert script.command_submit(submit_args()) == 0
    run_id = only_run(isolated.root)
    assert run_id.startswith("external-fake_manila-")
    record = qpu_run.load(qpu_run.RUNS_DIR / run_id)
    assert record["inputs"] == isolated.loaded.inputs
    assert record["inputs"]["subset_row_index"] == [3, 5, 8, 9]
    assert record["protocol"]["path"] == "configs/external-qpu-subset.json"
    assert [(job["start"], job["stop"]) for job in record["jobs"]] == [(0, 45)]

    assert script.command_retrieve(SimpleNamespace(run_id=run_id)) == 0
    assert script.command_evaluate(SimpleNamespace(run_id=run_id)) == 0
    report = external_qpu.load(script.REPORT_PATH)
    assert report["source"] == "ibm_quantum" and report["hardware"]["run_id"] == run_id
    assert report["runtime"]["repeats"] == 25
    assert [(r["model"], r["kernel_source"]) for r in report["models"]] == list(
        external_qpu.COMPARED
    )
    assert report["kernel"]["calibration"]["a"] == pytest.approx(0.9, abs=0.01)
    errors = report["kernel"]["test_kernel_error_vs_ideal"]
    assert errors["qpu_calibrated"] < errors["qpu"]
    assert report["runtime"]["qpu"]["execution_seconds"] == pytest.approx(0.25 * 45)
    assert report["models"][4]["runtime_seconds"]["test_kernel"] == pytest.approx(0.25 * 45)
    assert report["subset"]["group_hash"] == isolated.loaded.subset.group_hash

    evidence = script.EVIDENCE_DIR / run_id
    assert {p.name for p in evidence.iterdir()} == {"run.json"} | {
        f"counts-{job['job_id']}.json" for job in record["jobs"]
    }
    text = script.REPORT_PATH.read_text(encoding="utf-8") + "".join(
        p.read_text(encoding="utf-8") for p in evidence.iterdir()
    )
    assert ".example" not in text and "dns:" not in text and str(isolated.root) not in text
    assert digests() == before
    assert "n=4" in capsys.readouterr().out


def test_retrieve_waits_for_unfinished_jobs(isolated) -> None:
    script.command_submit(submit_args())
    run_id = only_run(isolated.root)
    for job in isolated.service.jobs.values():
        job.state = "QUEUED"
    assert script.command_retrieve(SimpleNamespace(run_id=run_id)) == 0
    with pytest.raises(qpu_run.RunError, match="retrieve all jobs"):
        script.command_evaluate(SimpleNamespace(run_id=run_id))
    assert not script.REPORT_PATH.exists()


def test_evaluate_rejects_a_run_of_another_subset(isolated, monkeypatch) -> None:
    script.command_submit(submit_args())
    run_id = only_run(isolated.root)
    script.command_retrieve(SimpleNamespace(run_id=run_id))
    changed = copy.copy(isolated.loaded)
    changed.subset = external_qpu.Subset(
        positions=np.arange(4),
        row_index=[3, 5, 8, 10],
        group_hash=isolated.loaded.subset.group_hash,
        labels=isolated.loaded.subset.labels,
    )
    monkeypatch.setattr(script, "load_subset_inputs", lambda: changed)
    with pytest.raises(qpu_run.RunError, match="other inputs"):
        script.command_evaluate(SimpleNamespace(run_id=run_id))
    assert not script.REPORT_PATH.exists()


@pytest.mark.parametrize(
    "run_id", ["full-ibm_fez-20261025T094010Z", "pilot-ibm_fez-1", "external-x/../../y"]
)
def test_only_external_runs_are_addressed(run_id: str) -> None:
    with pytest.raises(qpu_run.RunError, match="named external-"):
        script.run_dir_for(run_id)


def test_real_subset_plan_matches_the_frozen_protocol() -> None:
    try:
        loaded = script.load_subset_inputs()
    except Exception as error:  # the external snapshot and PhiUSIIL CSV are local downloads
        pytest.skip(f"External inputs unavailable: {error}")
    assert loaded.subset.class_counts == {"legitimate": 20, "phishing": 20}
    assert loaded.primary_rows == 6441
    assert loaded.subset.row_index[:3] == [353, 574, 814]
    assert external_qpu.plan_counts(loaded.plan) == {
        "test": 960,
        "calibration": 24,
        "diagonal": 24,
    }
    assert len(hardware.batches(len(loaded.plan), 252)) == 4


def test_rehearsal_writes_only_to_its_artifact_directory(isolated) -> None:
    before = digests()
    out = isolated.root / "rehearsal"
    report = rehearsal.rehearse(isolated.loaded, FakeManilaV2(), out, repeats=1, batch_size=20)
    assert report["source"] == "noisy_simulation"
    assert "not a hardware result" in report["label"]
    assert report["hardware"]["run_id"].startswith("rehearsal-fake_manila-")
    assert len(report["hardware"]["job_ids"]) == 3
    assert report["runtime"]["qpu"] == dict.fromkeys(external_qpu.USAGE_KEYS)
    assert external_qpu.load(out / "summary.json") == report
    assert sorted(p.name for p in isolated.root.iterdir()) == ["rehearsal"]
    assert digests() == before
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["status"] == "retrieved"


def test_submit_and_evaluate_take_no_batch_size_or_repeats(monkeypatch) -> None:
    for argv in (["submit", "--batch-size", "20"], ["evaluate", "external-x", "--repeats", "1"]):
        monkeypatch.setattr("sys.argv", ["run_qpu_external.py", *argv])
        with pytest.raises(SystemExit) as raised:
            script.main()
        assert raised.value.code == 2


def test_a_second_submission_is_refused_before_connecting(isolated, monkeypatch, capsys) -> None:
    assert script.command_submit(submit_args()) == 0
    run_id = only_run(isolated.root)
    capsys.readouterr()
    monkeypatch.setattr(script.ibm, "connect", lambda *_: pytest.fail("contacted IBM"))
    assert script.command_submit(submit_args()) == 1
    out = capsys.readouterr().out
    assert f"Run {run_id} of this subset already exists (status submitted)" in out
    assert f"retrieve {run_id}" in out
    assert only_run(isolated.root) == run_id


@pytest.mark.parametrize(
    ("status", "jobs", "refused"),
    [
        ("created", [], False),
        ("submitting", [], True),
        ("failed", [{"job_id": "j"}], True),
        ("retrieved", [{"job_id": "j"}], True),
    ],
)
def test_earlier_runs_block_submission_once_they_may_use_quota(
    isolated, status: str, jobs: list, refused: bool
) -> None:
    run_dir = qpu_run.RUNS_DIR / "external-ibm_fez-20261025T130000Z"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": status, "jobs": jobs}), "utf-8")
    other = qpu_run.RUNS_DIR / "full-ibm_fez-20261025T094010Z"
    other.mkdir()
    (other / "run.json").write_text(json.dumps({"status": "retrieved", "jobs": [1]}), "utf-8")
    assert script.earlier_run() == ((run_dir.name, status) if refused else None)


def retrieved_run(isolated) -> str:
    script.command_submit(submit_args())
    run_id = only_run(isolated.root)
    script.command_retrieve(SimpleNamespace(run_id=run_id))
    return run_id


def test_evaluate_never_replaces_the_report_of_another_run(isolated, capsys) -> None:
    run_id = retrieved_run(isolated)
    script.REPORT_PATH.parent.mkdir(parents=True)
    published = {"hardware": {"run_id": "external-ibm_fez-20261025T120000Z"}}
    script.REPORT_PATH.write_text(json.dumps(published), encoding="utf-8")
    assert script.command_evaluate(SimpleNamespace(run_id=run_id)) == 1
    assert "already holds run external-ibm_fez-20261025T120000Z" in capsys.readouterr().out
    assert json.loads(script.REPORT_PATH.read_text(encoding="utf-8")) == published
    assert not script.EVIDENCE_DIR.exists()


def test_evaluate_never_adds_evidence_of_a_second_run(isolated, capsys) -> None:
    run_id = retrieved_run(isolated)
    (script.EVIDENCE_DIR / "external-ibm_fez-20261025T120000Z").mkdir(parents=True)
    assert script.command_evaluate(SimpleNamespace(run_id=run_id)) == 1
    assert "already holds run external-ibm_fez-20261025T120000Z" in capsys.readouterr().out
    assert not script.REPORT_PATH.exists()


def test_evaluating_the_same_run_again_is_allowed(isolated) -> None:
    run_id = retrieved_run(isolated)
    assert script.command_evaluate(SimpleNamespace(run_id=run_id)) == 0
    assert script.command_evaluate(SimpleNamespace(run_id=run_id)) == 0
    assert external_qpu.load(script.REPORT_PATH)["hardware"]["run_id"] == run_id


def test_evidence_with_an_instance_identifier_is_not_copied(isolated) -> None:
    run_id = retrieved_run(isolated)
    run_dir = qpu_run.RUNS_DIR / run_id
    record = qpu_run.load(run_dir)
    record["jobs"][0]["metrics"]["instance"] = "crn:v1:bluemix:public:quantum-computing"
    qpu_run.save(run_dir, record)
    with pytest.raises(external_qpu.ExternalQpuError, match="run.json looks like"):
        script.copy_evidence(run_dir, record, script.EVIDENCE_DIR)
    assert not script.EVIDENCE_DIR.exists()


def test_an_invalid_calibration_publishes_five_rows_with_the_deviation(
    isolated, monkeypatch
) -> None:
    run_id = retrieved_run(isolated)
    monkeypatch.setattr(
        hardware, "fit_calibration", lambda *_: {"valid": False, "reason": "non-positive slope"}
    )
    assert script.command_evaluate(SimpleNamespace(run_id=run_id)) == 0
    report = external_qpu.load(script.REPORT_PATH)
    assert [row["kernel_source"] for row in report["models"]][-1] == "qpu"
    assert external_qpu.CALIBRATION_DEVIATION in report["limitations"]
