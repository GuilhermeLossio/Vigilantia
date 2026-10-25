"""External hardware subset: frozen protocol, subset draw, budget guard, and report schema."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.benchmark_split import group_hash
from src.evaluation import external, external_qpu
from src.quantum import hardware

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports" / "qpu-evidence-external"
EVIDENCE_RUNS = (
    sorted(path for path in EVIDENCE.iterdir() if path.is_dir()) if EVIDENCE.is_dir() else []
)
PROTOCOL = external_qpu.load_protocol()
METRICS = dict.fromkeys(external.METRICS, 0.5)
INTERVALS = dict.fromkeys(external.METRICS, [0.25, 0.75])
SECONDS = {"fit": 0.01, "test_kernel": None, "predict": 0.001}


def primary_rows(labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": np.arange(len(labels)) * 3 + 1,
            "is_phishing": labels,
            "group": [f"dns:site{i}.example" for i in range(len(labels))],
        }
    )


def model_row(model: str, source: str) -> dict:
    return {
        "name": external_qpu.NAMES[model],
        "model": model,
        "kernel_source": source,
        "metrics": dict(METRICS),
        "confidence_intervals": copy.deepcopy(INTERVALS),
        "confusion_matrix": [[1, 1], [1, 1]],
        "runtime_seconds": dict(SECONDS),
    }


def report(source: str = "ibm_quantum") -> dict:
    return external_qpu.build_report(
        source=source,
        protocol={
            "path": "configs/external-qpu-subset.json",
            "sha256": "a" * 64,
            "frozen_at_utc": "t",
        },
        parent_protocol={"path": "configs/external-validation.json", "sha256": "b" * 64},
        benchmark={"model_version": "v1", "training_examples": 3, "input_hashes": {}},
        subset={
            "size": 4,
            "primary_rows": 10,
            "class_counts": {"legitimate": 2, "phishing": 2},
            "row_index": [1, 4, 7, 9],
            "group_hash": [group_hash(f"dns:{i}") for i in range(4)],
        },
        hardware_run={
            "backend": "fake_backend",
            "run_id": "external-fake_backend-20260101T000000Z",
            "sampler": "v2",
            "shots": 1024,
            "job_ids": ["job-0"],
            "circuits": 18,
            "plan": {"test": 12, "calibration": 3, "diagonal": 3},
        },
        kernel={
            "mean_abs_kernel_error_vs_ideal": 0.03,
            "test_kernel_error_vs_ideal": {"qpu": 0.04, "qpu_calibrated": 0.02},
            "calibration": {
                "valid": True,
                "a": 0.9,
                "c": 0.02,
                "rmse": 0.01,
                "clipped_test_values": 1,
            },
            "diagonal_controls_mean": 0.95,
        },
        runtime={
            "method": "median",
            "repeats": 25,
            "environment": "test machine",
            "qpu": {"execution_seconds": 2.0, "charged_seconds": 3.0, "resource_units": None},
        },
        bootstrap={"method": "m", "level": 0.95, "resamples": 10, "resamples_used": 9, "seed": 42},
        models=[model_row(*pair) for pair in external_qpu.COMPARED],
    )


def test_frozen_protocol_matches_its_parent_and_the_tooling() -> None:
    assert PROTOCOL["subset"]["per_class"] == {"legitimate": 20, "phishing": 20}
    assert PROTOCOL["hardware"]["batch_size"] == 252
    plan = hardware.plan_pairs(24, PROTOCOL["subset"]["size"])
    assert len(plan) == PROTOCOL["hardware"]["circuits"] == 1008
    assert external_qpu.plan_counts(plan) == {"test": 960, "calibration": 24, "diagonal": 24}


def test_protocol_with_another_parent_is_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="utf-8")
    with pytest.raises(external_qpu.ExternalQpuError, match="parent protocol"):
        external_qpu.load_protocol(parent_path=parent)


@pytest.mark.parametrize(("key", "value"), [("shots", 512), ("batch_size", 20)])
def test_protocol_with_other_settings_is_rejected(tmp_path: Path, key: str, value: int) -> None:
    changed = copy.deepcopy(PROTOCOL)
    changed["hardware"][key] = value
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(external_qpu.ExternalQpuError, match="shots, transpiler, sampler, or batch"):
        external_qpu.load_protocol(path)


def test_frozen_constants_match_the_protocol() -> None:
    assert PROTOCOL["hardware"]["batch_size"] == external_qpu.BATCH_SIZE
    assert f"over {external_qpu.FROZEN_REPEATS} repeats" in PROTOCOL["runtime"]["local"]


def test_protocol_hash_is_line_ending_independent(tmp_path: Path) -> None:
    text = external_qpu.PROTOCOL_PATH.read_bytes().replace(b"\r\n", b"\n")
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(text)
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    assert external.protocol_sha256(lf) == external.protocol_sha256(crlf)
    reference = external_qpu.protocol_reference(lf, PROTOCOL, tmp_path)
    assert reference == {
        "path": "lf.json",
        "sha256": external.protocol_sha256(external_qpu.PROTOCOL_PATH),
        "frozen_at_utc": PROTOCOL["frozen_at_utc"],
    }


def test_subset_follows_the_frozen_rule() -> None:
    labels = [0, 1] * 30 + [0] * 5
    rows = primary_rows(labels)
    subset = external_qpu.select_subset(rows, {"legitimate": 4, "phishing": 3})
    rng = np.random.default_rng(42)
    legitimate = rng.choice(np.flatnonzero(np.array(labels) == 0), size=4, replace=False)
    phishing = rng.choice(np.flatnonzero(np.array(labels) == 1), size=3, replace=False)
    expected = np.sort(np.concatenate([legitimate, phishing]))
    assert subset.positions.tolist() == expected.tolist()
    assert subset.row_index == rows["row_index"].to_numpy()[expected].tolist()
    assert subset.group_hash == [group_hash(rows["group"][p]) for p in expected]
    assert subset.labels.tolist() == np.array(labels)[expected].tolist()
    assert subset.class_counts == {"legitimate": 4, "phishing": 3}
    again = external_qpu.select_subset(rows, {"legitimate": 4, "phishing": 3})
    assert again.positions.tolist() == subset.positions.tolist()


def test_subset_identifiers_hold_no_hosts() -> None:
    subset = external_qpu.select_subset(primary_rows([0, 1] * 5), {"legitimate": 2, "phishing": 2})
    assert all(len(value) == 16 and "example" not in value for value in subset.group_hash)


def test_subset_needs_enough_rows_per_class() -> None:
    with pytest.raises(external_qpu.ExternalQpuError, match="fewer than 3 phishing"):
        external_qpu.select_subset(primary_rows([0, 0, 0, 1, 1]), {"legitimate": 2, "phishing": 3})


@pytest.mark.parametrize(
    ("remaining", "refused"),
    [
        (600, False),
        (336.2, False),
        (336.1, True),
        (336.192, True),
        (300, True),
        (None, True),
        ("600", True),
    ],
)
def test_budget_guard_needs_projected_billed_time_plus_a_minute(remaining, refused) -> None:
    assert external_qpu.projected_billed_seconds(1008) == pytest.approx(276.192)
    assert external_qpu.required_quota(1008) == pytest.approx(336.192)
    assert (external_qpu.budget_refusal(remaining, 1008) is not None) == refused


def test_billed_rate_matches_the_recorded_benchmark_run() -> None:
    results = json.loads((ROOT / "reports" / "benchmark-results.json").read_text(encoding="utf-8"))
    record = json.loads(
        (ROOT / "reports" / "qpu-evidence" / results["run_id"] / "run.json").read_text(
            encoding="utf-8"
        )
    )
    charged = sum(job["metrics"]["usage"]["qpu_charge_time_seconds"] for job in record["jobs"])
    assert round(charged / len(record["plan"]), 3) == external_qpu.BILLED_SECONDS_PER_CIRCUIT


def test_rows_share_resamples_and_hardware_rows_use_ibm_execution_time() -> None:
    labels = np.array([0, 0, 1, 1])
    evaluation = {
        "metrics": external.point_metrics(
            labels, np.array([0, 1, 1, 1]), np.array([0.1, 0.6, 0.7, 0.9])
        ),
        "confusion_matrix": [[1, 1], [0, 2]],
        "predictions": [0, 1, 1, 1],
        "decision_scores": [0.1, 0.6, 0.7, 0.9],
    }
    rows, used = external_qpu.rows_with_intervals(
        [(m, s, evaluation) for m, s in external_qpu.COMPARED],
        labels,
        {"resamples": 100, "seed": 42, "level": 0.95},
    )
    assert 0 < used <= 100
    assert [(r["model"], r["kernel_source"]) for r in rows] == list(external_qpu.COMPARED)
    assert rows[1]["confidence_intervals"] == rows[5]["confidence_intervals"]
    measured = {
        key: {"fit": 1.0, "test_kernel": 2.0 if key == "simulation" else None, "predict": 3.0}
        for key in (
            "majority",
            "logistic_regression",
            "rbf_svm",
            "simulation",
            "qpu",
            "qpu_calibrated",
        )
    }
    timed = external_qpu.with_runtime(rows, measured, {"execution_seconds": 9.5})
    assert [r["runtime_seconds"]["test_kernel"] for r in timed] == [None, None, None, 2.0, 9.5, 9.5]
    assert "runtime_seconds" not in rows[0]


def test_valid_report_round_trips(tmp_path: Path) -> None:
    built = report()
    assert built["label"] == "IBM Quantum hardware"
    assert any("2.5 percentage points" in text for text in built["limitations"])
    path = tmp_path / "out" / "external-qpu-subset.json"
    external_qpu.write(built, path)
    assert external_qpu.load(path) == built


def test_rehearsal_report_is_labeled_noisy_simulation() -> None:
    built = report("noisy_simulation")
    assert "not a hardware result" in built["label"]
    assert built["limitations"][0] == external_qpu.REHEARSAL_LIMITATION


def test_missing_and_malformed_reports_are_errors(tmp_path: Path) -> None:
    with pytest.raises(external_qpu.ExternalQpuError, match="no hardware results"):
        external_qpu.load(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(external_qpu.ExternalQpuError, match="not valid JSON"):
        external_qpu.load(bad)


def drop_calibrated(d: dict) -> None:
    d["models"].pop()


def invalid_fit_keeping_row(d: dict) -> None:
    d["kernel"]["calibration"] = {"valid": False, "reason": "non-positive slope"}
    d["kernel"]["test_kernel_error_vs_ideal"].pop("qpu_calibrated")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version=2),
        lambda d: d.update(source="simulation"),
        lambda d: d.update(label="IBM Quantum hardware (best run)"),
        lambda d: d["protocol"].update(sha256="xyz"),
        lambda d: d["parent_protocol"].pop("path"),
        lambda d: d["subset"].update(size=5),
        lambda d: d["subset"]["row_index"].reverse(),
        lambda d: d["subset"]["group_hash"].__setitem__(0, "www.example.com"),
        lambda d: d["subset"]["group_hash"].__setitem__(1, d["subset"]["group_hash"][0]),
        lambda d: d["subset"].update(class_counts={"legitimate": 4, "phishing": 0}),
        lambda d: d["hardware"].update(job_ids=[]),
        lambda d: d["hardware"]["plan"].update(test=11),
        lambda d: d["hardware"].update(circuits=17),
        lambda d: d["kernel"].update(diagonal_controls_mean=1.5),
        lambda d: d["kernel"]["calibration"].pop("a"),
        lambda d: d["runtime"]["qpu"].update(charged_seconds=-1),
        lambda d: d["runtime"].update(repeats=0),
        lambda d: d["runtime"].update(repeats=24),
        lambda d: d["hardware"].update(job_ids=["job-0", "job-1"]),
        lambda d: d["bootstrap"].update(resamples_used=11),
        drop_calibrated,
        invalid_fit_keeping_row,
        lambda d: d["models"].reverse(),
        lambda d: d["models"][4].update(kernel_source="simulation"),
        lambda d: d["models"][0].update(name="Majority"),
        lambda d: d["models"][1]["metrics"].update(accuracy=1.2),
        lambda d: d["models"][1]["confidence_intervals"].update(f1=[0.8, 0.2]),
        lambda d: d["models"][1]["confidence_intervals"].update(f1=None),
        lambda d: d["models"][1].update(confusion_matrix=[[1, 1], [1, 2]]),
        lambda d: d["models"][1]["runtime_seconds"].pop("predict"),
        lambda d: d["models"][1]["runtime_seconds"].update(fit=float("inf")),
        lambda d: d.update(limitations=[]),
        lambda d: d["limitations"].append("See https://example.com"),
        lambda d: d["limitations"].append("Saved to C:\\Users\\someone\\run.json"),
    ],
)
def test_validation_rejects_broken_reports(mutate) -> None:
    broken = copy.deepcopy(report())
    mutate(broken)
    with pytest.raises(external_qpu.ExternalQpuError):
        external_qpu.validate(broken)


def test_invalid_calibration_reports_five_rows_as_a_stated_deviation() -> None:
    broken = copy.deepcopy(report())
    invalid_fit_keeping_row(broken)
    drop_calibrated(broken)
    with pytest.raises(external_qpu.ExternalQpuError, match="protocol deviation"):
        external_qpu.validate(broken)
    broken["limitations"].append(external_qpu.CALIBRATION_DEVIATION)
    assert len(external_qpu.validate(broken)["models"]) == 5


def test_rehearsal_reports_may_use_other_batches_and_repeats() -> None:
    rehearsal = copy.deepcopy(report("noisy_simulation"))
    rehearsal["hardware"]["job_ids"] = ["job-0", "job-1", "job-2"]
    rehearsal["runtime"]["repeats"] = 1
    assert external_qpu.validate(rehearsal) is rehearsal


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('{"crn": "crn:v1:bluemix:public"}', "credential"),
        ('{"instance": "x"}', "credential"),
        ('{"token": "abc"}', "credential"),
        ('{"api_key": "abc"}', "credential"),
        ('{"server": "https://example.com"}', "URL"),
        ('{"saved": "C:\\Users\\someone"}', "URL"),
    ],
)
def test_evidence_with_secrets_urls_or_paths_is_refused(text: str, reason: str) -> None:
    clean = json.dumps([{"0000": 1024}])
    refusal = external_qpu.evidence_refusal({"counts-job.json": clean, "run.json": text})
    assert refusal is not None and refusal.startswith("run.json") and reason in refusal
    assert external_qpu.evidence_refusal({"counts-job.json": clean}) is None


def test_recorded_benchmark_evidence_passes_the_evidence_check() -> None:
    texts = {
        f"{path.parent.name}/{path.name}": path.read_text(encoding="utf-8")
        for path in (ROOT / "reports" / "qpu-evidence").glob("*/*.json")
    }
    assert texts and external_qpu.evidence_refusal(texts) is None


@pytest.mark.skipif(not EVIDENCE_RUNS, reason="No external subset hardware run is published yet.")
@pytest.mark.parametrize("run_dir", EVIDENCE_RUNS, ids=lambda path: path.name)
def test_external_evidence_holds_no_credentials_and_matches_the_report(run_dir: Path) -> None:
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"external-ibm_fez-\w+", run_dir.name)
    assert record["run_id"] == run_dir.name and record["status"] == "retrieved"
    assert {path.name for path in run_dir.iterdir()} == {"run.json"} | {
        f"counts-{job['job_id']}.json" for job in record["jobs"]
    }
    texts = {path.name: path.read_text(encoding="utf-8") for path in run_dir.iterdir()}
    assert external_qpu.evidence_refusal(texts) is None
    published = external_qpu.load(ROOT / "reports" / "external-qpu-subset.json")
    assert published["hardware"]["run_id"] == run_dir.name
    assert published["hardware"]["job_ids"] == [job["job_id"] for job in record["jobs"]]
    assert len(EVIDENCE_RUNS) == 1
