"""Versioned IBM Quantum evidence: complete, consistent with the published results, no secrets."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports" / "qpu-evidence"
RUNS = sorted(path for path in EVIDENCE.iterdir() if path.is_dir())
RESULTS = json.loads((ROOT / "reports" / "benchmark-results.json").read_text(encoding="utf-8"))
SECRET = re.compile(r"crn:|token|api[_-]?key|instance", re.IGNORECASE)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def counts(run_dir: Path, record: dict) -> list[dict]:
    collected: list[dict] = []
    for job in sorted(record["jobs"], key=lambda entry: entry["start"]):
        collected += read(run_dir / f"counts-{job['job_id']}.json")
    return collected


def test_the_published_run_has_evidence() -> None:
    assert (EVIDENCE / RESULTS["run_id"] / "run.json").is_file()
    assert {path.name.split("-")[0] for path in RUNS} == {"full", "pilot"}


@pytest.mark.parametrize("run_dir", RUNS, ids=lambda path: path.name)
def test_counts_cover_the_plan_with_every_shot(run_dir: Path) -> None:
    record = read(run_dir / "run.json")
    assert record["run_id"] == run_dir.name and record["status"] == "retrieved"
    collected = counts(run_dir, record)
    assert len(collected) == len(record["plan"])
    for entry in collected:
        assert sum(entry.values()) == record["shots"]
        assert all(len(bits) == 4 and set(bits) <= {"0", "1"} for bits in entry)


@pytest.mark.parametrize("run_dir", RUNS, ids=lambda path: path.name)
def test_evidence_holds_no_credentials(run_dir: Path) -> None:
    for path in run_dir.iterdir():
        assert not SECRET.search(path.read_text(encoding="utf-8")), path.name
    assert {path.name for path in run_dir.iterdir()} == {"run.json"} | {
        f"counts-{job['job_id']}.json" for job in read(run_dir / "run.json")["jobs"]
    }


def test_full_run_matches_the_published_hardware_section() -> None:
    record = read(EVIDENCE / RESULTS["run_id"] / "run.json")
    hardware = RESULTS["hardware"]
    assert record["backend"] == hardware["backend"]
    assert record["shots"] == hardware["shots"]
    assert [job["job_id"] for job in record["jobs"]] == hardware["job_ids"]
    assert not record["pilot"] and len(record["plan"]) == record["full_plan_circuits"] == 336
    kinds = [spec["kind"] for spec in record["plan"]]
    assert (kinds.count("test"), kinds.count("calibration"), kinds.count("diagonal")) == (
        288,
        24,
        24,
    )


def test_full_run_usage_matches_the_published_runtime() -> None:
    record = read(EVIDENCE / RESULTS["run_id"] / "run.json")
    metrics = [job["metrics"] for job in record["jobs"]]
    execution = sum(m["circuits_execution_time_ns"] for m in metrics) / 1e9
    charged = sum(m["usage"]["qpu_charge_time_seconds"] for m in metrics)
    qpu = RESULTS["runtime"]["qpu"]
    assert execution == pytest.approx(qpu["execution_seconds"])
    assert charged == qpu["charged_seconds"]
