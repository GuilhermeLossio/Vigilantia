"""IBM Quantum benchmark runs in three persisted steps: submit, retrieve, evaluate.

- submit: transpile the planned circuits for one backend, send them with SamplerV2 in job
  mode, and save every job ID as soon as it exists. Nothing is ever resubmitted.
- retrieve: fetch finished jobs by ID; safe to repeat until all jobs are done.
- evaluate: build the hardware test kernel, calibrate it on training pairs, classify the
  test set, and publish the results. A pilot run only reports QPU time and kernel error.

Each run lives in artifacts/benchmark/qpu-runs/<run_id>/ (Git-ignored): run.json holds the
plan, backend, transpilation statistics, and jobs; counts-<job_id>.json holds raw counts.
Runs of the external subset (scripts/run_qpu_external.py) are named external-*; they share
submit and retrieve and are evaluated with measure_kernel, never by evaluate_run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.evaluation import publish
from src.evaluation.inputs import ARTIFACT_DIR, BenchmarkInputs
from src.evaluation.metrics import evaluate
from src.quantum import hardware
from src.quantum import kernel as qkernel

RUNS_DIR = ARTIFACT_DIR / "qpu-runs"
SIMULATION_EVALUATION = ARTIFACT_DIR / "evaluation-simulation.json"
SAMPLER = "v2"  # SamplerV2: the established Open Plan path; jobs are retrievable by ID.
DONE = "DONE"
FAILED = {"ERROR", "CANCELLED"}


class RunError(Exception):
    """A run cannot proceed in its current state."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def status_name(status) -> str:
    return str(getattr(status, "name", status)).upper()


def full_plan(
    data: BenchmarkInputs, test_angles: np.ndarray | None = None
) -> list[hardware.PairSpec]:
    """The plan for the benchmark test set, or for `test_angles` when given."""
    test = data.test_angles if test_angles is None else test_angles
    n_train, n_test = len(data.train_angles), len(test)
    calibration = min(hardware.CALIBRATION_PAIRS, n_train * (n_train - 1) // 2)
    return hardware.plan_pairs(n_train, n_test, n_calibration=calibration)


def plan_for(
    data: BenchmarkInputs, pilot: int | None, test_angles: np.ndarray | None = None
) -> list[hardware.PairSpec]:
    """The full plan, or its first `pilot` circuits (test pairs) for a quota probe."""
    plan = full_plan(data, test_angles)
    if pilot is None:
        return plan
    if not 0 < pilot < len(plan):
        raise RunError(f"A pilot needs between 1 and {len(plan) - 1} circuits.")
    return plan[:pilot]


def save(run_dir: Path, record: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = run_dir / "run.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, run_dir / "run.json")


def load(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunError(f"No run record in {run_dir}.") from error


def new_record(
    data: BenchmarkInputs,
    backend_name: str,
    plan: list[hardware.PairSpec],
    pilot: bool,
    transpilation: dict,
    full_size: int,
    inputs: dict | None = None,
    prefix: str | None = None,
    extra: dict | None = None,
) -> tuple[str, dict]:
    """A run record. `inputs` defaults to the benchmark input hashes and `prefix` to pilot
    or full; `extra` adds further fields to the record."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{prefix or ('pilot' if pilot else 'full')}-{backend_name}-{stamp}"
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": now(),
        "pilot": pilot,
        "full_plan_circuits": full_size,
        "backend": backend_name,
        "sampler": SAMPLER,
        "shots": hardware.SHOTS,
        "inputs": data.input_hashes if inputs is None else inputs,
        "plan": [asdict(spec) for spec in plan],
        "transpilation": transpilation,
        "status": "created",
        "jobs": [],
    }
    return run_id, {**record, **(extra or {})}


def submit_run(
    run_dir: Path,
    record: dict,
    backend,
    isa_circuits: list,
    batch_size: int | None = None,
    submit: Callable = hardware.submit,
) -> dict:
    """Submit the ISA circuits in batches, saving each job ID immediately."""
    if record["jobs"]:
        raise RunError("This run already has jobs; retrieve them instead of resubmitting.")
    if len(isa_circuits) != len(record["plan"]):
        raise RunError("The ISA circuits do not match the run plan.")
    record["status"] = "submitting"
    save(run_dir, record)
    for start, stop in hardware.batches(len(isa_circuits), batch_size):
        job = submit(backend, isa_circuits[start:stop], record["shots"], record["sampler"])
        record["jobs"].append(
            {
                "job_id": job.job_id(),
                "start": start,
                "stop": stop,
                "submitted_at_utc": now(),
                "status": "SUBMITTED",
            }
        )
        save(run_dir, record)
    record["status"] = "submitted"
    save(run_dir, record)
    return record


def retrieve_run(run_dir: Path, service) -> dict:
    """Fetch every unfinished job by ID; store counts and IBM's usage metrics."""
    record = load(run_dir)
    if not record["jobs"]:
        raise RunError("This run has no submitted jobs.")
    for entry in record["jobs"]:
        if entry["status"] == DONE:
            continue
        job = service.job(entry["job_id"])
        entry["status"] = status_name(job.status())
        entry["checked_at_utc"] = now()
        if entry["status"] == DONE:
            counts = hardware.counts_from_result(job.result())
            if len(counts) != entry["stop"] - entry["start"]:
                raise RunError(f"Job {entry['job_id']} returned an unexpected number of results.")
            (run_dir / f"counts-{entry['job_id']}.json").write_text(
                json.dumps(counts) + "\n", encoding="utf-8"
            )
            entry["metrics"] = json.loads(json.dumps(job.metrics(), default=str))
        save(run_dir, record)
    statuses = {entry["status"] for entry in record["jobs"]}
    record["status"] = (
        "retrieved" if statuses == {DONE} else "failed" if statuses & FAILED else "waiting"
    )
    save(run_dir, record)
    return record


def collected_counts(run_dir: Path, record: dict) -> list[dict]:
    counts: list[dict] = []
    for entry in sorted(record["jobs"], key=lambda job: job["start"]):
        path = run_dir / f"counts-{entry['job_id']}.json"
        counts += json.loads(path.read_text(encoding="utf-8"))
    if len(counts) != len(record["plan"]):
        raise RunError("Collected counts do not cover the run plan.")
    return counts


def _job_usage(metrics: dict) -> dict:
    """Normalize IBM job metrics. IBM reports billed time as `usage.qpu_charge_time_seconds`
    (older responses: `usage.quantum_seconds`), the Open Plan cost in resource units as
    `usage.value`, and the circuits' own run time as `circuits_execution_time_ns`."""
    usage = metrics.get("usage", {}) or {}
    charged = usage.get("qpu_charge_time_seconds", usage.get("quantum_seconds"))
    units = usage.get("value") if usage.get("unit") == "resource_units" else None
    execution_ns = metrics.get("circuits_execution_time_ns")
    return {
        "charged_seconds": charged,
        "resource_units": units,
        "execution_seconds": None if execution_ns is None else execution_ns / 1e9,
    }


def usage_totals(record: dict) -> dict:
    """Sum each usage measure over the run's jobs; a measure missing from any job is None."""
    per_job = [_job_usage(job.get("metrics", {})) for job in record["jobs"]]
    totals = {}
    for key in ("charged_seconds", "resource_units", "execution_seconds"):
        values = [usage[key] for usage in per_job]
        totals[key] = None if not values or None in values else float(sum(values))
    return totals


def _require_retrieved(record: dict) -> None:
    if record["status"] != "retrieved":
        raise RunError(f"Run status is {record['status']}; retrieve all jobs first.")


def _ideal_kernels(train_angles: np.ndarray, test_angles: np.ndarray) -> tuple:
    return (
        qkernel.statevector_kernel(train_angles),
        qkernel.statevector_kernel(test_angles, train_angles),
    )


def _mean_circuit_error(
    plan: list[hardware.PairSpec], counts: list[dict], ideal_train, ideal_test
) -> float:
    """Mean |measured - ideal| over every circuit; a diagonal control is ideally 1."""

    def ideal(spec: hardware.PairSpec) -> float:
        if spec.kind == "test":
            return float(ideal_test[spec.left, spec.right])
        return 1.0 if spec.kind == "diagonal" else float(ideal_train[spec.left, spec.right])

    measured = [qkernel.kernel_from_counts(c) for c in counts]
    return float(np.mean([abs(m - ideal(spec)) for spec, m in zip(plan, measured, strict=True)]))


def measure_kernel(
    run_dir: Path,
    record: dict,
    train_angles: np.ndarray,
    test_angles: np.ndarray,
    labels: np.ndarray,
    model,
) -> dict:
    """Build a full run's hardware test kernel, calibrate it on the run's own training
    pairs, and classify `labels` with `model`. Saves the kernels next to the run record.

    Returns the circuit count, shots returned, mean error over all circuits, calibration,
    diagonal-control mean, test-kernel errors, evaluations and kernels per source ("qpu",
    and "qpu_calibrated" when the calibration is valid), the ideal test kernel, and
    limitations.
    """
    _require_retrieved(record)
    if record.get("pilot"):
        raise RunError("A pilot run has no complete test kernel.")
    plan = [hardware.PairSpec(**spec) for spec in record["plan"]]
    counts = collected_counts(run_dir, record)
    ideal_train, ideal_test = _ideal_kernels(train_angles, test_angles)

    assembled = hardware.assemble(plan, counts, len(train_angles), len(test_angles))
    pairs = assembled["calibration"]
    calibration = hardware.fit_calibration(
        np.array([ideal_train[i, j] for i, j, _ in pairs]), np.array([v for _, _, v in pairs])
    )
    raw = assembled["test_kernel"]
    evaluations = {"qpu": evaluate(model, raw, labels)}
    kernels_error = {"qpu": float(np.abs(raw - ideal_test).mean())}
    kernels = {"qpu": raw}
    limitations = []
    if calibration["valid"]:
        calibrated, clipped = hardware.apply_calibration(raw, calibration)
        evaluations["qpu_calibrated"] = evaluate(model, calibrated, labels)
        kernels_error["qpu_calibrated"] = float(np.abs(calibrated - ideal_test).mean())
        kernels["qpu_calibrated"] = calibrated
        calibration["clipped_test_values"] = clipped
        np.save(run_dir / "test-kernel-calibrated.npy", calibrated)
    else:
        limitations.append(
            f"Kernel calibration failed ({calibration['reason']}); only uncorrected hardware "
            "results are reported."
        )
    np.save(run_dir / "test-kernel-raw.npy", raw)
    diagonal = [value for _, value in assembled["diagonal"]]
    return {
        "circuits": len(plan),
        "shots_returned": sorted({sum(c.values()) for c in counts}),
        "mean_abs_kernel_error_vs_ideal": _mean_circuit_error(
            plan, counts, ideal_train, ideal_test
        ),
        "calibration": calibration,
        "diagonal_controls_mean": float(np.mean(diagonal)),
        "test_kernel_error_vs_ideal": kernels_error,
        "evaluations": evaluations,
        "kernels": kernels,
        "ideal_test_kernel": ideal_test,
        "limitations": limitations,
    }


def evaluate_run(
    run_dir: Path,
    data: BenchmarkInputs,
    results_path: Path | None = None,
    simulation_path: Path = SIMULATION_EVALUATION,
) -> dict:
    """Evaluate a retrieved run. Full runs are published to `results_path` when given."""
    record = load(run_dir)
    _require_retrieved(record)
    if record["inputs"] != data.input_hashes:
        raise RunError("The run used other inputs than the current benchmark artifacts.")

    plan = [hardware.PairSpec(**spec) for spec in record["plan"]]
    counts = collected_counts(run_dir, record)
    usage = usage_totals(record)
    summary = {
        "run_id": record["run_id"],
        "backend": record["backend"],
        "pilot": record["pilot"],
        "circuits": len(plan),
        "shots_returned": sorted({sum(c.values()) for c in counts}),
        "usage": usage,
        "mean_abs_kernel_error_vs_ideal": _mean_circuit_error(
            plan, counts, *_ideal_kernels(data.train_angles, data.test_angles)
        ),
    }
    if record["pilot"]:
        # Linear per-circuit projection. Billed seconds are rounded per job, so the
        # execution-time projection is the closer estimate of the full run's QPU time.
        scale = record["full_plan_circuits"] / len(plan)
        summary["projected_full_run"] = {
            key: None if value is None else value * scale for key, value in usage.items()
        }
        _write_summary(run_dir, summary)
        return summary

    n_train, n_test = len(data.train_angles), len(data.test_angles)
    measured = measure_kernel(
        run_dir,
        record,
        data.train_angles,
        data.test_angles,
        data.test_labels,
        data.models["quantum_kernel_svm"],
    )
    evaluations, limitations = measured["evaluations"], measured["limitations"]
    summary.update(
        {
            key: measured[key]
            for key in (
                "calibration",
                "diagonal_controls_mean",
                "test_kernel_error_vs_ideal",
                "evaluations",
            )
        }
    )
    _write_summary(run_dir, summary)

    if results_path is not None:
        simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
        if simulation["inputs"] != data.input_hashes:
            raise RunError("The simulation evaluation used other inputs; re-run it first.")
        rows = publish.simulation_rows(simulation["evaluations"]) + [
            publish.row("Quantum-kernel SVM", source, evaluation)
            for source, evaluation in evaluations.items()
        ]
        hardware_info = {
            "backend": record["backend"],
            "shots": record["shots"],
            "job_ids": [job["job_id"] for job in record["jobs"]],
        }
        split = {"train": n_train, "test": n_test, "seed": 42}
        results = publish.build(record["run_id"], rows, split, hardware_info, limitations)
        publish.write(results, results_path)
        summary["published"] = str(results_path)
    return summary


def _write_summary(run_dir: Path, summary: dict) -> None:
    (run_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
