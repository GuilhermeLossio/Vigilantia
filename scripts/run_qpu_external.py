"""Run the 40-row external subset on IBM Quantum hardware (configs/external-qpu-subset.json).

Uses real QPU quota when submitting. The protocol is frozen; this script only follows it.

Commands:
  plan
      Local only. Draw the subset, build the plan, and print its class counts, circuit
      count, and projected execution and billed time. Never connects to IBM.
  submit [--yes]
      Check IBM_QUANTUM_BACKEND and that no earlier subset run exists before connecting,
      check the instance usage limit and the protocol's budget guard, transpile, ask for
      confirmation, and submit with SamplerV2 in job mode, in jobs of the protocol's batch
      size. The run is named external-<backend>-<timestamp>. The protocol allows one run:
      submit refuses while an earlier run of the subset with jobs exists.
  retrieve RUN_ID
      Fetch finished jobs by ID. Safe to repeat; nothing is resubmitted.
  evaluate RUN_ID
      Local only. Build and calibrate the hardware kernel, evaluate the six frozen rows on
      the subset with bootstrap intervals and runtimes, write reports/external-qpu-subset.json,
      and copy the run record and counts to reports/qpu-evidence-external/<run_id>/.
      Local rows are timed over the protocol's 25 repeats. Refuses to replace a report of
      another run. reports/benchmark-results.json is never touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts import download_external, evaluate_external, run_qpu  # noqa: E402
from src.config import ibm  # noqa: E402
from src.evaluation import external, external_qpu, runtime  # noqa: E402
from src.evaluation import inputs as benchmark_inputs  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.features.scaling import ScalingError  # noqa: E402
from src.inference.predictor import MODELS_PATH, _version  # noqa: E402
from src.quantum import hardware, qpu_run  # noqa: E402

PROTOCOL_PATH = ROOT / "configs" / "external-qpu-subset.json"
PARENT_PATH = evaluate_external.CONFIG_PATH
REPORT_PATH = ROOT / "reports" / "external-qpu-subset.json"
EVIDENCE_DIR = ROOT / "reports" / "qpu-evidence-external"
PREFIX = "external"
CLASSICAL = ("majority", "logistic_regression", "rbf_svm")


@dataclass
class SubsetInputs:
    """Everything a subset run is planned and evaluated from."""

    protocol: dict
    references: dict
    data: benchmark_inputs.BenchmarkInputs
    subset: external_qpu.Subset
    angles: np.ndarray
    clipped: dict
    primary_rows: int
    plan: list
    model_version: str

    @property
    def inputs(self) -> dict:
        """The run record's inputs: everything a later evaluation must find unchanged."""
        return {
            "parent_protocol_sha256": self.references["parent_protocol"]["sha256"],
            "subset_protocol_sha256": self.references["protocol"]["sha256"],
            **self.data.input_hashes,
            "subset_row_index": list(self.subset.row_index),
            "subset_group_hash": list(self.subset.group_hash),
        }


def load_subset_inputs() -> SubsetInputs:
    """Verify the local snapshot (no download), prepare the primary set, and draw the subset."""
    protocol = external_qpu.load_protocol(PROTOCOL_PATH, PARENT_PATH)
    parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
    evaluate_external.verified_snapshot(parent["dataset"])
    data, prepared, scaler = evaluate_external.load_inputs(parent["dataset"])
    subset = external_qpu.select_subset(prepared.rows, protocol["subset"]["per_class"])
    _, angles, clipped = external.featurize(prepared.rows["host"].iloc[subset.positions], scaler)
    plan = qpu_run.full_plan(data, angles)
    if len(plan) != protocol["hardware"]["circuits"]:
        raise external_qpu.ExternalQpuError(
            f"The plan has {len(plan)} circuits; the protocol froze "
            f"{protocol['hardware']['circuits']}."
        )
    return SubsetInputs(
        protocol=protocol,
        references={
            "protocol": external_qpu.protocol_reference(PROTOCOL_PATH, protocol, ROOT),
            "parent_protocol": {
                "path": PARENT_PATH.relative_to(ROOT).as_posix(),
                "sha256": external.protocol_sha256(PARENT_PATH),
            },
        },
        data=data,
        subset=subset,
        angles=angles,
        clipped=clipped,
        primary_rows=int(len(prepared.rows)),
        plan=plan,
        model_version=_version(benchmark_inputs.SCALER_PATH.read_bytes(), MODELS_PATH.read_bytes()),
    )


def describe_plan(loaded: SubsetInputs, batch_size: int) -> list[str]:
    counts = external_qpu.plan_counts(loaded.plan)
    circuits = len(loaded.plan)
    settings = loaded.protocol["hardware"]
    jobs = len(hardware.batches(circuits, batch_size))
    classes = loaded.subset.class_counts
    return [
        f"Subset: {len(loaded.subset.labels)} of {loaded.primary_rows} primary rows "
        f"(legitimate {classes['legitimate']}, phishing {classes['phishing']})",
        f"Row indexes: {loaded.subset.row_index}",
        f"Circuits: {circuits} ({counts['test']} test, {counts['calibration']} calibration, "
        f"{counts['diagonal']} diagonal); {jobs} job(s) of up to {batch_size}",
        f"Backend {settings['backend']}, {settings['shots']} shots each, sampler {settings['sampler']}",
        f"Projected QPU time: {settings['projected_execution_seconds']} s execution, "
        f"{external_qpu.projected_billed_seconds(circuits):.1f} s billed "
        f"({external_qpu.BILLED_SECONDS_PER_CIRCUIT} s per circuit)",
        f"Budget guard: submit needs more than {external_qpu.required_quota(circuits):.1f} s "
        "of remaining instance quota",
    ]


def command_plan(_: argparse.Namespace) -> int:
    loaded = load_subset_inputs()
    print("\n".join(describe_plan(loaded, loaded.protocol["hardware"]["batch_size"])))
    print("No IBM connection was made.")
    return 0


def earlier_run() -> tuple[str, str] | None:
    """(run_id, status) of an earlier subset run, on any backend, that used or may use quota."""
    for path in sorted(qpu_run.RUNS_DIR.glob(f"{PREFIX}-*/run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("jobs") or record.get("status") != "created":
            return path.parent.name, str(record.get("status"))
    return None


def command_submit(args: argparse.Namespace) -> int:
    settings = ibm.settings()
    backend_name = run_qpu.required_backend(settings)
    if backend_name is None:
        return 1
    protocol = external_qpu.load_protocol(PROTOCOL_PATH, PARENT_PATH)
    if backend_name != protocol["hardware"]["backend"]:
        print(f"The protocol froze backend {protocol['hardware']['backend']}; set it first.")
        return 1
    previous = earlier_run()
    if previous:
        run_id, status = previous
        print(
            f"Run {run_id} of this subset already exists (status {status}); the protocol "
            "allows one run, so nothing was submitted. Retrieve and evaluate it: "
            f"python scripts/run_qpu_external.py retrieve {run_id}"
        )
        return 1
    loaded = load_subset_inputs()
    circuits = len(loaded.plan)

    opened = run_qpu.open_instance(settings)
    if opened is None:
        return 1
    service, usage = opened
    refusal = external_qpu.budget_refusal(usage.get("usage_remaining_seconds"), circuits)
    if refusal:
        print(refusal)
        return 1
    backend = service.backend(backend_name)
    built = hardware.build_circuits(loaded.plan, loaded.data.train_angles, loaded.angles)
    isa, stats = hardware.transpile_for(backend, built)
    batch_size = protocol["hardware"]["batch_size"]

    print(f"Backend: {backend.name}; instance remaining: {usage.get('usage_remaining_seconds')} s")
    print("\n".join(describe_plan(loaded, batch_size)))
    print(f"Transpiled depth up to {stats['depth']['max']}.")
    if not run_qpu.confirmed(args.yes):
        return 1

    run_id, record = qpu_run.new_record(
        loaded.data,
        backend.name,
        loaded.plan,
        False,
        stats,
        circuits,
        inputs=loaded.inputs,
        prefix=PREFIX,
        extra={"protocol": loaded.references["protocol"]},
    )
    qpu_run.submit_run(qpu_run.RUNS_DIR / run_id, record, backend, isa, batch_size)
    print(f"Submitted run {run_id}: jobs {[job['job_id'] for job in record['jobs']]}")
    print(f"Next: python scripts/run_qpu_external.py retrieve {run_id}")
    return 0


def run_dir_for(run_id: str) -> Path:
    if not run_id.startswith(f"{PREFIX}-") or Path(run_id).name != run_id:
        raise qpu_run.RunError(f"External subset runs are named {PREFIX}-<backend>-<timestamp>.")
    return qpu_run.RUNS_DIR / run_id


def command_retrieve(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.run_id)
    record = qpu_run.retrieve_run(run_dir, ibm.connect(ibm.settings()))
    for job in record["jobs"]:
        print(f"  {job['job_id']}: {job['status']}")
    print(f"Run status: {record['status']}")
    if record["status"] == "retrieved":
        print(f"Next: python scripts/run_qpu_external.py evaluate {args.run_id}")
        return 0
    return 0 if record["status"] == "waiting" else 1


def evaluate_subset(
    run_dir: Path, record: dict, loaded: SubsetInputs, repeats: int, source: str
) -> dict:
    """The six frozen rows on the subset rows, with intervals, runtimes, and kernel checks."""
    if record["inputs"] != loaded.inputs:
        raise qpu_run.RunError("The run used other inputs than the current subset and artifacts.")
    data, angles, labels = loaded.data, loaded.angles, loaded.subset.labels
    quantum = data.models["quantum_kernel_svm"]
    measured = qpu_run.measure_kernel(run_dir, record, data.train_angles, angles, labels, quantum)
    evaluations = [
        (name, "classical", evaluate(data.models[name], angles, labels)) for name in CLASSICAL
    ]
    evaluations.append(
        (
            "quantum_kernel_svm",
            "simulation",
            evaluate(quantum, measured["ideal_test_kernel"], labels),
        )
    )
    evaluations += [
        ("quantum_kernel_svm", kernel_source, evaluation)
        for kernel_source, evaluation in measured["evaluations"].items()
    ]
    settings = loaded.protocol["metrics"]["confidence_intervals"]
    rows, used = external_qpu.rows_with_intervals(evaluations, labels, settings)
    limitations = list(measured["limitations"])
    if not measured["calibration"]["valid"]:
        limitations.append(external_qpu.CALIBRATION_DEVIATION)
    usage = qpu_run.usage_totals(record)
    timings = runtime.measure(data, measured["kernels"], repeats, test_angles=angles)
    report = external_qpu.build_report(
        source=source,
        protocol=loaded.references["protocol"],
        parent_protocol=loaded.references["parent_protocol"],
        benchmark={
            "model_version": loaded.model_version,
            "training_examples": int(len(data.train_labels)),
            "input_hashes": data.input_hashes,
        },
        subset={
            "size": len(labels),
            "primary_rows": loaded.primary_rows,
            "class_counts": loaded.subset.class_counts,
            "seed": external_qpu.SUBSET_SEED,
            "row_index": list(loaded.subset.row_index),
            "group_hash": list(loaded.subset.group_hash),
            "clipped_values": loaded.clipped,
        },
        hardware_run={
            "backend": record["backend"],
            "run_id": record["run_id"],
            "sampler": record["sampler"],
            "shots": record["shots"],
            "job_ids": [job["job_id"] for job in record["jobs"]],
            "circuits": measured["circuits"],
            "plan": external_qpu.plan_counts([hardware.PairSpec(**s) for s in record["plan"]]),
            "shots_returned": measured["shots_returned"],
        },
        kernel={
            "mean_abs_kernel_error_vs_ideal": measured["mean_abs_kernel_error_vs_ideal"],
            "test_kernel_error_vs_ideal": measured["test_kernel_error_vs_ideal"],
            "calibration": measured["calibration"],
            "diagonal_controls_mean": measured["diagonal_controls_mean"],
        },
        runtime={
            "method": external_qpu.RUNTIME_METHOD.format(repeats=repeats),
            "repeats": repeats,
            "environment": runtime.environment(),
            "qpu": {key: usage[key] for key in external_qpu.USAGE_KEYS},
        },
        bootstrap={
            "method": external_qpu.BOOTSTRAP_METHOD,
            "level": settings["level"],
            "resamples": settings["resamples"],
            "resamples_used": used,
            "seed": settings["seed"],
        },
        models=external_qpu.with_runtime(rows, timings, usage),
        limitations=limitations,
    )
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def copy_evidence(run_dir: Path, record: dict, destination: Path) -> Path:
    """Copy run.json and every counts file, and nothing else, to destination/<run_id>/,
    after checking every file for credentials, instance identifiers, URLs, and local paths."""
    names = ["run.json"] + [f"counts-{job['job_id']}.json" for job in record["jobs"]]
    refusal = external_qpu.evidence_refusal(
        {name: (run_dir / name).read_text(encoding="utf-8") for name in names}
    )
    if refusal:
        raise external_qpu.ExternalQpuError(f"Evidence was not copied: {refusal}")
    target = destination / record["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(run_dir / name, target / name)
    return target


def print_report(report: dict) -> None:
    size = report["subset"]["size"]
    print(f"{report['label']}: {report['hardware']['run_id']} on {report['hardware']['backend']}")
    print(f"{'model':<22}{'source':<16}" + "".join(f"{m[:9]:>11}" for m in external.METRICS))
    for row in report["models"]:
        values = "".join(
            f"{'n/a' if v is None else f'{v:.3f}':>11}" for v in row["metrics"].values()
        )
        print(f"{row['model']:<22}{row['kernel_source']:<16}{values}   n={size}")
    print(f"95% bootstrap intervals (n={size}):")
    for row in report["models"]:
        spans = "".join(
            f"{'n/a' if ci is None else f'{ci[0]:.2f}-{ci[1]:.2f}':>11}"
            for ci in row["confidence_intervals"].values()
        )
        print(f"{row['model'] + ' ' + row['kernel_source']:<38}{spans}")
    kernel = report["kernel"]
    print(f"Test kernel error vs exact simulation: {kernel['test_kernel_error_vs_ideal']}")
    print(f"Calibration: {kernel['calibration']}")
    print(f"Diagonal controls mean: {kernel['diagonal_controls_mean']:.3f} (ideal 1)")
    print(f"IBM usage: {report['runtime']['qpu']}")


def shown(path: Path) -> str:
    """A repository-relative path for printing; never an absolute local path."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def published_run_conflict(run_id: str) -> str | None:
    """Why publishing run_id would replace the report or evidence of another run."""
    if REPORT_PATH.exists():
        try:
            published = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["hardware"]["run_id"]
        except (ValueError, KeyError, TypeError):
            return f"{shown(REPORT_PATH)} exists and names no run; it was not replaced."
        if published != run_id:
            return f"{shown(REPORT_PATH)} already holds run {published}; it was not replaced."
    if EVIDENCE_DIR.is_dir():
        others = sorted(p.name for p in EVIDENCE_DIR.iterdir() if p.is_dir() and p.name != run_id)
        if others:
            return f"{shown(EVIDENCE_DIR)} already holds run {others[0]}; nothing was published."
    return None


def command_evaluate(args: argparse.Namespace) -> int:
    run_dir = run_dir_for(args.run_id)
    conflict = published_run_conflict(args.run_id)
    if conflict:
        print(f"{conflict} The protocol publishes one run only.")
        return 1
    record = qpu_run.load(run_dir)
    report = evaluate_subset(
        run_dir, record, load_subset_inputs(), external_qpu.FROZEN_REPEATS, "ibm_quantum"
    )
    external_qpu.write(report, REPORT_PATH)
    evidence = copy_evidence(run_dir, record, EVIDENCE_DIR)
    print_report(report)
    print(f"Report: {shown(REPORT_PATH)}")
    print(f"Evidence: {shown(evidence)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").set_defaults(handler=command_plan)
    submit = commands.add_parser("submit")
    submit.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    submit.set_defaults(handler=command_submit)
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("run_id")
    retrieve.set_defaults(handler=command_retrieve)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("run_id")
    evaluate_parser.set_defaults(handler=command_evaluate)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (
        qpu_run.RunError,
        benchmark_inputs.InputError,
        external.ExternalError,
        external_qpu.ExternalQpuError,
        download_external.AcquisitionError,
        ScalingError,
    ) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
