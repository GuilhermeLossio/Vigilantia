"""Run the v2 page-feature quantum kernel on a 40-row test subset (configs/benchmark-v2-qpu-subset.json).

Uses real QPU quota when submitting. The protocol is frozen; this script only follows it.

Commands:
  Every command but retrieve and evaluate accepts --rows N (even, 4 to 40) to measure only
  the first N/2 rows of each class in the frozen draw order, so the run fits a smaller QPU
  quota. evaluate reads the row count from the run record.

  plan [--rows N]
      Local only. Rebuild the frozen inputs from artifacts/benchmark-v2, draw the subset, and
      print its class counts, circuit count, and projected billed time. Never connects to IBM.
  rehearse [--rows N]
      Local only. Run the identical plan on the FakeFez noisy simulator and evaluate it.
      Outputs go to the Git-ignored artifacts/benchmark-v2-qpu-rehearsal/; nothing is published.
  submit [--rows N] [--yes]
      Check IBM_QUANTUM_BACKEND and that no earlier run exists before connecting, check the
      instance usage limit and the budget guard (which names the largest --rows that fits),
      transpile, ask for confirmation, and submit
      with SamplerV2 in job mode. The run is named v2page-<backend>-<timestamp>. The protocol
      allows one run.
  retrieve RUN_ID
      Fetch finished jobs by ID. Safe to repeat; nothing is resubmitted.
  evaluate RUN_ID
      Local only. Build and calibrate the hardware kernel, evaluate the six frozen rows, time
      them locally, write reports/benchmark-v2-qpu-subset.json and docs/benchmark-v2-qpu.svg,
      and copy the run record and counts to reports/qpu-evidence-v2/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import render_v2_qpu_svg, run_qpu  # noqa: E402
from scripts.rehearse_external_qpu import LocalJob  # noqa: E402
from src.config import ibm  # noqa: E402
from src.evaluation import benchmark_v2_qpu as v2qpu  # noqa: E402
from src.evaluation import external_qpu  # noqa: E402
from src.quantum import hardware, qpu_run  # noqa: E402

REHEARSAL_DIR = ROOT / "artifacts" / "benchmark-v2-qpu-rehearsal"


def describe(loaded: v2qpu.Inputs) -> list[str]:
    settings = loaded.protocol["hardware"]
    circuits = len(loaded.plan)
    counts = external_qpu.plan_counts(loaded.plan)
    labels = loaded.labels
    jobs = len(hardware.batches(circuits, settings["batch_size"]))
    return [
        f"Model: {loaded.protocol['benchmark']['quantum_run']} "
        f"({len(loaded.train_angles)} training references)",
        f"Subset: {len(labels)} test rows (legitimate {int((labels == 0).sum())}, "
        f"phishing {int(labels.sum())}); no domain shared with development or calibration",
        f"Circuits: {circuits} ({counts['test']} test, {counts['calibration']} calibration, "
        f"{counts['diagonal']} diagonal); {jobs} job(s) of up to {settings['batch_size']}",
        f"Backend {settings['backend']}, {settings['shots']} shots each, sampler {settings['sampler']}",
        f"Projected billed time: {external_qpu.projected_billed_seconds(circuits):.1f} s; "
        f"submit needs more than {external_qpu.required_quota(circuits):.1f} s remaining",
    ]


def load(rows: int | None = None) -> v2qpu.Inputs:
    return v2qpu.load_inputs(v2qpu.load_protocol(), rows=rows)


def command_plan(args: argparse.Namespace) -> int:
    print("\n".join(describe(load(args.rows))))
    print("No IBM connection was made.")
    return 0


def print_report(report: dict) -> None:
    size = report["subset"]["size"]
    print(f"{report['source']}: {report['hardware']['run_id']} on {report['hardware']['backend']}")
    print(f"{'model':<22}{'source':<16}" + "".join(f"{m[:9]:>11}" for m in v2qpu.METRICS))
    for row in report["models"]:
        values = "".join(f"{v:>11.3f}" for v in row["metrics"].values())
        print(f"{row['model']:<22}{row['kernel_source']:<16}{values}   n={size}")
    print(f"95% bootstrap intervals for accuracy (n={size}):")
    for row in report["models"]:
        low, high = row["confidence_intervals"]["accuracy"]
        print(f"  {row['model']} {row['kernel_source']}: {low:.3f}-{high:.3f}")
    kernel = report["kernel"]
    print(f"Test kernel error vs exact simulation: {kernel['test_kernel_error_vs_ideal']}")
    print(f"Ideal test kernel range: {kernel['ideal_test_kernel_spread']}")
    print(f"Calibration: {kernel['calibration']}")
    print(f"Diagonal controls mean: {kernel['diagonal_controls_mean']:.3f} (ideal 1)")
    print(f"IBM usage: {report['hardware']['usage']}")
    print("Runtime (median seconds: fit / test kernel / predict):")
    for row in report["models"]:
        times = " / ".join(
            "-" if v is None else f"{v:.6f}" for v in row["runtime_seconds"].values()
        )
        print(f"  {row['model']} {row['kernel_source']}: {times}")


def rehearse(loaded: v2qpu.Inputs, backend, output_dir: Path) -> dict:
    """Run the plan on a local backend through the same submit, retrieve, and evaluate code."""
    for stale in output_dir.glob("counts-*.json"):
        stale.unlink()
    circuits = hardware.build_circuits(loaded.plan, loaded.train_angles, loaded.test_angles)
    isa, stats = hardware.transpile_for(backend, circuits)
    _, record = qpu_run.new_record(
        None, backend.name, loaded.plan, False, stats, len(loaded.plan),
        inputs=loaded.inputs, prefix="rehearsal", extra={"protocol": loaded.reference},
    )  # fmt: skip
    jobs: dict[str, LocalJob] = {}

    def submit(target, batch, shots, sampler):
        job = LocalJob(hardware.submit(target, batch, shots, sampler))
        jobs[job.job_id()] = job
        return job

    size = loaded.protocol["hardware"]["batch_size"]
    qpu_run.submit_run(output_dir, record, backend, isa, size, submit=submit)
    for job in jobs.values():
        job.result()  # wait for the local simulation before retrieving
    record = qpu_run.retrieve_run(output_dir, SimpleNamespace(job=jobs.__getitem__))
    if record["status"] != "retrieved":
        raise qpu_run.RunError(f"Rehearsal jobs ended with status {record['status']}.")
    report = v2qpu.evaluate_run(output_dir, record, loaded, "noisy_simulation")
    v2qpu.write(report, output_dir / "summary.json")
    return report


def command_rehearse(args: argparse.Namespace) -> int:
    from qiskit_ibm_runtime import fake_provider

    print_report(rehearse(load(args.rows), fake_provider.FakeFez(), REHEARSAL_DIR))
    print(
        f"Artifacts: {REHEARSAL_DIR.relative_to(ROOT).as_posix()} (noisy simulation; not published)"
    )
    return 0


def earlier_run() -> tuple[str, str] | None:
    """(run_id, status) of an earlier run that used or may use quota."""
    for path in sorted(qpu_run.RUNS_DIR.glob(f"{v2qpu.PREFIX}-*/run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("jobs") or record.get("status") != "created":
            return path.parent.name, str(record.get("status"))
    return None


def command_submit(args: argparse.Namespace) -> int:
    settings = ibm.settings()
    backend_name = run_qpu.required_backend(settings)
    if backend_name is None:
        return 1
    protocol = v2qpu.load_protocol()
    if backend_name != protocol["hardware"]["backend"]:
        print(f"The protocol froze backend {protocol['hardware']['backend']}; set it first.")
        return 1
    previous = earlier_run()
    if previous:
        run_id, status = previous
        print(
            f"Run {run_id} already exists (status {status}); the protocol allows one run, so "
            f"nothing was submitted. Retrieve it: python scripts/run_qpu_v2.py retrieve {run_id}"
        )
        return 1
    loaded = v2qpu.load_inputs(protocol, rows=args.rows)
    circuits = len(loaded.plan)
    opened = run_qpu.open_instance(settings)
    if opened is None:
        return 1
    service, usage = opened
    remaining = usage.get("usage_remaining_seconds")
    refusal = external_qpu.budget_refusal(remaining, circuits)
    if refusal:
        print(refusal)
        if isinstance(remaining, int | float):
            fits = v2qpu.max_rows(protocol, len(loaded.train_angles), remaining)
            print(
                f"The largest subset that fits is --rows {fits}."
                if fits
                else "No allowed subset fits the remaining quota."
            )
        return 1
    backend = service.backend(backend_name)
    built = hardware.build_circuits(loaded.plan, loaded.train_angles, loaded.test_angles)
    isa, stats = hardware.transpile_for(backend, built)
    print(f"Backend: {backend.name}; instance remaining: {usage.get('usage_remaining_seconds')} s")
    print("\n".join(describe(loaded)))
    print(f"Transpiled depth up to {stats['depth']['max']}.")
    if not run_qpu.confirmed(args.yes):
        return 1
    run_id, record = qpu_run.new_record(
        None, backend.name, loaded.plan, False, stats, circuits,
        inputs=loaded.inputs, prefix=v2qpu.PREFIX, extra={"protocol": loaded.reference},
    )  # fmt: skip
    qpu_run.submit_run(
        qpu_run.RUNS_DIR / run_id, record, backend, isa, protocol["hardware"]["batch_size"]
    )
    print(f"Submitted run {run_id}: jobs {[job['job_id'] for job in record['jobs']]}")
    print(f"Next: python scripts/run_qpu_v2.py retrieve {run_id}")
    return 0


def run_dir_for(run_id: str) -> Path:
    if not run_id.startswith(f"{v2qpu.PREFIX}-") or Path(run_id).name != run_id:
        raise qpu_run.RunError(f"V2 runs are named {v2qpu.PREFIX}-<backend>-<timestamp>.")
    return qpu_run.RUNS_DIR / run_id


def command_retrieve(args: argparse.Namespace) -> int:
    record = qpu_run.retrieve_run(run_dir_for(args.run_id), ibm.connect(ibm.settings()))
    for job in record["jobs"]:
        print(f"  {job['job_id']}: {job['status']}")
    print(f"Run status: {record['status']}")
    if record["status"] == "retrieved":
        print(f"Next: python scripts/run_qpu_v2.py evaluate {args.run_id}")
        return 0
    return 0 if record["status"] == "waiting" else 1


def published_conflict(run_id: str) -> str | None:
    """Why publishing run_id would replace another run's report or evidence."""
    if v2qpu.REPORT_PATH.exists():
        published = json.loads(v2qpu.REPORT_PATH.read_text(encoding="utf-8"))["hardware"]["run_id"]
        if published != run_id:
            return f"reports/benchmark-v2-qpu-subset.json already holds run {published}."
    if v2qpu.EVIDENCE_DIR.is_dir():
        others = sorted(p.name for p in v2qpu.EVIDENCE_DIR.iterdir() if p.name != run_id)
        if others:
            return f"reports/qpu-evidence-v2 already holds run {others[0]}."
    return None


def copy_evidence(run_dir: Path, record: dict) -> Path:
    """Copy run.json and every counts file after checking them for secrets, URLs, and paths."""
    names = ["run.json"] + [f"counts-{job['job_id']}.json" for job in record["jobs"]]
    refusal = external_qpu.evidence_refusal(
        {name: (run_dir / name).read_text(encoding="utf-8") for name in names}
    )
    if refusal:
        raise v2qpu.V2QpuError(f"Evidence was not copied: {refusal}")
    target = v2qpu.EVIDENCE_DIR / record["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(run_dir / name, target / name)
    return target


def command_evaluate(args: argparse.Namespace) -> int:
    conflict = published_conflict(args.run_id)
    if conflict:
        print(f"{conflict} The protocol publishes one run only.")
        return 1
    run_dir = run_dir_for(args.run_id)
    record = qpu_run.load(run_dir)
    rows = len(record["inputs"]["subset_row_index"])
    report = v2qpu.evaluate_run(run_dir, record, load(rows), "ibm_quantum")
    v2qpu.write(report, v2qpu.REPORT_PATH)
    evidence = copy_evidence(run_dir, record)
    chart = render_v2_qpu_svg.write(v2qpu.REPORT_PATH, render_v2_qpu_svg.OUTPUT)
    print_report(report)
    print("Report: reports/benchmark-v2-qpu-subset.json")
    print(f"Chart: {chart.relative_to(ROOT).as_posix()}")
    print(f"Evidence: {evidence.relative_to(ROOT).as_posix()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("plan", command_plan),
        ("rehearse", command_rehearse),
        ("submit", command_submit),
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--rows", type=int, help="even subset size; default: the protocol's")
        sub.set_defaults(handler=handler)
        if name == "submit":
            sub.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    for name, handler in (("retrieve", command_retrieve), ("evaluate", command_evaluate)):
        sub = commands.add_parser(name)
        sub.add_argument("run_id")
        sub.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (qpu_run.RunError, v2qpu.V2QpuError, OSError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
