"""Run the quantum benchmark on IBM Quantum hardware. Uses real QPU quota when submitting.

Commands:
  submit [--pilot N] [--batch-size N] [--yes]
      Connect with the .env settings, check the instance quota, transpile for
      IBM_QUANTUM_BACKEND, ask for confirmation, and submit with SamplerV2 in job mode.
      --pilot N submits only the first N circuits to measure QPU time before the full run.
  retrieve RUN_ID
      Fetch finished jobs by ID. Safe to repeat; nothing is resubmitted.
  evaluate RUN_ID
      Local only. A pilot reports QPU time and projects the full run; a full run builds the
      hardware test kernel, calibrates it, classifies the test set, and publishes
      reports/benchmark-results.json.
  list
      Local only. Show recorded runs and their status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ibm  # noqa: E402
from src.evaluation import inputs as benchmark_inputs  # noqa: E402
from src.quantum import hardware, qpu_run  # noqa: E402

RESULTS_PATH = ROOT / "reports" / "benchmark-results.json"
CONFIRMATION = "SUBMIT"


def latest_pilot_projection(backend: str) -> dict | None:
    projections = []
    for path in sorted(qpu_run.RUNS_DIR.glob(f"pilot-{backend}-*/evaluation.json")):
        value = json.loads(path.read_text(encoding="utf-8")).get("projected_full_run")
        if value:
            projections.append(value)
    return projections[-1] if projections else None


def command_submit(args: argparse.Namespace) -> int:
    settings = ibm.settings()
    backend_name = settings.get("IBM_QUANTUM_BACKEND", "").strip()
    if not backend_name:
        print("Set IBM_QUANTUM_BACKEND (for example ibm_fez) so the whole run uses one backend.")
        return 1
    data = benchmark_inputs.load()
    plan = qpu_run.plan_for(data, args.pilot)

    service = ibm.connect(settings)
    usage = service.usage()
    if usage.get("usage_limit_reached"):
        print("The instance has reached its usage limit; nothing was submitted.")
        return 1
    backend = service.backend(backend_name)
    circuits = hardware.build_circuits(plan, data.train_angles, data.test_angles)
    isa, stats = hardware.transpile_for(backend, circuits)
    batch_size = args.batch_size or getattr(backend, "max_circuits", None)
    jobs = len(hardware.batches(len(isa), batch_size))

    print(f"Backend: {backend.name}; instance remaining: {usage.get('usage_remaining_seconds')} s")
    print(f"Circuits: {len(isa)} ({'pilot' if args.pilot else 'full run'}), {jobs} job(s), ")
    print(f"  {hardware.SHOTS} shots each, depth up to {stats['depth']['max']}.")
    projection = latest_pilot_projection(backend.name)
    if not args.pilot:
        if projection is None:
            print("No pilot projection yet; consider `submit --pilot 12` first.")
        else:
            print(
                "Latest pilot projects for the full run: "
                + ", ".join(
                    f"{key.replace('_', ' ')} {value:.2f}"
                    for key, value in projection.items()
                    if value is not None
                )
            )
    if not args.yes:
        answer = input(f"Type {CONFIRMATION} to use QPU quota: ").strip()
        if answer != CONFIRMATION:
            print("Cancelled; nothing was submitted.")
            return 1

    run_id, record = qpu_run.new_record(
        data, backend.name, plan, bool(args.pilot), stats, len(qpu_run.full_plan(data))
    )
    run_dir = qpu_run.RUNS_DIR / run_id
    qpu_run.submit_run(run_dir, record, backend, isa, batch_size)
    print(f"Submitted run {run_id}: jobs {[job['job_id'] for job in record['jobs']]}")
    print(f"Next: python scripts/run_qpu.py retrieve {run_id}")
    return 0


def command_retrieve(args: argparse.Namespace) -> int:
    run_dir = qpu_run.RUNS_DIR / args.run_id
    record = qpu_run.retrieve_run(run_dir, ibm.connect(ibm.settings()))
    for job in record["jobs"]:
        print(f"  {job['job_id']}: {job['status']}")
    print(f"Run status: {record['status']}")
    if record["status"] == "retrieved":
        print(f"Next: python scripts/run_qpu.py evaluate {args.run_id}")
        return 0
    return 0 if record["status"] == "waiting" else 1


def command_evaluate(args: argparse.Namespace) -> int:
    run_dir = qpu_run.RUNS_DIR / args.run_id
    summary = qpu_run.evaluate_run(run_dir, benchmark_inputs.load(), RESULTS_PATH)
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluations"}, indent=2))
    for source, evaluation in summary.get("evaluations", {}).items():
        m = evaluation["metrics"]
        print(
            f"  {source:<15} acc {m['accuracy']:.3f}  bal {m['balanced_accuracy']:.3f}  "
            f"CM {evaluation['confusion_matrix']}"
        )
    return 0


def command_list(_: argparse.Namespace) -> int:
    runs = sorted(qpu_run.RUNS_DIR.glob("*/run.json"))
    if not runs:
        print("No QPU runs recorded.")
    for path in runs:
        record = json.loads(path.read_text(encoding="utf-8"))
        print(f"{record['run_id']}: {record['status']}, {len(record['plan'])} circuits")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--pilot", type=int)
    submit.add_argument("--batch-size", type=int)
    submit.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    submit.set_defaults(handler=command_submit)
    for name, handler in (("retrieve", command_retrieve), ("evaluate", command_evaluate)):
        sub = commands.add_parser(name)
        sub.add_argument("run_id")
        sub.set_defaults(handler=handler)
    commands.add_parser("list").set_defaults(handler=command_list)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (qpu_run.RunError, benchmark_inputs.InputError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
