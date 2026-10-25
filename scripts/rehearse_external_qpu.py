"""Noisy rehearsal of the external subset run on a local fake IBM backend. No account or QPU.

Runs the identical plan of scripts/run_qpu_external.py (the frozen 40-row subset, 1,008
circuits, 1,024 shots, the protocol's batch size) through ISA transpilation, the same
submit and retrieve code with SamplerV2 on FakeFez, and the same evaluation. The summary is
labeled "noisy simulation" and is never published as a hardware result.

Usage: python scripts/rehearse_external_qpu.py [--repeats N]
Outputs go only to the Git-ignored artifacts/external/rehearsal-fake_fez/; nothing is
written to reports/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit_ibm_runtime import fake_provider  # noqa: E402

from scripts import run_qpu_external  # noqa: E402
from src.evaluation import external_qpu, runtime  # noqa: E402
from src.quantum import hardware, qpu_run  # noqa: E402

OUTPUT_DIR = ROOT / "artifacts" / "external" / "rehearsal-fake_fez"
PREFIX = "rehearsal"


class LocalJob:
    """A local primitive job seen through the interface retrieve_run expects of IBM jobs.
    A fake backend records no IBM usage, so its metrics are empty."""

    def __init__(self, job) -> None:
        self._job = job

    def job_id(self) -> str:
        return self._job.job_id()

    def status(self):
        return self._job.status()

    def result(self):
        return self._job.result()

    def metrics(self) -> dict:
        return {}


def rehearse(
    loaded: run_qpu_external.SubsetInputs,
    backend,
    output_dir: Path,
    repeats: int,
    batch_size: int | None = None,
) -> dict:
    """Run the subset plan on `backend` and evaluate it; write only under `output_dir`."""
    for stale in output_dir.glob("counts-*.json"):
        stale.unlink()  # counts of an earlier rehearsal carry other job IDs
    circuits = hardware.build_circuits(loaded.plan, loaded.data.train_angles, loaded.angles)
    isa, stats = hardware.transpile_for(backend, circuits)
    _, record = qpu_run.new_record(
        loaded.data,
        backend.name,
        loaded.plan,
        False,
        stats,
        len(loaded.plan),
        inputs=loaded.inputs,
        prefix=PREFIX,
        extra={
            "protocol": loaded.references["protocol"],
            "label": external_qpu.SOURCES["noisy_simulation"],
        },
    )
    jobs: dict[str, LocalJob] = {}

    def submit(target, batch, shots, sampler):
        job = LocalJob(hardware.submit(target, batch, shots, sampler))
        jobs[job.job_id()] = job
        return job

    size = batch_size or loaded.protocol["hardware"]["batch_size"]
    qpu_run.submit_run(output_dir, record, backend, isa, size, submit=submit)
    for job in jobs.values():
        job.result()  # wait for the local simulation before retrieving
    record = qpu_run.retrieve_run(output_dir, SimpleNamespace(job=jobs.__getitem__))
    if record["status"] != "retrieved":
        raise qpu_run.RunError(f"Rehearsal jobs ended with status {record['status']}.")
    report = run_qpu_external.evaluate_subset(
        output_dir, record, loaded, repeats, "noisy_simulation"
    )
    external_qpu.write(report, output_dir / "summary.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=runtime.REPEATS)
    args = parser.parse_args()
    loaded = run_qpu_external.load_subset_inputs()
    report = rehearse(loaded, fake_provider.FakeFez(), OUTPUT_DIR, args.repeats)
    run_qpu_external.print_report(report)
    print(f"Artifacts: {OUTPUT_DIR.relative_to(ROOT).as_posix()} (noisy simulation; not published)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
