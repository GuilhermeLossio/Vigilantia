"""Check IBM Quantum access and the client-side Sampler with a dry run. Nothing runs on a QPU.

Steps:
1. Connect with IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE, read from the repository's .env
   file and overridden by variables set in the terminal session. Without a token, the saved
   Qiskit account is used. The token is never printed.
2. Print the instance usage reported by IBM.
3. Pick IBM_QUANTUM_BACKEND, or the least busy operational hardware backend.
4. Submit a dry run of the client-side Sampler (qiskit_ibm_runtime.executor_sampler). IBM
   validates the job server-side and estimates its QPU time, but does not execute it;
   results are mock data. With --full-workload the dry run covers all 336 benchmark
   circuits, which estimates the QPU time the real benchmark would use.

SamplerV2 has no dry run, so its access cannot be checked without spending quota.

Usage: python scripts/check_ibm_access.py [--full-workload] [--timeout 600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import ibm  # noqa: E402
from src.quantum import hardware  # noqa: E402

ENV_PATH = ibm.ENV_PATH
service_kwargs = ibm.service_kwargs


def step(number: int, text: str) -> None:
    print(f"\n[{number}] {text}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full-workload", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    from qiskit_ibm_runtime.executor_sampler import Sampler

    step(1, "Connecting to IBM Quantum")
    settings = ibm.settings(ENV_PATH)
    kwargs = service_kwargs(settings)
    try:
        service = ibm.connect(settings)
    except Exception as error:  # noqa: BLE001 - report any connection failure plainly
        print(f"FAILED: could not connect ({type(error).__name__}: {error})")
        print("Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE in .env, or save an account")
        print("once with QiskitRuntimeService.save_account(...).")
        return 1
    account = service.active_account() or {}
    print("OK: connected using", ".env / environment variables" if kwargs else "the saved account")
    print(f"    instance: {account.get('instance', 'default')}")

    step(2, "Instance usage reported by IBM")
    try:
        for key, value in service.usage().items():
            print(f"    {key}: {value}")
    except Exception as error:  # noqa: BLE001
        print(f"    unavailable ({type(error).__name__}: {error})")

    step(3, "Selecting a backend")
    name = settings.get("IBM_QUANTUM_BACKEND", "").strip()
    try:
        backend = (
            service.backend(name)
            if name
            else service.least_busy(operational=True, simulator=False, min_num_qubits=4)
        )
    except Exception as error:  # noqa: BLE001
        print(f"FAILED: no usable backend ({type(error).__name__}: {error})")
        return 1
    print(f"OK: {backend.name} ({backend.num_qubits} qubits)")
    print("    accessible:", ", ".join(b.name for b in service.backends(simulator=False)))

    step(4, "Dry run of the client-side Sampler (no QPU execution)")
    if args.full_workload:
        from src.evaluation import inputs as benchmark_inputs

        data = benchmark_inputs.load()
        plan = hardware.plan_pairs(len(data.train_angles), len(data.test_angles))
        circuits = hardware.build_circuits(plan, data.train_angles, data.test_angles)
    else:
        x = np.full((2, 4), 0.5)
        circuits = hardware.build_circuits([hardware.PairSpec("calibration", 0, 1)], x, x)
    isa, stats = hardware.transpile_for(backend, circuits)
    print(f"    {len(isa)} circuit(s), {hardware.SHOTS} shots, depth up to {stats['depth']['max']}")
    try:
        job = Sampler(mode=backend).run(isa, shots=hardware.SHOTS, dry_run=True)
        print(f"    submitted dry-run job {job.job_id()}; waiting up to {args.timeout:.0f} s")
        result = job.result(timeout=args.timeout)
    except Exception as error:  # noqa: BLE001
        print(f"    {type(error).__name__}: {error}")
        if "dry_run" in str(error):
            print("NOT AVAILABLE: this account cannot use dry runs (no mock backend).")
            print("    This does not show whether the client-side Sampler works for real jobs.")
            print("    scripts/run_qpu.py uses SamplerV2, the established Open Plan path.")
            return 0
        print("FAILED: the client-side Sampler dry run was rejected or timed out.")
        return 1
    print(f"OK: dry run accepted; {len(result)} mock result(s) returned")
    try:
        print(f"    estimated QPU time: {job.usage_estimation}")
    except Exception as error:  # noqa: BLE001
        print(f"    usage estimate unavailable ({type(error).__name__})")
    print("\nThe client-side Sampler is available to this account. Nothing ran on hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
