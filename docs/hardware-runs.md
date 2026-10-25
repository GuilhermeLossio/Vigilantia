# IBM Quantum Hardware Runs

**Status: the first full run is complete and published.** Run `full-ibm_fez-20261025T094010Z` sent all 336 benchmark circuits to `ibm_fez`. Its raw counts are versioned in `reports/qpu-evidence/`, so the published hardware rows can be rebuilt offline without an IBM account; see the [reproduction guide](reproducing.md) and [notebook 04](../notebooks/04-ibm-hardware-results-from-recorded-counts.ipynb).

Only new hardware runs need an IBM Quantum account. These commands use real QPU quota. The rules they follow (job mode, quota checks, no paid fallback, no automatic resubmission) are defined in the [benchmark protocol](benchmark-protocol.md#runtime-and-resource-boundaries). The external validation runs no hardware jobs.

## Configure IBM Quantum access

[.env.example](../.env.example) documents the settings. Create a local `.env` from it once:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux, use `cp .env.example .env`.

The `.env.example` entries are commented out deliberately. `scripts/check_ibm_access.py` and `scripts/run_qpu.py` read `IBM_QUANTUM_TOKEN`, `IBM_QUANTUM_INSTANCE`, and `IBM_QUANTUM_BACKEND` from `.env` and pass them to the SDK; variables set in the terminal session take precedence, and a saved Qiskit account is used when no token is set. These are not automatic SDK environment variables. Local `.env` variants and project-local provider credential files are ignored by Git; the example remains versionable. Keep real secrets out of the example.

The access check submits only a dry run, which IBM validates without executing on a QPU:

```powershell
.\.venv\Scripts\python.exe scripts/check_ibm_access.py
.\.venv\Scripts\python.exe scripts/check_ibm_access.py --full-workload
```

The second form dry-runs all 336 benchmark circuits and prints IBM's estimate of the QPU time they would use. Dry runs depend on IBM making a mock backend available; for the current Open Plan account they are not, which says nothing about real jobs.

## Run the benchmark on IBM Quantum hardware

Set `IBM_QUANTUM_BACKEND` in `.env` first, for example `ibm_fez`, so every circuit runs on the same device. `submit` checks the instance quota, transpiles for that backend, and asks you to type `SUBMIT` before sending anything:

```powershell
.\.venv\Scripts\python.exe scripts/run_qpu.py submit --pilot 12
.\.venv\Scripts\python.exe scripts/run_qpu.py retrieve <run-id>
.\.venv\Scripts\python.exe scripts/run_qpu.py evaluate <run-id>
```

The pilot sends 12 circuits and `evaluate` reports their QPU time and a projection for the full run. If the projection fits the remaining quota, submit the full run with `submit` and no `--pilot`, then retrieve and evaluate it. `retrieve` can be repeated while jobs are queued; it never resubmits. `evaluate` of a full run calibrates the hardware kernel, classifies the test set, and publishes `reports/benchmark-results.json` with the QPU rows. `list` shows recorded runs. Run records, job IDs, and raw counts are kept in the ignored `artifacts/benchmark/qpu-runs/` directory. Pilot results are never published.

After `evaluate`, add measured runtimes to the published file. This is local and makes no IBM call; hardware rows reuse the execution time IBM recorded for the run:

```powershell
.\.venv\Scripts\python.exe scripts/time_benchmark.py
```

It times fit, test-kernel computation, and prediction for every row, as the median of 25 repeats, and records the machine and package versions. A new `evaluate` rewrites the results file without runtimes, so run it again afterwards, then refresh the replay with `scripts/publish_replay.py` and the chart with `scripts/render_results_svg.py`.

A new run measures new counts, so its kernel values differ slightly from the recorded ones because of shot noise and device drift.

## External subset (40 rows)

**Status: planned; no hardware job for the subset has been submitted and no results exist yet.** A stratified subset of 40 rows (20 legitimate, 20 phishing) of the [external validation](external-validation.md) primary set runs on IBM Quantum hardware under the frozen protocol [configs/external-qpu-subset.json](../configs/external-qpu-subset.json). The protocol fixes the subset rule, the backend (`ibm_fez`), the plan, the six compared rows, and the budget guard; neither it nor the parent protocol changes after results are seen. The commands below need the local external snapshot and PhiUSIIL CSV (see the [data guide](../data/README.md)). `plan` and `evaluate` are local; `submit` and `retrieve` contact IBM, and `submit` uses real QPU quota.

First check the plan without connecting:

```powershell
.\.venv\Scripts\python.exe scripts/run_qpu_external.py plan
```

Expected output:

```text
Subset: 40 of 6441 primary rows (legitimate 20, phishing 20)
Row indexes: [353, 574, 814, ..., 10918, 11068]
Circuits: 1008 (960 test, 24 calibration, 24 diagonal); 4 job(s) of up to 252
Backend ibm_fez, 1024 shots each, sampler v2
Projected QPU time: 267 s execution, 276.2 s billed (0.274 s per circuit)
Budget guard: submit needs more than 336.2 s of remaining instance quota
No IBM connection was made.
```

Optionally rehearse the identical plan on the local FakeFez backend. It needs no account, takes about a minute, and writes only to the ignored `artifacts/external/rehearsal-fake_fez/`, labeled noisy simulation; nothing goes to `reports/`:

```powershell
.\.venv\Scripts\python.exe scripts/rehearse_external_qpu.py
```

Then submit, retrieve, and evaluate. `submit` requires `IBM_QUANTUM_BACKEND=ibm_fez` before it connects, stops if the instance has reached its usage limit, refuses unless the remaining quota exceeds 336.2 s, and asks you to type `SUBMIT`. It sends 4 jobs of up to 252 circuits, the protocol's frozen batch size, and names the run `external-ibm_fez-<timestamp>`. The protocol allows one run: `submit` refuses, before connecting, while any earlier `external-*` run with jobs exists, and names that run so you retrieve it instead:

```powershell
.\.venv\Scripts\python.exe scripts/run_qpu_external.py submit
.\.venv\Scripts\python.exe scripts/run_qpu_external.py retrieve <run-id>
.\.venv\Scripts\python.exe scripts/run_qpu_external.py evaluate <run-id>
```

`submit` prints the submitted run ID and job IDs. `retrieve` prints each job's status and `Run status: retrieved` once every job is done; repeat it while jobs are queued, it never resubmits. `evaluate` builds the hardware kernel, fits a fresh calibration on the run's 24 training pairs, and prints the six rows (three classical models, the quantum-kernel SVM on the exact simulated kernel, and on the uncorrected and calibrated hardware kernels) with 95% bootstrap intervals and `n=40`, local runtimes over the protocol's 25 repeats, the kernel errors against the exact simulation, the calibration fit, the diagonal-control mean, and IBM's recorded usage. It writes `reports/external-qpu-subset.json` and copies `run.json` and the counts to `reports/qpu-evidence-external/<run-id>/` after checking them for credentials, instance identifiers, URLs, and local paths. It refuses to replace a report or evidence of another run. If the fresh calibration fit is invalid, the calibrated hardware row cannot be computed: the report then holds five rows and states this in its limitations as a protocol deviation. It never changes `reports/benchmark-results.json`, and these runs are ignored by the benchmark's pilot projection and results chart.

## Quota and recorded usage

Account checks on 2026-10-25 found an Open Plan instance with 600 seconds of QPU time per rolling 28-day window and access to `ibm_fez`, `ibm_kingston`, and `ibm_marrakesh`. `ibm_fez` matches the fake backend used in the rehearsal.

| Run | Circuits | Execution time | Billed QPU time | Resource units |
| --- | --- | --- | --- | --- |
| Pilot `pilot-ibm_fez-20261025T092140Z` | 12 | 3.2 s | 5 s | 0.082 |
| Full `full-ibm_fez-20261025T094010Z` | 336 | 89.2 s | 92 s | 1.53 of 10 |

The pilot projected about 89 s of execution for the full run, which matched the measurement. The full run's test kernel error against the exact simulation was 0.037 uncorrected and 0.029 calibrated, about three times the fake-backend rehearsal. The [benchmark protocol](benchmark-protocol.md#ibm-quantum-hardware-full-run) records the calibration fit, diagonal controls, and the resulting predictions.

Required resources for a new run are a compatible Python environment, the selected dataset, local simulator resources, and IBM Quantum Open Plan access with sufficient available quota. No IBM credentials are needed for local simulator checks or for reproducing the published results.

## Recorded evidence

`reports/qpu-evidence/` holds the two IBM runs on `ibm_fez`: the 12-circuit pilot and the 336-circuit full run. Each run has:
- `run.json`: the plan, transpilation summary, job IDs, timestamps, and IBM's usage metrics;
- `counts-<job-id>.json`: the raw counts of every circuit.

The files contain no token, instance, or account identifier. To evaluate the recorded run from scripts rather than the notebook, copy `reports/qpu-evidence/<run-id>/` to `artifacts/benchmark/qpu-runs/<run-id>/` and run `scripts/run_qpu.py evaluate <run-id>`, then `scripts/time_benchmark.py` and `scripts/publish_replay.py`. See [reports/README.md](../reports/README.md#qpu-evidence) for how the evidence is checked.
