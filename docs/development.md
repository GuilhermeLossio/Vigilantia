# Development

This guide runs each pipeline script individually and describes the local checks and CI. For the tested end-to-end path, use the [reproduction guide](reproducing.md); the notebooks call the same scripts. Local simulator experiments and the published results are reproducible without cloud deployment or an IBM account.

## Environment

Use a Python 3.12 environment for modeling; Qiskit Aer may not support newer versions yet. This is the same `.venv` as in the [reproduction guide](reproducing.md); `requirements-dev.txt` adds pytest and Ruff to the complete pinned environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Pipeline scripts step by step

### Download the dataset

```powershell
python scripts/download_dataset.py
```

This command uses only the standard library and writes the original ZIP, CSV, and `acquisition.json` to `data/raw/`. It can run before installing runtime dependencies. Repeating it verifies and reuses an intact snapshot without a new download. Partial or modified snapshots fail with an explanation and are preserved. The [data guide](../data/README.md) explains recovery and validation boundaries.

### Build the benchmark sample

```powershell
.\.venv\Scripts\python.exe scripts/prepare_benchmark.py
```

The script checks the CSV checksum, then writes `configs/benchmark-split.json` and the local `data/processed/benchmark-sample.csv`. Then fit the training-only angle scaler:

```powershell
.\.venv\Scripts\python.exe scripts/scale_benchmark.py
```

It writes `configs/feature-scaler.json` and the local `data/processed/benchmark-scaled.csv`. Then build the simulated training kernel and fit the four models; no test data is read:

```powershell
.\.venv\Scripts\python.exe scripts/train_benchmark.py
```

Fitted models, the training kernel, and a summary go to the ignored `artifacts/benchmark/` directory.

### Evaluate and rehearse

Classify the test set with the simulated kernel, and rehearse the hardware path on a local fake backend:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_benchmark.py
.\.venv\Scripts\python.exe scripts/rehearse_hardware.py
```

The first writes the published `reports/benchmark-results.json`, with status partial until QPU results exist; it keeps the published file when that file already holds hardware results. The second uses no IBM account and writes only local diagnostics. See the [benchmark protocol](benchmark-protocol.md) for the rules. Regenerate the results chart and the comparison chart after any new evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/render_results_svg.py
.\.venv\Scripts\python.exe scripts/render_comparison_svg.py
```

The comparison chart, `docs/model-comparison.svg`, reads the benchmark, external-validation, and hardware-subset reports, and marks a report that is not published yet as "report not published".

Hardware runs, runtimes, and the replay file are covered in [hardware runs](hardware-runs.md).

### External validation

```powershell
.\.venv\Scripts\python.exe scripts/download_external.py
.\.venv\Scripts\python.exe scripts/evaluate_external.py
```

`download_external.py` uses only the standard library and writes the verified CSV and `acquisition.json` to `data/external/hannousse-2020/`. `evaluate_external.py` makes no network access: it verifies that snapshot, the PhiUSIIL CSV, and the benchmark inputs, then publishes `reports/external-validation.json`. It needs the trained models in `artifacts/benchmark/`. On a missing or inconsistent input it exits with code 1 and publishes nothing. See [External validation](external-validation.md) for the frozen protocol.

The 40-row hardware subset uses `scripts/run_qpu_external.py`: `plan` and `evaluate` are local, and `submit` and `retrieve` contact IBM Quantum. `scripts/rehearse_external_qpu.py` rehearses the same plan on a local fake backend. See [hardware runs](hardware-runs.md#external-subset-40-rows).

### Classify new hosts

```powershell
.\.venv\Scripts\python.exe scripts/predict_host.py example.com
```

See the [service guide](service.md#classify-new-hosts) for the output and the Flask routes.

## Local checks

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_repository.py
.\.venv\Scripts\python.exe -m pip check
```

The unit tests need no data, network, or IBM account. See [tests/README.md](../tests/README.md) for test conventions and the [documentation checklist](documentation-checklist.md) for documentation changes.

## Development and CI

Development tooling is configured in `pyproject.toml`. GitHub Actions runs Ruff (including the notebooks), Pytest, and offline repository checks on Python 3.12 and 3.13 for PRs and pushes to `develop` and `main`. The unit tests cover data preparation, features, the quantum kernel and hardware tooling with stubbed IBM services, models, evaluation, the external validation, the external hardware subset, the results and comparison charts, the published numbers quoted in the documentation, inference, the Flask service, requirements, recorded QPU evidence, and notebook hygiene. CI does not execute the notebooks, because that needs the dataset downloads; the [reproduction guide](reproducing.md) records their tested run.

Use feature branches targeting `develop`, then a PR from `develop` to `main`. The importable ruleset requires the stable **CI required** check and PRs, and blocks force-push and deletion. **Remote branch protection is not activated by these files.** See [CI and branch protection](ci-and-branch-protection.md) for activation instructions and current limitations, and [CONTRIBUTING.md](../CONTRIBUTING.md) for the change workflow.

## Post-hackathon text study

The separate `scripts/run_benchmark_v2_text.py` CLI uses only local Qiskit Statevector and Aer. Run `prepare`, `classical`, `statevector`, `shots`, and `report` sequentially for an explicit run ID, or use `all`. Only `publish` updates the delimited README block. See the [text-study protocol](benchmark-v2-text.md) for configuration, constraints, artifacts and measured results.
