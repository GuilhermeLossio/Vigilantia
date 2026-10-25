# Reproducing the Benchmark

This guide rebuilds every published Vigilantia result from a fresh checkout: the dataset snapshot, the frozen split, the scaler, the four trained models, the simulated evaluation, the IBM Quantum hardware results, the replay, and the local service. The IBM hardware results are reproduced offline from the raw counts recorded during the run. No IBM Quantum account is needed, and no quantum job is submitted.

## Tested environment

The steps below were run end to end on 2026-10-25, from a copy of the repository without data, artifacts, `.env`, or a virtual environment.

| Item | Value |
| --- | --- |
| Operating system | Windows 10 Pro (10.0.19045), x86-64 |
| Python | 3.12.10 |
| Installation | `pip install -r requirements.txt` into a new virtual environment: 69 s, about 0.7 GB, `pip check` clean |
| Notebooks 01 to 05 | All executed headless in order, about 60 s in total, including the 15 MB dataset download |
| Result | Every replication check passed. The regenerated split, scaler, results, and replay files were byte-identical to the versioned ones. |

Use Python 3.12. The pinned Qiskit Aer release may not provide wheels for newer Python versions. CI also runs the unit tests on Python 3.13 under Ubuntu. The macOS and Linux commands below follow the same steps but were not run end to end.

You need internet access for `pip` and for the dataset download from the UCI Machine Learning Repository.

## 1. Install

Windows (PowerShell):

```powershell
git clone https://github.com/GuilhermeLossio/Vigilantia.git
cd Vigilantia
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

macOS and Linux:

```bash
git clone https://github.com/GuilhermeLossio/Vigilantia.git
cd Vigilantia
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

`requirements.txt` pins every package: the Flask service, data preparation, the classical models, Qiskit with Aer and Runtime, and the notebook tools. `requirements-dev.txt` adds pytest and Ruff.

## 2. Run the notebooks

Open them in JupyterLab and run each from top to bottom, in order:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks
```

Or execute all of them headless. Each notebook stops with an error at the first check that fails:

```powershell
foreach ($nb in Get-ChildItem notebooks\0*.ipynb) {
  .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace $nb.FullName
}
```

```bash
for nb in notebooks/0*.ipynb; do
  .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

| Notebook | What it rebuilds and checks |
| --- | --- |
| [01 · Data and Split](../notebooks/01-data-and-split.ipynb) | Downloads or verifies the PhiUSIIL snapshot against the documented checksums. Audits the collection shortcut that restricts features to the host. Regenerates the domain-disjoint 24/12 split and compares it with `configs/benchmark-split.json`. |
| [02 · Features, Scaling, and the Training Kernel](../notebooks/02-features-scaling-and-kernel.ipynb) | Computes the host features. Regenerates the training-fitted scaler and compares it with `configs/feature-scaler.json`. Draws the circuits, then simulates the 24 x 24 training kernel exactly and checks its diagnostics. |
| [03 · Training and Simulated Evaluation](../notebooks/03-training-and-simulated-evaluation.ipynb) | Fits the four fixed models and classifies the test set with the simulated kernel. Compares every metric and confusion matrix with `reports/benchmark-results.json`. Also reports the training-only leave-one-out diagnostic. |
| [04 · IBM Quantum Hardware Results from Recorded Counts](../notebooks/04-ibm-hardware-results-from-recorded-counts.ipynb) | Rebuilds the `ibm_fez` test kernel from the versioned counts, then calibrates it and classifies the test set. Compares the result with the published hardware rows, the protocol's hardware measures, and the replay file. It can also rerun the local noisy rehearsal. |
| [05 · Live Inference, Runtime, and Service](../notebooks/05-live-inference-runtime-and-service.ipynb) | Classifies new example hosts with the served RBF SVM and measures local runtimes. Calls every Flask route through the test client. |

The committed notebooks keep the outputs of the tested run, so you can compare your outputs with them. Runtimes vary with the machine. Every other checked value must match.

Running the notebooks writes the ignored `data/raw/`, `data/processed/`, and `artifacts/` directories. They also regenerate `configs/benchmark-split.json` and `configs/feature-scaler.json`. With the pinned environment, both come out identical, and `git status` stays clean. With other package or Python versions, the split manifest records a different `environment` block. Notebook 01 then warns you about it, and `git checkout -- configs` restores the versioned file.

## 3. Run the tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/check_repository.py
```

The unit tests need no data, network, or IBM account. They also check that the notebooks never call the IBM Quantum API and that their outputs contain no errors.

## 4. Script-only path

The notebooks call these scripts. You can run them directly, in this order:

```powershell
.\.venv\Scripts\python.exe scripts/download_dataset.py
.\.venv\Scripts\python.exe scripts/prepare_benchmark.py
.\.venv\Scripts\python.exe scripts/scale_benchmark.py
.\.venv\Scripts\python.exe scripts/train_benchmark.py
.\.venv\Scripts\python.exe scripts/evaluate_benchmark.py
.\.venv\Scripts\python.exe scripts/rehearse_hardware.py
.\.venv\Scripts\python.exe scripts/predict_host.py example.com
```

`evaluate_benchmark.py` keeps the published results file when it already holds hardware results. To evaluate the recorded hardware run from scripts, copy `reports/qpu-evidence/<run-id>/` to `artifacts/benchmark/qpu-runs/<run-id>/` first. Then run `scripts/run_qpu.py evaluate <run-id>`, which rewrites the published file. After it, run `scripts/time_benchmark.py` and `scripts/publish_replay.py` so the file gets its runtimes back.

## 5. Run the service

```powershell
$env:FLASK_APP = "app:create_app"
.\.venv\Scripts\python.exe -m flask run
```

Open `http://127.0.0.1:5000/`. To run the service in Docker, with the trained models mounted read-only, see [Run the Flask service](../README.md#run-the-flask-service).

## 6. Optional: a new IBM Quantum hardware run

This step is not needed to reproduce the published results, and it uses real QPU quota. The full run used 92 s of billed QPU time out of the 600 s that the Open Plan allows every 28 days. You need:
- an IBM Quantum Platform account;
- a `.env` file created from [.env.example](../.env.example), holding your token, instance, and backend.

The commands are in [Run the benchmark on IBM Quantum hardware](../README.md#run-the-benchmark-on-ibm-quantum-hardware). A new run measures new counts, so its kernel values differ slightly from the recorded ones because of shot noise and device drift.

## Recorded hardware evidence

`reports/qpu-evidence/` holds two IBM runs on `ibm_fez`: the 12-circuit pilot and the 336-circuit full run. Each run has:
- `run.json`: the plan, transpilation summary, job IDs, timestamps, and IBM's usage metrics;
- the raw counts for every circuit.

The files contain no token, instance, or account identifier.
