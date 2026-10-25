# Vigilantia Architecture

Vigilantia is a phishing-classification research prototype using IBM Quantum hardware, classical baselines, and a Flask service. Controlled exposure evidence and AWS deployment are later phases; see the [roadmap](docs/roadmap.md).

## Implementation status

The MVP components below are implemented and unit-tested: dataset acquisition, the frozen split and features, training, simulated and hardware evaluation, runtime measurement, the external validation, and the Flask service with its Docker image. The benchmark and a 40-row subset of the external validation ran on IBM Quantum hardware, and both are published. Model and service dependencies are pinned. Cloud infrastructure and the controlled exposure module remain proposed.

| Item | Decision |
| --- | --- |
| Primary task | Binary phishing classification |
| Product context | Financial-sector security and investigation workflows |
| Primary dataset | PhiUSIIL Phishing URL (Website) |
| External validation dataset | Web page phishing detection (Hannousse and Yahiouche, 2020), evaluation only |
| Positive modeling class | Phishing |
| Quantum approach | Quantum-kernel SVM / QSVC |
| Initial experiment | Four features, four qubits, 24 training and 12 test examples, seed 42 |
| Quantum hardware | IBM Quantum Open Plan, Qiskit Runtime sampler, job mode |
| API | Python and Flask |
| MVP environment | Local API; IBM Quantum hardware for the benchmark |
| Later deployment target | AWS |
| Later complementary scope | Controlled data-exposure evidence demonstration |

## Overview

![Architecture](docs/architecture-flow.svg)

The offline path prepares a PhiUSIIL snapshot, creates a domain-disjoint 24-training/12-test split with seed 42, and fits a shared four-feature pipeline using training data only. Classical baselines and a quantum-kernel SVM are evaluated under the same fixed protocol. The quantum SVM uses a hybrid design: it is trained on an exactly simulated kernel, and IBM Quantum hardware measures the kernel used to classify the test set. A separate offline step applies the frozen models, unchanged, to a second public dataset, and a 40-row subset of it is classified once from kernels measured on IBM Quantum hardware. The local API consumes classical inference artifacts and recorded results. A later controlled evidence module would expose deterministic observations for analyst review.

Training and benchmark jobs run separately from HTTP requests. The local API uses the classical model for new inputs. Saved QPU predictions are shown as explicitly labeled replay with their sample and source run. Live quantum inference requires additional reference-kernel evaluations and an asynchronous design, which is outside the initial API scope.

The [MVP criteria](README.md#mvp-completion-criteria) require complete hardware classification results, measured comparisons, validated local inference, and reproducible instructions. The [open decisions](docs/roadmap.md#open-decisions) record how each earlier choice was resolved.

## Components

The acquisition entry point `scripts/download_dataset.py` stages and validates the UCI archive, publishes a complete snapshot, and verifies an existing snapshot before reuse. `scripts/download_external.py` does the same for the external dataset. See the [data guide](data/README.md) for observed counts, checksums, and validation limits.

- **Data preparation (`src/data/`)**: implemented for the benchmark. Deduplication, target mapping, registrable-domain grouping, and the frozen domain-disjoint split are produced by `scripts/prepare_benchmark.py`.
- **Features (`src/features/`)**: host-only lexical features are implemented, and four are frozen: `host_entropy`, `host_label_count`, `host_digit_ratio`, and `host_hyphen_count`. Training-only scaling to `[0, pi/2]` is implemented and fitted.
- **Models (`src/models/`)**: majority baseline, logistic regression, RBF SVM, and SVM with a precomputed quantum kernel. The four models and the exact training kernel in `src/quantum/` are implemented and fitted. The training kernel comes from exact simulation; the test kernel comes from the QPU. Four-qubit circuit logic is validated in simulation before hardware acquisition.
- **Hardware acquisition (`src/quantum/hardware.py`, `src/quantum/qpu_run.py`)**: implemented, rehearsed on a fake backend, and run on `ibm_fez`. `scripts/run_qpu.py` submits a pilot or the full run with confirmation, saves job IDs immediately, retrieves without resubmitting, and publishes evaluated results. It uses Qiskit Runtime sampler jobs (SamplerV2 or its client-side replacement), ISA transpilation, job/result mapping, raw counts, and resumable retrieval on the intended Open Plan instance. See [hardware runs](docs/hardware-runs.md).
- **Evaluation (`reports/`, `configs/`, `src/evaluation/`)**: frozen configurations, one domain-disjoint held-out evaluation, predictive metrics, and runtime reporting (`src/evaluation/runtime.py`, `scripts/time_benchmark.py`). No tuning or fold/seed variability estimate in the initial run.
- **External validation (`src/evaluation/external.py`, `scripts/evaluate_external.py`)**: implemented. It verifies the downloaded snapshot offline, applies the frozen preparation in `configs/external-validation.json`, classifies the primary set with the four frozen models (the quantum kernel by exact simulation), adds bootstrap intervals, and publishes aggregates only to `reports/external-validation.json`. See [External validation](docs/external-validation.md).
- **External hardware subset (`src/evaluation/external_qpu.py`, `scripts/run_qpu_external.py`)**: implemented and run once on `ibm_fez`. It draws a stratified 40-row subset of the primary set by the rule frozen in `configs/external-qpu-subset.json`, plans 1,008 circuits, submits them with the same QPU tooling under a budget guard, and evaluates the six frozen rows with bootstrap intervals. It publishes `reports/external-qpu-subset.json` and copies the counts to `reports/qpu-evidence-external/`. `scripts/rehearse_external_qpu.py` rehearses the same plan on a fake backend without an account.
- **Serving (`app/`)**: implemented. The Flask service publishes the validated results and both charts, shows recorded quantum-kernel SVM test predictions as replay with their source runs, shows the external validation and its hardware subset, and serves live classical inference through `POST /v1/phishing/predict` and a page form. Inference lives in `src/inference/predictor.py`, with host validation, feature range reporting, and a model version derived from the artifacts; `scripts/predict_host.py` exposes it on the command line. The Docker image runs Gunicorn as a non-root user; trained models are mounted read-only rather than baked in. Routes are documented in the [service guide](docs/service.md#api).
- **Later evidence checks (`src/exposure/`)**: proposed. Consistency checks and reference-record matches on fictitious samples, with incident replay for human review.
- **Later infrastructure (`infra/`)**: proposed AWS deployment. The results container would run on Amazon ECS Express Mode, which provisions Fargate tasks and an HTTPS load balancer; ECR, S3, CloudWatch, and IAM support it, and DynamoDB is optional. Nothing has been deployed. See the [roadmap](docs/roadmap.md#deployment-strategy).

The `src/exposure/` and `infra/` paths are planned and do not currently exist.

## Technology stack

| Layer | Technology | Status |
| --- | --- | --- |
| Language | Python | Selected |
| API | Flask with Jinja templates; Gunicorn in the container | Implemented locally: results, replay, external validation and hardware subset, charts, and live prediction |
| Quantum ML | Qiskit and Qiskit Aer; fidelity kernel implemented directly | Implemented |
| Quantum hardware | IBM Quantum; qiskit-ibm-runtime sampler | Full benchmark run and the external subset run on `ibm_fez`; counts recorded |
| Classical ML | scikit-learn | Implemented |
| Data processing | pandas, NumPy, tldextract | Implemented |
| Experiments | Jupyter notebooks | Seven reproduction notebooks (01 to 07), executed in order from a fresh environment (see the [reproduction guide](docs/reproducing.md)) |
| Packaging | Docker | Service image builds and runs locally |
| Cloud | AWS | Later deployment target; services proposed in the [roadmap](docs/roadmap.md#deployment-strategy) |
| Demo interface | Server-rendered results page and JSON routes | Implemented locally: published results, replay, external validation, hardware subset, and both charts; a missing file is shown as pending |

Every dependency is pinned in the `requirements*.txt` files and was installed from scratch on Python 3.12; see the [reproduction guide](docs/reproducing.md).

## Pipeline

| Stage | Output | Validation |
| --- | --- | --- |
| Acquire | Immutable PhiUSIIL snapshot and source metadata | Checksum, license, schema, label counts |
| Prepare | Deduplicated rows and domain-group split manifest | No domain overlap across partitions |
| Transform | Versioned feature pipeline | Training-only fitting and target exclusion |
| Train | Classical models and SVM fitted to an exactly simulated quantum kernel | Identical examples and feature inputs |
| Evaluate | Metrics on the test set; quantum SVM classified from QPU-measured kernels | Untouched final test set |
| Validate externally | Aggregate metrics of the frozen models on a second public dataset | Frozen protocol, checksum, no overlap with PhiUSIIL domains |
| Validate on hardware | The frozen models on a 40-row external subset, with QPU-measured test kernels | Subset rule frozen before submission, one run, budget guard |
| Package | Model, preprocessing, reference data, and metadata | Artifact compatibility and inference smoke check |
| Serve | Local classical inference and labeled saved QPU predictions | Input contract, model availability, and replay identification |

## Data and serving contracts

Map the original dataset target to `is_phishing = 1 - label`. Keep equivalent registrable domains together across splits. Choose features and fit scaling to `[0, pi/2]` using training rows only; clip test/inference values and record clipping. Package the model with preprocessing, feature order, reference examples required by the kernel, and version metadata. See the [data guide](data/README.md) and [benchmark protocol](docs/benchmark-protocol.md).

Run training and hardware benchmark jobs outside HTTP requests. The local API uses the classical model for live inputs; recorded QPU test predictions must identify the sample and source run and be labeled replay. New quantum predictions require reference-kernel jobs and an asynchronous workflow outside the initial API scope. Return explicit failures for missing or incompatible artifacts. Describe scores according to their actual semantics; calibration is required before presenting SVM margins as probabilities.

## Quantum circuit and execution

![Four-qubit quantum kernel](docs/quantum-circuit.svg)

Use `zz_feature_map(4, reps=1, entanglement="linear", alpha=2.0)` and the compute-uncompute fidelity measurement with a Qiskit Runtime sampler. Encode four scaled values, apply the inverse map for a second example, and measure the probability of `0000` to estimate their fidelity kernel. Linear connections are 0–1, 1–2, and 2–3. The circuit encodes data; it has no trainable ansatz. See the [circuit specification](docs/quantum-circuit.md) for gate structure and logical counts.

Training is classical: the `24 x 24` training kernel is computed by exact statevector simulation, so it needs no QPU time and no PSD correction. The hardware classifies: the benchmark requests 1,024 shots for each of 288 test-to-training pairs, 24 training calibration pairs, and 24 diagonal controls, for 336 evaluations and 344,064 shots. The calibration pairs measure how hardware noise scales kernel values, and the same SVM is reported on the ideal, raw hardware, and calibrated hardware test kernels. Preserve raw measurements separately from calibrated values. The [protocol](docs/benchmark-protocol.md) defines the exact matrix treatment and acceptance checks.

Use job mode on the confirmed Open Plan instance, check quota and estimated usage before submission, and do not switch to paid execution. Persist job IDs before retrieval and record backend, physical layout, depth, shots, and available queue/execution timings. Missing results, quota exhaustion, or queue delays leave the benchmark incomplete. No jobs are submitted by this documentation.

The MVP accepts a host or URL and computes features from the host alone, rather than fetching URLs. Exposure observations cannot establish breach authenticity, source, or authorship. Replay data and precomputed predictions must be identified in the demonstration.

## Design decisions and tradeoffs

| Decision | Benefit | Tradeoff |
| --- | --- | --- |
| Shared feature contract | Comparable classical and quantum experiments | Small feature sets may discard useful signals |
| Offline training | Predictable API workload | Requires artifact versioning and deployment discipline |
| Host-only feature inputs | Reproducible inference without arbitrary URL fetching | Not yet a general-purpose URL scanner |
| Deferred exposure demonstration | Keeps the first release focused on the phishing experiment | Evidence workflows are unavailable in the MVP |
| Simulator checks, hardware benchmark | Validate circuit logic locally before collecting QPU measurements | Hardware completion depends on access, quota, and queues |
| Classical training, quantum classification | Halves QPU usage and keeps hardware noise out of the training matrix | The model is fitted to ideal similarities and applied to noisy ones, which requires calibration and a simulated reference |
| External validation with frozen models and a simulated kernel | Tests generalization to other data without any tuning or QPU quota | Measures the frozen 24-example models under distribution shift, not the method with more data; hardware noise is measured only on a 40-row subset |

- Shared features and splits make comparisons interpretable, while four-feature reduction can remove useful signal.
- Local simulation checks circuit logic; required hardware measurements expose noise and resource constraints without establishing quantum speedup.
- A single shallow linear feature map and small fixed sample bound the first experiment but limit statistical conclusions.

## Deployment boundaries

The initial API runs locally. IBM Quantum is the hardware execution provider, not the API host. AWS remains a later deployment target with no resources provisioned. Before public serving, implement protected analysis routes, HTTPS, request limits, private artifact storage, least-privilege task roles, and operational logging without sensitive payloads. Record request identifiers, model versions, latency, and outcomes. Define cost limits and teardown instructions when infrastructure is implemented. See the [roadmap](docs/roadmap.md#later-phases).

## Repository structure

Tracked files in the current checkout; the package `__init__.py` files under `src/` and its subpackages are omitted. The Git-ignored `data/raw/`, `data/processed/`, `data/external/`, and `artifacts/` directories are created by the scripts and notebooks.

```text
Vigilantia/
├── .github/
│   ├── rulesets/
│   │   └── protected-branches.json
│   ├── workflows/
│   │   └── ci.yml
│   └── dependabot.yml
├── app/
│   ├── static/
│   │   └── fonts/
│   │       ├── cinzel-latin.woff2
│   │       └── OFL.txt
│   ├── templates/
│   │   └── index.html
│   ├── __init__.py
│   ├── external.py
│   ├── external_qpu.py
│   ├── replay.py
│   └── results.py
├── configs/
│   ├── benchmark-split.json
│   ├── external-qpu-subset.json
│   ├── external-validation.json
│   └── feature-scaler.json
├── data/
│   └── README.md
├── docs/
│   ├── architecture-flow.svg
│   ├── benchmark-protocol.md
│   ├── benchmark-results.svg
│   ├── ci-and-branch-protection.md
│   ├── development.md
│   ├── documentation-checklist.md
│   ├── external-validation.md
│   ├── hardware-runs.md
│   ├── model-comparison.svg
│   ├── quantum-circuit.md
│   ├── quantum-circuit.svg
│   ├── README.md
│   ├── reproducing.md
│   ├── roadmap.md
│   └── service.md
├── notebooks/
│   ├── 01-data-and-split.ipynb
│   ├── 02-features-scaling-and-kernel.ipynb
│   ├── 03-training-and-simulated-evaluation.ipynb
│   ├── 04-ibm-hardware-results-from-recorded-counts.ipynb
│   ├── 05-live-inference-runtime-and-service.ipynb
│   ├── 06-external-validation.ipynb
│   └── 07-external-subset-on-ibm-hardware.ipynb
├── reports/
│   ├── qpu-evidence/
│   │   ├── full-ibm_fez-20261025T094010Z/
│   │   └── pilot-ibm_fez-20261025T092140Z/
│   ├── qpu-evidence-external/
│   │   └── external-ibm_fez-20261025T132451Z/
│   ├── benchmark-replay.json
│   ├── benchmark-results.json
│   ├── external-qpu-subset.json
│   ├── external-validation.json
│   └── README.md
├── scripts/
│   ├── check_ibm_access.py
│   ├── check_repository.py
│   ├── download_dataset.py
│   ├── download_external.py
│   ├── evaluate_benchmark.py
│   ├── evaluate_external.py
│   ├── predict_host.py
│   ├── prepare_benchmark.py
│   ├── publish_replay.py
│   ├── rehearse_external_qpu.py
│   ├── rehearse_hardware.py
│   ├── render_comparison_svg.py
│   ├── render_results_svg.py
│   ├── run_qpu.py
│   ├── run_qpu_external.py
│   ├── scale_benchmark.py
│   ├── time_benchmark.py
│   └── train_benchmark.py
├── src/
│   ├── config/
│   │   ├── env_file.py
│   │   └── ibm.py
│   ├── data/
│   │   └── benchmark_split.py
│   ├── evaluation/
│   │   ├── external.py
│   │   ├── external_qpu.py
│   │   ├── inputs.py
│   │   ├── metrics.py
│   │   ├── publish.py
│   │   ├── replay.py
│   │   └── runtime.py
│   ├── features/
│   │   ├── host_features.py
│   │   └── scaling.py
│   ├── inference/
│   │   └── predictor.py
│   ├── models/
│   │   └── benchmark_models.py
│   └── quantum/
│       ├── hardware.py
│       ├── kernel.py
│       └── qpu_run.py
├── tests/
│   ├── unit/
│   │   ├── test_app_inference_replay.py
│   │   ├── test_benchmark_split.py
│   │   ├── test_catalog.py
│   │   ├── test_check_ibm_access.py
│   │   ├── test_download_dataset.py
│   │   ├── test_download_dataset_cli.py
│   │   ├── test_external_app.py
│   │   ├── test_external_download.py
│   │   ├── test_external_preparation.py
│   │   ├── test_external_qpu.py
│   │   ├── test_external_qpu_app.py
│   │   ├── test_external_qpu_script.py
│   │   ├── test_external_report.py
│   │   ├── test_external_script.py
│   │   ├── test_hardware_and_metrics.py
│   │   ├── test_notebooks.py
│   │   ├── test_page_theme.py
│   │   ├── test_predictor.py
│   │   ├── test_published_docs.py
│   │   ├── test_qpu_evidence.py
│   │   ├── test_qpu_run.py
│   │   ├── test_quantum_kernel.py
│   │   ├── test_render_comparison_svg.py
│   │   ├── test_render_results_svg.py
│   │   ├── test_repository_checks.py
│   │   ├── test_requirements.py
│   │   ├── test_results_app.py
│   │   ├── test_runtime.py
│   │   └── test_scaling.py
│   └── README.md
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── Architecture.md
├── CATALOG.md
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-app.txt
├── requirements-dev.txt
├── requirements-model.txt
└── requirements.txt
```

**Responsibilities.** `src/exposure/` and `infra/` are planned for later phases and do not exist yet.

| Path | Responsibility |
| --- | --- |
| `app/` | Flask application, routes, templates, and validation of the published files |
| `src/data/` | Validation, deduplication, grouping, and splits |
| `src/features/` | Host features and training-fitted scaling |
| `src/models/` | Classical and quantum model adapters |
| `src/quantum/` | Fidelity kernel, hardware acquisition, and QPU run records |
| `src/evaluation/` | Metrics, publishing, replay, runtimes, the external validation, and its hardware subset |
| `src/inference/` | Live classical inference for new hosts |
| `src/exposure/` | Controlled evidence checks (planned) |
| `scripts/` | Command-line entry points for every pipeline step |
| `notebooks/` | Reproduction notebooks: assumptions, replication checks, and interpretation |
| `configs/` | Frozen split, scaler, external-validation protocol, and hardware-subset protocol |
| `reports/` | Published results, replay, external validation and its hardware subset, and QPU evidence |
| `infra/` | AWS deployment definitions (planned) |
| `Dockerfile` | Service image |

## Limitations and next steps

The benchmark is one small split with one seed, so its estimates are unstable. The external validation tests the frozen models on other data but does not change that. Next steps are the later exposure and AWS phases in the [roadmap](docs/roadmap.md). No quantum advantage, production readiness, or live incident monitoring is claimed.

Repository tooling includes unit-tested offline documentation/configuration checks and a Python CI workflow. See [development](docs/development.md) and [CI and branch protection](docs/ci-and-branch-protection.md).

See the [README](README.md) for results and scope, the [documentation catalog](CATALOG.md) for every guide, [CONTRIBUTING.md](CONTRIBUTING.md) for change workflow, and the [documentation checklist](docs/documentation-checklist.md) for pre-commit review.
