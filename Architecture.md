# Vigilantia Architecture

Vigilantia is a planned phishing-classification research prototype using IBM Quantum hardware, classical baselines, and a Flask service. Controlled exposure evidence and AWS deployment are later phases.

## Implementation status

The MVP components below are implemented and unit-tested: dataset acquisition, the frozen split and features, training, simulated and hardware evaluation, runtime measurement, and the Flask service with its Docker image. The first benchmark ran on IBM Quantum hardware and is published. Model and service dependencies are pinned. Cloud infrastructure and the controlled exposure module remain proposed.

## Overview

![Planned architecture](docs/architecture-flow.svg)

The offline path prepares a PhiUSIIL snapshot, creates a domain-disjoint 24-training/12-test split with seed 42, and fits a shared four-feature pipeline using training data only. Classical baselines and a quantum-kernel SVM are evaluated under the same fixed protocol. The quantum SVM uses a hybrid design: it is trained on an exactly simulated kernel, and IBM Quantum hardware measures the kernel used to classify the test set. The planned local API consumes classical inference artifacts and recorded benchmark results. A later controlled evidence module would expose deterministic observations for analyst review.

The [MVP criteria](README.md#mvp-completion-criteria) require complete hardware classification results, measured comparisons, validated local inference, and reproducible instructions. The [open decisions](README.md#open-decisions) record how each earlier choice was resolved.

## Planned components

The acquisition entry point `scripts/download_dataset.py` is implemented: it stages and validates the UCI archive, publishes a complete snapshot, and verifies an existing snapshot before reuse. See the [data guide](data/README.md) for observed counts, checksums, and validation limits. This does not implement the broader data pipeline below.

- **Data preparation (`src/data/`)**: implemented for the benchmark. Deduplication, target mapping, registrable-domain grouping, and the frozen domain-disjoint split are produced by `scripts/prepare_benchmark.py`.
- **Features (`src/features/`)**: host-only lexical features are implemented, and four are frozen: `host_entropy`, `host_label_count`, `host_digit_ratio`, and `host_hyphen_count`. Training-only scaling to `[0, pi/2]` is implemented and fitted; the inference schema remains planned.
- **Models (`src/models/`)**: majority baseline, logistic regression, RBF SVM, and SVM with a precomputed quantum kernel. The four models and the exact training kernel in `src/quantum/` are implemented and fitted. The training kernel comes from exact simulation; the test kernel comes from the QPU. Validate four-qubit circuit logic in simulation before hardware acquisition.
- **Hardware acquisition (`src/quantum/hardware.py`, `src/quantum/qpu_run.py`)**: implemented and rehearsed on a fake backend. `scripts/run_qpu.py` submits a pilot or the full run with confirmation, saves job IDs immediately, retrieves without resubmitting, and publishes evaluated results. Qiskit Runtime sampler jobs (SamplerV2 or its client-side replacement), ISA transpilation, job/result mapping, raw counts, and resumable retrieval on the intended Open Plan instance.
- **Evaluation (`reports/`, `configs/`, `src/evaluation/`)**: frozen configurations, one domain-disjoint held-out evaluation, predictive metrics, and runtime reporting (`src/evaluation/runtime.py`, `scripts/time_benchmark.py`). No tuning or fold/seed variability estimate in the initial run.
- **Serving (`app/`)**: implemented. The Flask service publishes the validated results and chart, shows recorded quantum-kernel SVM test predictions as replay with their source runs, and serves live classical inference through `POST /v1/phishing/predict` and a page form. Inference lives in `src/inference/predictor.py`, with host validation, feature range reporting, and a model version derived from the artifacts; `scripts/predict_host.py` exposes it on the command line. The Docker image runs Gunicorn as a non-root user; trained models are mounted read-only rather than baked in. Proposed routes are documented in the [README](README.md#api).
- **Later evidence checks (`src/exposure/`)**: consistency checks and reference-record matches on fictitious samples, with incident replay for human review.
- **Later infrastructure (`infra/`)**: proposed AWS deployment. The results container would run on Amazon ECS Express Mode, which provisions Fargate tasks and an HTTPS load balancer; ECR, S3, CloudWatch, and IAM support it, and DynamoDB is optional. Nothing has been deployed.

These paths are planned and do not currently exist.

## Data and serving contracts

Map the original dataset target to `is_phishing = 1 - label`. Keep equivalent registrable domains together across splits. Choose features and fit scaling to `[0, pi/2]` using training rows only; clip test/inference values and record clipping. Package the model with preprocessing, feature order, reference examples required by the kernel, and version metadata. See the [data guide](data/README.md) and [benchmark protocol](docs/benchmark-protocol.md).

Run training and hardware benchmark jobs outside HTTP requests. The local API uses the classical model for live inputs; recorded QPU test predictions must identify the sample and source run and be labeled replay. New quantum predictions require reference-kernel jobs and an asynchronous workflow outside the initial API scope. Return explicit failures for missing or incompatible artifacts. Describe scores according to their actual semantics; calibration is required before presenting SVM margins as probabilities.

## Quantum circuit and execution

![Planned four-qubit quantum kernel](docs/quantum-circuit.svg)

Use `zz_feature_map(4, reps=1, entanglement="linear", alpha=2.0)` and the compute-uncompute fidelity measurement with a Qiskit Runtime sampler. The circuit encodes data; it has no trainable ansatz. See the [circuit specification](docs/quantum-circuit.md) for gate structure and logical counts.

Training is classical: the `24 x 24` training kernel is computed by exact statevector simulation, so it needs no QPU time and no PSD correction. The hardware classifies: the benchmark requests 1,024 shots for each of 288 test-to-training pairs, 24 training calibration pairs, and 24 diagonal controls, for 336 evaluations and 344,064 shots. The calibration pairs measure how hardware noise scales kernel values, and the same SVM is reported on the ideal, raw hardware, and calibrated hardware test kernels. Preserve raw measurements separately from calibrated values. The [protocol](docs/benchmark-protocol.md) defines the exact matrix treatment and acceptance checks.

Use job mode on the confirmed Open Plan instance, check quota and estimated usage before submission, and do not switch to paid execution. Persist job IDs before retrieval and record backend, physical layout, depth, shots, and available queue/execution timings. Missing results, quota exhaustion, or queue delays leave the benchmark incomplete. No jobs are submitted by this documentation.

The MVP accepts structured features rather than fetching URLs. Exposure observations cannot establish breach authenticity, source, or authorship. Replay data and precomputed predictions must be identified in the demonstration.

## Design decisions and tradeoffs

- Shared features and splits make comparisons interpretable, while four-feature reduction can remove useful signal.
- Offline training separates experiments from serving, but requires compatible, versioned artifacts.
- Local simulation checks circuit logic; required hardware measurements expose noise and resource constraints without establishing quantum speedup.
- A single shallow linear feature map and small fixed sample bound the first experiment but limit statistical conclusions.
- Deferring the evidence module and AWS deployment keeps the MVP focused on the measured comparison and local API.

## Deployment boundaries

The initial API runs locally. IBM Quantum is the hardware execution provider, not the API host. AWS remains a later deployment target with no resources provisioned. Before public serving, implement protected analysis routes, HTTPS, request limits, private artifact storage, least-privilege task roles, and operational logging without sensitive payloads. Record request identifiers, model versions, latency, and outcomes. Define cost limits and teardown instructions when infrastructure is implemented.

## Current repository structure

```text
Vigilantia/
├── .github/
│   ├── rulesets/
│   │   └── protected-branches.json
│   ├── workflows/
│   │   └── ci.yml
│   └── dependabot.yml
├── app/
│   ├── templates/
│   │   └── index.html
│   ├── __init__.py
│   ├── replay.py
│   └── results.py
├── configs/
│   ├── benchmark-split.json
│   └── feature-scaler.json
├── data/
│   └── README.md
├── docs/
│   ├── architecture-flow.svg
│   ├── benchmark-protocol.md
│   ├── benchmark-results.svg
│   ├── ci-and-branch-protection.md
│   ├── documentation-checklist.md
│   ├── quantum-circuit.md
│   └── quantum-circuit.svg
├── reports/
│   ├── benchmark-replay.json
│   ├── benchmark-results.json
│   └── README.md
├── scripts/
│   ├── check_ibm_access.py
│   ├── check_repository.py
│   ├── download_dataset.py
│   ├── evaluate_benchmark.py
│   ├── predict_host.py
│   ├── prepare_benchmark.py
│   ├── publish_replay.py
│   ├── rehearse_hardware.py
│   ├── render_results_svg.py
│   ├── run_qpu.py
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
│   │   ├── test_check_ibm_access.py
│   │   ├── test_download_dataset.py
│   │   ├── test_download_dataset_cli.py
│   │   ├── test_hardware_and_metrics.py
│   │   ├── test_predictor.py
│   │   ├── test_qpu_run.py
│   │   ├── test_quantum_kernel.py
│   │   ├── test_render_results_svg.py
│   │   ├── test_repository_checks.py
│   │   ├── test_results_app.py
│   │   ├── test_runtime.py
│   │   └── test_scaling.py
│   └── README.md
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── Architecture.md
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

## Limitations and next steps

The benchmark is one small split with one seed, so its estimates are unstable. Next steps are a tested setup guide from a clean environment, then the later exposure and AWS phases. No quantum advantage, production readiness, or live incident monitoring is claimed.

Repository tooling now includes unit-tested offline documentation/configuration checks and a Python CI workflow. These do not implement or validate the proposed application. See [CI and branch protection](docs/ci-and-branch-protection.md) for the workflow and remote activation requirements.

See the [README](README.md) for scope and roadmap, [docs/](docs/) for diagrams and protocol, [CONTRIBUTING.md](CONTRIBUTING.md) for change workflow, and the [documentation checklist](docs/documentation-checklist.md) for pre-commit review.
