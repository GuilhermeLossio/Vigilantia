# Vigilantia Architecture

Vigilantia is a planned phishing-classification research prototype using IBM Quantum hardware, classical baselines, and a local FastAPI service. Controlled exposure evidence and AWS deployment are later phases.

## Implementation status

This repository currently contains documentation, Git configuration, the MIT license, an initial unpinned `requirements.txt`, and a proposed `.env.example`. No application, dataset, model, dependency lockfile, infrastructure, or benchmark results exist yet. Dependency compatibility has not been validated, and no code consumes the example settings. All components below describe the agreed design, not an existing implementation.

## Overview

![Planned architecture](docs/architecture-flow.svg)

The offline path prepares a PhiUSIIL snapshot, creates a domain-disjoint 24-training/12-test split with seed 42, and fits a shared four-feature pipeline using training data only. Classical baselines and a SVM using IBM Quantum hardware kernels are evaluated under the same fixed protocol. The planned local API consumes classical inference artifacts and recorded benchmark results. A later controlled evidence module would expose deterministic observations for analyst review.

The [MVP criteria](README.md#mvp-completion-criteria) require complete hardware results, measured comparisons, validated local inference, and reproducible instructions. The [open decisions](README.md#open-decisions) include package versions, feature names, backend, and available quota. Documentation exists; these runtime components do not.

## Planned components

- **Data preparation (`src/data/`)**: acquisition metadata, checksums, deduplication, target mapping, and domain-group splits.
- **Features (`src/features/`)**: training-only fitting, feature ordering, transformations, and inference schema. Begin with four numerical features; the actual names remain undecided.
- **Models (`src/models/`)**: majority baseline, logistic regression, RBF SVM, and SVM with a precomputed QPU-derived kernel. Validate four-qubit circuit logic in simulation before hardware acquisition.
- **Hardware acquisition (within `src/models/`)**: Qiskit Runtime SamplerV2 jobs, ISA transpilation, job/result mapping, raw counts, and resumable retrieval on the intended Open Plan instance.
- **Evaluation (`reports/`, `configs/`, `notebooks/`)**: frozen configurations, one domain-disjoint held-out evaluation, predictive metrics, and runtime reporting. No tuning or fold/seed variability estimate in the initial run.
- **Serving (`app/`)**: structured feature validation, classical live inference, artifact loading, model-version reporting, and recorded benchmark results. Proposed routes are documented in the [README](README.md#api).
- **Later evidence checks (`src/exposure/`)**: consistency checks and reference-record matches on fictitious samples, with incident replay for human review.
- **Later infrastructure (`infra/`)**: proposed AWS deployment definitions. ECR, ECS/Fargate, S3, ALB, CloudWatch, and IAM are candidates; DynamoDB is optional.

These paths are planned and do not currently exist.

## Data and serving contracts

Map the original dataset target to `is_phishing = 1 - label`. Keep equivalent registrable domains together across splits. Choose features and fit scaling to `[0, pi]` using training rows only; clip test/inference values and record clipping. Package the model with preprocessing, feature order, reference examples required by the kernel, and version metadata. See the [data guide](data/README.md) and [benchmark protocol](docs/benchmark-protocol.md).

Run training and hardware benchmark jobs outside HTTP requests. The local API uses the classical model for live inputs; recorded QPU test predictions must identify the sample and source run and be labeled replay. New quantum predictions require reference-kernel jobs and an asynchronous workflow outside the initial API scope. Return explicit failures for missing or incompatible artifacts. Describe scores according to their actual semantics; calibration is required before presenting SVM margins as probabilities.

## Quantum circuit and execution

![Planned four-qubit quantum kernel](docs/quantum-circuit.svg)

Use `zz_feature_map(4, reps=1, entanglement="linear", alpha=2.0)` and the compute-uncompute fidelity measurement with Qiskit Runtime SamplerV2. The circuit encodes data; it has no trainable ansatz. See the [circuit specification](docs/quantum-circuit.md) for gate structure and logical counts.

The benchmark requests 1,024 shots for each of 276 training pairs, 288 test-to-training pairs, and 24 diagonal controls: 588 evaluations and 602,112 shots. Preserve raw measurements and document imposed symmetry, unit modeling diagonals, and PSD correction separately from hardware results. The [protocol](docs/benchmark-protocol.md) defines the exact matrix treatment and acceptance checks.

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
├── data/
│   └── README.md
├── docs/
│   ├── architecture-flow.svg
│   ├── benchmark-protocol.md
│   ├── ci-and-branch-protection.md
│   ├── documentation-checklist.md
│   ├── quantum-circuit.md
│   └── quantum-circuit.svg
├── scripts/
│   └── check_repository.py
├── tests/
│   ├── unit/
│   │   └── test_repository_checks.py
│   └── README.md
├── .env.example
├── .gitattributes
├── .gitignore
├── Architecture.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

## Limitations and next steps

The README provides environment preparation commands, but installation compatibility and application startup remain unverified. Acquire and inspect the dataset, finalize the feature contract, implement baselines and grouped evaluation, then implement and validate inference. Pin compatible dependencies and document reproducible commands once they exist. No measured performance, quantum advantage, production readiness, or live incident monitoring is claimed.

Repository tooling now includes unit-tested offline documentation/configuration checks and a Python CI workflow. These do not implement or validate the proposed application. See [CI and branch protection](docs/ci-and-branch-protection.md) for the workflow and remote activation requirements.

See the [README](README.md) for scope and roadmap, [docs/](docs/) for diagrams and protocol, [CONTRIBUTING.md](CONTRIBUTING.md) for change workflow, and the [documentation checklist](docs/documentation-checklist.md) for pre-commit review.
