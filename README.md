<div align="center">

# Vigilantia
### Financial Security · IBM Quantum + Qiskit · Python + FastAPI

[![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qiskit](https://img.shields.io/badge/QML-Qiskit-6929C4?style=flat-square)](https://github.com/Qiskit/qiskit)
[![IBM Quantum](https://img.shields.io/badge/Quantum%20hardware-IBM%20Quantum-6929C4?style=flat-square)](https://quantum.cloud.ibm.com/)
![Status](https://img.shields.io/badge/Status-Design%20Stage-EAB308?style=flat-square)

**A research prototype for phishing detection and data-exposure evidence triage.**

**Dataset:** PhiUSIIL | **Track:** Quantum Machine Learning | **Event:** Qiskit Fall Fest Hackathon

</div>

---

## Why this project

Financial institutions and fintech security teams must prioritize suspicious websites and data-exposure claims using incomplete evidence. A convincing sample may contain fabricated records, modified information, or material recycled from an earlier incident.

Vigilantia explores whether quantum kernels improve phishing classification under a small training budget. The MVP combines a hardware benchmark on IBM Quantum with classical baselines and a local inference API. A later complementary demonstration will organize evidence from controlled exposure scenarios for analyst review.

**Research question:** Can a quantum-kernel classifier improve phishing detection compared with classical models using the same features, data splits, and tuning budget?

The project measures predictive quality and computational cost. Classical models outperforming the quantum model is a valid outcome. Statistical similarity alone cannot establish the authenticity, source, or authorship of a breach.

---

## Table of Contents

| Section | Description |
| --- | --- |
| [About](#about) | Scope and implementation status |
| [MVP Completion Criteria](#mvp-completion-criteria) | Evidence required for the first working release |
| [Open Decisions](#open-decisions) | Unresolved implementation choices |
| [Capabilities](#capabilities) | Product modules and boundaries |
| [Dataset](#dataset) | PhiUSIIL, labels, and data preparation |
| [Architecture](#architecture) | Planned training and serving paths |
| [Technology Stack](#technology-stack) | Application, ML, and cloud components |
| [Pipeline](#pipeline) | Reproducible data and model lifecycle |
| [Quantum Benchmark Strategy](#quantum-benchmark-strategy) | Fair comparisons and resource controls |
| [Quantum Circuit](#quantum-circuit) | Four-qubit encoding and hardware measurement |
| [Evaluation Metrics](#evaluation-metrics) | Quality, cost, and reporting |
| [API](#api) | Proposed FastAPI contract |
| [Repository Structure](#repository-structure) | Planned modules and responsibilities |
| [Getting Started](#getting-started) | Setup readiness and prerequisites |
| [Deployment Strategy](#deployment-strategy) | Target AWS infrastructure |
| [Security and Data Handling](#security-and-data-handling) | Input, access, and evidence boundaries |
| [Roadmap](#roadmap) | Delivery checklist |
| [Limitations](#limitations) | Interpretation boundaries |
| [Hackathon Submission](#hackathon-submission) | Deliverables and deadline |
| [Author](#author) | Project identification |
| [License and References](#license-and-references) | Attribution and supporting resources |
| [Related Docs](#related-docs) | Protocol, data guide, contribution, and review checklist |

---

## About

**Current status: design stage.** Documentation, an unpinned dependency proposal, and a commented configuration example exist. Dataset processing, models, API endpoints, hardware jobs, and cloud resources are not implemented. No benchmark results or working deployment are available yet.

| Item | Decision |
| --- | --- |
| Primary task | Binary phishing classification |
| Product context | Financial-sector security and investigation workflows |
| Primary dataset | PhiUSIIL Phishing URL (Website) |
| Positive modeling class | Phishing |
| Quantum approach | Quantum-kernel SVM / QSVC |
| Initial experiment | Four features, four qubits, 24 training and 12 test examples, seed 42 |
| Quantum hardware | IBM Quantum Open Plan, Qiskit Runtime SamplerV2, job mode |
| API | Python and FastAPI |
| MVP environment | Local API; IBM Quantum hardware for the benchmark |
| Later deployment target | AWS |
| Later complementary scope | Controlled data-exposure evidence demonstration |

## MVP Completion Criteria

- [ ] Acquire and attribute the dataset; record checksums, inspection results, and domain-disjoint splits.
- [ ] Freeze four ordered features, training-fitted preprocessing, and fixed model settings.
- [ ] Obtain complete training and test kernels from IBM Quantum hardware, with job IDs and raw counts.
- [ ] Compare majority, logistic regression, RBF SVM, and quantum-kernel SVM on the same examples; publish measured quality and runtime.
- [ ] Validate local classical inference for new structured inputs, input errors, and missing/incompatible artifacts.
- [ ] Identify any saved QPU predictions exposed in the demo as replay, with sample and source-run metadata.
- [ ] Document tested environment setup and reproducible acquisition, experiment, and API commands.

Simulator diagnostics alone do not complete the hardware criterion. Exposure analysis, incident replay, and AWS deployment belong to later phases. See the [benchmark protocol](docs/benchmark-protocol.md) for acceptance checks.

## Open Decisions

- **Python and dependency versions:** resolve compatibility in an isolated environment, run import/circuit/inference checks, then pin the tested environment before benchmarking.
- **Four numerical features:** inspect the snapshot and feature provenance; use training data only and freeze names/order before test evaluation.
- **Backend and quota:** confirm the accessible Open Plan instance, remaining quota, backend limits, and estimated execution usage before submission; record the selected backend and configuration.
- **Public authentication and AWS limits:** resolve during the later deployment phase, before exposing analysis routes publicly.

These are unresolved decisions, not implemented settings. The initial sample, seed, shots, circuit, and model configurations are defined in the [protocol](docs/benchmark-protocol.md).

## Capabilities

| Module | Planned capability | Boundary |
| --- | --- | --- |
| Phishing Shield | Classify structured website features using classical and quantum models | General phishing benchmark; financial-sector performance requires separate validation |
| Exposure Evidence | Later: inspect consistency and reference-record matches in controlled samples | Deferred from MVP; no proof of real-world breach authenticity |
| Incident Timeline | Later: replay prepared events with source, observation time, evidence, and review status | Deferred from MVP; not live breach discovery |
| Benchmark Reports | Compare predictive quality, variability, and runtime | No claim of quantum advantage before measurement |

The exposure demonstration will cover samples extracted from a fictitious reference database, modified records, older versions, and independently generated records. These are experimental scenarios, not mutually exclusive real-world incident categories. A dedicated exposure classifier is a future extension requiring trustworthy labels.

PhiUSIIL is the only selected training dataset. It does not provide labels for breach authenticity or establish a link between phishing and a subsequent exposure.

---

## Dataset

Dataset metadata below describes the published source. The dataset has not yet been downloaded or inspected in this repository. See the [data guide](data/README.md) for attribution, acquisition, storage, and pending checks.

**Selected source:** [PhiUSIIL Phishing URL (Website), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).

| Property | Published value |
| --- | --- |
| Instances | 235,795 |
| Legitimate examples | 134,850 |
| Phishing examples | 100,945 |
| Features | 54 |
| Format | CSV, approximately 54 MB uncompressed |
| Original labels | `1 = legitimate`, `0 = phishing` |
| Modeling target | `is_phishing = 1 - label` |
| License | CC BY 4.0 |
| Authors | Arvind Prasad and Shalini Chandra |
| Python retrieval option | `ucimlrepo.fetch_ucirepo(id=967)` |

### Data preparation rules

- Record the acquisition date, source version, file checksum, sampling seed, and attribution.
- Drop `FILENAME` from model inputs. Preserve URL and domain fields for deduplication and grouping, rather than treating them as memorized identifiers.
- Keep equivalent registrable domains in the same split, using a documented Public Suffix List rule. Handle missing or invalid domains explicitly.
- Select four reproducible numerical features using training data only. Freeze their names, ordering, training-fitted scaling to `[0, pi]`, and clipping behavior before evaluation and serving.
- Audit similarity, reputation, and derived probability features before use. Their construction must not expose test labels or unavailable inference-time information.
- Fit scaling and feature selection on training rows only. If cross-validation is introduced later, refit transformations inside each training fold.

The dataset contains pre-extracted URL and webpage features. Reproducing the benchmark does not require visiting suspicious URLs. The MVP API will accept structured features; accepting raw URLs later requires a compatible extraction pipeline.

---

## Architecture

The proposed design separates data preparation, offline experiments, and API serving. Both model families share the same feature contract and evaluation protocol.

![Planned architecture](docs/architecture-flow.svg)

See [Architecture.md](Architecture.md) for component responsibilities and implementation boundaries.

Training and benchmark jobs run separately from HTTP requests. The local MVP API will use the classical model for new inputs. Saved QPU predictions can be shown as explicitly labeled replay with their sample and source run. Live quantum inference requires additional reference-kernel evaluations and an asynchronous design, which is outside the initial API scope.

### Architectural decisions

| Decision | Benefit | Tradeoff |
| --- | --- | --- |
| Shared feature contract | Comparable classical and quantum experiments | Small feature sets may discard useful signals |
| Offline training | Predictable API workload | Requires artifact versioning and deployment discipline |
| Structured feature inputs | Reproducible inference without arbitrary URL fetching | Not yet a general-purpose URL scanner |
| Deferred exposure demonstration | Keeps the first release focused on the phishing experiment | Evidence workflows are unavailable in the MVP |
| Simulator checks, hardware benchmark | Validate circuit logic locally before collecting QPU measurements | Hardware completion depends on access, quota, and queues |

## Technology Stack

| Layer | Technology | Status |
| --- | --- | --- |
| Language | Python | Selected |
| API | FastAPI; Pydantic for schemas | FastAPI selected; schema design planned |
| Quantum ML | Qiskit, Qiskit Machine Learning, Qiskit Aer | Selected; implementation pending |
| Quantum hardware | IBM Quantum; qiskit-ibm-runtime SamplerV2 | Open Plan selected; jobs pending |
| Classical ML | scikit-learn | Planned |
| Data processing | pandas, NumPy | Planned |
| Experiments | Jupyter notebooks | Planned |
| Packaging | Docker | Planned |
| Cloud | AWS | Later deployment target; services proposed below |
| Demo interface | FastAPI OpenAPI interface; optional dashboard | Planned |

Dependency versions will be pinned after compatibility checks. Technology badges identify the intended stack, not a validated build.

---

## Pipeline

| Stage | Planned output | Validation |
| --- | --- | --- |
| Acquire | Immutable PhiUSIIL snapshot and source metadata | Checksum, license, schema, label counts |
| Prepare | Deduplicated rows and domain-group split manifest | No domain overlap across partitions |
| Transform | Versioned feature pipeline | Training-only fitting and target exclusion |
| Train | Classical models and SVM fitted to QPU-derived kernels | Identical examples and feature inputs |
| Evaluate | Metrics, plots, configuration, and runtime report | Untouched final test set |
| Package | Model, preprocessing, reference data, and metadata | Artifact compatibility and inference smoke check |
| Serve | Local classical inference and labeled saved QPU predictions | Input contract, model availability, and replay identification |

## Quantum Benchmark Strategy

| Model | Role |
| --- | --- |
| Majority-class classifier | Minimum reference |
| Logistic regression | Linear baseline |
| RBF-kernel SVM | Primary nonlinear baseline |
| Quantum-kernel SVM / QSVC | Experimental classifier |

Start with **four features, four qubits, 24 training examples, 12 test examples, seed 42, and 1,024 shots per circuit**. Seek balanced classes while preserving domain isolation. Use identical examples and transformations for all models.

1. Reserve a domain-disjoint final test set before model selection.
2. Freeze the small split and report its groups and class distribution.
3. Fit preprocessing on training rows only; clip test/inference values using that saved transformation.
4. Use the fixed model settings in the protocol, with no tuning or test-based threshold selection.
5. Collect QPU training and test kernels before final evaluation, retaining raw measurements and all matrix corrections.
6. Report all models and disclose that this first run has one seed and no fold/seed variability estimate.

The training kernel requires pairwise comparisons, so its matrix grows quadratically with the training sample count. Cache only configuration-compatible kernels and record sample counts, circuit depth, qubits, backend, shots when applicable, and execution time.

The initial hardware workload is **276 training pairs + 288 test-to-training pairs + 24 diagonal controls = 588 evaluations**, totaling **602,112 requested shots**. Check quota and estimated execution usage before submission. Use Open Plan job mode; insufficient quota or unfinished jobs leave the benchmark pending. Hardware execution is required, and simulator results cannot replace it. See the [complete protocol](docs/benchmark-protocol.md) for matrix processing, runtime evidence, and failure handling.

## Quantum Circuit

![Planned four-qubit quantum kernel](docs/quantum-circuit.svg)

Use `zz_feature_map(4, reps=1, entanglement="linear", alpha=2.0)`. Encode four scaled values, apply the inverse map for a second example, and measure the probability of `0000` to estimate their fidelity kernel. Linear connections are 0–1, 1–2, and 2–3. The circuit contains no trainable ansatz; the SVM is fitted classically to the hardware-derived kernel. See the [circuit specification](docs/quantum-circuit.md).

## Evaluation Metrics

| Metric | Purpose |
| --- | --- |
| PR-AUC / average precision | Ranking quality for the phishing class; specify the exact definition used |
| Precision and recall | False-alert burden and phishing detection coverage |
| F1 and balanced accuracy | Threshold-dependent classification comparison |
| Confusion matrix | Explicit false positives and false negatives |
| Training and inference runtime | Computational cost and serving feasibility |
| Fold and seed variability | Later extension; unavailable in the initial single-seed run |

**Results: pending.** No performance values or advantage claims are available. Reports must include class prevalence, split definitions, thresholds, feature choices, and runtime conditions. Scores will not be described as probabilities without appropriate calibration and evaluation.

---

## API

**Proposed contract — not implemented.**

| Method | Route | Planned behavior |
| --- | --- | --- |
| `GET` | `/health` | Application liveness |
| `POST` | `/v1/phishing/predict` | MVP: validate structured features and return classical live inference with model version and score semantics |
| `POST` | `/v1/exposure/analyze` | Later phase: consistency and reference-match checks on controlled samples |
| `GET` | `/v1/incidents` | Later phase: demonstration incidents and review states |
| `GET` | `/v1/benchmarks` | Return recorded benchmark results and configuration |

The feature schema will be finalized after inspecting PhiUSIIL. Invalid input and unavailable artifacts must be distinguishable from a classification result. Saved quantum benchmark predictions belong to recorded benchmark results and must identify their sample and run as replay. Training and hardware benchmark jobs will not run inside prediction requests. The later exposure module will return evidence observations, not authentication verdicts.

## Repository Structure

Current checkout (application code has not been created):

```text
Vigilantia/
├── data/
│   └── README.md
├── docs/
│   ├── architecture-flow.svg
│   ├── benchmark-protocol.md
│   ├── documentation-checklist.md
│   ├── quantum-circuit.md
│   └── quantum-circuit.svg
├── .env.example
├── .gitattributes
├── .gitignore
├── Architecture.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

**Planned layout; the implementation files have not been created yet.**

| Path | Responsibility |
| --- | --- |
| `app/` | FastAPI application, routes, schemas, and dependency wiring |
| `src/data/` | Acquisition, validation, deduplication, and splits |
| `src/features/` | Transformations and feature selection |
| `src/models/` | Classical and quantum model adapters |
| `src/exposure/` | Controlled evidence checks |
| `notebooks/` | Assumptions, experiments, and interpretation |
| `configs/` | Experiment configurations and seeds |
| `tests/` | Data, inference, and API contract checks |
| `reports/` | Metrics, plots, and limitations |
| `infra/` | AWS deployment definitions |
| `Dockerfile` | Application image |
| `pyproject.toml` | Dependency and Python configuration |

## Getting Started

The application scaffold and dependency lockfile are pending. There is no verified local startup command yet.

> **Reader notice:** [requirements.txt](requirements.txt) is an initial dependency proposal, not a tested or locked environment. Versions are unpinned and installations may change over time. [.env.example](.env.example) documents proposed settings only; no application reads them yet. Installing dependencies and copying the example does not produce a running service.

To prepare an isolated environment on Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
# Run once, if you do not already have a local .env file:
Copy-Item .env.example .env
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe` and `cp .env.example .env` to copy the example.

The dependency list covers API serving, data preparation, classical baselines, local quantum simulation, IBM Quantum Runtime, notebooks, and plots. Installation guidance: [FastAPI](https://fastapi.tiangolo.com/) and [Qiskit Machine Learning](https://github.com/qiskit-community/qiskit-machine-learning/blob/main/docs/getting_started.rst). Dependencies are proposals; no hardware integration is implemented yet.

The `.env.example` entries are commented out deliberately. No environment-file loader exists. Future code must explicitly read the proposed IBM token, Open Plan instance, and backend settings and pass them to the SDK; these are not automatic SDK environment variables. Confirm names against that implementation before enabling them. Local `.env` variants and project-local provider credential files are ignored by Git; the example remains versionable. Keep real secrets out of the example.

Before the first runnable release, validate the supported Python version and package compatibility, pin dependencies, and document dataset retrieval, API startup, and benchmark commands. Local simulator experiments should be reproducible independently of cloud deployment.

Required resources for the planned workflow are a compatible Python environment, the selected dataset, local simulator resources, and IBM Quantum Open Plan access with sufficient available quota for the hardware benchmark. No IBM credentials are needed for local simulator checks. AWS access and Docker belong to the later deployment phase.

---

## Deployment Strategy

**Later target environment: AWS. Infrastructure has not been provisioned and is outside the local MVP.**

| Service | Proposed responsibility |
| --- | --- |
| Amazon ECR | Docker image registry |
| Amazon ECS with AWS Fargate | API service and separate benchmark tasks |
| Application Load Balancer | HTTPS entry point |
| Amazon S3 | Private datasets, model artifacts, and reports |
| Amazon DynamoDB | Optional persistent incident metadata and review state |
| Amazon CloudWatch | Operational logs and metrics |
| AWS IAM | Least-privilege task permissions |

The initial demo can replay incidents from prepared files without DynamoDB. Service selection and resource sizing remain subject to the available budget and implementation time.

Deployment sequence:

1. Validate the local pipeline and pin dependencies.
2. Build the Docker image and publish it to ECR.
3. Configure private S3 storage and task roles.
4. Deploy the API service and HTTPS entry point.
5. Run benchmarks separately and package selected artifacts with their feature pipeline and metadata.
6. Verify inference, access control, artifact compatibility, and operational logs.
7. Set resource limits and cost alerts, and document teardown after the demonstration.

### Non-functional requirements

| Category | Planned requirement |
| --- | --- |
| Reproducibility | Versioned input snapshot, split manifest, seed, feature schema, and artifacts |
| Reliability | Explicit failures for incompatible or missing artifacts |
| Observability | Request ID, model version, latency, and outcome without sensitive payload logging |
| Security | Private storage, least-privilege IAM, input limits, and protected analysis routes |
| Cost | Bounded experiment sizes and separate training tasks |

## Security and Data Handling

- Use appropriately licensed public data for phishing research and fictitious records for the exposure laboratory.
- Keep credentials, personal records, and real breach samples out of the repository, logs, and demonstration video.
- Validate input size and schema, protect analysis routes before public exposure, and define artifact retention.
- Do not fetch arbitrary user-provided URLs in the MVP.
- Keep operational secrets outside source code and use task roles for AWS access.
- Present correlations as investigation leads. Require human review before actions affecting customers or accounts.

Tests should cover target mapping, domain isolation, training-only preprocessing, feature ordering, invalid input, and missing model artifacts. CI results will be documented once an implementation exists.

---

## Roadmap

- [x] Define the financial-security use case and QML track.
- [x] Select PhiUSIIL and identify its published CC BY 4.0 license.
- [x] Select Python, FastAPI, Qiskit, and IBM Quantum Open Plan; retain AWS as a later target.
- [x] Document the initial four-qubit circuit and 24/12 hardware benchmark protocol.
- [ ] Acquire and inspect data; record attribution and checksums.
- [ ] Freeze the split protocol and initial feature contract.
- [ ] Implement classical baselines, simulator checks, and hardware kernel acquisition.
- [ ] Collect complete QPU kernels and preserve job/result evidence within the verified Open Plan quota.
- [ ] Produce reproducible metrics, plots, and runtime comparisons.
- [ ] Implement and validate the FastAPI inference service.
- [ ] Validate classical live inference and clearly labeled saved quantum predictions.
- [ ] Complete notebooks and verified environment instructions.
- [x] Include the MIT code license.
- [ ] Publish the repository, record the demo, and submit.
- [ ] Later: build the controlled exposure and incident replay demonstration.
- [ ] Later: package and deploy on AWS with authentication and resource limits.

Delivery priority: data inspection, frozen protocol, local circuit checks, QPU acquisition, fair comparison, and local API. Exposure workflows and AWS deployment follow the MVP.

## Limitations

- Historical phishing data does not establish performance against current attacks or banking-specific campaigns.
- Small samples increase uncertainty; feature reduction may remove useful signals.
- Derived dataset attributes may introduce shortcuts or leakage unless their provenance is audited.
- Simulator results do not establish quantum speedup or production readiness.
- PhiUSIIL cannot validate breach authenticity. Controlled exposure scenarios only test the stated laboratory checks.
- Exposure experiments must separate source databases and their variants across splits; recognizing generator artifacts is not proof of authenticity.
- Incident replay does not provide continuous monitoring or establish causality between phishing and exposure.

## Hackathon Submission

The following event details were supplied in the project brief and have not been independently verified. Confirm the deadline, track, and submission requirements with the organizers before submitting.

| Item | Detail |
| --- | --- |
| Event | Qiskit Fall Fest Hackathon |
| Proposed track | Track 2 — Quantum Machine Learning |
| Deadline provided by organizers | October 25, 2026, 16:00 UTC / 13:00 America/Sao_Paulo |
| Required deliverables | Public repository, open-source license, README, environment instructions, explanatory notebooks, and a 1–3 minute demo video |
| Submission | Email team details, track, GitHub link, and demo video to `submission@aiqtlab.com`; confirm receipt in the submissions channel |

## Author

**Guilherme Lossio** — [GitHub](https://github.com/GuilhermeLossio)

Project: **Vigilantia**.

## License and References

**Code license: [MIT](LICENSE).** The repository already includes the MIT license. PhiUSIIL is distributed under CC BY 4.0; retain its attribution. Reused code and data remain subject to their respective licenses.

- [PhiUSIIL — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset), Arvind Prasad and Shalini Chandra.
- [Qiskit Machine Learning — Quantum Kernel Tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html).
- [Qiskit Machine Learning Repository](https://github.com/qiskit-community/qiskit-machine-learning).
- [Xanadu — qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks).


## Related Docs

- [Architecture.md](Architecture.md): planned components, data flow, and implementation gaps.
- [docs/](docs/): version-controlled SVG diagrams.
- [LICENSE](LICENSE): MIT code license.
- [Benchmark protocol](docs/benchmark-protocol.md): fixed comparison, hardware budget, and reporting.
- [Quantum circuit](docs/quantum-circuit.md): encoding, fidelity measurement, and execution design.
- [Data guide](data/README.md): attribution, acquisition, storage, and inspection status.
- [Contributing](CONTRIBUTING.md): branches, PRs, validation, and data handling.
- [Documentation checklist](docs/documentation-checklist.md): review before the initial commit.
