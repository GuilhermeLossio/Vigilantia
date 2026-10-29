# Roadmap

The original MVP is complete. The hackathon has ended; its delivery checklist below is historical, and no submission status is asserted. Active development is the [v2 text and local Qiskit expansion](benchmark-v2-text.md), a reproducible portfolio study with no IBM Quantum connection or cloud execution.

## Delivery checklist

### 1. Scope and design

- [x] Define the financial-security use case and QML track.
- [x] Select Python, Flask, Qiskit, and IBM Quantum Open Plan; retain AWS as a later target.
- [x] Document the four-qubit circuit and the 24/12 hardware benchmark protocol.
- [x] Adopt the hybrid design: classical training on a simulated kernel, quantum classification on hardware.

### 2. Data

- [x] Select PhiUSIIL and identify its published CC BY 4.0 license.
- [x] Acquire the original data; record attribution, checksums, structural checks, and class counts.
- [x] Inspect feature provenance, duplicates, and domain quality; exclude collection shortcuts.
- [x] Freeze the domain-disjoint 24/12 split and four host-only features.

### 3. Modeling and simulation

- [x] Implement the simulated training kernel and fit the four fixed models.
- [x] Classify the test set with the simulated kernel and publish the results.
- [x] Report a training-only leave-one-out diagnostic and the kernel's class separation.

### 4. IBM Quantum hardware

- [x] Implement hardware kernel acquisition and rehearse it on a fake backend.
- [x] Measure the cost with a 12-circuit pilot before the full run.
- [x] Collect the complete QPU test and calibration kernels on `ibm_fez` and preserve job IDs and raw counts within the Open Plan quota.
- [x] Protect the quota: typed confirmation, saved job IDs, retrieval that never resubmits, a budget guard, and a refusal to run a frozen protocol twice.
- [x] Version the raw counts so the hardware results can be rebuilt without an IBM account.

### 5. Service and packaging

- [x] Implement the Flask results service, its container image, and its unit tests.
- [x] Implement live classical inference for new hosts, with a command-line entry point and the Flask route.
- [x] Show the benchmark and comparison charts in the Flask results page.
- [x] Show the recorded quantum predictions as clearly labeled replay.
- [x] Build and smoke-test the Docker image with the trained models mounted read-only.

### 6. Validation beyond the benchmark

- [x] Produce reproducible metrics, plots, and runtime comparisons.
- [x] Validate the frozen models on a second public dataset under a protocol frozen beforehand, and show it in the service.
- [x] Run a 40-row subset of the external validation once on `ibm_fez` under a second frozen protocol, publish it with its evidence and notebook 07, and show it in the service.
- [x] Publish a comparison chart of accuracy and runtime across the benchmark, the external set, and the hardware subset.

### 7. Reproducibility and quality

- [x] Pin every dependency and test the installation from a fresh Python 3.12 environment.
- [x] Complete the seven reproduction notebooks and run them from a clean copy of the repository.
- [x] Run Ruff, the unit tests, and the repository checks in CI on Python 3.12 and 3.13, behind the stable **CI required** check.
- [x] Include the MIT code license and keep the CC BY 4.0 attribution of both datasets.
- [x] Organize the documentation with the [catalog](../CATALOG.md) and a test that requires every document and notebook to be listed.
- [x] Scan the full Git history for credentials, datasets, and local paths before publishing.

### 8. Submission (historical, closed event)

These unchecked items record the original delivery intentions; they are not outstanding requirements for the portfolio expansion and do not imply that a submission happened.

- [ ] Merge `develop` into `main` through a pull request with a passing **CI required** check.
- [ ] Confirm that the branch ruleset enforces **CI required** on `main` and `develop`; see [CI and branch protection](ci-and-branch-protection.md#activate-protection).
- [ ] Make the GitHub repository public (owner action; required by the submission rules), and update its description and topics.
- [ ] Record the 1–3 minute demo video.
- [ ] Submit to the hackathon (team details, track, GitHub link, video).

Delivery priority: data inspection, frozen protocol, local circuit checks, QPU acquisition, fair comparison, and local API. Exposure workflows and AWS deployment follow the MVP.

## Research after the hackathon

### Research

Each item below would run under a new protocol frozen before its results, like the existing ones.

- [ ] Estimate variability: repeat the benchmark over several seeds and domain-disjoint splits, and publish intervals for the benchmark rows.
- [ ] Train on more data: scale the training set beyond 24 examples, for example with a kernel approximation, and measure how the classical and quantum kernels change.
- [ ] Explore other feature maps and kernel settings, chosen on training data only.
- [ ] Measure hardware drift: repeat the recorded subset on `ibm_fez` and on another accessible backend when the quota window allows.
- [ ] Evaluate error mitigation, such as readout correction, against the current linear calibration.
- [ ] Extend the external validation to current phishing feeds with a legitimate reference list, under their licenses.

### Product

- [ ] Build the controlled exposure and incident replay demonstration; see [Later phases](#later-phases).
- [ ] Package and deploy on AWS with authentication and resource limits; see the [deployment strategy](#deployment-strategy).
- [ ] Add request logging with model version and latency, without sensitive payloads.

## Open decisions

The initial sample, seed, shots, circuit, and model configurations are defined in the [protocol](benchmark-protocol.md). The decisions left open during design were resolved as follows.

| Decision | Resolution criterion | Resolution |
| --- | --- | --- |
| Python and dependency versions | Resolve compatibility in an isolated environment, run import/circuit/inference checks, then pin the tested environment before benchmarking | **Resolved.** Every dependency is pinned in the `requirements*.txt` files and was installed from scratch on Python 3.12.10; see the [reproduction guide](reproducing.md#tested-environment). CI also runs the unit tests on Python 3.13. |
| Backend and quota | Confirm the accessible Open Plan instance, remaining quota, backend limits, and estimated execution usage before submission; record the selected backend and configuration | **Resolved.** The Open Plan instance had 600 s of QPU time per 28 days; `ibm_fez` was selected, a 12-circuit pilot measured the cost, the full run billed 92 s, and the external subset run billed 280 s, leaving about 223 s in the window. See [hardware runs](hardware-runs.md#quota-and-recorded-usage). |
| Public authentication and AWS limits | Resolve during the later deployment phase, before exposing analysis routes publicly | **Open.** Belongs to the [deployment phase](#deployment-strategy). |

## Capabilities

| Module | Capability | Boundary |
| --- | --- | --- |
| Phishing Shield | Classify a URL from four features of its host, with classical models live and the quantum-kernel SVM on recorded kernels | General phishing benchmark; financial-sector performance requires separate validation |
| Exposure Evidence | Later: inspect consistency and reference-record matches in controlled samples | Deferred from MVP; no proof of real-world breach authenticity |
| Incident Timeline | Later: replay prepared events with source, observation time, evidence, and review status | Deferred from MVP; not live breach discovery |
| Benchmark Reports | Compare predictive quality, variability, and runtime | Measured on the benchmark, the external set, and the 40-row `ibm_fez` subset; no quantum advantage observed or claimed |

PhiUSIIL is the only training dataset; the [external validation](external-validation.md) dataset is used for evaluation only. Neither provides labels for breach authenticity or establishes a link between phishing and a subsequent exposure.

## Later phases

### Controlled exposure demonstration

The exposure demonstration will cover samples extracted from a fictitious reference database, modified records, older versions, and independently generated records. These are experimental scenarios, not mutually exclusive real-world incident categories. A dedicated exposure classifier is a future extension requiring trustworthy labels. Exposure experiments must separate source databases and their variants across splits, and incident replay does not provide continuous monitoring.

### Deployment strategy

**Later target environment: AWS. Infrastructure has not been provisioned, and no deployment has been run.** The results service is containerized and ready to deploy. The planned path is Amazon ECS Express Mode, which runs the container on Fargate behind a managed HTTPS load balancer with a URL of the form `https://<service>.ecs.<region>.on.aws`. AWS App Runner is not used because it closed to new customers on April 30, 2026; AWS recommends Express Mode as its replacement.

| Service | Proposed responsibility |
| --- | --- |
| Amazon ECR | Docker image registry |
| Amazon ECS Express Mode on AWS Fargate | Results service container; separate benchmark tasks later |
| Application Load Balancer | HTTPS entry point, provisioned and managed by Express Mode |
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

See the [architecture](../Architecture.md#deployment-boundaries) and the [service guide](service.md#security-and-data-handling) for the current boundaries.

## Active portfolio phase: v2 text and local Qiskit

- [x] Recover hostnames against the frozen v2 membership and prepare text/numeric representations.
- [x] Implement train-only TF-IDF, four-component SVD and matched classical baselines.
- [x] Implement direct Qiskit Statevector and seeded local Aer measurement workflows.
- [x] Freeze model and hypothesis selections before evaluation, and isolate run artifacts.
- [x] Add a reproducible notebook, aggregate reports, uncertainty analysis and explicit README publication.
- [ ] Collect a genuinely new temporal evaluation dataset under a new protocol.
- [ ] Design a separately authorized hardware experiment if the local results justify that research question.

The study is complete when its experiment is verifiable, even if the recall/FPR target is not reached. The application keeps its existing served model. No new hackathon submission, IBM quota use or AWS deployment belongs to this phase.
