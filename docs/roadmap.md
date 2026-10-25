# Roadmap

The MVP is complete: the [completion criteria](../README.md#mvp-completion-criteria) are all met. This page records the delivery checklist, how the open decisions were resolved, the planned capabilities, and the later phases.

## Delivery checklist

- [x] Define the financial-security use case and QML track.
- [x] Select PhiUSIIL and identify its published CC BY 4.0 license.
- [x] Select Python, Flask, Qiskit, and IBM Quantum Open Plan; retain AWS as a later target.
- [x] Document the initial four-qubit circuit and 24/12 hardware benchmark protocol.
- [x] Acquire the original data; record attribution, checksums, structural checks, and class counts.
- [x] Inspect feature provenance, duplicates, and domain quality; exclude collection shortcuts.
- [x] Freeze the domain-disjoint 24/12 split and four host-only features.
- [x] Adopt the hybrid design: classical training on a simulated kernel, quantum classification on hardware.
- [x] Implement the simulated training kernel and fit the four fixed models.
- [x] Classify the test set with the simulated kernel and publish partial results.
- [x] Implement hardware kernel acquisition and rehearse it on a fake backend.
- [x] Collect the complete QPU test and calibration kernels and preserve job/result evidence within the verified Open Plan quota.
- [x] Produce reproducible metrics, plots, and runtime comparisons.
- [x] Implement the Flask results service, its container image, and its unit tests.
- [x] Implement live classical inference for new hosts, with a command-line entry point.
- [x] Implement and validate the Flask inference route.
- [x] Show the benchmark chart in the Flask results page.
- [x] Validate classical live inference and clearly labeled saved quantum predictions.
- [x] Complete notebooks and verified environment instructions.
- [x] Include the MIT code license.
- [x] Validate the frozen models on a second public dataset under a protocol frozen beforehand, and show it in the service.
- [ ] Publish the repository, record the demo, and submit.
- [ ] Later: build the controlled exposure and incident replay demonstration.
- [ ] Later: package and deploy on AWS with authentication and resource limits.

Delivery priority: data inspection, frozen protocol, local circuit checks, QPU acquisition, fair comparison, and local API. Exposure workflows and AWS deployment follow the MVP.

## Open decisions

The initial sample, seed, shots, circuit, and model configurations are defined in the [protocol](benchmark-protocol.md). The decisions left open during design were resolved as follows.

| Decision | Resolution criterion | Resolution |
| --- | --- | --- |
| Python and dependency versions | Resolve compatibility in an isolated environment, run import/circuit/inference checks, then pin the tested environment before benchmarking | **Resolved.** Every dependency is pinned in the `requirements*.txt` files and was installed from scratch on Python 3.12.10; see the [reproduction guide](reproducing.md#tested-environment). CI also runs the unit tests on Python 3.13. |
| Backend and quota | Confirm the accessible Open Plan instance, remaining quota, backend limits, and estimated execution usage before submission; record the selected backend and configuration | **Resolved.** The Open Plan instance had 600 s of QPU time per 28 days; `ibm_fez` was selected, a 12-circuit pilot measured the cost, and the full run billed 92 s. See [hardware runs](hardware-runs.md#quota-and-recorded-usage). |
| Public authentication and AWS limits | Resolve during the later deployment phase, before exposing analysis routes publicly | **Open.** Belongs to the [deployment phase](#deployment-strategy). |

## Capabilities

| Module | Planned capability | Boundary |
| --- | --- | --- |
| Phishing Shield | Classify structured website features using classical and quantum models | General phishing benchmark; financial-sector performance requires separate validation |
| Exposure Evidence | Later: inspect consistency and reference-record matches in controlled samples | Deferred from MVP; no proof of real-world breach authenticity |
| Incident Timeline | Later: replay prepared events with source, observation time, evidence, and review status | Deferred from MVP; not live breach discovery |
| Benchmark Reports | Compare predictive quality, variability, and runtime | No claim of quantum advantage before measurement |

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
