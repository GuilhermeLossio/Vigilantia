# Documentation Catalog

This catalog lists every readable document in Vigilantia. Each entry says who it is for and what it answers. Reading times are estimated at about 200 words per minute.

## Where to start

| If you want to... | Read, in this order | Time |
| --- | --- | --- |
| Understand the project and its results | [README](README.md) → [Benchmark protocol: recorded results](docs/benchmark-protocol.md#recorded-results) → [External validation](docs/external-validation.md) | ~25 min |
| Reproduce every result yourself | [Reproduction guide](docs/reproducing.md) → notebooks [01](notebooks/01-data-and-split.ipynb) to [06](notebooks/06-external-validation.ipynb) | ~45 min, plus about 2 min of execution |
| Understand the quantum part | [Quantum circuit](docs/quantum-circuit.md) → [Benchmark protocol](docs/benchmark-protocol.md) → [Hardware runs](docs/hardware-runs.md) → [notebook 04](notebooks/04-ibm-hardware-results-from-recorded-counts.ipynb) | ~40 min |
| Run the service or the API | [Service](docs/service.md) → [notebook 05](notebooks/05-live-inference-runtime-and-service.ipynb) | ~12 min |
| Contribute code | [Contributing](CONTRIBUTING.md) → [Development](docs/development.md) → [Tests](tests/README.md) → [Architecture](Architecture.md) | ~20 min |

## Documents

### Overview

| Document | For | What it answers | Time |
| --- | --- | --- | --- |
| [README](README.md) | Everyone | What Vigilantia is, the results at a glance, quick start, MVP criteria, and limitations | ~10 min |
| [Architecture](Architecture.md) | Developers, reviewers | Components, technology stack, pipeline, design decisions, and the repository tree | ~12 min |
| [Roadmap](docs/roadmap.md) | Reviewers, maintainers | What is done, resolved decisions, capabilities, and later phases (controlled exposure, AWS) | ~6 min |

### Method and results

| Document | For | What it answers | Time |
| --- | --- | --- | --- |
| [Benchmark protocol](docs/benchmark-protocol.md) | Evaluators, researchers | The role of each model, the split, features, fixed settings, hybrid design, metrics, hardware budget, and every recorded result with runtimes | ~21 min |
| [External validation](docs/external-validation.md) | Evaluators, researchers | How the frozen models perform on a second public dataset: frozen protocol, preparation funnel, metrics with 95% intervals | ~7 min |
| [Quantum circuit](docs/quantum-circuit.md) | Quantum readers | The four-qubit encoding, the fidelity measurement, and how it runs on IBM Quantum | ~4 min |
| [Data guide](data/README.md) | Evaluators, data reviewers | PhiUSIIL and the external dataset: licenses, attribution, acquisition, checksums, and preparation rules | ~8 min |
| [Published results](reports/README.md) | Developers, reviewers | The contracts of the results, replay, and external-validation files, and the recorded QPU evidence | ~6 min |

### Reproduce and run

| Document | For | What it answers | Time |
| --- | --- | --- | --- |
| [Reproduction guide](docs/reproducing.md) | Evaluators | The tested environment, installation, the six notebooks, tests, and the script-only path | ~6 min |
| [Service](docs/service.md) | Developers, demo viewers | API routes, running Flask locally and in Docker, live classification, security, and data handling | ~5 min |
| [Hardware runs](docs/hardware-runs.md) | Anyone with IBM Quantum access | Configuring `.env`, pilot and full QPU runs, quota, and how evidence is recorded | ~4 min |

### Development

| Document | For | What it answers | Time |
| --- | --- | --- | --- |
| [Development](docs/development.md) | Developers | Each pipeline script step by step, local checks, and CI | ~4 min |
| [Contributing](CONTRIBUTING.md) | Contributors | Branches, pull requests, validation, and data handling | ~2 min |
| [Tests](tests/README.md) | Developers | What the unit tests cover and their conventions | ~2 min |
| [CI and branch protection](docs/ci-and-branch-protection.md) | Maintainers | The CI workflow, rulesets, and Dependabot | ~3 min |
| [Documentation checklist](docs/documentation-checklist.md) | Maintainers | What to review before changing documentation | ~3 min |
| [Documentation folder index](docs/README.md) | Anyone browsing `docs/` | A short pointer back to this catalog | ~1 min |

## Notebooks

The notebooks are committed with the outputs of a tested run, so you can read them on GitHub without running anything. Run them in order. The [reproduction guide](docs/reproducing.md) explains how.

| Notebook | What it shows and verifies | Time |
| --- | --- | --- |
| [01 · Data and Split](notebooks/01-data-and-split.ipynb) | Dataset checksums, the PhiUSIIL collection shortcut, and the domain-disjoint split | ~5 min |
| [02 · Features, Scaling, and the Training Kernel](notebooks/02-features-scaling-and-kernel.ipynb) | Host features, the training-fitted scaler, the circuits, and the simulated training kernel | ~6 min |
| [03 · Training and Simulated Evaluation](notebooks/03-training-and-simulated-evaluation.ipynb) | The four fixed models and the test results with the simulated kernel | ~5 min |
| [04 · IBM Quantum Hardware Results from Recorded Counts](notebooks/04-ibm-hardware-results-from-recorded-counts.ipynb) | The `ibm_fez` results rebuilt offline from the recorded counts, with calibration | ~7 min |
| [05 · Live Inference, Runtime, and Service](notebooks/05-live-inference-runtime-and-service.ipynb) | Classifying new hosts, measured runtimes, and every Flask route | ~7 min |
| [06 · External Validation](notebooks/06-external-validation.ipynb) | The frozen models on the second public dataset, checked against the published file | ~5 min |

## Diagrams and data files

| File | What it is |
| --- | --- |
| [docs/benchmark-results.svg](docs/benchmark-results.svg) | Accuracy per model and source, kernel error, and QPU time, generated by `scripts/render_results_svg.py` |
| [docs/architecture-flow.svg](docs/architecture-flow.svg) | Training, evaluation, and serving paths |
| [docs/quantum-circuit.svg](docs/quantum-circuit.svg) | The four-qubit fidelity circuit |
| [reports/benchmark-results.json](reports/benchmark-results.json) | Published benchmark results, including runtimes |
| [reports/external-validation.json](reports/external-validation.json) | Published external-validation results |
| [reports/benchmark-replay.json](reports/benchmark-replay.json) | Recorded quantum-kernel SVM test predictions, shown as replay |
| [reports/qpu-evidence/](reports/qpu-evidence/) | Raw counts and job metadata of the IBM Quantum runs |
| [configs/](configs/) | The frozen split, the scaler, and the external-validation protocol |

`tests/unit/test_catalog.py` checks that every Markdown file and notebook in the repository is listed here.
