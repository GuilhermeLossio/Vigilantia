<div align="center">

# Vigilantia
### Financial Security · IBM Quantum + Qiskit · Python + Flask

[![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/API-Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Qiskit](https://img.shields.io/badge/QML-Qiskit-6929C4?style=flat-square)](https://github.com/Qiskit/qiskit)
[![IBM Quantum](https://img.shields.io/badge/Quantum%20hardware-IBM%20Quantum-6929C4?style=flat-square)](https://quantum.cloud.ibm.com/)
![Status](https://img.shields.io/badge/Status-Portfolio%20research-22C55E?style=flat-square)

**A research prototype for phishing detection and data-exposure evidence triage.**

**Dataset:** PhiUSIIL | **External validation:** Hannousse & Yahiouche (2020) | **Focus:** Hybrid machine learning research | **Origin:** Qiskit Fall Fest Hackathon

**[Documentation catalog](CATALOG.md)** · [Reproduce the results](docs/reproducing.md) · [Notebooks](notebooks/)

</div>

---

## Why this project

Financial institutions and fintech security teams must prioritize suspicious websites and data-exposure claims using incomplete evidence. A convincing sample may contain fabricated records, modified information, or material recycled from an earlier incident.

Vigilantia explores whether quantum kernels improve phishing classification under a small training budget. The MVP combines a hardware benchmark on IBM Quantum with classical baselines and a local inference API. A later complementary demonstration will organize evidence from controlled exposure scenarios for analyst review.

**Research question:** Can a quantum-kernel classifier improve phishing detection compared with classical models using the same features, data splits, and tuning budget?

The project measures predictive quality and computational cost. Classical models outperforming the quantum model is a valid outcome. Statistical similarity alone cannot establish the authenticity, source, or authorship of a breach.

**Current status: benchmark run complete; external validation and its 40-row hardware subset complete; local service implemented.** The dataset downloaders, the frozen split and features, the four models, the IBM Quantum run tooling, the external validation, and the Flask service with its Docker image are implemented and unit-tested. The full benchmark and a 40-row external subset ran on `ibm_fez`, and their quality, runtime, and evidence are published. Cloud resources are not implemented, and no deployment exists. The hackathon has ended. Active work is a post-hackathon portfolio study of text representations and local Qiskit simulation; see the [research guide](docs/benchmark-v2-text.md) and [roadmap](docs/roadmap.md#delivery-checklist). No hackathon submission is asserted.

---

## Results at a glance

![Benchmark results](docs/benchmark-results.svg)

**Benchmark: 24 training and 12 test examples from PhiUSIIL, seed 42, four host features.** The model rows come from [reports/benchmark-results.json](reports/benchmark-results.json); the hardware measures (billed QPU time, the pilot projection, and the kernel errors) come from the [benchmark protocol's recorded results](docs/benchmark-protocol.md#recorded-results), which also hold the full tables and runtimes.

| Model | Test kernel source | Accuracy | Balanced accuracy | Average precision | TN / FP / FN / TP |
| --- | --- | --- | --- | --- | --- |
| Majority baseline | Classical | 0.500 | 0.500 | n/a | 6 / 0 / 6 / 0 |
| Logistic regression | Classical | 0.833 | 0.833 | 0.976 | 6 / 0 / 2 / 4 |
| RBF SVM | Classical | 0.917 | 0.917 | 0.933 | 6 / 0 / 1 / 5 |
| Quantum-kernel SVM | Exact simulation | 0.500 | 0.500 | 0.526 | 2 / 4 / 2 / 4 |
| Quantum-kernel SVM | `ibm_fez`, uncorrected | 0.500 | 0.500 | 0.538 | 2 / 4 / 2 / 4 |
| Quantum-kernel SVM | `ibm_fez`, calibrated | 0.500 | 0.500 | 0.509 | 2 / 4 / 2 / 4 |

The RBF SVM scored highest, at 11 of 12. The quantum-kernel SVM scored 6 of 12, chance level, with the simulated kernel and with both hardware kernels; hardware noise changed no prediction. The full 336-circuit run on `ibm_fez` billed 92 s of the Open Plan's 600 s (89.2 s of execution, matching the pilot's projection of 89 s). Its test kernel error against the exact simulation was 0.037 uncorrected and 0.029 calibrated. Every model fits and predicts in milliseconds locally. With 12 test examples, each example moves accuracy by about 8.3 points, so every estimate here is unstable.

**External validation: the same frozen models on 6,441 rows of a second public dataset.** No retraining, recalibration, or threshold change; the quantum kernel is an exact simulation, not hardware. Values from [reports/external-validation.json](reports/external-validation.json), with 95% bootstrap intervals; details in [External validation](docs/external-validation.md).

| Model | Accuracy | Balanced accuracy | Average precision |
| --- | --- | --- | --- |
| Majority baseline | 0.539 (0.526–0.551) | 0.500 (0.500–0.500) | n/a |
| Logistic regression | 0.663 (0.651–0.674) | 0.653 (0.642–0.665) | 0.670 (0.653–0.687) |
| RBF SVM | 0.644 (0.632–0.656) | 0.637 (0.626–0.649) | 0.647 (0.629–0.666) |
| Quantum-kernel SVM (simulation) | 0.522 (0.510–0.535) | 0.528 (0.516–0.541) | 0.461 (0.445–0.479) |

The classical models keep a real but much smaller signal outside PhiUSIIL. The quantum-kernel SVM stays near chance: its average precision about equals the 46.1% phishing share of the evaluated rows.

**External subset on IBM Quantum hardware: 40 of those rows (20 per class), run once on `ibm_fez`.** 1,008 circuits in 4 jobs under a protocol frozen before submission. Values from [reports/external-qpu-subset.json](reports/external-qpu-subset.json), with 95% bootstrap intervals; details in [External validation: hardware subset](docs/external-validation.md#hardware-subset-40-rows-on-ibm_fez).

| Model | Test kernel source | Accuracy (n=40) | TN / FP / FN / TP |
| --- | --- | --- | --- |
| Majority baseline | Classical | 0.500 (0.350–0.650) | 20 / 0 / 20 / 0 |
| Logistic regression | Classical | 0.525 (0.375–0.675) | 13 / 7 / 12 / 8 |
| RBF SVM | Classical | 0.525 (0.375–0.675) | 13 / 7 / 12 / 8 |
| Quantum-kernel SVM | Exact simulation | 0.475 (0.325–0.625) | 7 / 13 / 8 / 12 |
| Quantum-kernel SVM | `ibm_fez`, uncorrected | 0.500 (0.350–0.650) | 7 / 13 / 7 / 13 |
| Quantum-kernel SVM | `ibm_fez`, calibrated | 0.525 (0.375–0.675) | 8 / 12 / 7 / 13 |

On 40 rows no model separates from chance, and each row moves accuracy by 2.5 points. Hardware noise flipped 1 prediction (uncorrected) and 2 (calibrated) relative to the simulation, all toward the correct label; that is well within the intervals and is not evidence that hardware helps. The run used 267.6 s of execution and billed 280 s, and its test kernel error against the exact simulation was 0.026 uncorrected and 0.018 calibrated. **No quantum advantage is claimed.**

![Accuracy by model on the benchmark test set, the full external set, and the 40-row hardware subset, and runtime on the benchmark test set and the subset](docs/model-comparison.svg)

The comparison chart draws every model's accuracy on the three evaluation sets, with intervals where published, and the time to classify the benchmark test set and the 40-row subset: milliseconds locally against minutes of QPU execution on `ibm_fez`. The full external set has no published runtime.

---

## Quick start

The [reproduction guide](docs/reproducing.md) is the tested path: a fresh Python 3.12 environment and the seven [notebooks](notebooks/) rebuild every published number, including the IBM hardware rows from the recorded counts. No IBM account is needed, and no quantum job is submitted.

```powershell
git clone https://github.com/GuilhermeLossio/Vigilantia.git
cd Vigilantia
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
foreach ($nb in Get-ChildItem notebooks\0*.ipynb) {
  .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace $nb.FullName
}
```

Then start the service and open `http://127.0.0.1:8000/`:

```powershell
.\.venv\Scripts\python.exe -m flask --app "app:create_app()" run --port 8000
```

The [service guide](docs/service.md) covers Docker and the JSON routes; the [development guide](docs/development.md) runs each pipeline script individually; [hardware runs](docs/hardware-runs.md) explains new IBM Quantum jobs.

---

## How it works

![Architecture](docs/architecture-flow.svg)

- **Data:** a PhiUSIIL snapshot is verified by checksum, deduplicated, grouped by registrable domain, and split into a domain-disjoint 24-training/12-test sample with seed 42. See the [data guide](data/README.md).
- **Features:** four host-only features (`host_entropy`, `host_label_count`, `host_digit_ratio`, `host_hyphen_count`), chosen on training-side data after a shortcut audit and scaled to `[0, pi/2]` on the 24 training rows only.
- **Models:** majority baseline, logistic regression, RBF SVM, and an SVM with a precomputed four-qubit fidelity kernel built on `zz_feature_map`. No tuning; all settings are frozen in the [protocol](docs/benchmark-protocol.md).
- **Hybrid execution:** training is classical, on an exactly simulated `24 x 24` kernel. IBM Quantum hardware measures the `12 x 24` test kernel plus 24 calibration pairs and 24 diagonal controls (336 circuits, 1,024 shots each). See the [circuit specification](docs/quantum-circuit.md).
- **External validation:** the frozen models are applied once to a second public dataset under a protocol frozen beforehand, and a 40-row subset of it runs once on `ibm_fez` (1,008 circuits) under a second frozen protocol. See [External validation](docs/external-validation.md).
- **Serving:** a Flask service shows the validated results, both charts, recorded quantum predictions labeled as replay, the external validation and its hardware subset, and live classical inference for new hosts. The page is never fetched; only the host is used.

[Architecture.md](Architecture.md) describes components, the technology stack, the pipeline, design tradeoffs, and the repository structure.

---

## Documentation

The **[documentation catalog](CATALOG.md)** lists every document and notebook, with reading paths by goal and estimated reading times. The main starting points:

| Goal | Start with |
| --- | --- |
| Understand the method and every recorded result | [Benchmark protocol](docs/benchmark-protocol.md) and [External validation](docs/external-validation.md) |
| Reproduce everything from scratch | [Reproduction guide](docs/reproducing.md) and the [notebooks](notebooks/) |
| Understand the quantum part | [Quantum circuit](docs/quantum-circuit.md) and [Hardware runs](docs/hardware-runs.md) |
| Run the service | [Service](docs/service.md) |
| See what is done and what comes next | [Roadmap](docs/roadmap.md) |
| Contribute | [Contributing](CONTRIBUTING.md) and [Architecture](Architecture.md) |

---

## MVP Completion Criteria

- [x] Acquire and attribute the dataset; record checksums, inspection results, and domain-disjoint splits.
- [x] Freeze four ordered features, training-fitted preprocessing, and fixed model settings.
- [x] Compute the training kernel by exact simulation, and obtain the complete test kernel and calibration pairs from IBM Quantum hardware, with job IDs and raw counts.
- [x] Compare majority, logistic regression, RBF SVM, and quantum-kernel SVM on the same examples; publish measured quality and runtime.
- [x] Validate local classical inference for new structured inputs, input errors, and missing/incompatible artifacts.
- [x] Identify any saved QPU predictions exposed in the demo as replay, with sample and source-run metadata.
- [x] Document tested environment setup and reproducible acquisition, experiment, and API commands.

Simulator diagnostics alone do not complete the hardware criterion. Exposure analysis, incident replay, and AWS deployment belong to later phases; see the [roadmap](docs/roadmap.md). See the [benchmark protocol](docs/benchmark-protocol.md#acceptance-checks) for acceptance checks.

## Limitations

- Historical phishing data does not establish performance against current attacks or banking-specific campaigns.
- Small samples increase uncertainty; feature reduction may remove useful signals.
- The quantum SVM is trained on ideal similarities and classifies with noisy hardware ones. Calibration is a linear approximation, and no model trained on hardware measurements is evaluated.
- PhiUSIIL has collection shortcuts: every legitimate URL is a bare `https://www.` homepage. The benchmark uses only host-based features, but those still reflect how the data was collected.
- The external validation tests the frozen models under distribution shift, with host-only features and an exactly simulated quantum kernel; only its 40-row subset, one run on one backend, uses hardware. It does not evaluate the method with more training data.
- Simulator results do not establish quantum speedup or production readiness.
- PhiUSIIL cannot validate breach authenticity. Controlled exposure scenarios only test the stated laboratory checks.
- Exposure experiments must separate source databases and their variants across splits; recognizing generator artifacts is not proof of authenticity.
- Incident replay does not provide continuous monitoring or establish causality between phishing and exposure.

## Hackathon origin (historical)

The following event details were supplied in the original project brief and were not independently verified. The hackathon has ended; these are historical requirements, not an active submission workflow. The portfolio expansion is not being submitted to that event.

| Item | Detail |
| --- | --- |
| Event | Qiskit Fall Fest Hackathon |
| Proposed track | Track 2 — Quantum Machine Learning |
| Deadline provided by organizers | October 25, 2026, 16:00 UTC / 13:00 America/Sao_Paulo |
| Required deliverables | Public repository, open-source license, README, environment instructions, explanatory notebooks, and a 1–3 minute demo video |
| Submission | Email team details, track, GitHub link, and demo video to `submission@aiqtlab.com`; confirm receipt in the submissions channel |

**Remaining:** make the GitHub repository public, record the 1–3 minute demo video, and submit. The license, README, environment instructions, and notebooks are in place; see the [delivery checklist](docs/roadmap.md#delivery-checklist).

## Author

**Guilherme Lossio** — [GitHub](https://github.com/GuilhermeLossio)

Project: **Vigilantia**.

## License and References

**Code license: [MIT](LICENSE).** Both datasets are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain their attribution. Neither dataset is included in the repository; the scripts download them from their original sources. Reused code and data remain subject to their respective licenses.

- [PhiUSIIL Phishing URL (Website) — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset), Arvind Prasad and Shalini Chandra (2024), CC BY 4.0. [Associated publication](https://doi.org/10.1016/j.cose.2023.103545).
- [Web page phishing detection, version 2 — Mendeley Data](https://doi.org/10.17632/c2gw7fy2j4.2), Abdelhakim Hannousse and Salima Yahiouche (2020), DOI 10.17632/c2gw7fy2j4.2, CC BY 4.0. Associated paper: *Towards benchmark datasets for machine learning based website phishing detection: An experimental study*, Engineering Applications of Artificial Intelligence 104 (2021) 104347. Only the `url` and `status` columns are used, filtered and deduplicated.
- [Qiskit Machine Learning — Quantum Kernel Tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html).
- [Qiskit Machine Learning Repository](https://github.com/qiskit-community/qiskit-machine-learning).
- [Xanadu — qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks).

## Benchmark v2: local improvement experiments

The [v2 protocol and reproduction guide](docs/benchmark-v2.md) describes the isolated CPU-only experiment comparing eight host features with four page-content features (`LineOfCode`, `NoOfSelfRef`, `LargestLineLength`, `NoOfImage`), classical model search, threshold calibration and an exact four-qubit Z/ZZ quantum kernel on the page features. Its research target is accuracy of at least 98%, recall of at least 95% and a false positive rate no greater than 1%. It is evaluated on an internal PhiUSIIL test and is not an operational-readiness criterion.

See the [aggregate v2 report](reports/benchmark-v2.json), [learning curves](docs/benchmark-v2-learning.svg), [operating points](docs/benchmark-v2-operating-points.svg), and [matched classical/quantum comparison](docs/benchmark-v2-kernels.svg). The existing service and original benchmark retain their original behavior and results.

The complete 114-run experiment selected a page-feature RBF SVM (2,000 examples) and a page-feature ZZ quantum kernel (500 examples). Mean internal-test accuracy was **98.8%** (classical) and **98.6%** (quantum), and every seed of both reached 98%. With only 24 training examples, the ZZ kernel already reached **98.5%**. False positives were **1.0–1.15%** per seed, slightly above the 1% point target, so the report's formal conclusions are `no_consistent_improvement` and `target_not_met`. Page features largely reflect how PhiUSIIL was collected and have no external check; host-only models stay near 80% internally. See the [interpretation](docs/benchmark-v2.md#interpretation-and-next-experiment).

**Page kernel on IBM Quantum hardware: 22 internal-test rows (11 per class), run once on `ibm_fez`.** Run `v2page-ibm_fez-20261029T010703Z` measured the 24-example ZZ page model (seed 42) with 576 circuits in 3 jobs. The protocol was frozen before submission and reduced from 40 to 22 rows, before any job was sent, to fit the remaining quota. None of the rows shares a domain with the model's training or calibration data. Each model uses its frozen v2 threshold. Values come from [reports/benchmark-v2-qpu-subset.json](reports/benchmark-v2-qpu-subset.json), with 95% bootstrap intervals for accuracy; details are in [Hardware check of the page kernel](docs/benchmark-v2.md#hardware-check-of-the-page-kernel).

| Model (24 training examples) | Kernel source | Accuracy | 95% interval | Recall | FPR |
|---|---|---|---|---|---|
| Logistic regression | classical | 95.5% (21/22) | 86.4–100% | 90.9% | 0.0% |
| RBF SVM | classical | 95.5% (21/22) | 86.4–100% | 90.9% | 0.0% |
| Random Forest | classical | 100% (22/22) | 100–100% | 100% | 0.0% |
| Quantum-kernel SVM (ZZ) | exact simulation | 100% (22/22) | 100–100% | 100% | 0.0% |
| Quantum-kernel SVM (ZZ) | `ibm_fez`, uncorrected | 100% (22/22) | 100–100% | 100% | 0.0% |
| Quantum-kernel SVM (ZZ) | `ibm_fez`, calibrated | 100% (22/22) | 100–100% | 100% | 0.0% |

The hardware kernel differed from the exact simulation by a mean of 0.073 uncorrected and 0.013 after the linear calibration on the run's own training pairs. Diagonal controls averaged 0.943 (ideal 1). IBM billed 161 seconds. With 22 rows, each row moves accuracy by about 4.5 points, so this shows that hardware noise did not change the model's decisions on these rows. It is not a precise accuracy estimate or a quantum-advantage claim.

Runtime on the same 22 rows. Fit and predict are local medians over 25 runs on one machine. The hardware test-kernel time is IBM's recorded execution of all 576 circuits, including calibration pairs and diagonal controls, without queueing. Both hardware rows reuse the same jobs.

| Model (24 training examples) | Kernel source | Fit | Test kernel | Predict |
|---|---|---|---|---|
| Logistic regression | classical | 2.0 ms | — | 400 µs |
| RBF SVM | classical | 1.1 ms | — | 484 µs |
| Random Forest | classical | 183 ms | — | 10 ms |
| Quantum-kernel SVM (ZZ) | exact simulation | 577 µs | 43 µs | 87 µs |
| Quantum-kernel SVM (ZZ) | `ibm_fez`, uncorrected | 577 µs | 153.1 s | 86 µs |
| Quantum-kernel SVM (ZZ) | `ibm_fez`, calibrated | 577 µs | 153.1 s (same jobs) | 85 µs |

Computing the test kernel takes 43 µs in exact simulation and 153.1 s on `ibm_fez`, about 3.5 million times longer. A four-qubit kernel is cheap to simulate, so the hardware run tests fidelity, not speed.

![Benchmark v2 page kernel on IBM Quantum: accuracy and 95% intervals, and fit, test-kernel and predict runtime per model](docs/benchmark-v2-qpu.svg)

## Why expand the second experiment?

The earlier host-only v2 run, preserved as [reports/benchmark-v2-20261028-093841.json](reports/benchmark-v2-20261028-093841.json), tested more training data, richer numeric host features, parameter search and calibrated thresholds. It exposed limited gains from increasing the sample and poor transfer of false-positive control to external data. The text expansion asks whether hostname character patterns help, how much information four-component SVD discards, and whether a local quantum kernel changes the result. This is a continuation of v2, not a replacement of the historical evidence or a new production model.

See the [rationale and full protocol](docs/benchmark-v2-text.md), [local pipeline diagram](docs/benchmark-v2-text-flow.svg), and [notebook 08](notebooks/08-text-representation-and-local-qiskit.ipynb). The original numerical v2 uses vectorized analytic states; this expansion executes Qiskit Statevector and local Aer circuits directly, without IBM Quantum connections.

<!-- BENCHMARK-V2-TEXT:BEGIN -->
## Benchmark v2 expansion: hostname text and local Qiskit

**Portfolio research, post-hackathon.** Run `portfolio-text-20261028-verified`, generated 2026-10-28 (UTC): **target_not_met**.

The selected classical path is `full_logistic` with 100 training examples. Mean internal continuity accuracy is **78.52%**, recall **55.11%**, and false positive rate **9.86%**. External continuity accuracy is **65.24%**, recall **47.35%**, and false positive rate **19.43%**.

The study separates text representation, four-component reduction, ideal Qiskit kernels and local Aer sampling. These are previously inspected continuity datasets; no quantum advantage or production readiness is claimed. No IBM Quantum connection is used in this phase.

[Protocol and interpretation](docs/benchmark-v2-text.md) | [Run report](reports/benchmark-v2-text/portfolio-text-20261028-verified.json) | [representations](docs/benchmark-v2-text/portfolio-text-20261028-verified-representations.svg) | [external](docs/benchmark-v2-text/portfolio-text-20261028-verified-external.svg) | [measurements](docs/benchmark-v2-text/portfolio-text-20261028-verified-measurements.svg)
<!-- BENCHMARK-V2-TEXT:END -->
