# Phishing Benchmark Protocol

**Status: planned, not executed.** This is the initial exploratory benchmark specification. It uses a hybrid design: classical training on a simulated kernel and quantum classification on IBM Quantum hardware. Completion requires the hardware classification results; simulator results are a separate reference.

## Dataset and split

Use the PhiUSIIL snapshot and acquisition record described in [data/README.md](../data/README.md). Record its SHA-256, source, acquisition time, schema, label counts, and exclusions. Define `is_phishing = 1 - label` so phishing is the positive class.

Deduplicate equivalent URLs before sampling. Group rows by registrable domain using a recorded Public Suffix List snapshot and extraction configuration. Group IP hosts by their normalized IP address; exclude rows whose host cannot be resolved from the stored URL/domain fields and record the count. Do not visit dataset URLs. Keep every group wholly within one partition, including conflicting-label duplicates, which must be excluded and counted.

Use seed **42** to build a deterministic domain-disjoint split and select **24 training examples and 12 test examples**. Target 12 legitimate / 12 phishing training examples and 6 / 6 test examples. Preserve domain isolation if exact balance is infeasible; report actual counts, and stop if either partition lacks a class or the requested size cannot be obtained. Freeze row identifiers, group identifiers, ordering, and class counts before fitting. Do not redraw the split after inspecting model results.

**Implemented and frozen.** `scripts/prepare_benchmark.py` builds the sample and writes the versioned manifest `configs/benchmark-split.json`. The manifest stores the CSV checksum, package versions, rules, exclusion counts, the feature-selection record, and each row's index, hashed group, and label. It contains no URLs or hosts. Running the script again reproduces the same manifest. The implemented rules are:

- Exact duplicate URLs are removed, keeping the first occurrence: 425 rows.
- Hosts come from the `Domain` field, lowercased, with a trailing dot, a port, and one leading `www.` removed.
- Groups use the registrable domain from the Public Suffix List snapshot bundled with `tldextract`, with private suffixes included. Each customer subdomain of a hosting suffix such as `web.app` is therefore its own group. A host that is itself a suffix, such as `cloudflare-ipfs.com`, is its own group. IP hosts are grouped by address.
- Rows without a usable host are excluded: 2 rows. Groups with both labels are excluded: 106 groups, 606 rows.
- The test rows are drawn first, one row from each of 6 + 6 distinct groups. Features are then selected on the remaining pool, which excludes every test group. The training rows are drawn last from that pool, one row from each of 12 + 12 distinct groups. Exact balance was obtained.

This deliberately small, approximately balanced sample supports an exploratory comparison, not an estimate of production prevalence or banking-sector performance.

## Shared features and preprocessing

### Dataset shortcut audit

PhiUSIIL contains collection shortcuts that separate the classes without describing phishing:

| Pattern in the source data | Legitimate | Phishing |
| --- | --- | --- |
| URL has a path or query | 0% | 60% |
| Host starts with `www.` | 100% | 41% |
| `IsHTTPS` equals 1 | 100% | 49% |
| `URLSimilarityIndex` equals 100 | 100% | 0.8% |

Every legitimate URL is a bare `https://www.<host>` homepage. Features built from the path, query, scheme, or `www.` prefix would learn this collection process. Page features, such as `LineOfCode` or `HasTitle`, also require fetching the page, which the MVP does not do. The provided derived features `URLSimilarityIndex`, `TLDLegitimateProb`, and `URLCharProb` depend on reference statistics whose construction is not reproducible here. All of these are excluded.

### Host-only candidate features

Eight candidates are computed by `src/features/host_features.py` from the host alone, after removing a leading `www.`: host length, digit ratio, hyphen count, label count, Shannon entropy of its characters, longest label length, TLD length, and whether the host is an IP address. The same function can compute them for a new host at inference time.

### Selection rule and frozen features

Selection uses only the training-side pool, which excludes every test group. A class-balanced sample of 20,000 pool rows, drawn with seed 42, ranks the candidates by mutual information with the target. Candidates are taken in that order, skipping any with an absolute Spearman correlation of 0.7 or more to an already selected feature. The rule stops at four.

The frozen ordered features are:

1. `host_entropy`
2. `host_label_count`
3. `host_digit_ratio`
4. `host_hyphen_count`

`host_length` ranked third but was skipped as redundant with `host_entropy`. The manifest records every score. These host features still reflect how this dataset was collected, for example phishing pages on free hosting subdomains. They are not evidence of performance on current attacks.

### Preprocessing

All four models receive the same four features, sample ordering, and training-fitted transformation. Fit a MinMax transformation to `[0, pi/2]` on the 24 training rows only. The upper bound is pi/2, not pi: the circuit applies a phase of `2 * x`, and with `x = pi` that phase is a full turn, identical to `x = 0`. Capping at pi/2 keeps the smallest and largest training values distinguishable on every qubit. This change was made before any model was trained or evaluated. Clip test and inference values to that interval and record clipping counts. Map a feature constant in training to zero for every partition and record it. Reject missing or non-finite values until an explicit training-only imputation policy is introduced. Save training minima, maxima, constant-feature flags, and feature order with the artifact.

**Implemented.** `src/features/scaling.py` holds the scaler, and `scripts/scale_benchmark.py` fits it on the 24 training rows after checking the sample against the split manifest. The fitted parameters and test clipping counts are versioned in `configs/feature-scaler.json`. No feature is constant in training, and no training value is clipped.

| Feature | Training minimum | Training maximum | Test values clipped |
| --- | --- | --- | --- |
| `host_entropy` | 2.3219 | 4.5739 | 1 |
| `host_label_count` | 2 | 3 | 2 |
| `host_digit_ratio` | 0 | 0.1558 | 1 |
| `host_hyphen_count` | 0 | 6 | 0 |

`host_label_count` takes only two values in training, so it becomes a binary angle of 0 or pi/2. Test hosts with more labels are clipped to pi/2.

## Fixed model configurations

The first run performs **no hyperparameter search, cross-validation model selection, or threshold tuning**. Freeze package versions and explicit settings before evaluating the held-out test set.

- Majority baseline: `DummyClassifier(strategy="most_frequent")`.
- Logistic regression: L2 regularization, `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, no class weights. Treat a convergence warning as an unresolved run failure.
- Classical SVM: `SVC(kernel="rbf", C=1.0, gamma="scale")`, no class weights. The default `probability=False` is kept; scikit-learn 1.9 deprecates passing it explicitly.
- Quantum-kernel SVM: `SVC(kernel="precomputed", C=1.0)`, no class weights, fitted to the simulated training kernel and applied to the test kernels specified below. The SVM optimization runs classically.

Use each classifier's default decision rule; report continuous decision scores for ranking metrics when available. Do not label SVM margins as probabilities. Larger simulator-only experiments, alternative seeds, and parameter sweeps are later extensions, separately labeled and never substituted for this QPU benchmark.

## Hybrid execution design

Training is classical, and quantum hardware is used only for classification. The SVM is fitted to a training kernel computed by exact statevector simulation of the [four-qubit circuit](quantum-circuit.md). IBM Quantum hardware measures the kernel between each test example and each training example. The fitted SVM then classifies the test set from those measurements.

Rationale:

- The number of training circuits grows quadratically with the training sample count, so a hardware training kernel consumes most of the QPU budget.
- Hardware noise in a measured training matrix can break the positive semidefinite property the SVM requires, which forces a heuristic correction before fitting.
- An exact training kernel isolates the hardware effect in the classification step, where it can be compared with an ideal reference on the same examples.

Tradeoff: the model is fitted to noise-free similarities and applied to noisy ones. Hardware noise tends to shrink kernel values while the fitted SVM bias stays fixed, so decisions can shift. The protocol measures this mismatch with a simulated reference and corrects it with a training-only calibration, both defined below. This design does not evaluate a model trained on hardware measurements.

## Kernel computation

**Training kernel.** Compute all `24 x 24` entries by exact statevector overlap, using the same circuit and pinned package versions. The matrix is symmetric with a unit diagonal by construction. Verify that its minimum eigenvalue is at least `-1e-10`; a lower value indicates an implementation error and stops the run. No PSD correction is applied. Report the mean and standard deviation of the off-diagonal entries as a concentration diagnostic: values clustered near one constant mean the kernel cannot separate examples. This diagnostic uses training data only and does not change the frozen settings of the first run.

*Implemented.* `src/quantum/kernel.py` computes the kernel and `scripts/train_benchmark.py` fits the four models without reading test data. The 24 x 24 kernel passed every check:

| Check | Value |
| --- | --- |
| Minimum eigenvalue | -7.3e-16, within tolerance |
| Largest asymmetry | 4.4e-16 |
| Largest diagonal error | 1.8e-15 |
| Off-diagonal mean and standard deviation | 0.221 and 0.309 |
| Off-diagonal range | 0.000 to 1.000 |

The values are spread out, so the kernel is not concentrated. Off-diagonal values of exactly 1 come from 4 pairs of training examples with identical feature vectors, all with matching labels; the four host features are coarse and often zero. The training classes are tied at 12 and 12, so the majority baseline resolves the tie to the first class, legitimate. Fitted artifacts are stored locally under the ignored `artifacts/benchmark/` directory.

**Ideal test kernel.** Compute the `12 x 24` test-to-training matrix by the same exact simulation. It is the simulation reference for classification, not a hardware result.

**Hardware workload.** Use **1,024 shots per circuit** and one selected backend for the run:

- 288 test-to-training pairs: `12 * 24`.
- 24 calibration pairs: off-diagonal training pairs drawn uniformly with seed 42 from the 276 available, frozen before submission.
- 24 training diagonal controls.
- **336 circuit evaluations and 344,064 requested shots in total**, excluding any separately approved diagnostic or retry jobs.

**Calibration.** Fit `measured = a * ideal + c` by ordinary least squares on the 24 calibration pairs. Map the hardware test kernel with `calibrated = clip((measured - c) / a, 0, 1)`. Record `a`, `c`, the residual error, and clipping counts. If `a <= 0`, the calibration has failed: report it and omit the calibrated variant. Calibration uses training pairs only; test labels and test kernel values never influence `a` or `c`. Diagonal circuits simplify toward an identity and mainly measure readout error, so they are reported as controls and not used for calibration.

Preserve the raw counts, the hardware test kernel, the calibrated test kernel, and the ideal test kernel separately. Report the mean absolute difference between the hardware and ideal test kernels, before and after calibration. Do not present calibrated values as raw hardware output.

**Quantum SVM variants.** The same fitted model is applied to three test kernels:

- **Ideal:** the simulated test kernel. This is the simulation reference.
- **QPU:** the uncorrected hardware test kernel.
- **QPU calibrated:** the calibrated hardware test kernel.

Report all three. No variant is selected using test results. Only the two QPU variants are hardware results.

## Runtime and resource boundaries

Use the Qiskit Runtime sampler in **job mode** on the user's Open Plan instance. qiskit-ibm-runtime 0.50 deprecated `SamplerV2` in favor of a client-side `Sampler` (`qiskit_ibm_runtime.executor_sampler`). `src/quantum/hardware.py` supports both and records which one ran. Confirm at submission time which one the Open Plan account accepts; the local rehearsal uses `SamplerV2` because the client-side sampler took about 600 seconds locally for the 336 circuits, against about 12 seconds. Confirm the instance, remaining quota, backend access, and estimated execution usage before submission. Do not select a paid instance or silently reduce the agreed sample or shots. IBM's [instance guidance](https://quantum.cloud.ibm.com/docs/en/guides/instances) describes quota; [execution modes](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes) describe job and session availability. Verify account-specific limits at execution time.

Transpile for the selected backend and record optimization level, transpiler seed, layout, physical qubits, operation counts, and depth. Bind the eight independent pair parameters before transpilation so each circuit can be optimized for its data. Use optimization level 1 and seed 42 for the first run. Diagonal circuits can simplify to an identity and therefore test readout more than full-depth fidelity; label this limitation.

*Implemented in `scripts/run_qpu.py` and `src/quantum/qpu_run.py`.* The run uses SamplerV2 in job mode on the backend named by `IBM_QUANTUM_BACKEND`. `submit` checks the instance quota and requires typing `SUBMIT`; a `--pilot N` run sends only the first N circuits and projects the full run's QPU time from IBM's reported usage. Every job ID is saved to the run record as soon as the job exists, before the next batch is sent. `retrieve` fetches jobs by ID and never resubmits; `evaluate` builds the hardware kernel, calibrates it on training pairs, and publishes the QPU rows. Pilot results are never published.

Account checks on 2026-10-25 found an Open Plan instance with 600 seconds of QPU time per rolling 28-day window, none used, and access to `ibm_fez`, `ibm_kingston`, and `ibm_marrakesh`. Server-side dry runs were not available to the account, so no QPU-time estimate exists before the pilot. `ibm_fez` matches the fake backend used in the rehearsal.

The pilot run `pilot-ibm_fez-20261025T092140Z` sent the first 12 test pairs to `ibm_fez` in one job. IBM reports usage as billed seconds (`qpu_charge_time_seconds`), resource units, and circuit execution time; the run script records all three.

| Measure | Pilot, 12 circuits | Linear projection, 336 circuits |
| --- | --- | --- |
| Circuit execution time | 3.2 s | about 89 s |
| Billed QPU time | 5 s | up to 140 s; the pilot's billing is rounded per job |
| Resource units | 0.082 | about 2.3 of the 10 available |

The pilot's mean absolute kernel error against the exact kernel was 0.034, about 2.6 times the 0.013 seen on the fake backend. Real hardware noise is therefore larger than the rehearsal suggested. Pilot results are not published.

Batch circuit publications within the selected backend's current job limits; record pair-to-job/result-index mappings. Save job IDs immediately and resume retrieval instead of automatically resubmitting. No automatic paid fallback, quota increase, or retry loop is part of this protocol. If access, quota, queue, or execution failures prevent complete matrices, the hardware benchmark remains **pending/incomplete**. Preserve completed work and report the failure explicitly.

## Evidence and reporting

Save dataset/split/feature fingerprints, package and Python versions, model settings, circuit definition, backend name, job IDs, requested and actual shots, counts, kernel matrices, calibration parameters, and artifact checksums. Keep tokens and account credentials out of reports. Record timestamps and distinguish client wall time, queue time, and provider-reported execution usage; unavailable timing fields are reported as unavailable.

Report average precision (the chosen PR summary), accuracy, precision, recall, F1, balanced accuracy, confusion matrix, class counts, and training/inference runtime. Record zero-denominator metrics using `zero_division=0` with a note when applicable. This first run has one split and one seed; no fold/seed variability estimate is available. Twelve test examples yield coarse and uncertain estimates; each test example changes accuracy by about 8.3 percentage points. Publish all four models and all three quantum SVM variants regardless of which performs best.

## Recorded results

![Benchmark results](benchmark-results.svg)

The chart is generated by `scripts/render_results_svg.py` from the published results, the rehearsal summary, and the latest pilot and full QPU runs. Gray marks classical models, blue marks simulation, orange marks IBM Quantum hardware, and hatched bars are projections.

### Simulation: test classification

`scripts/evaluate_benchmark.py` classified the 12 test examples once, with the frozen models and settings. The quantum-kernel SVM used the exact simulated test kernel. These are the first and only test evaluations; nothing was changed after them.

| Model | Source | Accuracy | Balanced accuracy | Precision | Recall | F1 | Average precision | TN / FP / FN / TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | Classical | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | n/a | 6 / 0 / 6 / 0 |
| Logistic regression | Classical | 0.833 | 0.833 | 1.000 | 0.667 | 0.800 | 0.976 | 6 / 0 / 2 / 4 |
| RBF SVM | Classical | 0.917 | 0.917 | 1.000 | 0.833 | 0.909 | 0.933 | 6 / 0 / 1 / 5 |
| Quantum-kernel SVM | Exact simulation | 0.500 | 0.500 | 0.500 | 0.667 | 0.571 | 0.526 | 2 / 4 / 2 / 4 |

The quantum-kernel SVM performed at chance level on this split. A training-only diagnostic, run after the test evaluation and changing nothing, helps interpret it:

- Leave-one-out accuracy on the 24 training examples is 0.54 for the quantum-kernel SVM and 0.50 for the RBF SVM. Neither is reliably above chance on training data alone, so the RBF SVM's 0.917 on 12 test examples is an unstable estimate.
- The quantum kernel separates the classes weakly: mean similarity is 0.245 within a class and 0.199 across classes. The quantum-kernel SVM fits 83% of its training examples but does not generalize.

These results describe one small split with coarse features. They are not evidence about quantum kernels in general.

### Noisy rehearsal on a fake backend

`scripts/rehearse_hardware.py` ran the full hardware workload on `FakeFez`, a local fake backend with the noise model and connectivity of a 156-qubit IBM Heron device. No account or QPU was used. These are diagnostics labeled noisy simulation and are not published as hardware results.

| Check | Value |
| --- | --- |
| Circuits and shots | 336 circuits, 1,024 shots each |
| Transpiled depth / two-qubit gates | 1 to 54 / 0 to 10 per circuit |
| Diagonal controls | mean 0.982, ideal 1 |
| Calibration fit | slope 0.960, intercept 0.007, RMSE 0.009 |
| Mean absolute kernel error vs ideal | 0.013 raw, 0.009 calibrated |
| Quantum-kernel SVM on noisy kernels | same predictions as the ideal kernel |

The rehearsal confirms that the hardware code path works end to end. The fake backend's noise did not change any prediction; real hardware noise can be larger.

### IBM Quantum hardware: full run

Run `full-ibm_fez-20261025T094010Z` sent all 336 circuits to `ibm_fez` in one SamplerV2 job, `dbessv7pc3tc739mbc6g`, with 1,024 shots each; every circuit returned all its shots. The same trained quantum-kernel SVM then classified the 12 test examples from the measured kernels.

| Model and test kernel | Accuracy | Balanced accuracy | Average precision | TN / FP / FN / TP |
| --- | --- | --- | --- | --- |
| Quantum-kernel SVM, exact simulation | 0.500 | 0.500 | 0.526 | 2 / 4 / 2 / 4 |
| Quantum-kernel SVM, `ibm_fez` uncorrected | 0.500 | 0.500 | 0.538 | 2 / 4 / 2 / 4 |
| Quantum-kernel SVM, `ibm_fez` calibrated | 0.500 | 0.500 | 0.509 | 2 / 4 / 2 / 4 |

| Hardware measure | Value |
| --- | --- |
| Billed QPU time | 92 s; the pilot projected 89 s of execution time, and 89.2 s were measured |
| Resource units | 1.53 of 10 |
| Diagonal controls | mean 0.924, ideal 1 |
| Calibration fit on 24 training pairs | slope 0.820, intercept 0.018, RMSE 0.026 |
| Test kernel error vs exact simulation | 0.037 uncorrected, 0.029 calibrated |
| Calibrated test values clipped to [0, 1] | 79 of 288 |
| Correlation of SVM decision scores with the simulation | 0.992 uncorrected, 0.985 calibrated |

Hardware noise shrank the kernel values, as the calibration slope shows, and calibration reduced the kernel error by about a fifth. Many ideal test values are near zero (37% are below 0.02), so mapping noisy values back below the measured intercept clips them to 0; that explains most of the 79 clipped values. The noise did not change any prediction, and the decision scores stayed almost perfectly correlated with the simulation. The quantum-kernel SVM's chance-level result therefore comes from the kernel and features on this split, not from the hardware. No quantum advantage is claimed.

## Acceptance checks

- Snapshot attribution and checksum exist; target mapping and feature order are verified.
- Train/test group sets do not overlap; both classes appear in both partitions.
- Preprocessing uses training rows only, with clipping and constants handled as specified.
- All 336 hardware evaluations have mapped results with expected shots, finite probabilities in `[0, 1]`, and a complete `12 x 24` hardware test matrix.
- The simulated `24 x 24` training kernel passes the eigenvalue check, and its concentration diagnostic is reported.
- Raw, calibrated, and simulated matrices are distinguishable; model settings and scores are reproducible from saved artifacts.
- The report identifies simulation, QPU execution, and replay separately and makes no quantum-advantage claim from this small experiment.

See [MVP completion criteria](../README.md#mvp-completion-criteria), [architecture](../Architecture.md), and the [documentation checklist](documentation-checklist.md).
