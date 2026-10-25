# Phishing Benchmark Protocol

**Status: planned, not executed.** This is the initial exploratory benchmark specification. It uses a hybrid design: classical training on a simulated kernel and quantum classification on IBM Quantum hardware. Completion requires the hardware classification results; simulator results are a separate reference.

## Dataset and split

Use the PhiUSIIL snapshot and acquisition record described in [data/README.md](../data/README.md). Record its SHA-256, source, acquisition time, schema, label counts, and exclusions. Define `is_phishing = 1 - label` so phishing is the positive class.

Deduplicate equivalent URLs before sampling. Group rows by registrable domain using a recorded Public Suffix List snapshot and extraction configuration. Group IP hosts by their normalized IP address; exclude rows whose host cannot be resolved from the stored URL/domain fields and record the count. Do not visit dataset URLs. Keep every group wholly within one partition, including conflicting-label duplicates, which must be excluded and counted.

Use seed **42** to build a deterministic domain-disjoint split and select **24 training examples and 12 test examples**. Target 12 legitimate / 12 phishing training examples and 6 / 6 test examples. Preserve domain isolation if exact balance is infeasible; report actual counts, and stop if either partition lacks a class or the requested size cannot be obtained. Freeze row identifiers, group identifiers, ordering, and class counts before fitting. Do not redraw the split after inspecting model results.

This deliberately small, approximately balanced sample supports an exploratory comparison, not an estimate of production prevalence or banking-sector performance.

## Shared features and preprocessing

The four numerical feature names remain pending dataset inspection. Exclude the target and identifiers; audit derived similarity, probability, and reputation features for unavailable inference-time information or leakage. Select features using training data and provenance only, document the rationale, and freeze the ordered names before accessing test outcomes.

All four models receive the same four features, sample ordering, and training-fitted transformation. Fit a MinMax transformation to `[0, pi]` on the 24 training rows only. Clip test and inference values to that interval and record clipping counts. Map a feature constant in training to zero for every partition and record it. Reject missing or non-finite values until an explicit training-only imputation policy is introduced. Save training minima, maxima, constant-feature flags, and feature order with the artifact.

## Fixed model configurations

The first run performs **no hyperparameter search, cross-validation model selection, or threshold tuning**. Freeze package versions and explicit settings before evaluating the held-out test set.

- Majority baseline: `DummyClassifier(strategy="most_frequent")`.
- Logistic regression: L2 regularization, `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, no class weights. Treat a convergence warning as an unresolved run failure.
- Classical SVM: `SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)`, no class weights.
- Quantum-kernel SVM: `SVC(kernel="precomputed", C=1.0, probability=False)`, no class weights, fitted to the simulated training kernel and applied to the test kernels specified below. The SVM optimization runs classically.

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

Use Qiskit Runtime `SamplerV2` in **job mode** on the user's Open Plan instance. Confirm the instance, remaining quota, backend access, and estimated execution usage before submission. Do not select a paid instance or silently reduce the agreed sample or shots. IBM's [instance guidance](https://quantum.cloud.ibm.com/docs/en/guides/instances) describes quota; [execution modes](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes) describe job and session availability. Verify account-specific limits at execution time.

Transpile for the selected backend and record optimization level, transpiler seed, layout, physical qubits, operation counts, and depth. Bind the eight independent pair parameters before transpilation so each circuit can be optimized for its data. Use optimization level 1 and seed 42 for the first run. Diagonal circuits can simplify to an identity and therefore test readout more than full-depth fidelity; label this limitation.

Batch circuit publications within the selected backend's current job limits; record pair-to-job/result-index mappings. Save job IDs immediately and resume retrieval instead of automatically resubmitting. No automatic paid fallback, quota increase, or retry loop is part of this protocol. If access, quota, queue, or execution failures prevent complete matrices, the hardware benchmark remains **pending/incomplete**. Preserve completed work and report the failure explicitly.

## Evidence and reporting

Save dataset/split/feature fingerprints, package and Python versions, model settings, circuit definition, backend name, job IDs, requested and actual shots, counts, kernel matrices, calibration parameters, and artifact checksums. Keep tokens and account credentials out of reports. Record timestamps and distinguish client wall time, queue time, and provider-reported execution usage; unavailable timing fields are reported as unavailable.

Report average precision (the chosen PR summary), accuracy, precision, recall, F1, balanced accuracy, confusion matrix, class counts, and training/inference runtime. Record zero-denominator metrics using `zero_division=0` with a note when applicable. This first run has one split and one seed; no fold/seed variability estimate is available. Twelve test examples yield coarse and uncertain estimates; each test example changes accuracy by about 8.3 percentage points. Publish all four models and all three quantum SVM variants regardless of which performs best.

## Acceptance checks

- Snapshot attribution and checksum exist; target mapping and feature order are verified.
- Train/test group sets do not overlap; both classes appear in both partitions.
- Preprocessing uses training rows only, with clipping and constants handled as specified.
- All 336 hardware evaluations have mapped results with expected shots, finite probabilities in `[0, 1]`, and a complete `12 x 24` hardware test matrix.
- The simulated `24 x 24` training kernel passes the eigenvalue check, and its concentration diagnostic is reported.
- Raw, calibrated, and simulated matrices are distinguishable; model settings and scores are reproducible from saved artifacts.
- The report identifies simulation, QPU execution, and replay separately and makes no quantum-advantage claim from this small experiment.

See [MVP completion criteria](../README.md#mvp-completion-criteria), [architecture](../Architecture.md), and the [documentation checklist](documentation-checklist.md).
