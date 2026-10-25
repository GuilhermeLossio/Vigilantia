# Phishing Benchmark Protocol

**Status: planned, not executed.** This is the initial exploratory benchmark specification. Completion requires real IBM Quantum hardware results; simulator results are separate diagnostics.

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
- Quantum-kernel SVM: `SVC(kernel="precomputed", C=1.0, probability=False)`, no class weights, using the QPU matrices specified below. The SVM optimization itself runs classically.

Use each classifier's default decision rule; report continuous decision scores for ranking metrics when available. Do not label SVM margins as probabilities. Larger simulator-only experiments, alternative seeds, and parameter sweeps are later extensions, separately labeled and never substituted for this QPU benchmark.

## Hardware kernel acquisition

Use the [four-qubit circuit](quantum-circuit.md), **1,024 shots per circuit**, and one selected backend for the run:

- 276 unique off-diagonal training pairs: `24 * 23 / 2`.
- 288 test-to-training pairs: `12 * 24`.
- 24 training diagonal controls.
- **588 circuit evaluations and 602,112 requested shots in total**, excluding any separately approved diagnostic or retry jobs.

Measure the training upper triangle and mirror it to the lower triangle; this imposes symmetry by construction rather than measuring both physical directions. Preserve the measured diagonals as controls. Set the modeling diagonal to the ideal self-similarity value 1 and report this as a modeling choice, not a measured result.

Preserve the raw matrix, the mirrored matrix, and the matrix supplied to the SVM separately. For the modeling training matrix, use the symmetric eigendecomposition and clip negative eigenvalues to zero. Record the original minimum eigenvalue, number of clipped eigenvalues, correction norm, and tolerance (`1e-10` for reporting materially negative eigenvalues). Retain the measured rectangular test-to-training matrix without PSD correction; explicitly report that correcting training only is a heuristic under hardware noise and does not guarantee a jointly PSD train/test kernel. Do not conceal the correction as hardware output.

## Runtime and resource boundaries

Use Qiskit Runtime `SamplerV2` in **job mode** on the user's Open Plan instance. Confirm the instance, remaining quota, backend access, and estimated execution usage before submission. Do not select a paid instance or silently reduce the agreed sample or shots. IBM's [instance guidance](https://quantum.cloud.ibm.com/docs/en/guides/instances) describes quota; [execution modes](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes) describe job and session availability. Verify account-specific limits at execution time.

Transpile for the selected backend and record optimization level, transpiler seed, layout, physical qubits, operation counts, and depth. Bind the eight independent pair parameters before transpilation so each circuit can be optimized for its data. Use optimization level 1 and seed 42 for the first run. Diagonal circuits can simplify to an identity and therefore test readout more than full-depth fidelity; label this limitation.

Batch circuit publications within the selected backend's current job limits; record pair-to-job/result-index mappings. Save job IDs immediately and resume retrieval instead of automatically resubmitting. No automatic paid fallback, quota increase, or retry loop is part of this protocol. If access, quota, queue, or execution failures prevent complete matrices, the hardware benchmark remains **pending/incomplete**. Preserve completed work and report the failure explicitly.

## Evidence and reporting

Save dataset/split/feature fingerprints, package and Python versions, model settings, circuit definition, backend name, job IDs, requested and actual shots, counts, kernel matrices, corrections, and artifact checksums. Keep tokens and account credentials out of reports. Record timestamps and distinguish client wall time, queue time, and provider-reported execution usage; unavailable timing fields are reported as unavailable.

Report average precision (the chosen PR summary), precision, recall, F1, balanced accuracy, confusion matrix, class counts, and training/inference runtime. Record zero-denominator metrics using `zero_division=0` with a note when applicable. This first run has one split and one seed; no fold/seed variability estimate is available. Twelve test examples yield coarse and uncertain estimates. Publish all four models regardless of which performs best.

## Acceptance checks

- Snapshot attribution and checksum exist; target mapping and feature order are verified.
- Train/test group sets do not overlap; both classes appear in both partitions.
- Preprocessing uses training rows only, with clipping and constants handled as specified.
- All 588 hardware evaluations have mapped results with expected shots, finite probabilities in `[0, 1]`, and complete `24 x 24` / `12 x 24` matrices.
- Raw and corrected matrices are distinguishable; model settings and scores are reproducible from saved artifacts.
- The report identifies simulation, QPU execution, and replay separately and makes no quantum-advantage claim from this small experiment.

See [MVP completion criteria](../README.md#mvp-completion-criteria), [architecture](../Architecture.md), and the [documentation checklist](documentation-checklist.md).
