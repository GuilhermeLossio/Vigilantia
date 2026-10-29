# Benchmark v2: local phishing detection experiments

Benchmark v2 evaluates whether page-content features let a four-qubit quantum kernel reach high phishing-detection accuracy, alongside host-only features, classical model search and a calibrated decision threshold. The research target is, on the internal test and for every fitted seed of a selection, accuracy of at least 98%, recall of at least 95% and a false positive rate (FPR) no greater than 1%. This is an exploratory target, not a deployment approval criterion.

The v1 hardware benchmark, its datasets, published results, scripts and serving model retain their original contracts. V2 does not update the Flask application or submit quantum jobs. An earlier host-only v2 run is preserved as [reports/benchmark-v2-20261028-093841.json](../reports/benchmark-v2-20261028-093841.json) because the [text expansion](benchmark-v2-text.md) uses it as its reference; it was produced by the previous v2 code and configuration.

## Reproduce

Use Python 3.12 and the existing pinned dependencies. A working local environment can be reused; otherwise create one:

```powershell
py -3.12 -m venv .venv-qml
.\.venv-qml\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-qml\Scripts\python.exe -m pip check
.\.venv-qml\Scripts\python.exe scripts/download_dataset.py
.\.venv-qml\Scripts\python.exe scripts/download_external.py
.\.venv-qml\Scripts\python.exe scripts/run_benchmark_v2.py prepare
.\.venv-qml\Scripts\python.exe scripts/run_benchmark_v2.py classical
.\.venv-qml\Scripts\python.exe scripts/run_benchmark_v2.py quantum
.\.venv-qml\Scripts\python.exe scripts/run_benchmark_v2.py report
```

`all` executes the four stages in order. `--config <path>` selects an explicit JSON configuration; the default is [configs/benchmark-v2.json](../configs/benchmark-v2.json). The two download commands are the only steps above that may need the network. The v2 stages are CPU-only and require neither IBM credentials nor AWS resources.

Preparation verifies both CSV SHA-256 hashes against the original dataset protocols. Missing, unreadable or modified snapshots stop execution with exit code 1. Restore access to the local files or use the existing dataset downloaders; never bypass verification or edit the expected hashes to accept a different dataset.

For Windows environments with an inaccessible default temporary directory, run checks using a fresh project-local temporary directory:

```powershell
.\.venv-qml\Scripts\python.exe -m pytest --basetemp artifacts/pytest-v2 -o cache_dir=artifacts/pytest-cache-v2
.\.venv-qml\Scripts\python.exe -m ruff check .
.\.venv-qml\Scripts\python.exe -m ruff format --check .
.\.venv-qml\Scripts\python.exe scripts/check_repository.py
```

## Frozen data and selection protocol

Preparation uses the existing host normalization, private-public-suffix grouping, duplicate removal and conflicting-label exclusions. It keeps the first row in original file order for each domain group. The twelve historical test groups are removed **before all v2 splits**, which is stricter than excluding them only from development.

The remaining domains are stratified into 60% development, 20% threshold calibration and 20% internal test with split seed 42. The first full preparation produced 118,555 development, 39,518 calibration and 39,519 test examples. External filtering retains the same 6,441 domain-disjoint continuity examples as the original validation.

Balanced training samples contain 24, 100, 500 or 2,000 examples, using seeds 42, 43 and 44. Smaller samples are prefixes of larger samples within each class and seed. Training-group membership is saved in local run records before fitting; partition CSV hashes and membership hashes are recorded before training starts. The test and calibration distributions retain their natural class proportions after deduplication; only training samples are balanced.

Each classical model receives one of two feature sets:

- `host`: all eight implemented hostname features, computed without fetching the page.
- `page`: four PhiUSIIL page-content columns, `LineOfCode`, `NoOfSelfRef`, `LargestLineLength` and `NoOfImage`, transformed with `log1p` because they are heavy-tailed counts.

The page candidates are the 27 PhiUSIIL columns describing the fetched page. URL-derived columns are excluded because they encode the collection process (scheme, path, `www.` prefix) or, like `URLSimilarityIndex`, effectively the label. The four page features were first identified in an exploratory analysis that also looked at the internal test. To remove that dependence, preparation ranks all 27 candidates by permutation importance of a gradient-boosting model fitted and evaluated **only on development data** (75/25 inner split, seed 42) and records the ranking in `partitions.json`. The published-report test requires its top four to equal the configured set.

The external dataset has no comparable page columns, so its prepared table contains only host features and page-feature models are never evaluated there.

Logistic regression and RBF SVM use a `StandardScaler` inside their fitted pipelines, without clipping. Random Forest receives original values and uses 300 trees. Three-fold stratified validation selects hyperparameters by mean average precision (AP), breaking exact ties by measured fit time. No fold sees a transform fitted on its validation examples.

The fixed searches are six C values for logistic regression; three C values crossed with two gamma values for RBF SVM; and three depth values crossed with two leaf-size values for Random Forest, as recorded in the configuration. The majority baseline predicts the most frequent training class; balanced training ties select class 0, legitimate. Its constant-score AP equals the positive prevalence and is explicitly reported as a constant-ranking reference, unlike v1's null AP convention.

For every non-baseline fitted model, calibration chooses a score threshold maximizing recall at FPR ≤ 1%; ties prefer lower FPR and then a higher threshold. Scores equal to the threshold are positive. A JSON `null` threshold means predict no positives, making the all-negative operating point representable without non-standard infinite JSON numbers.

Classical configuration selection maximizes mean calibration recall across training seeds, then mean calibration AP, then prefers smaller training samples. Remaining exact ties use model and feature-set names in lexical order. Quantum configuration selection applies the same rule separately. A third selection applies it to classical `host` runs only; it is the configuration evaluated on the external dataset. All selections and the hashes of all completed training records are written to `frozen-selection.json` **before the report opens either evaluation partition**. No test result changes a model, threshold or selection.

## Quantum simulation

Z without entanglement and linearly entangled ZZ maps use four qubits, one repetition and Qiskit's alpha of 2. They encode the four `page` features, one per qubit, and share the exact same training examples as the matched classical `page` runs, at sizes 24, 100 and 500.

Training-fold min/max scaling of the `log1p` values maps features into `[0, angle_max]` for angle maxima π/8, π/4 and π/2; out-of-range inputs are clipped and counted. Constant training features encode zero, and shifts in such features are recorded separately. Each map searches these three scales crossed with C values 0.1, 1, 10 and 100: twelve configurations per map, twenty-four across both maps. Each classical family searches six configurations; these are different search budgets, explicitly reported rather than presented as equal-budget comparisons.

For these one-repetition circuits, Hadamard gates create uniform amplitudes, individual phase gates encode `2*x[j]`, and each neighboring ZZ pair contributes `2*(π-x[j])*(π-x[j+1])` on the pair's bit parity. The implementation computes these exact 16-amplitude statevectors in NumPy and batches cross-kernel evaluation to limit memory. Unit tests compare the states and reordered cross-kernels directly with Qiskit `Statevector` for both maps. Runtime reflects this CPU implementation and does not establish quantum speedup.

## Hardware check of the page kernel

[configs/benchmark-v2-qpu-subset.json](../configs/benchmark-v2-qpu-subset.json) freezes one IBM Quantum run of the 24-example ZZ page model with seed 42 (`quantum-zz-page-24-42`). It runs on 40 rows (20 per class) drawn from the v2 internal test partition. The external dataset cannot be used because it has no page columns. The subset's domain groups are checked to be absent from the development and calibration partitions, and the protocol also pins the hashes of the v2 configuration, test partition and model bundle.

[scripts/run_qpu_v2.py](../scripts/run_qpu_v2.py) rebuilds the training angles from the run's recorded training groups. It refuses to continue unless they reproduce the bundle's stored states. It then plans 960 test pairs, 24 calibration pairs and 24 diagonal controls (1,008 circuits). The evaluation uses each model's frozen v2 threshold and reports six rows: three classical page models with the same 24 examples, plus the quantum kernel from exact simulation, from uncorrected hardware and from linearly calibrated hardware.

```powershell
.\.venv-qml\Scripts\python.exe scripts/run_qpu_v2.py plan        # local, no IBM connection
.\.venv-qml\Scripts\python.exe scripts/run_qpu_v2.py rehearse    # local FakeFez noisy simulation
.\.venv-qml\Scripts\python.exe scripts/run_qpu_v2.py submit      # uses QPU quota; asks to type SUBMIT
.\.venv-qml\Scripts\python.exe scripts/run_qpu_v2.py retrieve <run_id>
.\.venv-qml\Scripts\python.exe scripts/run_qpu_v2.py evaluate <run_id>
```

`submit` requires `IBM_QUANTUM_BACKEND=ibm_fez` and allows only one run. For the full 40 rows, it refuses unless the instance has more than 336.2 seconds of remaining quota: 1,008 circuits at 0.274 billed seconds each, plus 60 seconds. `plan`, `rehearse` and `submit` accept `--rows N` (even, 4 to 40) to measure only the first N/2 rows of each class in the frozen draw order, so a run fits a smaller remaining quota. When the budget guard refuses, `submit` names the largest `--rows` that fits. `evaluate` reads the row count from the run record, times every row locally (median of 25 runs), writes `reports/benchmark-v2-qpu-subset.json` and the chart `docs/benchmark-v2-qpu.svg` (`scripts/render_v2_qpu_svg.py` redraws it from the report), states any reduction in its limitations, and copies the run record and counts to `reports/qpu-evidence-v2/<run_id>/` after checking them for credentials, URLs and local paths. In a local FakeFez rehearsal of the full 40 rows, all three quantum rows classified every row correctly, with mean test-kernel errors of 0.026 uncorrected and 0.008 calibrated. The rehearsal is not a hardware result.

**Hardware result.** Run `v2page-ibm_fez-20261029T010703Z` was submitted once on October 29, 2026, with `--rows 22` (11 per class). The instance had 223 seconds remaining, below the 336.2 seconds the full plan requires; the reduction was chosen before submission. It sent 576 circuits in 3 jobs, and IBM billed 161 seconds (153.1 seconds of circuit execution).

| Model (24 training examples) | Kernel source | Accuracy, n=22 | 95% interval | Recall | FPR |
|---|---|---|---|---|---|
| Logistic regression | classical | 95.5% | 86.4–100% | 90.9% | 0.0% |
| RBF SVM | classical | 95.5% | 86.4–100% | 90.9% | 0.0% |
| Random Forest | classical | 100% | 100–100% | 100% | 0.0% |
| Quantum-kernel SVM (ZZ) | exact simulation | 100% | 100–100% | 100% | 0.0% |
| Quantum-kernel SVM (ZZ) | `ibm_fez`, uncorrected | 100% | 100–100% | 100% | 0.0% |
| Quantum-kernel SVM (ZZ) | `ibm_fez`, calibrated | 100% | 100–100% | 100% | 0.0% |

The mean absolute difference between the hardware and exact-simulation test kernels was 0.073 uncorrected and 0.013 after calibration. The calibration fit on the run's 24 training pairs gave slope 0.769, intercept 0.011 and RMSE 0.0098, and clipped 99 of 528 calibrated test values to [0, 1]. Diagonal controls averaged 0.943 (ideal 1). Uncorrected hardware compresses kernel values toward zero, as the slope below 1 shows. Even so, the SVM's decisions with the frozen threshold matched the exact simulation on all 22 rows. With 22 rows, each row moves accuracy by about 4.5 points. The result shows that hardware noise did not change the decisions on these rows; it is not a precise accuracy estimate or a quantum-advantage claim.

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

![Benchmark v2 page kernel on IBM Quantum: accuracy and 95% intervals, and fit, test-kernel and predict runtime per model](benchmark-v2-qpu.svg)


## Artifacts and report contract

`artifacts/benchmark-v2/` contains verified partition tables, their manifest, run identity, JSON checkpoints, fitted model bundles, calibration scores, frozen selection and per-example evaluation predictions. These files are ignored by Git. Prepared tables and predictions contain row indices and hashed groups; they do not need raw hosts. Only trusted local, hash-verified pickle bundles may be loaded.

A checkpoint can be resumed only with identical configuration, dataset hashes, source hashes and Python/package versions. Completed model and calibration-score artifacts are also hash-checked. Failed or interrupted training runs are retried; completed runs are reused. Once selection is frozen, training is closed and only verified completed searches can be reused. A changed identity requires a separate output directory in a copied configuration. An interrupted preparation without a complete identity must also use a fresh directory. Reporting may be rerun against the same frozen choices.

[reports/benchmark-v2.json](../reports/benchmark-v2.json) uses schema version 1 and contains:

- `status`, execution timestamp and `identity`, including the exact configuration, verified datasets, source hashes and environment versions.
- `partitions`, with exclusion counts, aggregate class counts, membership/file hashes and the development-only `page_feature_ranking`; raw memberships remain local.
- `selection` and `selection_sha256`, identifying the frozen classical, quantum and host-only classical (`classical_host`) configurations.
- `runs`, with each seed's hyperparameter trials, chosen settings, threshold, metrics, confusion matrix, clipping diagnostics and timings.
- `calibration_summary`, `test_summary` and `external_summary`, with means and population standard deviations across fitted seeds.
- `selected_confidence_intervals` and `quantum_confidence_intervals`, with 95% paired percentile bootstrap intervals from 2,000 shared test-row resamples, for the mean selected-model metric and its difference against the same-size `host` majority baseline.
- `target`, `conclusion`, `quantum_conclusion`, the improvement reference, limitations and the next recommended experiment.

Metrics are accuracy, balanced accuracy, precision, recall, FPR, F1 and AP, with phishing positive and confusion matrices ordered `[[TN, FP], [FN, TP]]`. AP uses continuous scores, independently of the calibrated threshold. Subgroup metrics include digits, hyphens and host-label counts; a missing positive/negative class makes the corresponding subgroup recall/FPR undefined (`null`). Selected classical and quantum seeds receive internal-test subgroup analysis. The external stage evaluates only the `classical_host` selection and its same-size majority baseline, without recalibration.

Bootstrap resamples use identical row indices across fitted seeds and the reference model. Their intervals quantify test-row uncertainty conditional on those fitted models; they do not replace seed dispersion or estimate uncertainty over all possible training samples. The vectorized weighted AP implementation retains score ties and is tested against scikit-learn.

The conclusion is `target_met` only if **every classical selected seed** meets all three point targets on the internal test. Otherwise, `partial_improvement` requires a positive lower bound for the paired recall difference versus majority and a mean test FPR ≤ 1%; all other outcomes are `no_consistent_improvement`. `quantum_conclusion` applies the same three point targets to every quantum selected seed and is `target_met` or `target_not_met`. Improvement against majority is deliberately identified as that comparison and must not be interpreted as proof of improvement over the historical v1 SVM.

The generated figures show the mean across training seeds:

- [Learning curves](benchmark-v2-learning.svg): internal-test recall at independently calibrated thresholds, `host` versus `page` features.
- [Operating points](benchmark-v2-operating-points.svg): internal-test recall versus FPR at frozen thresholds for `page` models; connecting points indicate increasing training size, not an ROC sweep. FPR values beyond the axis are drawn at its edge.
- [Matched classical/quantum comparison](benchmark-v2-kernels.svg): AP for identical `page` inputs and matched training sizes.

## Interpretation and next experiment

The complete local run on October 28, 2026 produced 96 classical and 18 exact simulated quantum runs. Selection was frozen before internal-test and external evaluation. The development-only ranking placed `LineOfCode`, `NoOfSelfRef`, `LargestLineLength` and `NoOfImage` first, reproducing the configured page features.

**The 98% accuracy target was met by every fitted seed of both the classical and the quantum selection, and so was the 95% recall target. The 1% FPR point target was not:** internal-test FPR ranged from 1.00% to 1.15% per seed. The report therefore concludes `no_consistent_improvement` for the classical selection (the `partial_improvement` rule also requires a mean test FPR ≤ 1%) and `target_not_met` for the quantum selection. Thresholds were calibrated to exactly 1% FPR on a separate partition, so test FPR naturally lands on either side of that limit. The thresholds were not retuned after observing the test.

- The selected classical configuration is RBF SVM with `page` features and 2,000 balanced training examples. On the 39,519-row internal test, mean accuracy is **98.83%**, recall **98.56%**, FPR **1.04%** and AP **0.9987**, across three fitted seeds. The 95% paired row-bootstrap interval for mean accuracy is **98.72–98.93%**; for FPR it is **0.93–1.16%**.
- The separately selected quantum configuration is the linearly entangled ZZ map with angle maximum π/8 and 500 examples. Its internal-test mean accuracy is **98.64%**, recall **98.14%**, FPR **1.11%** and AP **0.9981**. The bootstrap interval for mean accuracy is **98.53–98.75%**. At the same 500 examples and `page` inputs, classical RBF SVM reaches 98.68% accuracy and logistic regression 98.51%.
- The quantum kernel is most useful with very few examples. With only 24 training examples, mean internal accuracy is 98.49% for ZZ and 98.18% for Z, against 97.78% for logistic regression, 97.42% for Random Forest and 97.32% for RBF SVM. For IBM hardware, measuring a test example costs one circuit per training reference, so the 24-example ZZ model is about 20 times cheaper than the selected 500-example model for a 0.15 percentage-point accuracy difference. This is a matched internal observation, not a quantum-advantage claim: all kernels are exact classical simulations.
- Host-only models remain far weaker. The best `host` configuration, Random Forest with 2,000 examples, reaches 79.72% internal accuracy at the 1% FPR threshold. On the 6,441 external examples, it reaches mean accuracy **63.84%**, recall **30.34%**, FPR **7.47%** and AP **0.6808**. The calibrated internal FPR constraint does not transfer to the external distribution.

Classical search, refitting and calibration scoring took approximately **135.8 seconds** in total; the vectorized quantum stage took **4.7 seconds**. These totals exclude preparation, final held-out scoring, prediction export and bootstrap reporting. They compare different workloads and search budgets, not computational advantage.

The page-feature results must be read together with their limitation. In PhiUSIIL, legitimate rows are homepages of popular sites, with long pages and many internal links, while phishing pages are often small. Four page-size features separate that collection process almost perfectly, so the internal accuracy does not show how the models would behave on phishing pages that copy a legitimate page's structure. The external dataset has no comparable page columns, so this run has no external check of the page models.

Next, measure the 24- or 100-example ZZ page kernel on IBM hardware with the existing hybrid design, and validate page features on independently collected, preferably temporal data that records the same page measurements. Application integration requires a separate decision supported by those results.
