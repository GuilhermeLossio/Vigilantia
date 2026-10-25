# External Validation

**Status: executed on 2026-10-25 under the frozen protocol.** The four frozen benchmark models were applied, without any change, to a second public phishing dataset collected by other authors. The protocol, [configs/external-validation.json](../configs/external-validation.json), was frozen before any external result existed. Nothing in it or in the benchmark was changed after the results below were seen. The published aggregates are in [reports/external-validation.json](../reports/external-validation.json). A 40-row subset of the same primary set was also run once on IBM Quantum hardware, under its own frozen protocol; see [Hardware subset](#hardware-subset-40-rows-on-ibm_fez).

## Dataset

Hannousse, A. and Yahiouche, S. (2020). *Web page phishing detection*, version 2. Mendeley Data. DOI [10.17632/c2gw7fy2j4.2](https://doi.org/10.17632/c2gw7fy2j4.2). Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Associated paper: *Towards benchmark datasets for machine learning based website phishing detection: An experimental study*, Engineering Applications of Artificial Intelligence 104 (2021) 104347.

The file `dataset_B_05_2020.csv` has 11,430 rows (5,715 legitimate, 5,715 phishing), 3,661,166 bytes, SHA-256 `21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c`. Only the `url` and `status` columns are used; the 87 published features are ignored. Changes made: rows are filtered and deduplicated as described below.

## Protocol summary

- **Models:** the four frozen models in `artifacts/benchmark/models.pkl`, trained on the 24 PhiUSIIL training examples. No retraining, recalibration, or threshold change; each model's default `predict` is used.
- **Features:** the four frozen host features, scaled with the unchanged [configs/feature-scaler.json](../configs/feature-scaler.json). Values outside the training range are clipped and counted.
- **Quantum kernel:** exact statevector simulation between each external example and the 24 training examples. **No IBM Quantum hardware job was run** for the primary set; only the [40-row hardware subset](#hardware-subset-40-rows-on-ibm_fez) ran on `ibm_fez`.
- **Preparation, in order:**
  1. `invalid_host`: derive the host with `extract_host` (which applies `normalize_host`); exclude rejected rows. A valid host without a registrable domain cannot be grouped and is counted here too.
  2. `conflicting_labels`: group by registrable domain with the benchmark's `group_key`; exclude groups carrying both labels.
  3. `phiusiil_overlap`: exclude every group that occurs among the registrable domains of any PhiUSIIL row (all 235,795 rows, not only the benchmark sample).
  4. `duplicate_group`: keep the first row of each remaining group in file order.
- **Metrics:** accuracy, balanced accuracy, precision, recall, F1, and average precision, with phishing as the positive class and zero division set to 0. Average precision uses decision scores and is not available for the majority baseline.
- **Confidence intervals:** 95% percentile bootstrap over rows of the primary set, 2,000 resamples, seed 42. The same resampled rows are used for every model. A resample with only one class would leave balanced accuracy, recall, and average precision undefined; such resamples are skipped for every metric and counted. None occurred: all 2,000 resamples were used.

## Reproducing

```powershell
.\.venv\Scripts\python.exe scripts/download_dataset.py    # PhiUSIIL, needed for the overlap step
.\.venv\Scripts\python.exe scripts/download_external.py   # verified download to data/external/hannousse-2020/
.\.venv\Scripts\python.exe scripts/evaluate_external.py   # offline: verify, prepare, evaluate, publish
```

Run these inside the pinned environment from the [reproduction guide](reproducing.md); the two downloaders need only the standard library, but `evaluate_external.py` needs pandas, scikit-learn, and Qiskit.

`download_external.py` uses only the standard library. It checks the size and SHA-256 against the protocol and the row and label counts, writes `acquisition.json`, and reuses an intact snapshot without downloading again. On any mismatch it fails and publishes nothing. The evaluation also checks the PhiUSIIL checksum and the benchmark input hashes. Per-row predictions, keyed by file row and hashed group, go to the Git-ignored `artifacts/external/`. The published report holds aggregates only, with no host or URL.

## Recorded results

### Preparation

| Step | Rows excluded | Groups |
| --- | ---: | ---: |
| `invalid_host` | 0 | |
| `conflicting_labels` | 397 | 17 |
| `phiusiil_overlap` | 1,805 | 802 |
| `duplicate_group` | 2,787 | |
| **Primary set** | **6,441** (3,470 legitimate, 2,971 phishing) | |

Phishing is 46.1% of the primary set, so a model that ranks at random has an average precision of about 0.461.

Values outside the training range, clipped by the scaler: `host_entropy` 59, `host_label_count` 396, `host_digit_ratio` 317, `host_hyphen_count` 3. In total, 600 rows (9.3%) had at least one clipped feature.

### Metrics on the 6,441-row primary set

Point estimate with the 95% bootstrap interval in parentheses.

| Model | Kernel | Accuracy | Balanced accuracy | Precision | Recall | F1 | Average precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | classical | 0.539 (0.526–0.551) | 0.500 (0.500–0.500) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | n/a |
| Logistic regression | classical | 0.663 (0.651–0.674) | 0.653 (0.642–0.665) | 0.669 (0.651–0.689) | 0.530 (0.513–0.548) | 0.592 (0.577–0.607) | 0.670 (0.653–0.687) |
| RBF SVM | classical | 0.644 (0.632–0.656) | 0.637 (0.626–0.649) | 0.629 (0.611–0.647) | 0.555 (0.538–0.572) | 0.590 (0.575–0.604) | 0.647 (0.629–0.666) |
| Quantum-kernel SVM | exact simulation | 0.522 (0.510–0.535) | 0.528 (0.516–0.541) | 0.486 (0.469–0.502) | 0.611 (0.594–0.629) | 0.541 (0.527–0.556) | 0.461 (0.445–0.479) |

Confusion matrices, `[[TN, FP], [FN, TP]]`: majority `[[3470, 0], [2971, 0]]`; logistic regression `[[2692, 778], [1395, 1576]]`; RBF SVM `[[2497, 973], [1322, 1649]]`; quantum-kernel SVM `[[1546, 1924], [1155, 1816]]`.

### Comparison with the 12-example benchmark

| Model | Benchmark test accuracy (12 rows) | External accuracy (6,441 rows) |
| --- | ---: | ---: |
| Majority baseline | 0.500 | 0.539 |
| Logistic regression | 0.833 | 0.663 |
| RBF SVM | 0.917 | 0.644 |
| Quantum-kernel SVM (simulation) | 0.500 | 0.522 |

## Interpretation

- The two classical models keep a real but much smaller signal outside PhiUSIIL. Their balanced accuracy, about 0.64 to 0.65, is well above the majority baseline's 0.500, and their intervals exclude it. It is far below the 11/12 (0.917) the RBF SVM reached on the benchmark test set. Possible causes include the 12-row test set, distribution shift between the datasets, and collection artifacts in PhiUSIIL; this validation does not separate them.
- On this dataset, logistic regression and the RBF SVM are close. Their accuracy intervals overlap, so this validation does not rank them.
- The quantum-kernel SVM is near chance. Its balanced accuracy of 0.528 is only slightly above 0.5, and its average precision (0.461) about equals the phishing share of the primary set, so its scores do not rank phishing above legitimate hosts. Its accuracy (0.522) is not above the majority baseline's (0.539, which only reflects the 53.9% legitimate share); the two intervals overlap. This agrees with its 6/12 result on the benchmark test set, in both simulation and hardware.
- These results do not support any claim of quantum advantage.

## Limitations

- The models were trained on only 24 examples and were not adapted to this dataset in any way.
- Features use only the host. Paths, page content, and the dataset's own features are ignored, so this does not evaluate phishing detection in general.
- The external data come from different authors, a different collection method, and a different period (the file is dated May 2020; the dataset was published in September 2020). This is a test under distribution shift: 9.3% of rows fall outside the training range on at least one feature.
- Keeping one row per registrable domain and removing every domain present in PhiUSIIL reduces the 11,430 rows to 6,441 and changes the class balance to 46.1% phishing.
- On the primary set, the quantum kernel is an exact simulation and hardware noise is not included. On the benchmark, hardware and simulation gave the same test accuracy; on the 40-row hardware subset they differ by one or two rows, well within the intervals.
- The evaluation was run twice. After the first run, a missing-value bug in host handling was fixed. It changed nothing, because no host was rejected, and the re-run gave identical results. Nothing that affects the results was tuned. Later robustness fixes (offline snapshot verification in the evaluation script, and not counting a constant training feature as clipped) do not affect these numbers, since the frozen scaler has no constant feature, and the evaluation was not run again.
- It measures the frozen models only, not the method with more training data.

## Hardware subset (40 rows on `ibm_fez`)

**Status: executed on 2026-10-25 under a second frozen protocol.** The primary set above uses an exactly simulated quantum kernel. To also measure the quantum-kernel SVM on IBM Quantum hardware within the Open Plan quota, a stratified subset of 40 primary-set rows ran once on `ibm_fez`. The protocol, [configs/external-qpu-subset.json](../configs/external-qpu-subset.json), was frozen before any hardware job for the subset was submitted. It fixes the subset rule, the backend, the circuit plan, the six compared rows, the metrics, the bootstrap, and the budget guard. Neither it nor the parent protocol changed after the results were seen. The published aggregates are in [reports/external-qpu-subset.json](../reports/external-qpu-subset.json), and the recorded counts are in [reports/qpu-evidence-external/](../reports/qpu-evidence-external/).

### Subset and run

- **Subset rule:** one `numpy.random.default_rng(42)` generator draws 20 legitimate and then 20 phishing rows, without replacement, from the 6,441-row primary set; the 40 rows are then sorted by file position. The rule was fixed before any result on these rows existed, so it cannot select rows by how a model scores on them. Rows are published only by `row_index` and hashed group. Seven rows had at least one feature clipped by the scaler.
- **Run:** `external-ibm_fez-20261025T132451Z`, 4 SamplerV2 jobs of at most 252 circuits, 1,024 shots each. The 1,008 circuits are 960 test pairs (each subset row against each of the 24 training examples), 24 calibration pairs, and 24 diagonal controls. Training is unchanged: the same frozen models, fitted on the exactly simulated 24 x 24 kernel.
- **Calibration:** a fresh linear fit on this run's 24 calibration pairs; the benchmark run's calibration is not reused.
- **Bootstrap:** 95% percentile intervals over the 40 rows, 2,000 resamples, seed 42, with the same resampled rows for every model; all 2,000 resamples were used.

### Metrics on the 40-row subset

Point estimate with the 95% bootstrap interval in parentheses. With 40 rows, each row changes accuracy by 2.5 points.

| Model | Test kernel | Accuracy | Balanced accuracy | Precision | Recall | F1 | Average precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | classical | 0.500 (0.350–0.650) | 0.500 (0.500–0.500) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | n/a |
| Logistic regression | classical | 0.525 (0.375–0.675) | 0.525 (0.370–0.675) | 0.533 (0.273–0.786) | 0.400 (0.185–0.632) | 0.457 (0.231–0.649) | 0.603 (0.412–0.803) |
| RBF SVM | classical | 0.525 (0.375–0.675) | 0.525 (0.370–0.675) | 0.533 (0.273–0.786) | 0.400 (0.185–0.632) | 0.457 (0.231–0.649) | 0.605 (0.404–0.798) |
| Quantum-kernel SVM | exact simulation | 0.475 (0.325–0.625) | 0.475 (0.325–0.625) | 0.480 (0.286–0.680) | 0.600 (0.381–0.821) | 0.533 (0.341–0.692) | 0.499 (0.339–0.734) |
| Quantum-kernel SVM | `ibm_fez`, uncorrected | 0.500 (0.350–0.650) | 0.500 (0.350–0.649) | 0.500 (0.304–0.696) | 0.650 (0.429–0.850) | 0.565 (0.372–0.717) | 0.524 (0.353–0.743) |
| Quantum-kernel SVM | `ibm_fez`, calibrated | 0.525 (0.375–0.675) | 0.525 (0.372–0.675) | 0.520 (0.320–0.714) | 0.650 (0.429–0.850) | 0.578 (0.378–0.731) | 0.504 (0.341–0.741) |

Confusion matrices, `[[TN, FP], [FN, TP]]`: majority `[[20, 0], [20, 0]]`; logistic regression and RBF SVM `[[13, 7], [12, 8]]`; quantum-kernel SVM with the exact simulated kernel `[[7, 13], [8, 12]]`, with the uncorrected `ibm_fez` kernel `[[7, 13], [7, 13]]`, and with the calibrated one `[[8, 12], [7, 13]]`.

### Hardware kernel and IBM usage

| Measure | Value |
| --- | --- |
| Test kernel error vs exact simulation (mean absolute) | 0.026 uncorrected, 0.018 calibrated (benchmark run: 0.037 and 0.029, from the [benchmark protocol](benchmark-protocol.md#ibm-quantum-hardware-full-run), rebuilt from `reports/qpu-evidence/` by notebook 04) |
| Calibration fit `measured = a * ideal + c` | a = 0.867, c = 0.0079, RMSE 0.021; 218 of the 960 calibrated test values clipped to `[0, 1]` |
| Diagonal controls (ideal 1) | mean 0.955 |
| Circuit execution time | 267.6 s (projected 267 s) |
| Billed QPU time | 280 s (projected 276.2 s), 4.61 resource units |

The report also holds local runtimes, as medians of 25 repeats: the exactly simulated test kernel for the 40 rows takes about 21 ms, against 267.6 s of circuit execution on `ibm_fez`, queueing excluded. The [comparison chart](model-comparison.svg) draws accuracy for the benchmark, the full external set, and this subset, and runtime for the benchmark and this subset; the full external set has no published runtime.

### Interpretation

- **On 40 rows, no model separates from chance.** Every row's accuracy lies inside every other row's 95% interval, and the intervals are 30 points wide.
- **Hardware noise changed 1 prediction (uncorrected) and 2 (calibrated), all toward the correct label.** A difference of one or two rows of 40 is well within the intervals, so it is not evidence that hardware improves the model. Another run, with new shot noise and device drift, could move them the other way.
- **The hardware kernel was closer to the exact kernel than in the benchmark run**: 0.026 and 0.018 against 0.037 and 0.029. The clipped values are mostly exact values near zero that noise placed below the intercept.
- **The classical models scored lower on this subset than on the full external set.** Logistic regression and the RBF SVM reach 0.663 and 0.644 accuracy on all 6,441 rows, but 0.525 here. The gap lies within the subset intervals and is consistent with sampling variation in 40 rows; the subset rule was frozen before any result, so it is not selection.
- **These results do not support any claim of quantum advantage.** Four qubits are trivial to simulate exactly.

### Limitations of the subset

- 40 rows give wide intervals; this subset cannot rank the models.
- It is one run on one backend. Shot noise and device drift change kernel values between runs, and the protocol allows this one run only. It used almost half of the Open Plan's 600 s per rolling 28 days.
- The models are the frozen benchmark models, trained on 24 PhiUSIIL examples with host-only features. The calibration is a linear approximation, and no model is trained on hardware measurements.

### Reproducing the subset offline

[Notebook 07](../notebooks/07-external-subset-on-ibm-hardware.ipynb) rebuilds the subset by the frozen rule, re-evaluates the six rows from the recorded counts, and compares every published number with the report, without an IBM account. The script path is in the [reproduction guide](reproducing.md#4-script-only-path). The submission commands, which use real QPU quota, are in [hardware runs](hardware-runs.md#external-subset-40-rows).

## Related documents

- [Notebook 06](../notebooks/06-external-validation.ipynb) recomputes every published number and compares it with the report.
- The [data guide](../data/README.md#external-validation-dataset) describes the dataset and its download.
- [Notebook 07](../notebooks/07-external-subset-on-ibm-hardware.ipynb) rebuilds the 40-row hardware subset offline from the recorded counts and compares it with its report.
- The [reports guide](../reports/README.md#external-validation-file) defines the report's contract, and its [hardware subset section](../reports/README.md#external-hardware-subset-file) the subset's. The [service](service.md#api) serves the report at `GET /v1/external-validation`, the subset at `GET /v1/external-validation/qpu`, and both on the results page.
- The [benchmark protocol](benchmark-protocol.md) defines the frozen models and features evaluated here.
