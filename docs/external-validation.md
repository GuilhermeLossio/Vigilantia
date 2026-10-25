# External Validation

**Status: executed on 2026-10-25 under the frozen protocol.** The four frozen benchmark models were applied, without any change, to a second public phishing dataset collected by other authors. The protocol, [configs/external-validation.json](../configs/external-validation.json), was frozen before any external result existed. Nothing in it or in the benchmark was changed after the results below were seen. The published aggregates are in [reports/external-validation.json](../reports/external-validation.json).

## Dataset

Hannousse, A. and Yahiouche, S. (2020). *Web page phishing detection*, version 2. Mendeley Data. DOI [10.17632/c2gw7fy2j4.2](https://doi.org/10.17632/c2gw7fy2j4.2). Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Associated paper: *Towards benchmark datasets for machine learning based website phishing detection: An experimental study*, Engineering Applications of Artificial Intelligence 104 (2021) 104347.

The file `dataset_B_05_2020.csv` has 11,430 rows (5,715 legitimate, 5,715 phishing), 3,661,166 bytes, SHA-256 `21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c`. Only the `url` and `status` columns are used; the 87 published features are ignored. Changes made: rows are filtered and deduplicated as described below.

## Protocol summary

- **Models:** the four frozen models in `artifacts/benchmark/models.pkl`, trained on the 24 PhiUSIIL training examples. No retraining, recalibration, or threshold change; each model's default `predict` is used.
- **Features:** the four frozen host features, scaled with the unchanged [configs/feature-scaler.json](../configs/feature-scaler.json). Values outside the training range are clipped and counted.
- **Quantum kernel:** exact statevector simulation between each external example and the 24 training examples. **No IBM Quantum hardware job was run** for this validation.
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
| Majority baseline | classical | 0.539 (0.526–0.551) | 0.500 (0.500–0.500) | 0.000 | 0.000 | 0.000 | n/a |
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
- The quantum kernel is an exact simulation. Hardware noise is not included; on the benchmark, hardware and simulation gave the same test accuracy.
- The evaluation was run twice. After the first run, a missing-value bug in host handling was fixed. It changed nothing, because no host was rejected, and the re-run gave identical results. Nothing that affects the results was tuned. Later robustness fixes (offline snapshot verification in the evaluation script, and not counting a constant training feature as clipped) do not affect these numbers, since the frozen scaler has no constant feature, and the evaluation was not run again.
- It measures the frozen models only, not the method with more training data.

## Related documents

- [Notebook 06](../notebooks/06-external-validation.ipynb) recomputes every published number and compares it with the report.
- The [data guide](../data/README.md#external-validation-dataset) describes the dataset and its download.
- The [reports guide](../reports/README.md#external-validation-file) defines the report's contract; the [service](service.md#api) serves it at `GET /v1/external-validation` and on the results page.
- The [benchmark protocol](benchmark-protocol.md) defines the frozen models and features evaluated here.
