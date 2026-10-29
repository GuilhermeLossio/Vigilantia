# Published Benchmark Results

**Status: complete results published** for run `full-ibm_fez-20261025T094010Z`, and the [external validation](#external-validation-file) and its [hardware subset](#external-hardware-subset-file) are published separately. `scripts/evaluate_benchmark.py` writes `reports/benchmark-results.json` with the three classical models and the quantum-kernel SVM on the exact simulated kernel. `scripts/run_qpu.py evaluate` adds the uncorrected and calibrated IBM Quantum hardware rows after a full QPU run and marks the file complete; the simulation script then leaves a complete file untouched. `scripts/time_benchmark.py` then adds measured runtimes; a new evaluation removes them until it runs again. Without the file, the service reports the results as pending and shows no values.

Only measured results go in this file. Do not add simulated values labeled as hardware, placeholder numbers, or hand-edited metrics. Measured kernels and other generated files stay under the ignored `artifacts/` directory.

## QPU evidence

`reports/qpu-evidence/<run-id>/` holds the recorded IBM Quantum runs, the 12-circuit pilot and the 336-circuit full run on `ibm_fez`: `run.json` with the plan, transpilation summary, job IDs, timestamps, and IBM's usage metrics, and `counts-<job-id>.json` with the raw counts of every circuit. The files contain no token, instance, or account identifier. They let anyone rebuild the hardware rows offline; [notebook 04](../notebooks/04-ibm-hardware-results-from-recorded-counts.ipynb) does so and compares the result with this file. `tests/unit/test_qpu_evidence.py` checks that the evidence is complete and consistent with the published hardware section and runtimes. The external subset run has its own evidence directory, described under [External QPU evidence](#external-qpu-evidence).

## Replay file

`scripts/publish_replay.py` writes `reports/benchmark-replay.json` from the frozen test rows, the simulated evaluation, and the latest full QPU run. It holds, for each of the 12 test samples, the true label and the quantum-kernel SVM prediction and decision margin per source: simulation, QPU uncorrected, and QPU calibrated. Samples are identified by test position, dataset row index, and hashed domain group, never by host or URL. The file is labeled replay and names its source runs; `app/replay.py` validates it, and the page and `GET /v1/benchmarks/replay` present it as replay, not live inference.

## External validation file

`scripts/evaluate_external.py` writes `reports/external-validation.json`, the aggregates of the frozen benchmark models on the external dataset defined in [configs/external-validation.json](../configs/external-validation.json). It is separate from the benchmark results and never changes them. `src/evaluation/external.py` validates it before publishing, and `app/external.py` validates it again with the standard library on every request; the page and `GET /v1/external-validation` show it, or report it as pending (`503`) or invalid (`500`). Per-row predictions stay in the ignored `artifacts/external/`. See [External validation](../docs/external-validation.md) for the protocol and results.

| Field | Rule |
| --- | --- |
| `schema_version` | Must be `1`. |
| `generated_at_utc` | Non-empty string, an ISO 8601 UTC timestamp. |
| `protocol` | `path`, the protocol's `sha256` hex digest, and `frozen_at_utc`. |
| `dataset` | Non-empty `title`, `doi`, `file`, `sha256`, and `license`, and a positive `rows` count; `attribution` records the citation. |
| `benchmark` | `model_version` of the frozen models and a positive `training_examples` count; the split manifest and scaler hashes are recorded under `input_hashes`. |
| `preparation` | `exclusions` lists `invalid_host`, `conflicting_labels`, `phiusiil_overlap`, and `duplicate_group` in that order, each with a non-negative `rows` count. `input_rows` equals the exclusions plus `primary_rows`, and `class_counts` splits `primary_rows` into `legitimate` and `phishing`. |
| `clipped_values` | `per_feature` clipped-value counts and `rows_with_any_clipped`, none above `primary_rows`. |
| `bootstrap` | `method`, `level`, `resamples`, `resamples_used` (at most `resamples`), and `seed`. |
| `models` | Each of `majority`, `logistic_regression`, `rbf_svm`, and `quantum_kernel_svm` exactly once, with its fixed `name` and `kernel_source` (`classical`, or `simulation` for the quantum-kernel SVM). |
| `metrics`, `confidence_intervals` | Exactly the six benchmark metrics. Each metric is in `[0, 1]` or `null`; each interval is `[low, high]` within `[0, 1]`, and `null` exactly when the metric is. |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]`, summing to `primary_rows`. |
| `limitations` | Non-empty list of strings. |

No string in the file, keys included, may contain a URL or a local path; the report holds aggregates only, and no host from either dataset is published.

## External hardware subset file

`scripts/run_qpu_external.py evaluate <run-id>` writes `reports/external-qpu-subset.json`: the six frozen rows on the 40-row external subset defined in [configs/external-qpu-subset.json](../configs/external-qpu-subset.json), with the quantum-kernel SVM classified on the exact simulated kernel and on the uncorrected and calibrated `ibm_fez` kernels. The published file holds run `external-ibm_fez-20261025T132451Z`. It is separate from the benchmark results and the external-validation report and never changes them. The protocol allows one run, and `evaluate` refuses to replace the report or evidence of another run. `src/evaluation/external_qpu.py` validates the file before writing it, and `app/external_qpu.py` validates it again with the standard library on every request; the page and `GET /v1/external-validation/qpu` show it, or report it as pending (`503`) or invalid (`500`). See [External validation: hardware subset](../docs/external-validation.md#hardware-subset-40-rows-on-ibm_fez) for the results.

| Field | Rule |
| --- | --- |
| `schema_version` | Must be `1`. |
| `generated_at_utc` | Non-empty string, an ISO 8601 UTC timestamp. |
| `source`, `label` | `ibm_quantum` with the label `IBM Quantum hardware` for a published run. The local rehearsal uses `noisy_simulation` and is never written to `reports/`. |
| `protocol` | `path`, `sha256`, and `frozen_at_utc` of the subset protocol. |
| `parent_protocol` | `path` and `sha256` of the external-validation protocol. |
| `benchmark` | `model_version` of the frozen models and a positive `training_examples` count; the split manifest and scaler hashes are recorded under `input_hashes`. |
| `subset` | `size` (at least 2), `primary_rows` (at least `size`), `class_counts` splitting `size` into `legitimate` and `phishing`, `size` increasing `row_index` values, and `size` distinct 16-hex `group_hash` values; `seed` and `clipped_values` record the draw and the scaler clipping. |
| `hardware` | `backend`, `run_id`, `sampler`, `shots`, and `job_ids`; `plan` counts `test` (subset size times training examples), `calibration`, and `diagonal` (training examples) pairs, summing to `circuits`. A published run has one job per 252 circuits, the frozen batch size. |
| `kernel` | `calibration` states `valid`; a valid fit holds `a`, `c`, `rmse`, and `clipped_test_values`. `test_kernel_error_vs_ideal` holds `qpu` and, with a valid fit, `qpu_calibrated`; it and `mean_abs_kernel_error_vs_ideal` and `diagonal_controls_mean` are in `[0, 1]`. |
| `runtime` | `method`, `environment`, `repeats` (the frozen 25 for a published run), and `qpu` with exactly `execution_seconds`, `charged_seconds`, and `resource_units`. |
| `bootstrap` | `method`, `level`, `resamples`, `resamples_used` (at most `resamples`), and `seed`. |
| `models` | The compared rows in order: `majority`, `logistic_regression`, and `rbf_svm` (`classical`), then `quantum_kernel_svm` with `simulation`, `qpu`, and `qpu_calibrated`. The calibrated row is present only with a valid fit; without it, `limitations` must state the protocol deviation. Each row has `model`, `kernel_source`, and the fixed `name`. |
| `metrics`, `confidence_intervals` | Exactly the six benchmark metrics. Each metric is in `[0, 1]` or `null`; each interval is `[low, high]` within `[0, 1]`, and `null` exactly when the metric is. |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]`, summing to `subset.size`. |
| `runtime_seconds` | Per row, exactly `fit`, `test_kernel`, and `predict`, each non-negative seconds or `null`. Hardware rows take `test_kernel` from IBM's recorded execution time. |
| `limitations` | Non-empty list of strings. |

No string in the file, keys included, may contain a URL or a local path; rows are identified only by `row_index` and hashed group, never by host or URL.

### V2 page-kernel QPU evidence

`scripts/run_qpu_v2.py evaluate <run-id>` writes `reports/benchmark-v2-qpu-subset.json` for the protocol in [configs/benchmark-v2-qpu-subset.json](../configs/benchmark-v2-qpu-subset.json). It covers six rows on the measured internal-test subset: three classical page models, plus the quantum-kernel SVM on the exact simulated kernel and on the uncorrected and calibrated `ibm_fez` kernels. Each row also carries `runtime_seconds` (local medians for fit, test kernel and predict; IBM's recorded execution time as the hardware test kernel), described under `runtime`, and `scripts/render_v2_qpu_svg.py` draws [docs/benchmark-v2-qpu.svg](../docs/benchmark-v2-qpu.svg) from the file. The published file holds run `v2page-ibm_fez-20261029T010703Z` (22 rows). `reports/qpu-evidence-v2/<run-id>/` holds that run's `run.json` and `counts-<job-id>.json` files, copied after the same credential, URL and local-path checks. See [Hardware check of the page kernel](../docs/benchmark-v2.md#hardware-check-of-the-page-kernel).

### External QPU evidence

`reports/qpu-evidence-external/<run-id>/` holds the recorded subset run, copied by `evaluate`: `run.json` with the plan, the inputs it was planned from, the transpilation summary, job IDs, timestamps, and IBM's usage metrics, and `counts-<job-id>.json` with the raw counts of every circuit, one file per job. `evaluate` copies only those files, after checking each one for credentials, instance identifiers, URLs, and local paths. [Notebook 07](../notebooks/07-external-subset-on-ibm-hardware.ipynb) rebuilds every number of the report from this evidence and compares it with the published file.

## Results contract (schema version 1)

The service validates the file on every request; `app/results.py` is the reference implementation. A file that fails validation is reported as invalid and is never partially displayed.

| Field | Rule |
| --- | --- |
| `schema_version` | Must be `1`. |
| `run_id`, `generated_at_utc` | Non-empty strings. Use an ISO 8601 UTC timestamp. |
| `status` | `complete` or `partial`. Use `partial` when any planned model or variant is missing, and explain why in `limitations`. |
| `split` | Integers `train`, `test`, and `seed`. |
| `hardware` | `backend` and `shots`, or `null` when no QPU result is included; `job_ids` as a list of strings. Any QPU result requires a backend, shots, and at least one job ID. |
| `models` | Non-empty list. Each entry has `name`, `kernel_source`, `metrics`, and `confusion_matrix`. |
| `kernel_source` | `classical`, `simulation`, `qpu`, or `qpu_calibrated`. It labels where each result came from. |
| `metrics` | Exactly `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, and `average_precision`. Each is a number in `[0, 1]`, or `null` when unavailable. |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]` with phishing as the positive class. The four cells must sum to `split.test`. |
| `runtime_seconds` | Optional, per model: exactly `fit`, `test_kernel`, and `predict`, each a non-negative number of seconds or `null` when not applicable. |
| `runtime` | Required when any model has `runtime_seconds`: non-empty `method` and `environment` strings, and optionally `qpu` with exactly `execution_seconds`, `charged_seconds`, and `resource_units`, each non-negative or `null`. |
| `limitations` | List of strings shown with the results. |

Structure, with illustrative field values only:

```json
{
  "schema_version": 1,
  "run_id": "<run identifier>",
  "generated_at_utc": "<ISO 8601 UTC timestamp>",
  "status": "complete",
  "split": {"train": 24, "test": 12, "seed": 42},
  "hardware": {"backend": "<backend name>", "shots": 1024, "job_ids": ["<job id>"]},
  "models": [
    {
      "name": "Quantum-kernel SVM",
      "kernel_source": "qpu_calibrated",
      "metrics": {"accuracy": null, "balanced_accuracy": null, "precision": null,
                  "recall": null, "f1": null, "average_precision": null},
      "confusion_matrix": [[0, 0], [0, 0]]
    }
  ],
  "limitations": ["<limitation shown next to the results>"]
}
```

The example is not a valid results file: its confusion matrix does not sum to the test size. List all four models and all three quantum SVM variants defined in the [benchmark protocol](../docs/benchmark-protocol.md).
