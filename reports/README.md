# Published Benchmark Results

**Status: complete results published** for run `full-ibm_fez-20261025T094010Z`, and the [external validation](#external-validation-file) is published separately. `scripts/evaluate_benchmark.py` writes `reports/benchmark-results.json` with the three classical models and the quantum-kernel SVM on the exact simulated kernel. `scripts/run_qpu.py evaluate` adds the uncorrected and calibrated IBM Quantum hardware rows after a full QPU run and marks the file complete; the simulation script then leaves a complete file untouched. `scripts/time_benchmark.py` then adds measured runtimes; a new evaluation removes them until it runs again. Without the file, the service reports the results as pending and shows no values.

Only measured results go in this file. Do not add simulated values labeled as hardware, placeholder numbers, or hand-edited metrics. Measured kernels and other generated files stay under the ignored `artifacts/` directory.

## QPU evidence

`reports/qpu-evidence/<run-id>/` holds the recorded IBM Quantum runs, the 12-circuit pilot and the 336-circuit full run on `ibm_fez`: `run.json` with the plan, transpilation summary, job IDs, timestamps, and IBM's usage metrics, and `counts-<job-id>.json` with the raw counts of every circuit. The files contain no token, instance, or account identifier. They let anyone rebuild the hardware rows offline; [notebook 04](../notebooks/04-ibm-hardware-results-from-recorded-counts.ipynb) does so and compares the result with this file. `tests/unit/test_qpu_evidence.py` checks that the evidence is complete and consistent with the published hardware section and runtimes.

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
