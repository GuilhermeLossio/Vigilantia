# Published Benchmark Results

**Status: partial results published.** `scripts/evaluate_benchmark.py` writes `reports/benchmark-results.json` with the three classical models and the quantum-kernel SVM on the exact simulated kernel. `scripts/run_qpu.py evaluate` adds the uncorrected and calibrated IBM Quantum hardware rows after a full QPU run and marks the file complete; the simulation script then leaves a complete file untouched. Without the file, the service reports the results as pending and shows no values.

Only measured results go in this file. Do not add simulated values labeled as hardware, placeholder numbers, or hand-edited metrics. Raw counts, kernels, and job evidence stay under the ignored `reports/runs/` directory, or in private storage.

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
