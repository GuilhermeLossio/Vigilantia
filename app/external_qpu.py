"""Load and validate the published external hardware subset (schema version 1).

The report is produced by scripts/run_qpu_external.py, whose validator lives in
src/evaluation/external_qpu.py. That module needs numpy, pandas, and the quantum tooling,
which the service image does not ship, so the service checks the same contract here with
the standard library only.
"""

from __future__ import annotations

import json
import re
from math import ceil, isfinite
from pathlib import Path

from app.external import _FORBIDDEN_TEXT, METRICS, _is_int, _is_text, _is_unit, _require, _strings
from app.results import ResultsInvalid, ResultsUnavailable

SCHEMA_VERSION = 1
CLASSES = ("legitimate", "phishing")
BATCH_SIZE = 252
FROZEN_REPEATS = 25
NAMES = {
    "majority": "Majority baseline",
    "logistic_regression": "Logistic regression",
    "rbf_svm": "RBF SVM",
    "quantum_kernel_svm": "Quantum-kernel SVM",
}
COMPARED = (
    ("majority", "classical"),
    ("logistic_regression", "classical"),
    ("rbf_svm", "classical"),
    ("quantum_kernel_svm", "simulation"),
    ("quantum_kernel_svm", "qpu"),
    ("quantum_kernel_svm", "qpu_calibrated"),
)
HARDWARE_SOURCES = ("qpu", "qpu_calibrated")
HARDWARE_LABELS = {"qpu": "uncorrected hardware", "qpu_calibrated": "calibrated hardware"}
CLASSICAL_MODELS = ("logistic_regression", "rbf_svm")
RUNTIME_KEYS = ("fit", "test_kernel", "predict")
USAGE_KEYS = ("execution_seconds", "charged_seconds", "resource_units")
SOURCES = {
    "ibm_quantum": "IBM Quantum hardware",
    "noisy_simulation": "noisy simulation (fake backend); not a hardware result",
}
CALIBRATION_DEVIATION = (
    "Protocol deviation: the fresh kernel calibration fit on this run's 24 training pairs was "
    "invalid, so the calibrated hardware row could not be computed and only five of the six "
    "frozen rows are reported."
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GROUP_HASH = re.compile(r"[0-9a-f]{16}")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value)


def _is_seconds(value: object) -> bool:
    return value is None or _is_number(value) and value >= 0


def _validate_reference(name: str, value: object, frozen: bool) -> None:
    _require(
        isinstance(value, dict)
        and _is_text(value.get("path"))
        and isinstance(value.get("sha256"), str)
        and _SHA256.fullmatch(value["sha256"])
        and (not frozen or _is_text(value.get("frozen_at_utc"))),
        f"{name} must hold path and a SHA-256 hex digest"
        + (", and frozen_at_utc." if frozen else "."),
    )


def _validate_subset(subset: object) -> int:
    _require(isinstance(subset, dict), "subset must be an object.")
    size = subset.get("size")
    _require(_is_int(size, 2), "subset.size must be an integer of at least 2.")
    _require(
        _is_int(subset.get("primary_rows"), 1) and subset["primary_rows"] >= size,
        "subset.primary_rows must be at least subset.size.",
    )
    counts = subset.get("class_counts")
    _require(
        isinstance(counts, dict)
        and list(counts) == list(CLASSES)
        and all(_is_int(value, 1) for value in counts.values())
        and sum(counts.values()) == size,
        "subset.class_counts must split subset.size into legitimate and phishing.",
    )
    rows = subset.get("row_index")
    _require(
        isinstance(rows, list)
        and len(rows) == size
        and all(_is_int(value) for value in rows)
        and all(a < b for a, b in zip(rows, rows[1:], strict=False)),
        "subset.row_index must list subset.size file rows in increasing order.",
    )
    groups = subset.get("group_hash")
    _require(
        isinstance(groups, list)
        and len(groups) == size
        and all(isinstance(g, str) and _GROUP_HASH.fullmatch(g) for g in groups)
        and len(set(groups)) == size,
        "subset.group_hash must list one distinct 16-hex group hash per row.",
    )
    return size


def _validate_hardware(run: object, size: int, training: int, published: bool) -> None:
    _require(
        isinstance(run, dict)
        and all(_is_text(run.get(key)) for key in ("backend", "run_id", "sampler"))
        and _is_int(run.get("shots"), 1)
        and isinstance(run.get("job_ids"), list)
        and run["job_ids"]
        and all(_is_text(job) for job in run["job_ids"]),
        "hardware must hold backend, run_id, sampler, shots, and job_ids.",
    )
    plan = run.get("plan")
    _require(
        isinstance(plan, dict)
        and list(plan) == ["test", "calibration", "diagonal"]
        and all(_is_int(value) for value in plan.values())
        and plan["test"] == size * training
        and plan["diagonal"] == training
        and run.get("circuits") == sum(plan.values()),
        "hardware.plan must count test, calibration, and diagonal pairs summing to circuits.",
    )
    _require(
        not published or len(run["job_ids"]) == ceil(run["circuits"] / BATCH_SIZE),
        f"A hardware run must send its circuits in jobs of the frozen batch size {BATCH_SIZE}.",
    )


def _validate_kernel(kernel: object) -> bool:
    _require(isinstance(kernel, dict), "kernel must be an object.")
    calibration = kernel.get("calibration")
    _require(
        isinstance(calibration, dict) and isinstance(calibration.get("valid"), bool),
        "kernel.calibration must state whether the fit is valid.",
    )
    valid = calibration["valid"]
    if valid:
        _require(
            all(_is_number(calibration.get(key)) for key in ("a", "c", "rmse"))
            and _is_int(calibration.get("clipped_test_values")),
            "A valid calibration must hold a, c, rmse, and clipped_test_values.",
        )
    errors = kernel.get("test_kernel_error_vs_ideal")
    expected = list(HARDWARE_SOURCES) if valid else ["qpu"]
    _require(
        isinstance(errors, dict)
        and list(errors) == expected
        and all(_is_unit(v) for v in errors.values()),
        f"kernel.test_kernel_error_vs_ideal must hold {expected} in [0, 1].",
    )
    _require(
        _is_unit(kernel.get("mean_abs_kernel_error_vs_ideal"))
        and _is_unit(kernel.get("diagonal_controls_mean")),
        "kernel must hold mean_abs_kernel_error_vs_ideal and diagonal_controls_mean in [0, 1].",
    )
    return valid


def _validate_model(index: int, row: object, size: int) -> tuple[str, str]:
    where = f"models[{index}]"
    _require(isinstance(row, dict), f"{where} must be an object.")
    pair = (row.get("model"), row.get("kernel_source"))
    _require(pair in COMPARED, f"{where} must be one of the compared rows.")
    _require(row.get("name") == NAMES[pair[0]], f"{where}.name must be {NAMES[pair[0]]!r}.")
    metrics, intervals = row.get("metrics"), row.get("confidence_intervals")
    _require(
        isinstance(metrics, dict)
        and set(metrics) == set(METRICS)
        and isinstance(intervals, dict)
        and set(intervals) == set(METRICS),
        f"{where} metrics and confidence_intervals must contain exactly {list(METRICS)}.",
    )
    for metric in METRICS:
        value, interval = metrics[metric], intervals[metric]
        _require(value is None or _is_unit(value), f"{where}.metrics.{metric} must be in [0, 1].")
        _require(
            (value is None) == (interval is None)
            and (
                interval is None
                or isinstance(interval, list)
                and len(interval) == 2
                and all(_is_unit(bound) for bound in interval)
                and interval[0] <= interval[1]
            ),
            f"{where}.confidence_intervals.{metric} must be [low, high], null with the metric.",
        )
    matrix = row.get("confusion_matrix")
    _require(
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(line, list) and len(line) == 2 for line in matrix)
        and all(_is_int(cell) for line in matrix for cell in line)
        and sum(sum(line) for line in matrix) == size,
        f"{where}.confusion_matrix must be [[TN, FP], [FN, TP]] summing to subset.size.",
    )
    seconds = row.get("runtime_seconds")
    _require(
        isinstance(seconds, dict)
        and list(seconds) == list(RUNTIME_KEYS)
        and all(_is_seconds(value) for value in seconds.values()),
        f"{where}.runtime_seconds must hold fit, test_kernel, and predict in seconds or null.",
    )
    return pair


def validate(report: object) -> dict:
    """Return the report unchanged when it satisfies schema version 1."""
    _require(isinstance(report, dict), "Report must be a JSON object.")
    _require(
        report.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}."
    )
    _require(_is_text(report.get("generated_at_utc")), "generated_at_utc must be a string.")
    _require(
        report.get("source") in SOURCES and report.get("label") == SOURCES[report["source"]],
        f"source must be one of {list(SOURCES)} with its fixed label.",
    )
    _validate_reference("protocol", report.get("protocol"), frozen=True)
    _validate_reference("parent_protocol", report.get("parent_protocol"), frozen=False)
    benchmark = report.get("benchmark")
    _require(
        isinstance(benchmark, dict)
        and _is_text(benchmark.get("model_version"))
        and _is_int(benchmark.get("training_examples"), 1),
        "benchmark must hold model_version and a positive training_examples count.",
    )
    size = _validate_subset(report.get("subset"))
    published = report["source"] == "ibm_quantum"
    _validate_hardware(report.get("hardware"), size, benchmark["training_examples"], published)
    calibrated = _validate_kernel(report.get("kernel"))

    runtime = report.get("runtime")
    _require(
        isinstance(runtime, dict)
        and _is_text(runtime.get("method"))
        and _is_text(runtime.get("environment"))
        and _is_int(runtime.get("repeats"), 1)
        and isinstance(runtime.get("qpu"), dict)
        and list(runtime["qpu"]) == list(USAGE_KEYS)
        and all(_is_seconds(value) for value in runtime["qpu"].values()),
        "runtime must hold method, environment, repeats, and qpu usage.",
    )
    _require(
        not published or runtime["repeats"] == FROZEN_REPEATS,
        f"A hardware report must time the local rows over the frozen {FROZEN_REPEATS} repeats.",
    )
    bootstrap = report.get("bootstrap")
    _require(
        isinstance(bootstrap, dict)
        and _is_text(bootstrap.get("method"))
        and _is_unit(bootstrap.get("level"))
        and _is_int(bootstrap.get("resamples"), 1)
        and _is_int(bootstrap.get("resamples_used"))
        and bootstrap["resamples_used"] <= bootstrap["resamples"]
        and _is_int(bootstrap.get("seed")),
        "bootstrap must hold method, level, resamples, resamples_used, and seed.",
    )
    models = report.get("models")
    _require(isinstance(models, list), "models must be a list.")
    pairs = [_validate_model(index, row, size) for index, row in enumerate(models)]
    expected = [pair for pair in COMPARED if calibrated or pair[1] != "qpu_calibrated"]
    _require(
        pairs == expected,
        "models must list the compared rows in order; qpu_calibrated only with a valid fit.",
    )
    limitations = report.get("limitations")
    _require(
        isinstance(limitations, list) and limitations and all(_is_text(i) for i in limitations),
        "limitations must be a non-empty list of strings.",
    )
    _require(
        calibrated or CALIBRATION_DEVIATION in limitations,
        "A report without the calibrated row must state the protocol deviation.",
    )
    _require(
        not any(_FORBIDDEN_TEXT.search(text) for text in _strings(report)),
        "The report must not contain URLs or local paths.",
    )
    return report


def _correct(row: dict) -> int:
    matrix = row["confusion_matrix"]
    return matrix[0][0] + matrix[1][1]


def _classical(report: dict) -> list[dict]:
    return [
        {
            "model": row["model"],
            "name": row["name"],
            "accuracy": row["metrics"]["accuracy"],
            "interval": (row.get("confidence_intervals") or {}).get("accuracy"),
        }
        for row in report["models"]
        if row["model"] in CLASSICAL_MODELS
    ]


def _full_set(report: dict, external: dict) -> dict:
    """Classical accuracies on the whole primary set, and whether each lies in its subset CI."""
    subset = {row["model"]: row for row in _classical(report)}
    classical = _classical(external)
    within = all(
        row["model"] in subset
        and subset[row["model"]]["interval"] is not None
        and subset[row["model"]]["interval"][0]
        <= row["accuracy"]
        <= subset[row["model"]]["interval"][1]
        for row in classical
    )
    return {
        "rows": external["preparation"]["primary_rows"],
        "classical": classical,
        "within_subset_intervals": within,
    }


def interpretation(report: dict, external: dict | None = None) -> dict:
    """Facts behind the page's reading of the hardware rows, all taken from the reports.

    Each hardware row is compared with the exact-simulation row by the net change in correct
    predictions. `within_intervals` holds only when every hardware accuracy lies inside the
    simulation interval and the simulation accuracy lies inside every hardware interval.
    `full_set` carries the classical accuracies on the whole external primary set, when that
    report is available, and whether each lies inside the subset interval of the same model,
    so a gap is described as sampling variation only when the intervals support it.
    """
    quantum = {
        row["kernel_source"]: row
        for row in report["models"]
        if row["model"] == "quantum_kernel_svm"
    }
    simulation = quantum["simulation"]
    hardware = [quantum[source] for source in HARDWARE_SOURCES if source in quantum]
    sim_accuracy = simulation["metrics"]["accuracy"]
    low, high = simulation["confidence_intervals"]["accuracy"]
    within = all(
        low <= row["metrics"]["accuracy"] <= high
        and row["confidence_intervals"]["accuracy"][0]
        <= sim_accuracy
        <= row["confidence_intervals"]["accuracy"][1]
        for row in hardware
    )
    size = report["subset"]["size"]
    return {
        "size": size,
        "points_per_row": 100 / size,
        "changes": [
            {
                "label": HARDWARE_LABELS[row["kernel_source"]],
                "delta": _correct(row) - _correct(simulation),
            }
            for row in hardware
        ],
        "within_intervals": within,
        "classical": _classical(report),
        "full_set": None if external is None else _full_set(report, external),
    }


def load(path: Path) -> dict:
    """Read and validate the report; distinguish absent from invalid."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ResultsUnavailable("The external subset has no hardware results yet.") from error
    try:
        report = json.loads(text)
    except json.JSONDecodeError as error:
        raise ResultsInvalid(f"External subset file is not valid JSON: {error.msg}.") from error
    return validate(report)
