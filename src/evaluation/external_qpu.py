"""The 40-row external subset on IBM Quantum hardware (configs/external-qpu-subset.json).

- `load_protocol` reads the frozen protocol and checks it against its parent protocol and
  the hardware tooling it relies on.
- `select_subset` draws the subset by the frozen rule and identifies its rows by
  row_index and group_hash only.
- `budget_refusal` is the protocol's budget guard on the instance's remaining quota.
- `evidence_refusal` checks hardware evidence for credentials, URLs, and local paths
  before it is copied into reports/.
- `rows_with_intervals`, `with_runtime`, `build_report`, and `validate` produce and check
  reports/external-qpu-subset.json (schema version 1), which holds no host, URL, or path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil, isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.benchmark_split import group_hash
from src.evaluation import external
from src.evaluation.external import METRICS
from src.evaluation.publish import DISPLAY
from src.quantum import hardware

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "external-qpu-subset.json"
PARENT_PATH = ROOT / "configs" / "external-validation.json"
SCHEMA_VERSION = 1
SUBSET_SEED = 42
CLASSES = ("legitimate", "phishing")  # label 0, then label 1: the frozen draw order
SAMPLER = "v2"
# The benchmark run's measured billed time: 92 s for 336 circuits, per the frozen protocol.
BILLED_SECONDS_PER_CIRCUIT = 0.274
GUARD_MARGIN_SECONDS = 60.0
BATCH_SIZE = 252  # the protocol's hardware.batch_size: 1,008 circuits in 4 jobs
FROZEN_REPEATS = 25  # the protocol's runtime.local repeats for the published report
COMPARED = (
    ("majority", "classical"),
    ("logistic_regression", "classical"),
    ("rbf_svm", "classical"),
    ("quantum_kernel_svm", "simulation"),
    ("quantum_kernel_svm", "qpu"),
    ("quantum_kernel_svm", "qpu_calibrated"),
)
HARDWARE_SOURCES = ("qpu", "qpu_calibrated")
NAMES = {key: name for key, (name, _) in DISPLAY.items()}
RUNTIME_KEYS = ("fit", "test_kernel", "predict")
USAGE_KEYS = ("execution_seconds", "charged_seconds", "resource_units")
SOURCES = {
    "ibm_quantum": "IBM Quantum hardware",
    "noisy_simulation": "noisy simulation (fake backend); not a hardware result",
}
RUNTIME_METHOD = (
    "Local rows: median wall-clock seconds over {repeats} repeats of fit, test-kernel "
    "computation, and prediction for the subset rows on the machine named in environment. "
    "Hardware rows take their test-kernel time from IBM's recorded circuit execution time "
    "for the run's jobs; queueing is excluded."
)
BOOTSTRAP_METHOD = (
    "Percentile bootstrap over the subset rows; the same resampled row indices are used for "
    "every row of the table. Resamples containing only one class are skipped for every "
    "metric and counted in resamples_used. Intervals are null when the point metric is null."
)
LIMITATIONS = [
    "Small sample: the subset holds 40 rows (20 per class) of the external primary set, so "
    "each row changes accuracy by 2.5 percentage points and the intervals are wide.",
    "The models were trained on only 24 PhiUSIIL examples (12 per class) and were not "
    "retrained, recalibrated, or re-thresholded; the hardware kernel calibration is a fresh "
    "linear fit on this run's 24 training pairs.",
    "Features use only the host of each URL, and the external dataset was collected by other "
    "authors with another method, so this is a test under distribution shift.",
    "The hardware rows come from one run on one backend; shot noise and device drift change "
    "kernel values between runs.",
    "No claim of quantum advantage is made.",
]
CALIBRATION_DEVIATION = (
    "Protocol deviation: the fresh kernel calibration fit on this run's 24 training pairs was "
    "invalid, so the calibrated hardware row could not be computed and only five of the six "
    "frozen rows are reported."
)
REHEARSAL_LIMITATION = (
    "These values come from a noisy simulation on a local fake backend; no IBM Quantum "
    "hardware job was run and they are not hardware results."
)
_HEX = re.compile(r"[0-9a-f]{64}")
_GROUP_HASH = re.compile(r"[0-9a-f]{16}")
# The same credential pattern tests/unit/test_qpu_evidence.py applies to reports/qpu-evidence/.
EVIDENCE_SECRET = re.compile(r"crn:|token|api[_-]?key|instance", re.IGNORECASE)


class ExternalQpuError(Exception):
    """The subset protocol, its inputs, or the report break their rules."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ExternalQpuError(message)


def load_protocol(path: Path = PROTOCOL_PATH, parent_path: Path = PARENT_PATH) -> dict:
    """The frozen subset protocol, checked against its parent and the hardware tooling."""
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExternalQpuError("The subset protocol is missing.") from error
    _require(protocol.get("schema_version") == 1, "The subset protocol must be schema 1.")
    parent = protocol["parent_protocol"]
    _require(
        external.protocol_sha256(parent_path) == parent["sha256"],
        "The parent protocol differs from the one the subset protocol was frozen against.",
    )
    subset = protocol["subset"]
    _require(
        list(subset["per_class"]) == list(CLASSES)
        and sum(subset["per_class"].values()) == subset["size"],
        "subset.per_class must split subset.size into legitimate and phishing.",
    )
    settings = protocol["hardware"]
    _require(
        settings["shots"] == hardware.SHOTS
        and settings["optimization_level"] == hardware.OPTIMIZATION_LEVEL
        and settings["seed_transpiler"] == hardware.SEED
        and settings["sampler"] == SAMPLER
        and settings["batch_size"] == BATCH_SIZE,
        "The hardware tooling no longer matches the frozen shots, transpiler, sampler, or "
        "batch size.",
    )
    _require(
        [(entry["model"], entry["kernel_source"]) for entry in protocol["compared"]]
        == list(COMPARED),
        "The compared rows differ from the six frozen rows.",
    )
    _require(list(protocol["metrics"]["reported"]) == list(METRICS), "Metrics differ.")
    return protocol


def protocol_reference(path: Path, protocol: dict, root: Path = ROOT) -> dict:
    """Relative path, LF-normalized SHA-256, and freeze time of a protocol file."""
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": external.protocol_sha256(path),
        "frozen_at_utc": protocol["frozen_at_utc"],
    }


@dataclass(frozen=True)
class Subset:
    """Chosen rows by primary-set position, file row, and hashed group, with labels."""

    positions: np.ndarray
    row_index: list[int]
    group_hash: list[str]
    labels: np.ndarray

    @property
    def class_counts(self) -> dict[str, int]:
        phishing = int(self.labels.sum())
        return {"legitimate": len(self.labels) - phishing, "phishing": phishing}


def select_subset(rows: pd.DataFrame, per_class: dict[str, int], seed: int = SUBSET_SEED) -> Subset:
    """The frozen draw: one generator, legitimate then phishing, each by rng.choice over that
    class's primary-set positions without replacement; then sorted by position.

    `rows` is the primary set in file order with row_index, is_phishing, and group columns.
    """
    labels = rows["is_phishing"].to_numpy()
    rng = np.random.default_rng(seed)
    chosen = []
    for label, name in enumerate(CLASSES):
        positions = np.flatnonzero(labels == label)
        size = per_class[name]
        _require(len(positions) >= size, f"The primary set has fewer than {size} {name} rows.")
        chosen.append(rng.choice(positions, size=size, replace=False))
    positions = np.sort(np.concatenate(chosen))
    return Subset(
        positions=positions,
        row_index=[int(value) for value in rows["row_index"].to_numpy()[positions]],
        group_hash=[group_hash(value) for value in rows["group"].to_numpy()[positions]],
        labels=labels[positions],
    )


def projected_billed_seconds(circuits: int) -> float:
    return circuits * BILLED_SECONDS_PER_CIRCUIT


def required_quota(circuits: int) -> float:
    """The remaining quota the budget guard demands: projected billed time plus 60 s."""
    return projected_billed_seconds(circuits) + GUARD_MARGIN_SECONDS


def budget_refusal(remaining: object, circuits: int) -> str | None:
    """Why the guard refuses submission, or None when the remaining quota exceeds it."""
    required = required_quota(circuits)
    if not isinstance(remaining, int | float) or isinstance(remaining, bool):
        return "The instance did not report its remaining quota; nothing was submitted."
    if not remaining > required:
        return (
            f"The instance has {remaining} s remaining; the budget guard requires more than "
            f"{required:.1f} s ({circuits} circuits at {BILLED_SECONDS_PER_CIRCUIT} billed s "
            f"each plus {GUARD_MARGIN_SECONDS:.0f} s). Nothing was submitted."
        )
    return None


def evidence_refusal(texts: dict[str, str]) -> str | None:
    """Why evidence files (name -> text) must not be published, or None when they are clean."""
    for name, text in texts.items():
        if EVIDENCE_SECRET.search(text):
            return f"{name} looks like it holds a credential or instance identifier."
        if external._FORBIDDEN_TEXT.search(text):
            return f"{name} holds a URL or a local path."
    return None


def plan_counts(plan: list[hardware.PairSpec]) -> dict[str, int]:
    return {
        kind: sum(spec.kind == kind for spec in plan)
        for kind in ("test", "calibration", "diagonal")
    }


def rows_with_intervals(
    evaluations: list[tuple[str, str, dict]], labels: np.ndarray, settings: dict
) -> tuple[list[dict], int]:
    """Table rows with bootstrap intervals; every row shares the same resampled indices."""
    indices = external.resample_indices(len(labels), settings["resamples"], settings["seed"])
    rows, used = [], 0
    for model, source, evaluation in evaluations:
        scores = evaluation["decision_scores"]
        intervals, used = external.bootstrap_intervals(
            labels,
            np.asarray(evaluation["predictions"]),
            None if scores is None else np.asarray(scores),
            indices,
            settings["level"],
        )
        rows.append(
            {
                "name": NAMES[model],
                "model": model,
                "kernel_source": source,
                "metrics": evaluation["metrics"],
                "confusion_matrix": evaluation["confusion_matrix"],
                "confidence_intervals": intervals,
            }
        )
    return rows, used


def with_runtime(rows: list[dict], measured: dict, usage: dict) -> list[dict]:
    """Attach runtime_seconds from src.evaluation.runtime.measure; hardware rows take
    their test-kernel time from IBM's recorded execution time."""
    timed = []
    for row in rows:
        source = row["kernel_source"]
        seconds = dict(measured[row["model"] if source == "classical" else source])
        if source in HARDWARE_SOURCES:
            seconds["test_kernel"] = usage.get("execution_seconds")
        timed.append({**row, "runtime_seconds": {key: seconds[key] for key in RUNTIME_KEYS}})
    return timed


def build_report(
    *,
    source: str,
    protocol: dict,
    parent_protocol: dict,
    benchmark: dict,
    subset: dict,
    hardware_run: dict,
    kernel: dict,
    runtime: dict,
    bootstrap: dict,
    models: list[dict],
    limitations: list[str] | None = None,
) -> dict:
    extra = [REHEARSAL_LIMITATION] if source == "noisy_simulation" else []
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "label": SOURCES.get(source, ""),
        "protocol": protocol,
        "parent_protocol": parent_protocol,
        "benchmark": benchmark,
        "subset": subset,
        "hardware": hardware_run,
        "kernel": kernel,
        "runtime": runtime,
        "bootstrap": bootstrap,
        "models": models,
        "limitations": extra + list(LIMITATIONS) + list(limitations or []),
    }
    return validate(report)


def _is_int(value: object, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value)


def _is_unit(value: object) -> bool:
    return _is_number(value) and 0 <= value <= 1


def _is_seconds(value: object) -> bool:
    return value is None or _is_number(value) and value >= 0


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _validate_reference(name: str, value: object, frozen: bool) -> None:
    _require(
        isinstance(value, dict)
        and _is_text(value.get("path"))
        and isinstance(value.get("sha256"), str)
        and _HEX.fullmatch(value["sha256"])
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
        not any(external._FORBIDDEN_TEXT.search(text) for text in _strings(report)),
        "The report must not contain URLs or local paths.",
    )
    return report


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validate(report), indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as error:
        raise ExternalQpuError("The external subset has no hardware results yet.") from error
    except json.JSONDecodeError as error:
        raise ExternalQpuError(f"Report is not valid JSON: {error.msg}.") from error
