"""Frozen IBM Quantum check of the v2 page-feature quantum kernel (configs/benchmark-v2-qpu-subset.json).

The protocol fixes one trained v2 bundle, a 40-row draw from the v2 internal test partition,
and the hardware plan. Inputs are rebuilt from the hash-checked v2 artifacts: training angles
come from the run's recorded training groups and must reproduce the bundle's stored states,
so the circuits measure exactly the kernel the model was trained on. Evaluation uses each
run's frozen v2 threshold; nothing is retrained or recalibrated after measurement, except the
protocol's linear kernel calibration on the run's own training pairs.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.evaluation import external, runtime
from src.evaluation.benchmark_v2 import core
from src.evaluation.benchmark_v2.experiment import bundle_scores
from src.quantum import hardware, qpu_run
from src.quantum import kernel as qkernel

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "benchmark-v2-qpu-subset.json"
REPORT_PATH = ROOT / "reports" / "benchmark-v2-qpu-subset.json"
EVIDENCE_DIR = ROOT / "reports" / "qpu-evidence-v2"
PREFIX = "v2page"
CLASSES = ("legitimate", "phishing")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "false_positive_rate",
    "f1",
    "average_precision",
)
INTERVAL_METRICS = ("accuracy", "recall", "false_positive_rate")
COMPARED = (
    ("logistic_regression", "classical"),
    ("rbf_svm", "classical"),
    ("random_forest", "classical"),
    ("quantum_kernel_svm", "simulation"),
    ("quantum_kernel_svm", "qpu"),
    ("quantum_kernel_svm", "qpu_calibrated"),
)
MIN_ROWS = 4
LIMITATIONS = [
    "Rows come from the v2 internal PhiUSIIL test partition, which shares the training data's collection process; page features largely reflect that process.",
    "One hardware run of one fitted seed; it measures kernel fidelity on hardware, not general phishing detection.",
]


class V2QpuError(Exception):
    """The protocol, artifacts, or run do not match."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise V2QpuError(message)


def load_protocol(path: Path = PROTOCOL_PATH, root: Path = ROOT) -> dict:
    """The frozen protocol, checked against the v2 configuration and the hardware tooling."""
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise V2QpuError("The v2 QPU protocol is missing.") from error
    _require(protocol.get("schema_version") == 1, "The v2 QPU protocol must be schema 1.")
    benchmark = protocol["benchmark"]
    _require(
        external.protocol_sha256(root / benchmark["config"]) == benchmark["config_sha256"],
        "The v2 configuration differs from the one the protocol was frozen against.",
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
        and settings["sampler"] == qpu_run.SAMPLER,
        "The hardware tooling no longer matches the frozen shots, transpiler, or sampler.",
    )
    _require(
        [(entry["model"], entry["kernel_source"]) for entry in protocol["compared"]]
        == list(COMPARED),
        "The compared rows differ from the six frozen rows.",
    )
    _require(list(protocol["metrics"]["reported"]) == list(METRICS), "Metrics differ.")
    return protocol


def select_subset(
    labels: np.ndarray, per_class: dict[str, int], seed: int, take: int | None = None
) -> np.ndarray:
    """The frozen draw: one generator, legitimate then phishing, sorted by position.

    `take` keeps the first `take` rows of each class in draw order, so every smaller subset
    is a prefix of the full frozen draw rather than a new one.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for label, name in enumerate(CLASSES):
        positions = np.flatnonzero(labels == label)
        _require(len(positions) >= per_class[name], f"Too few {name} rows for the subset.")
        chosen.append(rng.choice(positions, size=per_class[name], replace=False)[:take])
    return np.sort(np.concatenate(chosen))


def check_rows(protocol: dict, rows: int) -> None:
    size = protocol["subset"]["size"]
    _require(
        rows % 2 == 0 and MIN_ROWS <= rows <= size,
        f"Rows must be even, between {MIN_ROWS} and {size}.",
    )


def max_rows(protocol: dict, n_train: int, remaining_seconds: float) -> int | None:
    """The largest allowed subset whose plan passes the budget guard, or None."""
    from src.evaluation import external_qpu

    for rows in range(protocol["subset"]["size"], MIN_ROWS - 1, -2):
        circuits = len(hardware.plan_pairs(n_train, rows))
        if remaining_seconds > external_qpu.required_quota(circuits):
            return rows
    return None


def _load_bundle(runs: Path, run_id: str, expected: str | None = None) -> tuple[dict, dict]:
    """A completed run record and its bundle, hash-checked before unpickling."""
    record = json.loads((runs / f"{run_id}.json").read_text(encoding="utf-8"))
    _require(record.get("status") == "complete", f"Run {run_id} is not complete.")
    path = runs / f"{run_id}.pkl"
    actual = core.digest(path)
    _require(actual == record["artifact_hashes"][path.name], f"Bundle {path.name} changed.")
    _require(expected in {None, actual}, f"Bundle {path.name} differs from the frozen hash.")
    with path.open("rb") as stream:
        return record, pickle.load(stream)  # trusted local artifact, hash checked above


@dataclass
class Inputs:
    """Everything a run is planned and evaluated from."""

    protocol: dict
    reference: dict
    bundle: dict
    classical: dict[str, dict]
    rows: pd.DataFrame
    train_angles: np.ndarray
    test_angles: np.ndarray
    clipping: dict
    plan: list[hardware.PairSpec]
    train_x: np.ndarray
    train_labels: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        return self.rows.is_phishing.to_numpy()

    @property
    def inputs(self) -> dict:
        """The run record's inputs: everything a later evaluation must find unchanged."""
        benchmark = self.protocol["benchmark"]
        return {
            "protocol_sha256": self.reference["sha256"],
            "config_sha256": benchmark["config_sha256"],
            "test_partition_sha256": benchmark["test_partition_sha256"],
            "quantum_bundle_sha256": benchmark["quantum_bundle_sha256"],
            "subset_row_index": [int(v) for v in self.rows.row_index],
            "subset_group_hash": list(self.rows.group_hash),
        }


def load_inputs(
    protocol: dict, root: Path = ROOT, path: Path = PROTOCOL_PATH, rows: int | None = None
) -> Inputs:
    """The frozen inputs; `rows` reduces the subset to a prefix of the frozen draw."""
    benchmark = protocol["benchmark"]
    subset = protocol["subset"]
    rows = subset["size"] if rows is None else rows
    check_rows(protocol, rows)
    output = root / benchmark["output"]
    runs = output / "runs"
    test_path = output / "test.csv"
    _require(test_path.exists(), f"Missing {benchmark['output']}/test.csv; run benchmark v2.")
    _require(
        core.digest(test_path) == benchmark["test_partition_sha256"],
        "The v2 test partition differs from the frozen one.",
    )
    record, bundle = _load_bundle(
        runs, benchmark["quantum_run"], benchmark["quantum_bundle_sha256"]
    )
    # The hardware circuits implement only the linear ZZ map of src.quantum.kernel.
    _require(
        bundle["encoding"] == "zz" and bundle["features"] == "page", "Expected a zz page model."
    )
    classical = {}
    for run_id in benchmark["classical_runs"]:
        _, model = _load_bundle(runs, run_id)
        classical[run_id.split("-")[1]] = model

    def table(name: str) -> pd.DataFrame:
        return pd.read_csv(output / f"{name}.csv")

    development, calibration, test = table("development"), table("calibration"), table("test")
    by_group = development.set_index("group_hash")
    train = by_group.loc[record["training_group_hashes"]]
    train_angles, _ = core.angles(
        core.matrix(train, "page"), bundle["low"], bundle["high"], bundle["angle_max"]
    )
    _require(
        np.allclose(core.states(train_angles, "zz"), bundle["states"], atol=1e-12),
        "Rebuilt training angles do not reproduce the bundle's states.",
    )
    positions = select_subset(
        test.is_phishing.to_numpy(), subset["per_class"], subset["seed"], rows // 2
    )
    chosen = test.iloc[positions].reset_index(drop=True)
    seen = set(development.group_hash) | set(calibration.group_hash)
    _require(not seen & set(chosen.group_hash), "Subset rows share domains with model data.")
    test_angles, clipping = core.angles(
        core.matrix(chosen, "page"), bundle["low"], bundle["high"], bundle["angle_max"]
    )
    full = len(hardware.plan_pairs(len(train_angles), subset["size"]))
    _require(
        full == protocol["hardware"]["circuits"],
        f"The full plan has {full} circuits; the protocol froze {protocol['hardware']['circuits']}.",
    )
    plan = hardware.plan_pairs(len(train_angles), rows)
    reference = {
        "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name,
        "sha256": external.protocol_sha256(path),
        "frozen_at_utc": protocol["frozen_at_utc"],
    }
    return Inputs(
        protocol,
        reference,
        bundle,
        classical,
        chosen,
        train_angles,
        test_angles,
        clipping,
        plan,
        core.matrix(train, "page"),
        train.is_phishing.to_numpy(),
    )


def bootstrap(labels: np.ndarray, rows: list[tuple], settings: dict) -> tuple[list[dict], int]:
    """Percentile intervals over shared resampled row indices; one-class draws are skipped."""
    rng = np.random.default_rng(settings["seed"])
    draws = rng.integers(0, len(labels), (settings["resamples"], len(labels)))
    draws = [d for d in draws if 0 < labels[d].sum() < len(d)]
    _require(draws, "Bootstrap has no two-class resamples.")
    level = settings["level"]
    bounds = [50 * (1 - level), 100 - 50 * (1 - level)]
    output = []
    for prediction, score in rows:
        values = np.array(
            [
                [core.metrics(labels[d], prediction[d], score[d])[m] for m in INTERVAL_METRICS]
                for d in draws
            ]
        )
        output.append(
            {
                m: np.percentile(values[:, i], bounds).tolist()
                for i, m in enumerate(INTERVAL_METRICS)
            }
        )
    return output, len(draws)


RUNTIME_METHOD = (
    "median wall-clock seconds over {repeats} repeats on one machine, for fitting on the "
    "training examples, computing the test kernel, and scoring the subset rows; quantum fit "
    "includes simulating the training kernel. Hardware rows take their test-kernel time from "
    "IBM's recorded execution time for all of the run's circuits, including calibration pairs "
    "and diagonal controls; queueing is excluded and both hardware rows share the same jobs."
)


def measure_runtime(
    loaded: Inputs, kernels: dict[str, np.ndarray], execution: float | None, repeats: int
) -> dict[str, dict]:
    """{kernel source or classical model: {"fit", "test_kernel", "predict"}} in seconds."""
    median = runtime.median_seconds
    output = {}
    for name, bundle in loaded.classical.items():
        output[name] = {
            "fit": median(
                lambda b=bundle: clone(b["model"]).fit(loaded.train_x, loaded.train_labels),
                repeats,
            ),
            "test_kernel": None,
            "predict": median(lambda b=bundle: bundle_scores(b, loaded.rows), repeats),
        }
    quantum = loaded.bundle["model"]

    def fit_quantum():
        states = core.states(loaded.train_angles, "zz")
        clone(quantum).fit(core.kernel(states, states), loaded.train_labels)

    fit = median(fit_quantum, repeats)
    for source, kernel in kernels.items():
        output[source] = {
            "fit": fit,
            "test_kernel": median(
                lambda: core.kernel(core.states(loaded.test_angles, "zz"), loaded.bundle["states"]),
                repeats,
            )
            if source == "simulation"
            else execution,
            "predict": median(lambda k=kernel: quantum.decision_function(k), repeats),
        }
    return output


def evaluate_run(
    run_dir: Path, record: dict, loaded: Inputs, source: str, repeats: int = runtime.REPEATS
) -> dict:
    """The six frozen rows on the subset, with kernel checks, intervals, and runtimes."""
    _require(record["inputs"] == loaded.inputs, "The run used other inputs than the protocol.")
    _require(record["status"] == "retrieved", f"Run status is {record['status']}; retrieve first.")
    plan = [hardware.PairSpec(**spec) for spec in record["plan"]]
    counts = qpu_run.collected_counts(run_dir, record)
    n_train, n_test = len(loaded.train_angles), len(loaded.test_angles)
    assembled = hardware.assemble(plan, counts, n_train, n_test)
    train_states = core.states(loaded.train_angles, "zz")
    ideal_train = core.kernel(train_states, train_states)
    ideal_test = core.kernel(core.states(loaded.test_angles, "zz"), train_states)
    pairs = assembled["calibration"]
    calibration = hardware.fit_calibration(
        np.array([ideal_train[i, j] for i, j, _ in pairs]), np.array([v for _, _, v in pairs])
    )
    kernels = {"simulation": ideal_test, "qpu": assembled["test_kernel"]}
    limitations = [
        f"{n_test} rows: each row changes accuracy by {100 / n_test:.1f} percentage points; "
        "intervals are wide.",
        *LIMITATIONS,
    ]
    if n_test < loaded.protocol["subset"]["size"]:
        limitations.append(
            f"The subset was reduced from {loaded.protocol['subset']['size']} to {n_test} rows "
            "before submission to fit the remaining QPU quota; it is the first half of each "
            "class in the frozen draw order."
        )
    if calibration["valid"]:
        kernels["qpu_calibrated"], calibration["clipped_test_values"] = hardware.apply_calibration(
            assembled["test_kernel"], calibration
        )
    else:
        limitations.append(f"Kernel calibration failed ({calibration['reason']}).")
    labels = loaded.labels
    scored = []
    for model, kernel_source in COMPARED:
        if kernel_source == "classical":
            bundle = loaded.classical[model]
            score, _, _ = bundle_scores(bundle, loaded.rows)
        elif kernel_source in kernels:
            bundle = loaded.bundle
            score = bundle["model"].decision_function(kernels[kernel_source])
        else:
            continue
        scored.append((model, kernel_source, core.predict(score, bundle["threshold"]), score))
    settings = loaded.protocol["metrics"]["confidence_intervals"]
    intervals, used = bootstrap(labels, [(p, s) for _, _, p, s in scored], settings)
    models = []
    usage = qpu_run.usage_totals(record)
    timings = measure_runtime(loaded, kernels, usage["execution_seconds"], repeats)
    for (model, kernel_source, prediction, score), interval in zip(scored, intervals, strict=True):
        measured = core.metrics(labels, prediction, score)
        models.append(
            {
                "model": model,
                "kernel_source": kernel_source,
                "metrics": {m: measured[m] for m in METRICS},
                "confusion_matrix": measured["confusion_matrix"],
                "confidence_intervals": interval,
                "runtime_seconds": timings[
                    model if kernel_source == "classical" else kernel_source
                ],
            }
        )
    measured_values = [qkernel.kernel_from_counts(c) for c in counts]

    def ideal(spec: hardware.PairSpec) -> float:
        if spec.kind == "test":
            return float(ideal_test[spec.left, spec.right])
        return 1.0 if spec.kind == "diagonal" else float(ideal_train[spec.left, spec.right])

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "protocol": loaded.reference,
        "inputs": loaded.inputs,
        "subset": {
            "size": n_test,
            "class_counts": {
                "legitimate": int((labels == 0).sum()),
                "phishing": int(labels.sum()),
            },
            "clipping": loaded.clipping,
        },
        "hardware": {
            "backend": record["backend"],
            "run_id": record["run_id"],
            "sampler": record["sampler"],
            "shots": record["shots"],
            "job_ids": [job["job_id"] for job in record["jobs"]],
            "circuits": len(plan),
            "shots_returned": sorted({sum(c.values()) for c in counts}),
            "usage": usage,
        },
        "kernel": {
            "mean_abs_circuit_error_vs_ideal": float(
                np.mean([abs(m - ideal(s)) for s, m in zip(plan, measured_values, strict=True)])
            ),
            "test_kernel_error_vs_ideal": {
                name: float(np.abs(kernel - ideal_test).mean())
                for name, kernel in kernels.items()
                if name != "simulation"
            },
            "ideal_test_kernel_spread": {
                "min": float(ideal_test.min()),
                "max": float(ideal_test.max()),
            },
            "calibration": calibration,
            "diagonal_controls_mean": float(np.mean([v for _, v in assembled["diagonal"]])),
        },
        "bootstrap": {**settings, "resamples_used": used},
        "runtime": {
            "method": RUNTIME_METHOD.format(repeats=repeats),
            "repeats": repeats,
            "environment": runtime.environment(),
        },
        "models": models,
        "limitations": limitations,
    }
    serialized = json.dumps(report, allow_nan=False)
    _require('"host":' not in serialized and '"URL":' not in serialized, "Report holds hosts.")
    return report


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
