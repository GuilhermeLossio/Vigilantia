"""Training-only model search and checkpointed benchmark stages."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import time
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.benchmark_v2 import core
from src.evaluation.external import phiusiil_groups
from src.evaluation.external import prepare as prepare_external

ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict:
    config = load_json(path)
    required = set(load_json(ROOT / "configs/benchmark-v2.json"))
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("Invalid v2 configuration fields or schema version.")
    sizes = config["train_sizes"]
    if not sizes or sizes != sorted(set(sizes)) or any(n < 6 or n % 2 for n in sizes):
        raise ValueError("Training sizes must be sorted, unique, even and >= 6.")
    if not set(config["quantum_train_sizes"]).issubset(sizes):
        raise ValueError("Quantum sizes must be a subset of classical sizes.")
    if not config["quantum_train_sizes"]:
        raise ValueError("At least one quantum training size is required.")
    if config["folds"] < 2 or min(sizes) // 2 < config["folds"]:
        raise ValueError("Too few training examples for stratified folds.")
    if not 0 < config["max_fpr"] < 1 or not 0 < config["target_recall"] <= 1:
        raise ValueError("Invalid recall/FPR targets.")
    if not 0 < config["target_accuracy"] <= 1:
        raise ValueError("Invalid accuracy target.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and unique.")
    if config["bootstrap_resamples"] < 1 or config["forest_trees"] < 1:
        raise ValueError("Bootstrap resamples and forest trees must be positive.")
    if set(config["quantum_maps"]) != {"z", "zz"}:
        raise ValueError("Both z and zz maps are required.")
    if not config["angle_maxima"] or any(v <= 0 for v in config["angle_maxima"]):
        raise ValueError("Positive angle maxima are required.")
    if not config["quantum_c"] or any(v <= 0 for v in config["quantum_c"]):
        raise ValueError("Positive quantum C values are required.")
    if config["n_jobs"] == 0:
        raise ValueError("n_jobs cannot be zero.")
    if set(config["grids"]) != {"logistic_regression", "rbf_svm", "random_forest"}:
        raise ValueError("All three classical search grids are required.")
    for grid in config["grids"].values():
        list(ParameterGrid(grid))  # reject empty parameter lists before any preparation
    return config


def rank_page_features(rows: pd.DataFrame, seed: int) -> list[dict]:
    """Development-only permutation ranking of page candidates on an inner holdout."""
    x = np.log1p(rows[list(core.PAGE_CANDIDATES)].to_numpy(float))
    y = rows.is_phishing.to_numpy()
    fit_x, held_x, fit_y, held_y = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=seed
    )
    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(fit_x, fit_y)
    importance = permutation_importance(
        model, held_x, held_y, n_repeats=5, random_state=seed
    ).importances_mean
    order = sorted(range(len(importance)), key=lambda i: (-importance[i], i))
    return [{"feature": core.PAGE_CANDIDATES[i], "importance": float(importance[i])} for i in order]


def scores(model, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(values)
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


def estimator(name: str, seed: int, config: dict):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["forest_trees"], random_state=seed, n_jobs=config["n_jobs"]
        )
    raise ValueError(f"Unknown estimator: {name}")


def search_classical(name: str, x: np.ndarray, y: np.ndarray, seed: int, config: dict) -> tuple:
    base = estimator(name, seed, config)
    candidates = [{}] if name == "majority" else list(ParameterGrid(config["grids"][name]))
    trials = []
    for params in candidates:
        fitted_params = params
        if name in {"logistic_regression", "rbf_svm"}:
            prefix = "logisticregression" if name == "logistic_regression" else "svc"
            fitted_params = {f"{prefix}__{key}": value for key, value in params.items()}
        aps, fit_seconds = [], 0.0
        for train, validation in StratifiedKFold(
            config["folds"], shuffle=True, random_state=seed
        ).split(x, y):
            model = clone(base).set_params(**fitted_params)
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x[train], y[train])
            fit_seconds += time.perf_counter() - started
            aps.append(float(average_precision_score(y[validation], scores(model, x[validation]))))
        trials.append(
            {
                "params": params,
                "fitted_params": fitted_params,
                "fold_average_precision": aps,
                "mean_ap": float(np.mean(aps)),
                "fit_seconds": fit_seconds,
            }
        )
    chosen = min(trials, key=lambda row: (-row["mean_ap"], row["fit_seconds"]))
    model = clone(base).set_params(**chosen["fitted_params"])
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model, trials, chosen["params"], time.perf_counter() - started


def search_quantum(x: np.ndarray, y: np.ndarray, encoding: str, seed: int, config: dict) -> tuple:
    trials = []
    for maximum in config["angle_maxima"]:
        folds = []
        for train, validation in StratifiedKFold(
            config["folds"], shuffle=True, random_state=seed
        ).split(x, y):
            started = time.perf_counter()
            low, high = x[train].min(axis=0), x[train].max(axis=0)
            a, train_clip = core.angles(x[train], low, high, maximum)
            b, val_clip = core.angles(x[validation], low, high, maximum)
            train_states, val_states = core.states(a, encoding), core.states(b, encoding)
            folds.append(
                (
                    train,
                    validation,
                    core.kernel(train_states, train_states),
                    core.kernel(val_states, train_states),
                    time.perf_counter() - started,
                    train_clip,
                    val_clip,
                )
            )
        for c in config["quantum_c"]:
            aps, elapsed = [], 0.0
            for train, validation, k_train, k_validation, seconds, _, _ in folds:
                model = SVC(kernel="precomputed", C=c)
                started = time.perf_counter()
                model.fit(k_train, y[train])
                elapsed += seconds + time.perf_counter() - started
                aps.append(
                    float(
                        average_precision_score(
                            y[validation], model.decision_function(k_validation)
                        )
                    )
                )
            trials.append(
                {
                    "params": {"angle_max": maximum, "C": c, "encoding": encoding},
                    "fold_average_precision": aps,
                    "mean_ap": float(np.mean(aps)),
                    "fit_seconds": elapsed,
                    "fold_clipping": [{"train": f[5], "validation": f[6]} for f in folds],
                }
            )
    chosen = min(trials, key=lambda row: (-row["mean_ap"], row["fit_seconds"]))
    params = chosen["params"]
    started = time.perf_counter()
    low, high = x.min(axis=0), x.max(axis=0)
    a, _ = core.angles(x, low, high, params["angle_max"])
    reference = core.states(a, encoding)
    matrix = core.kernel(reference, reference)
    model = SVC(kernel="precomputed", C=params["C"]).fit(matrix, y)
    bundle = {"model": model, "low": low, "high": high, "states": reference, **params}
    return bundle, trials, params, time.perf_counter() - started, core.kernel_diagnostics(matrix, y)


def bundle_scores(bundle: dict, rows: pd.DataFrame) -> tuple:
    x = core.matrix(rows, bundle["features"])
    started = time.perf_counter()
    clipping = None
    if bundle["family"] == "quantum":
        a, clipping = core.angles(x, bundle["low"], bundle["high"], bundle["angle_max"])
        # Bound memory for large held-out partitions.
        output = np.concatenate(
            [
                scores(
                    bundle["model"],
                    core.kernel(core.states(chunk, bundle["encoding"]), bundle["states"]),
                )
                for chunk in np.array_split(a, max(1, (len(a) + 2047) // 2048))
            ]
        )
    else:
        output = scores(bundle["model"], x)
    return output, time.perf_counter() - started, clipping


class Experiment:
    """Stages lock source/data/environment identity and never evaluate test during search."""

    def __init__(self, config: dict, root: Path = ROOT):
        self.config, self.root = config, root
        self.output = root / config["paths"]["output"]
        for key in ("output", "report", "charts"):
            path = (root / config["paths"][key]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Output paths must remain inside the project root.")
        protected = [root / "artifacts/benchmark", root / "configs", root / "data"]
        resolved = self.output.resolve()
        if any(resolved == p.resolve() or resolved.is_relative_to(p.resolve()) for p in protected):
            raise ValueError("Output directory overlaps historical artifacts or input data.")
        report = (root / config["paths"]["report"]).resolve()
        if report.parent != (root / "reports").resolve() or not report.name.startswith(
            "benchmark-v2"
        ):
            raise ValueError("Report must be a benchmark-v2*.json file inside reports/.")
        if report.suffix != ".json":
            raise ValueError("Report must use the .json extension.")
        if (root / config["paths"]["charts"]).resolve() != (root / "docs").resolve():
            raise ValueError("Charts must be written to docs/.")

    def identity(self) -> dict:
        inputs = {}
        for key, manifest_path, expected_key in (
            ("phiusiil", "configs/benchmark-split.json", "csv_sha256"),
            ("external", "configs/external-validation.json", "sha256"),
        ):
            path = self.root / self.config["paths"][key]
            expected = load_json(self.root / manifest_path)["dataset"][expected_key]
            try:
                actual = core.digest(path)
            except OSError as error:
                raise ValueError(
                    f"Cannot read {key} dataset; restore access or run its downloader."
                ) from error
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {key}; restore the verified snapshot.")
            inputs[key] = actual
        sources = list(Path(__file__).parent.glob("*.py")) + [
            ROOT / "scripts/run_benchmark_v2.py",
            ROOT / "src/data/benchmark_split.py",
            ROOT / "src/features/host_features.py",
            ROOT / "src/evaluation/external.py",
            ROOT / "src/features/scaling.py",
            ROOT / "src/inference/predictor.py",
            self.root / "configs/benchmark-split.json",
            self.root / "configs/external-validation.json",
            self.root / "reports/external-validation.json",
        ]
        return {
            "config": self.config,
            "data": inputs,
            "sources": {
                str(p.relative_to(ROOT)).replace("\\", "/"): core.digest(p) for p in sources
            },
            "environment": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("numpy", "pandas", "scipy", "scikit-learn", "qiskit", "tldextract")
                },
            },
        }

    def check(self) -> dict:
        current = self.identity()
        saved = load_json(self.output / "identity.json")
        if current != saved:
            raise ValueError(
                "Run identity changed (data/config/source/environment); use a new output directory."
            )
        prepared = load_json(self.output / "partitions.json")
        for name, expected in prepared["files"].items():
            if core.digest(self.output / name) != expected:
                raise ValueError(f"Prepared partition changed: {name}.")
        return saved

    def prepare(self) -> None:
        identity = self.identity()
        if (self.output / "identity.json").exists():
            self.check()
            print("prepare: compatible checkpoint reused", flush=True)
            return
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise ValueError(
                "Output contains an incomplete preparation; choose a new output directory."
            )
        raw = pd.read_csv(self.root / self.config["paths"]["phiusiil"], keep_default_na=False)
        old = load_json(self.root / "configs/benchmark-split.json")
        excluded = {row["group_hash"] for row in old["test"]}
        parts, exclusions = core.partitions(raw, excluded, self.config["split_seed"])
        for rows in parts.values():
            page = raw.iloc[rows.row_index][list(core.PAGE_CANDIDATES)].astype(float)
            rows[list(core.PAGE_CANDIDATES)] = page.to_numpy()
        ranking = rank_page_features(parts["development"], self.config["split_seed"])
        for seed in self.config["seeds"]:
            core.nested_samples(parts["development"], self.config["train_sizes"], seed)
        external_config = load_json(self.root / "configs/external-validation.json")["dataset"]
        external_raw = pd.read_csv(
            self.root / self.config["paths"]["external"], keep_default_na=False
        )
        external = prepare_external(
            external_raw, external_config["label_map"], phiusiil_groups(raw.Domain)
        )
        from src.features.host_features import host_features

        external.rows = pd.concat(
            [external.rows, pd.DataFrame([host_features(h) for h in external.rows.host])], axis=1
        )
        external.rows["group_hash"] = external.rows.group.map(core.group_hash)
        parts["external"] = external.rows
        manifest = {
            "schema_version": 1,
            "exclusions": exclusions,
            "external_exclusions": external.exclusions,
            "page_feature_ranking": ranking,
            "page_features": list(core.PAGE_FEATURES),
            "partitions": {},
            "files": {},
        }
        for name, rows in parts.items():
            # Store hashes and features, never hosts, even in the prepared tables. The
            # external dataset has no PhiUSIIL page columns, so it keeps host features only.
            page = [] if name == "external" else list(core.PAGE_FEATURES)
            table = rows[
                ["row_index", "group_hash", "is_phishing", *core.CANDIDATE_FEATURES, *page]
            ]
            filename = f"{name}.csv"
            table.to_csv(self.output / filename, index=False)
            manifest["files"][filename] = core.digest(self.output / filename)
            manifest["partitions"][name] = {
                "rows": len(rows),
                "phishing": int(rows.is_phishing.sum()),
                "membership_sha256": hashlib.sha256(
                    "\n".join(rows.group_hash).encode()
                ).hexdigest(),
            }
        core.write_json(self.output / "partitions.json", manifest)
        core.write_json(self.output / "identity.json", identity)
        print("prepare: verified datasets and saved disjoint partitions", flush=True)

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.output / f"{name}.csv", float_precision="round_trip")

    def expected_ids(self, family: str) -> list[str]:
        if family == "classical":
            combinations = [
                (n, f, m)
                for n in self.config["train_sizes"]
                for f in core.FEATURE_SETS
                for m in ("majority", "logistic_regression", "rbf_svm", "random_forest")
            ]
        else:
            combinations = [
                (n, "page", m)
                for n in self.config["quantum_train_sizes"]
                for m in self.config["quantum_maps"]
            ]
        return [
            f"{family}-{m}-{f}-{n}-{s}" for n, f, m in combinations for s in self.config["seeds"]
        ]

    def train(self, family: str) -> None:
        self.check()
        if (self.output / "frozen-selection.json").exists():
            self.records()  # verify all complete checkpoints, never reopen a frozen search
            print(f"{family}: frozen, completed search reused", flush=True)
            return
        development, calibration = self.table("development"), self.table("calibration")
        samples = {
            s: core.nested_samples(development, self.config["train_sizes"], s)
            for s in self.config["seeds"]
        }
        runs = self.output / "runs"
        runs.mkdir(exist_ok=True)
        ids = self.expected_ids(family)
        for i, run_id in enumerate(ids):
            checkpoint = runs / f"{run_id}.json"
            if checkpoint.exists() and load_json(checkpoint)["status"] == "complete":
                record = load_json(checkpoint)
                for filename, expected in record["artifact_hashes"].items():
                    if core.digest(runs / filename) != expected:
                        raise ValueError(f"Run artifact changed: {filename}")
                continue
            _, name, features, size, seed = run_id.split("-")
            size, seed = int(size), int(seed)
            rows = development.iloc[samples[seed][size]]
            x = core.matrix(rows, features)
            y = rows.is_phishing.to_numpy()
            record = {
                "id": run_id,
                "family": family,
                "model": name,
                "features": features,
                "train_size": size,
                "seed": seed,
                "status": "running",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "training_group_hashes": rows.group_hash.tolist(),
            }
            core.write_json(checkpoint, record)
            started = time.perf_counter()
            try:
                if family == "classical":
                    model, trials, params, fit_time = search_classical(
                        name, x, y, seed, self.config
                    )
                    bundle = {"model": model}
                else:
                    bundle, trials, params, fit_time, diagnostics = search_quantum(
                        x, y, name, seed, self.config
                    )
                    record["kernel_diagnostics"] = diagnostics
                bundle.update({"family": family, "features": features})
                calibration_scores, score_time, clipping = bundle_scores(bundle, calibration)
                labels = calibration.is_phishing.to_numpy()
                point = core.operating_point(labels, calibration_scores, self.config["max_fpr"])
                if name == "majority":
                    # The most-frequent-class baseline is not threshold-optimized.
                    point["threshold"] = 0.5
                predictions = core.predict(calibration_scores, point["threshold"])
                bundle["threshold"] = point["threshold"]
                with (runs / f"{run_id}.pkl").open("wb") as stream:
                    pickle.dump(bundle, stream)
                np.save(runs / f"{run_id}-calibration.npy", calibration_scores)
                record.update(
                    {
                        "status": "complete",
                        "params": params,
                        "search": trials,
                        "threshold": point["threshold"],
                        "calibration": core.metrics(labels, predictions, calibration_scores),
                        "calibration_clipping": clipping,
                        "fit_seconds": fit_time,
                        "calibration_predict_seconds": score_time,
                        "total_seconds": time.perf_counter() - started,
                        "artifact_hashes": {
                            filename: core.digest(runs / filename)
                            for filename in (f"{run_id}.pkl", f"{run_id}-calibration.npy")
                        },
                    }
                )
                core.write_json(checkpoint, record)
            except Exception as error:
                core.write_json(
                    checkpoint, {**record, "status": "failed", "error_type": type(error).__name__}
                )
                raise
            print(f"{family}: {i + 1}/{len(ids)} {run_id} complete", flush=True)

    def records(self) -> list[dict]:
        ids = self.expected_ids("classical") + self.expected_ids("quantum")
        rows = []
        for run_id in ids:
            path = self.output / "runs" / f"{run_id}.json"
            if not path.exists() or load_json(path)["status"] != "complete":
                raise ValueError(f"Missing completed experiment: {run_id}")
            row = load_json(path)
            for filename, expected in row["artifact_hashes"].items():
                if core.digest(self.output / "runs" / filename) != expected:
                    raise ValueError(f"Run artifact changed: {filename}")
            rows.append(row)
        return rows
