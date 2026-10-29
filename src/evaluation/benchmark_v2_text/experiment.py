"""Verified hostname recovery, local checkpoints, training and frozen evaluations."""

from __future__ import annotations

import hashlib
import pickle
import platform
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.benchmark_v2 import core
from src.evaluation.benchmark_v2.experiment import load_json
from src.evaluation.benchmark_v2_text import features, quantum
from src.evaluation.external import phiusiil_groups
from src.evaluation.external import prepare as external_prepare

ROOT = Path(__file__).resolve().parents[3]


def load_config(path):
    config = load_json(path)
    required = set(load_json(ROOT / "configs/benchmark-v2-text.json"))
    if set(config) != required or config["schema_version"] != 1 or config["components"] != 4:
        raise ValueError("Invalid text configuration; this phase requires four components.")
    sizes = config["train_sizes"]
    if not sizes or sizes != sorted(set(sizes)) or any(n < 12 or n % 2 for n in sizes):
        raise ValueError("Training sizes must be unique, increasing, even and at least 12.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Training seeds must be unique and nonempty.")
    if config["folds"] < 2 or min(sizes) // 2 < config["folds"]:
        raise ValueError("Invalid fold count.")
    for name in ("shots", "batch_size", "subset_per_class", "bootstrap_resamples", "max_features"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 < config["max_fpr"] < 1 or not 0 < config["target_recall"] <= 1:
        raise ValueError("Invalid operating targets.")
    if set(config["maps"]) != {"z", "zz"}:
        raise ValueError("Both Z and ZZ maps are required.")
    if any(not config[k] for k in ("c_values", "gamma_values", "angle_maxima")):
        raise ValueError("Search lists must be nonempty.")
    if any(not np.isfinite(v) or v <= 0 for k in ("c_values", "angle_maxima") for v in config[k]):
        raise ValueError("C values and angle maxima must be finite and positive.")
    return config


def dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
    temporary.replace(path)


def load_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)  # only local bundles checked by their manifest


def membership(rows):
    return hashlib.sha256("\n".join(rows.group_hash).encode()).hexdigest()


def verify_recovery(rows, reference, saved=None):
    if len(rows) != reference["rows"] or int(rows.is_phishing.sum()) != reference["phishing"]:
        raise ValueError("Recovered row counts or labels differ from the v2 reference.")
    if membership(rows) != reference["membership_sha256"]:
        raise ValueError("Recovered domain membership differs from the v2 reference.")
    if saved is not None:
        for column in ("row_index", "group_hash", "is_phishing"):
            if rows[column].tolist() != saved[column].tolist():
                raise ValueError(f"Recovered {column} differs from saved v2 partition.")


class Experiment:
    def __init__(self, config, run_id, root=ROOT):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", run_id):
            raise ValueError("Run ID must contain lowercase letters, digits and hyphens (max 80).")
        self.config, self.run_id, self.root = config, run_id, Path(root)
        self.output = self.root / "artifacts/benchmark-v2-text" / run_id
        self.report_path = self.root / "reports/benchmark-v2-text" / f"{run_id}.json"
        self.charts = self.root / "docs/benchmark-v2-text"
        for field in ("reference_report", "reference_config"):
            if not (self.root / config[field]).resolve().is_relative_to(self.root.resolve()):
                raise ValueError("Reference paths must remain inside the repository.")

    def identity(self):
        ref_config = load_json(self.root / self.config["reference_config"])
        ref = load_json(self.root / self.config["reference_report"])
        snapshots = {}
        for key in ("phiusiil", "external"):
            path = self.root / ref_config["paths"][key]
            if core.digest(path) != ref["identity"]["data"][key]:
                raise ValueError(f"Verified {key} snapshot differs from the reference.")
            snapshots[key] = core.digest(path)
        source_paths = list(Path(__file__).parent.glob("*.py")) + [
            ROOT / "scripts/run_benchmark_v2_text.py",
            ROOT / "src/evaluation/benchmark_v2/core.py",
            ROOT / "src/evaluation/benchmark_v2/report.py",
            ROOT / "src/data/benchmark_split.py",
            ROOT / "src/features/host_features.py",
            ROOT / "src/evaluation/external.py",
            ROOT / "src/inference/predictor.py",
            ROOT / "src/features/scaling.py",
        ]
        return {
            "config": self.config,
            "run_id": self.run_id,
            "datasets": snapshots,
            "reference_report_sha256": core.digest(self.root / self.config["reference_report"]),
            "reference_config_sha256": core.digest(self.root / self.config["reference_config"]),
            "historical_split_sha256": core.digest(self.root / "configs/benchmark-split.json"),
            "external_protocol_sha256": core.digest(self.root / "configs/external-validation.json"),
            "sources": {p.relative_to(ROOT).as_posix(): core.digest(p) for p in source_paths},
            "environment": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in (
                        "numpy",
                        "pandas",
                        "scipy",
                        "scikit-learn",
                        "tldextract",
                        "qiskit",
                        "qiskit-aer",
                    )
                },
            },
        }

    def check(self):
        identity = load_json(self.output / "identity.json")
        if identity != self.identity():
            raise ValueError("Text run identity changed; choose a new run ID.")
        manifest = load_json(self.output / "partitions.json")
        for filename, expected in manifest["files"].items():
            if core.digest(self.output / filename) != expected:
                raise ValueError(f"Prepared artifact modified: {filename}")
        return identity

    def prepare(self):
        identity = self.identity()
        if (self.output / "identity.json").exists():
            self.check()
            print("prepare: verified checkpoint reused", flush=True)
            return
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError("Incomplete preparation exists; use a new run ID.")
        reference = load_json(self.root / self.config["reference_report"])
        ref_config = load_json(self.root / self.config["reference_config"])
        raw = pd.read_csv(self.root / ref_config["paths"]["phiusiil"], keep_default_na=False)
        old = load_json(self.root / "configs/benchmark-split.json")
        parts, exclusions = core.partitions(
            raw, {r["group_hash"] for r in old["test"]}, ref_config["split_seed"]
        )
        ext_config = load_json(self.root / "configs/external-validation.json")["dataset"]
        ext_raw = pd.read_csv(self.root / ref_config["paths"]["external"], keep_default_na=False)
        ext = external_prepare(ext_raw, ext_config["label_map"], phiusiil_groups(raw.Domain))
        from src.features.host_features import host_features

        ext.rows = pd.concat(
            [ext.rows, pd.DataFrame([host_features(h) for h in ext.rows.host])], axis=1
        )
        ext.rows["group_hash"] = ext.rows.group.map(core.group_hash)
        parts["external"] = ext.rows
        manifest = {
            "schema_version": 1,
            "files": {},
            "partitions": {},
            "exclusions": exclusions,
            "external_exclusions": ext.exclusions,
            "subsets": {},
            "training": {},
        }
        seen = set()
        self.output.mkdir(parents=True, exist_ok=True)
        for name, rows in parts.items():
            saved_path = self.root / ref_config["paths"]["output"] / f"{name}.csv"
            saved = pd.read_csv(saved_path) if saved_path.exists() else None
            verify_recovery(rows, reference["partitions"]["partitions"][name], saved)
            if seen.intersection(rows.group_hash):
                raise ValueError("Domain overlap in recovered partitions.")
            seen.update(rows.group_hash)
            columns = ["row_index", "group_hash", "is_phishing", "host", *core.CANDIDATE_FEATURES]
            filename = f"{name}.csv"
            rows[columns].to_csv(self.output / filename, index=False)
            manifest["files"][filename] = core.digest(self.output / filename)
            manifest["partitions"][name] = reference["partitions"]["partitions"][name]
            if name in {"test", "external"}:
                rng = np.random.default_rng(self.config["subset_seed"])
                indices = np.concatenate(
                    [
                        rng.choice(
                            np.flatnonzero(rows.is_phishing.to_numpy() == c),
                            self.config["subset_per_class"],
                            replace=False,
                        )
                        for c in (0, 1)
                    ]
                )
                manifest["subsets"][name] = indices.astype(int).tolist()
        development = parts["development"]
        for seed in self.config["seeds"]:
            manifest["training"][str(seed)] = {
                str(n): indices.astype(int).tolist()
                for n, indices in core.nested_samples(
                    development, self.config["train_sizes"], seed
                ).items()
            }
        core.write_json(self.output / "partitions.json", manifest)
        core.write_json(self.output / "identity.json", identity)
        print(
            "prepare: verified hostname recovery and fixed training/measurement subsets", flush=True
        )

    def table(self, name):
        return pd.read_csv(self.output / f"{name}.csv", keep_default_na=False)

    def training_rows(self, size, seed):
        manifest = load_json(self.output / "partitions.json")
        return (
            self.table("development")
            .iloc[manifest["training"][str(seed)][str(size)]]
            .reset_index(drop=True)
        )

    def unit(self, size, seed):
        directory = self.output / "features" / f"{size}-{seed}"
        bundle = directory / "representation.pkl"
        if bundle.exists() and (directory / "manifest.json").exists():
            expected = load_json(directory / "manifest.json")
            if core.digest(bundle) != expected["sha256"]:
                raise ValueError("Fitted representation modified.")
            return load_pickle(bundle), expected["sha256"]
        rep = features.Representation(seed, self.config["max_features"]).fit(
            self.training_rows(size, seed)
        )
        dump_pickle(bundle, rep)
        core.write_json(
            directory / "manifest.json", {"sha256": core.digest(bundle), **rep.diagnostics()}
        )
        return rep, core.digest(bundle)

    def folds(self, size, seed):
        rows = self.training_rows(size, seed)
        y = rows.is_phishing.to_numpy()
        result = []
        for number, (train, validation) in enumerate(
            StratifiedKFold(self.config["folds"], shuffle=True, random_state=seed).split(rows, y)
        ):
            path = self.output / "folds" / f"{size}-{seed}-{number}.pkl"
            meta = path.with_suffix(".json")
            if path.exists() and meta.exists():
                if core.digest(path) != load_json(meta)["sha256"]:
                    raise ValueError("Training fold cache modified.")
                item = load_pickle(path)
            else:
                rep = features.Representation(seed, self.config["max_features"]).fit(
                    rows.iloc[train]
                )
                a, clip_a = rep.transform(rows.iloc[train])
                b, clip_b = rep.transform(rows.iloc[validation])
                item = {
                    "train": a,
                    "validation": b,
                    "y_train": y[train],
                    "y_validation": y[validation],
                    "clipping": {"train": clip_a, "validation": clip_b},
                    "transform_hash": hashlib.sha256(pickle.dumps(rep)).hexdigest(),
                }
                dump_pickle(path, item)
                core.write_json(meta, {"sha256": core.digest(path)})
            result.append(item)
        return result

    def run_ids(self, family=None):
        names = (
            features.MODELS
            if family == "classical"
            else ("quantum",)
            if family == "statevector"
            else (*features.MODELS, "quantum")
        )
        return [
            f"{name}-{n}-{seed}"
            for n in self.config["train_sizes"]
            for seed in self.config["seeds"]
            for name in names
        ]

    def record(self, run_id):
        path = self.output / "runs" / f"{run_id}.json"
        record = load_json(path)
        if record["status"] != "complete":
            raise ValueError(f"Incomplete training run: {run_id}")
        for filename, expected in record["artifacts"].items():
            if core.digest(self.output / filename) != expected:
                raise ValueError(f"Training artifact modified: {filename}")
        return record

    def train(self, family):
        self.check()
        if (self.output / "selection.json").exists():
            for run_id in self.run_ids():
                self.record(run_id)
            print(f"{family}: frozen search reused", flush=True)
            return
        calibration = self.table("calibration")
        y_calibration = calibration.is_phishing.to_numpy()
        for run_id in self.run_ids(family):
            path = self.output / "runs" / f"{run_id}.json"
            if path.exists() and load_json(path)["status"] == "complete":
                self.record(run_id)
                continue
            name, size, seed = run_id.split("-")
            size, seed = int(size), int(seed)
            base = {
                "id": run_id,
                "model": name,
                "train_size": size,
                "seed": seed,
                "status": "running",
                "started_at_utc": datetime.now(UTC).isoformat(),
            }
            core.write_json(path, base)
            started = time.perf_counter()
            try:
                folds = self.folds(size, seed)
                trials = []
                cache = self.output / "cache/states"
                for params in features.candidates(name, self.config):
                    aps = []
                    for fold in folds:
                        if name == "quantum":
                            encoding, maximum = params["encoding"], params["angle_max"]
                            a = quantum.states(
                                fold["train"]["reduced"] * maximum,
                                encoding,
                                fold["transform_hash"],
                                cache,
                            )
                            b = quantum.states(
                                fold["validation"]["reduced"] * maximum,
                                encoding,
                                fold["transform_hash"],
                                cache,
                            )
                            x, v = core.kernel(a, a), core.kernel(b, a)
                        else:
                            x, v = (
                                fold["train"][features.view(name)],
                                fold["validation"][features.view(name)],
                            )
                        model = features.fit_model(
                            features.estimator(name, params, seed), x, fold["y_train"]
                        )
                        aps.append(
                            float(
                                average_precision_score(
                                    fold["y_validation"], features.scores(model, v)
                                )
                            )
                        )
                    trials.append(
                        {
                            "params": params,
                            "fold_average_precision": aps,
                            "mean_ap": float(np.mean(aps)),
                        }
                    )
                chosen = max(range(len(trials)), key=lambda i: trials[i]["mean_ap"])
                params = trials[chosen]["params"]
                rep, transform_hash = self.unit(size, seed)
                training = self.training_rows(size, seed)
                train_views, _ = rep.transform(training)
                cal_views, clipping = rep.transform(calibration)
                bundle = {
                    "name": name,
                    "size": size,
                    "seed": seed,
                    "transform_hash": transform_hash,
                    "params": params,
                }
                if name == "quantum":
                    reference = quantum.states(
                        train_views["reduced"] * params["angle_max"],
                        params["encoding"],
                        transform_hash,
                        cache,
                    )
                    x = core.kernel(reference, reference)
                    cal_x = quantum.cross_kernel(
                        cal_views["reduced"] * params["angle_max"],
                        reference,
                        params["encoding"],
                        transform_hash,
                        cache,
                    )
                    bundle["reference_states"] = reference
                    bundle["reference_angles"] = train_views["reduced"] * params["angle_max"]
                    base["kernel_diagnostics"] = core.kernel_diagnostics(
                        x, training.is_phishing.to_numpy()
                    )
                else:
                    x, cal_x = train_views[features.view(name)], cal_views[features.view(name)]
                bundle["model"] = features.fit_model(
                    features.estimator(name, params, seed), x, training.is_phishing.to_numpy()
                )
                score = features.scores(bundle["model"], cal_x)
                point = core.operating_point(y_calibration, score, self.config["max_fpr"])
                bundle["threshold"] = 0.5 if name == "majority" else point["threshold"]
                model_path = self.output / "runs" / f"{run_id}.pkl"
                dump_pickle(model_path, bundle)
                score_path = self.output / "runs" / f"{run_id}-calibration.npy"
                np.save(score_path, score)
                base.update(
                    {
                        "status": "complete",
                        "params": params,
                        "search": trials,
                        "selected_trial": chosen,
                        "threshold": bundle["threshold"],
                        "calibration": core.metrics(
                            y_calibration, core.predict(score, bundle["threshold"]), score
                        ),
                        "representation": rep.diagnostics(),
                        "calibration_clipping": clipping,
                        "fold_clipping": [fold["clipping"] for fold in folds],
                        "transform_hash": transform_hash,
                        "total_seconds": time.perf_counter() - started,
                        "artifacts": {
                            p.relative_to(self.output).as_posix(): core.digest(p)
                            for p in (model_path, score_path)
                        },
                    }
                )
                core.write_json(path, base)
                print(f"{family}: {run_id} complete", flush=True)
            except Exception as error:
                core.write_json(
                    path, {**base, "status": "failed", "error_type": type(error).__name__}
                )
                raise

    def freeze(self):
        self.check()
        records = [self.record(run_id) for run_id in self.run_ids()]

        def choose(names):
            groups = {}
            for row in records:
                if row["model"] not in names:
                    continue
                groups.setdefault((row["model"], row["train_size"]), []).append(row)
            chosen = min(
                groups,
                key=lambda k: (
                    -np.mean([r["calibration"]["recall"] for r in groups[k]]),
                    -np.mean([r["calibration"]["average_precision"] for r in groups[k]]),
                    k[1],
                    k[0],
                ),
            )
            return {"model": chosen[0], "train_size": chosen[1]}

        selected = {
            "classical": choose(set(features.MODELS) - {"majority"}),
            "quantum": choose({"quantum"}),
        }
        full = choose({"full_linear", "full_logistic"})
        value = {
            "selected": selected,
            "comparisons": {
                "text_benefit": [full, {"model": "numeric_rbf", "train_size": full["train_size"]}],
                "reduction_effect": [
                    {"model": "reduced_linear", "train_size": max(self.config["train_sizes"])},
                    {"model": "full_linear", "train_size": max(self.config["train_sizes"])},
                ],
                "kernel_effect": [
                    selected["quantum"],
                    {"model": "reduced_rbf", "train_size": selected["quantum"]["train_size"]},
                ],
            },
            "training_records": {
                r["id"]: core.digest(self.output / "runs" / f"{r['id']}.json") for r in records
            },
        }
        path = self.output / "selection.json"
        if path.exists() and load_json(path) != value:
            raise ValueError("Frozen selection changed.")
        if not path.exists():
            core.write_json(path, value)
        return records, value

    def load_bundle(self, record):
        self.record(record["id"])
        bundle = load_pickle(self.output / "runs" / f"{record['id']}.pkl")
        rep, expected = self.unit(record["train_size"], record["seed"])
        if expected != bundle["transform_hash"]:
            raise ValueError("Model/representation identity mismatch.")
        return bundle, rep

    def score(self, record, rows):
        bundle, rep = self.load_bundle(record)
        views, clipping = rep.transform(rows)
        if record["model"] == "quantum":
            params = bundle["params"]
            values = quantum.cross_kernel(
                views["reduced"] * params["angle_max"],
                bundle["reference_states"],
                params["encoding"],
                bundle["transform_hash"],
                self.output / "cache/states",
            )
        else:
            values = views[features.view(record["model"])]
        return features.scores(bundle["model"], values), clipping

    def shots(self):
        records, _ = self.freeze()  # no evaluation partition opened until all training is frozen
        manifest = load_json(self.output / "partitions.json")
        for record in records:
            if record["model"] != "quantum":
                continue
            bundle, rep = self.load_bundle(record)
            params = bundle["params"]
            for part in ("test", "external"):
                rows = self.table(part).iloc[manifest["subsets"][part]]
                directory = self.output / "shots" / record["id"] / part
                summary = directory / "evaluation.json"
                if summary.exists():
                    evidence = load_json(summary)
                    for filename, expected in evidence["evidence_hashes"].items():
                        if core.digest(directory / filename) != expected:
                            raise ValueError("Measured evidence modified.")
                    continue
                started = time.perf_counter()
                views, clipping = rep.transform(rows)
                measured = quantum.measured_kernel(
                    views["reduced"] * params["angle_max"],
                    bundle["reference_angles"],
                    params["encoding"],
                    self.config["shots"],
                    self.config["shot_seed"] + record["seed"],
                    self.config["batch_size"],
                    directory,
                )
                ideal_score, _ = self.score(record, rows)
                measured_score = features.scores(bundle["model"], measured)
                ideal_pred = core.predict(ideal_score, bundle["threshold"])
                measured_pred = core.predict(measured_score, bundle["threshold"])
                ideal = quantum.cross_kernel(
                    views["reduced"] * params["angle_max"],
                    bundle["reference_states"],
                    params["encoding"],
                    bundle["transform_hash"],
                    self.output / "cache/states",
                )
                np.save(directory / "kernel-ideal.npy", ideal)
                pd.DataFrame(
                    {
                        "row_index": rows.row_index,
                        "group_hash": rows.group_hash,
                        "is_phishing": rows.is_phishing,
                        "ideal_prediction": ideal_pred,
                        "measured_prediction": measured_pred,
                        "ideal_score": ideal_score,
                        "measured_score": measured_score,
                    }
                ).to_csv(directory / "predictions.csv", index=False)
                evidence_files = [
                    p for p in directory.iterdir() if p.is_file() and p.name != "evaluation.json"
                ]
                core.write_json(
                    summary,
                    {
                        "source": "local Aer finite-shot simulation; no device noise model",
                        "partition": part,
                        "rows": len(rows),
                        "pairs": int(measured.size),
                        "shots": self.config["shots"],
                        "seed_simulator_base": self.config["shot_seed"] + record["seed"],
                        "threshold": bundle["threshold"],
                        "ideal": core.metrics(rows.is_phishing.to_numpy(), ideal_pred, ideal_score),
                        "measured": core.metrics(
                            rows.is_phishing.to_numpy(), measured_pred, measured_score
                        ),
                        "kernel_mean_absolute_error": float(np.abs(measured - ideal).mean()),
                        "prediction_flips": int((ideal_pred != measured_pred).sum()),
                        "clipping": clipping,
                        "seconds": time.perf_counter() - started,
                        "evidence_hashes": {p.name: core.digest(p) for p in evidence_files},
                    },
                )
                print(f"shots: {record['id']} / {part} complete", flush=True)
