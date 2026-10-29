"""Frozen selection, held-out evaluation, paired intervals and static SVG figures."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.evaluation.benchmark_v2 import core
from src.evaluation.benchmark_v2.experiment import Experiment, bundle_scores, load_json

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "false_positive_rate",
    "f1",
    "average_precision",
)
LIMITATIONS = [
    "Page-content features separate PhiUSIIL classes partly through its collection process: legitimate rows are popular homepages, phishing rows are often small pages.",
    "The external dataset has no comparable page columns, so page-feature models have no external evaluation; only the host-feature selection is checked externally.",
    "The four page features were first identified in an exploratory analysis that also looked at the internal test; the prepare stage records a development-only ranking that must reproduce them.",
    "Page-feature inference requires fetching the page, unlike host-only inference.",
    "Internal partitions are domain-disjoint but share one collection process; the external dataset has already informed project decisions.",
    "Bootstrap intervals describe test-example uncertainty conditional on fitted models; seed dispersion is reported separately.",
    "Calibration is used for threshold and configuration selection; its metrics are not held-out performance estimates.",
    "Quantum results are exact CPU simulation; there is no claim of quantum speedup or advantage.",
    "Fresh independently collected, preferably temporal data is required before operational adoption.",
]


def key(row: dict) -> tuple:
    return row["family"], row["model"], row["features"], row["train_size"]


def aggregate(rows: list[dict], partition: str) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    output = []
    for (family, model, features, size), members in sorted(groups.items()):
        output.append(
            {
                "family": family,
                "model": model,
                "features": features,
                "train_size": size,
                "seeds": [m["seed"] for m in members],
                "metrics": {
                    metric: {
                        "mean": float(np.mean([m[partition][metric] for m in members])),
                        "std": float(np.std([m[partition][metric] for m in members], ddof=0)),
                    }
                    for metric in METRICS
                },
            }
        )
    return output


def choose(rows: list[dict], family: str, max_fpr: float, features: str | None = None) -> dict:
    groups = aggregate(
        [
            r
            for r in rows
            if r["family"] == family
            and r["model"] != "majority"
            and features in {None, r["features"]}
        ],
        "calibration",
    )
    feasible = [r for r in groups if r["metrics"]["false_positive_rate"]["mean"] <= max_fpr]
    if not feasible:
        raise ValueError("No configuration satisfies the calibration FPR constraint.")
    selected = min(
        feasible,
        key=lambda r: (
            -r["metrics"]["recall"]["mean"],
            -r["metrics"]["average_precision"]["mean"],
            r["train_size"],
            r["model"],
            r["features"],
        ),
    )
    return {field: selected[field] for field in ("family", "model", "features", "train_size")}


def freeze(experiment: Experiment, rows: list[dict]) -> dict:
    path = experiment.output / "frozen-selection.json"
    fingerprints = {
        r["id"]: core.digest(experiment.output / "runs" / f"{r['id']}.json") for r in rows
    }
    selection = {
        "schema_version": 1,
        "classical": choose(rows, "classical", experiment.config["max_fpr"]),
        "quantum": choose(rows, "quantum", experiment.config["max_fpr"]),
        # The external dataset only supports host features, so it gets its own selection.
        "classical_host": choose(rows, "classical", experiment.config["max_fpr"], "host"),
        "record_hashes": fingerprints,
        "rule": "mean calibration recall, then mean AP, then smaller training size; lexical tie-break",
    }
    if path.exists():
        if load_json(path) != selection:
            raise ValueError(
                "Frozen selection or training records changed; refusing test evaluation."
            )
    else:
        core.write_json(path, selection)
    return selection


def subgroup_analysis(rows: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray) -> dict:
    labels = rows.is_phishing.to_numpy()
    masks = {
        "digits_present": rows.host_digit_ratio.to_numpy() > 0,
        "no_digits": rows.host_digit_ratio.to_numpy() == 0,
        "hyphens_present": rows.host_hyphen_count.to_numpy() > 0,
        "no_hyphens": rows.host_hyphen_count.to_numpy() == 0,
        "labels_le_2": rows.host_label_count.to_numpy() <= 2,
        "labels_eq_3": rows.host_label_count.to_numpy() == 3,
        "labels_ge_4": rows.host_label_count.to_numpy() >= 4,
    }
    output = {}
    for name, mask in masks.items():
        if mask.any():
            point = core.metrics(labels[mask], predictions[mask], scores[mask])
            # Avoid interpreting missing-class rates as measured zero.
            if not (labels[mask] == 1).any():
                point["recall"] = point["average_precision"] = None
            if not (labels[mask] == 0).any():
                point["false_positive_rate"] = None
            output[name] = point
    return output


def weighted_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    tp = weights[(y == 1) & (prediction == 1)].sum()
    fp = weights[(y == 0) & (prediction == 1)].sum()
    positives, negatives = weights[y == 1].sum(), weights[y == 0].sum()
    fn, tn = positives - tp, negatives - fp
    recall, fpr = tp / positives, fp / negatives
    w = weights[order]
    positive = w * y[order]
    cumulative_tp = np.cumsum(positive)[ends]
    cumulative_all = np.cumsum(w)[ends]
    increments = np.diff(np.r_[0, cumulative_tp])
    precision = np.divide(
        cumulative_tp,
        cumulative_all,
        out=np.zeros_like(cumulative_tp, dtype=float),
        where=cumulative_all != 0,
    )
    ap = np.sum(increments * precision) / positives
    return np.array(
        [
            (tp + tn) / weights.sum(),
            (recall + 1 - fpr) / 2,
            tp / (tp + fp) if tp + fp else 0,
            recall,
            fpr,
            2 * tp / (2 * tp + fp + fn) if tp else 0,
            ap,
        ]
    )


def paired_intervals(
    y: np.ndarray, selected: list[tuple], baseline: list[tuple], resamples: int, seed: int
) -> dict:
    """Shared row bootstrap across all seeds and baseline, with exact score-tie AP."""
    prepared = []
    for prediction, score in selected + baseline:
        order = np.argsort(-score, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y) - 1]
        prepared.append((prediction, score, order, ends))
    rng = np.random.default_rng(seed)
    estimates, differences = [], []
    count = len(selected)
    for _ in range(resamples):
        weights = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        if not weights[y == 0].sum() or not weights[y == 1].sum():
            continue
        values = [weighted_metrics(y, p, s, weights, o, e) for p, s, o, e in prepared]
        chosen = np.mean(values[:count], axis=0)
        reference = np.mean(values[count:], axis=0)
        estimates.append(chosen)
        differences.append(chosen - reference)
    if not estimates:
        raise ValueError("Bootstrap has no two-class resamples.")
    return {
        "method": "paired percentile row bootstrap of the mean metric across fitted seeds",
        "confidence_level": 0.95,
        "resamples_used": len(estimates),
        "seed": seed,
        "selected": {
            m: np.percentile(np.array(estimates)[:, i], [2.5, 97.5]).tolist()
            for i, m in enumerate(METRICS)
        },
        "difference_vs_majority": {
            m: np.percentile(np.array(differences)[:, i], [2.5, 97.5]).tolist()
            for i, m in enumerate(METRICS)
        },
    }


def matches(row: dict, selection: dict) -> bool:
    return all(row[field] == value for field, value in selection.items())


def majority_for(records: list[dict], selection: dict) -> list[dict]:
    return [
        r
        for r in records
        if r["model"] == "majority"
        and r["features"] == "host"
        and r["train_size"] == selection["train_size"]
    ]


def meets_target(rows: list[dict], config: dict) -> bool:
    return all(
        r["test"]["recall"] >= config["target_recall"]
        and r["test"]["false_positive_rate"] <= config["max_fpr"]
        and r["test"]["accuracy"] >= config["target_accuracy"]
        for r in rows
    )


def build_report(experiment: Experiment) -> dict:
    identity = experiment.check()
    records = experiment.records()
    selection = freeze(experiment, records)  # Persist choices before opening the held-out data.
    test, external = experiment.table("test"), experiment.table("external")
    predictions = {}
    external_ids = {r["id"] for r in records if matches(r, selection["classical_host"])} | {
        r["id"] for r in majority_for(records, selection["classical_host"])
    }
    for i, row in enumerate(records):
        with (experiment.output / "runs" / f"{row['id']}.pkl").open("rb") as stream:
            bundle = pickle.load(stream)  # trusted local artifact, hash checked by records()
        for name, table in (("test", test), ("external", external)):
            if name == "external" and row["id"] not in external_ids:
                continue
            score, seconds, clipping = bundle_scores(bundle, table)
            prediction = core.predict(score, row["threshold"])
            row[name] = core.metrics(table.is_phishing.to_numpy(), prediction, score)
            row[f"{name}_predict_seconds"] = seconds
            if clipping is not None:
                row[f"{name}_clipping"] = clipping
            if matches(row, selection[row["family"]]) or (
                name == "external" and matches(row, selection["classical_host"])
            ):
                row[f"{name}_subgroups"] = subgroup_analysis(table, prediction, score)
            directory = experiment.output / "predictions"
            directory.mkdir(exist_ok=True)
            pd.DataFrame(
                {
                    "row_index": table.row_index,
                    "group_hash": table.group_hash,
                    "is_phishing": table.is_phishing,
                    "prediction": prediction,
                    "score": score,
                }
            ).to_csv(directory / f"{row['id']}-{name}.csv", index=False)
            if name == "test":
                predictions[row["id"]] = (prediction, score)
        if (i + 1) % 12 == 0:
            print(f"report: evaluated {i + 1}/{len(records)} frozen runs", flush=True)
    config = experiment.config
    selected = [r for r in records if matches(r, selection["classical"])]
    selected_quantum = [r for r in records if matches(r, selection["quantum"])]
    intervals, quantum_intervals = (
        paired_intervals(
            test.is_phishing.to_numpy(),
            [predictions[r["id"]] for r in chosen],
            [predictions[r["id"]] for r in majority_for(records, selection[family])],
            config["bootstrap_resamples"],
            config["split_seed"],
        )
        for family, chosen in (("classical", selected), ("quantum", selected_quantum))
    )
    achieved = meets_target(selected, config)
    partial = (
        intervals["difference_vs_majority"]["recall"][0] > 0
        and float(np.mean([r["test"]["false_positive_rate"] for r in selected]))
        <= config["max_fpr"]
    )
    conclusion = (
        "target_met"
        if achieved
        else "partial_improvement"
        if partial
        else "no_consistent_improvement"
    )
    # Membership is recorded in local partitions/manifests, not the published aggregates.
    for row in records:
        row.pop("training_group_hashes")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "identity": identity,
        "partitions": load_json(experiment.output / "partitions.json"),
        "selection": {k: v for k, v in selection.items() if k != "record_hashes"},
        "selection_sha256": core.digest(experiment.output / "frozen-selection.json"),
        "target": {
            "recall": config["target_recall"],
            "max_false_positive_rate": config["max_fpr"],
            "accuracy": config["target_accuracy"],
            "assessment": "all three point targets must hold for every fitted seed of the selection on internal test",
        },
        "conclusion": conclusion,
        "quantum_conclusion": "target_met"
        if meets_target(selected_quantum, config)
        else "target_not_met",
        "improvement_reference": "same-size majority baseline on identical internal test rows",
        "target_scope": "internal PhiUSIIL only; external continuity evidence is reported separately",
        "runs": records,
        "calibration_summary": aggregate(records, "calibration"),
        "test_summary": aggregate(records, "test"),
        "external_summary": aggregate([r for r in records if "external" in r], "external"),
        "selected_confidence_intervals": intervals,
        "quantum_confidence_intervals": quantum_intervals,
        "historical_external_reference": load_json(
            experiment.root / "reports/external-validation.json"
        )["models"],
        "search_budget": {
            "classical_candidates_per_family": {
                name: len(list(ParameterGrid(grid))) for name, grid in config["grids"].items()
            },
            "majority_candidates": 1,
            "quantum_candidates_per_map": len(config["angle_maxima"]) * len(config["quantum_c"]),
            "quantum_maps": len(config["quantum_maps"]),
            "classical_runs": len(experiment.expected_ids("classical")),
            "quantum_runs": len(experiment.expected_ids("quantum")),
        },
        "limitations": LIMITATIONS,
        "next_experiment": "Measure the selected page-feature quantum kernel on IBM hardware, then validate page features on newly collected temporal data.",
    }
    # Protect publication from accidental non-aggregate fields and non-finite values.
    serialized = json.dumps(report, allow_nan=False)
    # Row-level fields appear as keys; "host" is also a legitimate feature-set value.
    if '"host":' in serialized or '"URL":' in serialized or '"training_group_hashes"' in serialized:
        raise ValueError("Published report contains row-level identity.")
    core.write_json(experiment.root / config["paths"]["report"], report)
    render_charts(report, experiment.root / config["paths"]["charts"])
    return report


def svg_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[dict],
    x_max: float,
    x_ticks: list[float],
    caption: str,
) -> None:
    width, height, left, top, plot_w, plot_h = 1120, 740, 100, 100, 760, 400

    def x(value):
        return left + float(value) / x_max * plot_w

    def y(value):
        return top + (1 - float(value)) * plot_h

    def text(px, py, label, extra=""):
        return f'<text x="{px}" y="{py}" fill="#1f2328" {extra}>{escape(str(label))}</text>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title><desc>{escape(caption)}</desc>",
        '<rect width="1120" height="740" fill="#ffffff"/>',
        '<g font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="13">',
        text(40, 45, title, 'font-size="23" font-weight="600"'),
    ]
    for tick in (0, 0.2, 0.4, 0.6, 0.8, 1):
        parts += [
            f'<line x1="{left}" x2="{left + plot_w}" y1="{y(tick)}" y2="{y(tick)}" stroke="#d0d7de"/>',
            text(left - 12, y(tick) + 4, f"{tick:.0%}", 'text-anchor="end"'),
        ]
    for tick in x_ticks:
        parts += [text(x(tick), 524, f"{tick:g}", 'text-anchor="middle"')]
    parts += [
        text(left + plot_w / 2, 555, xlabel, 'text-anchor="middle"'),
        text(26, 300, ylabel, 'transform="rotate(-90 26 300)" text-anchor="middle"'),
    ]
    for i, item in enumerate(series):
        color = "#2563eb" if item.get("quantum") else "#57606a"
        dash = ["", "6 3", "2 3", "9 3 2 3"][i % 4]
        points = " ".join(f"{x(a):.2f},{y(b):.2f}" for a, b in item["points"])
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
        )
        for a, b in item["points"]:
            parts.append(f'<circle cx="{x(a)}" cy="{y(b)}" r="4" fill="{color}"/>')
        # Two-column legend keeps labels readable, outside the plotting rectangle.
        lx, ly = 105 + (i % 2) * 490, 592 + (i // 2) * 25
        parts += [
            f'<line x1="{lx}" x2="{lx + 28}" y1="{ly - 4}" y2="{ly - 4}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>',
            text(lx + 38, ly, item["name"]),
        ]
    parts += [text(40, 710, caption, 'font-size="12"'), "</g></svg>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_charts(report: dict, directory: Path) -> None:
    summary = report["test_summary"]
    caption = "Source: reports/benchmark-v2.json | Historical PhiUSIIL internal test | Mean across training seeds"
    learning = []
    for model in ("logistic_regression", "rbf_svm", "random_forest"):
        for features in core.FEATURE_SETS:
            rows = [r for r in summary if r["model"] == model and r["features"] == features]
            learning.append(
                {
                    "name": f"{model} / {features}",
                    "points": [
                        (r["train_size"], r["metrics"]["recall"]["mean"])
                        for r in sorted(rows, key=lambda r: r["train_size"])
                    ],
                }
            )
    maximum = max(r["train_size"] for r in summary)
    ticks = sorted({r["train_size"] for r in summary})
    svg_plot(
        directory / "benchmark-v2-learning.svg",
        "Learning curves at calibrated operating points",
        "Training examples (count)",
        "Internal test recall (%)",
        learning,
        maximum,
        ticks,
        caption,
    )
    tradeoff = []
    target = report["target"]
    fpr_axis = max(10.0, target["max_false_positive_rate"] * 500)
    for model in ("logistic_regression", "rbf_svm", "random_forest", "z", "zz"):
        rows = [r for r in summary if r["model"] == model and r["features"] == "page"]
        tradeoff.append(
            {
                "name": f"{model} / page; sizes increase along series",
                "quantum": model in {"z", "zz"},
                "points": [
                    (
                        min(r["metrics"]["false_positive_rate"]["mean"] * 100, fpr_axis),
                        r["metrics"]["recall"]["mean"],
                    )
                    for r in sorted(rows, key=lambda r: r["train_size"])
                ],
            }
        )
    max_fpr, recall = target["max_false_positive_rate"], target["recall"]
    tradeoff.append(
        {
            "name": f"Target: FPR <= {max_fpr:.0%}, recall >= {recall:.0%}",
            "points": [(max_fpr * 100, recall)],
        }
    )
    svg_plot(
        directory / "benchmark-v2-operating-points.svg",
        "Recall and false positives at frozen thresholds",
        "Internal test false positive rate (%, values beyond the axis are clipped)",
        "Internal test recall (%)",
        tradeoff,
        fpr_axis,
        [0, fpr_axis / 4, fpr_axis / 2, 3 * fpr_axis / 4, fpr_axis],
        caption,
    )
    comparison = []
    q_sizes = report["identity"]["config"]["quantum_train_sizes"]
    for model in ("logistic_regression", "rbf_svm", "random_forest", "z", "zz"):
        rows = [
            r
            for r in summary
            if r["model"] == model and r["features"] == "page" and r["train_size"] in q_sizes
        ]
        comparison.append(
            {
                "name": f"{model} / same four page features",
                "quantum": model in {"z", "zz"},
                "points": [
                    (r["train_size"], r["metrics"]["average_precision"]["mean"])
                    for r in sorted(rows, key=lambda r: r["train_size"])
                ],
            }
        )
    svg_plot(
        directory / "benchmark-v2-kernels.svg",
        "Classical and exact quantum simulation: matched inputs",
        "Training examples (count)",
        "Internal test average precision (%)",
        comparison,
        max(q_sizes),
        q_sizes,
        caption,
    )
