"""Aggregate-only study reports, matched comparisons and explicit README publication."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.evaluation.benchmark_v2 import core
from src.evaluation.benchmark_v2.experiment import load_json
from src.evaluation.benchmark_v2.report import (
    METRICS,
    paired_intervals,
    subgroup_analysis,
    svg_plot,
)
from src.evaluation.benchmark_v2_text import features

BEGIN = "<!-- BENCHMARK-V2-TEXT:BEGIN -->"
END = "<!-- BENCHMARK-V2-TEXT:END -->"


def group_summary(records, partition):
    groups = {}
    for row in records:
        groups.setdefault((row["model"], row["train_size"]), []).append(row)
    return [
        {
            "model": name,
            "train_size": size,
            "seeds": [r["seed"] for r in rows],
            "metrics": {
                m: {
                    "mean": float(np.mean([r[partition][m] for r in rows])),
                    "std": float(np.std([r[partition][m] for r in rows])),
                }
                for m in METRICS
            },
        }
        for (name, size), rows in sorted(groups.items())
    ]


def match(row, selection):
    return all(row[k] == value for k, value in selection.items())


def bootstrapped_comparison(experiment, records, predictions, labels, left, right):
    a = [predictions[r["id"]] for r in records if match(r, left)]
    b = [predictions[r["id"]] for r in records if match(r, right)]
    intervals = paired_intervals(labels, a, b, experiment.config["bootstrap_resamples"], 2026)
    intervals["difference"] = intervals.pop("difference_vs_majority")
    lower, upper = intervals["difference"]["average_precision"]
    return {
        "left": left,
        "right": right,
        "intervals": intervals,
        "ap_assessment": "positive_difference"
        if lower > 0
        else "negative_difference"
        if upper < 0
        else "inconclusive",
    }


def report(experiment):
    started = time.perf_counter()
    records, selection = experiment.freeze()
    # Validate and finish measurement checkpoints before publishing a complete study.
    for row in records:
        if row["model"] == "quantum":
            for part in ("test", "external"):
                directory = experiment.output / "shots" / row["id"] / part
                summary = load_json(directory / "evaluation.json")
                for filename, expected in summary["evidence_hashes"].items():
                    if core.digest(directory / filename) != expected:
                        raise ValueError("Shot evidence failed integrity verification.")
    manifest = load_json(experiment.output / "partitions.json")
    # Hypothesis choices were frozen before the first Aer/test evaluation.
    comparisons = selection["comparisons"]
    uncertainty, shot_uncertainty = {}, {}
    for part in ("test", "external"):
        table = experiment.table(part)
        y = table.is_phishing.to_numpy()
        predictions = {}
        for row in records:
            score_started = time.perf_counter()
            score, clipping = experiment.score(row, table)
            row[f"{part}_predict_seconds"] = time.perf_counter() - score_started
            predicted = core.predict(score, row["threshold"])
            row[part] = core.metrics(y, predicted, score)
            row[f"{part}_clipping"] = clipping
            row[f"{part}_subgroups"] = subgroup_analysis(table, predicted, score)
            # Undefined subgroup balanced accuracy is also undefined, not a fabricated zero.
            for subgroup in row[f"{part}_subgroups"].values():
                if subgroup["recall"] is None or subgroup["false_positive_rate"] is None:
                    subgroup["balanced_accuracy"] = None
            predictions[row["id"]] = (predicted, score)
            directory = experiment.output / "predictions"
            directory.mkdir(exist_ok=True)
            pd.DataFrame(
                {
                    "row_index": table.row_index,
                    "group_hash": table.group_hash,
                    "is_phishing": y,
                    "prediction": predicted,
                    "score": score,
                }
            ).to_csv(directory / f"{row['id']}-{part}.csv", index=False)
        uncertainty[part] = {
            name: bootstrapped_comparison(experiment, records, predictions, y, *sides)
            for name, sides in comparisons.items()
        }
        classical = selection["selected"]["classical"]
        uncertainty[part]["selected_classical"] = bootstrapped_comparison(
            experiment,
            records,
            predictions,
            y,
            classical,
            {"model": "numeric_rbf", "train_size": classical["train_size"]},
        )
        subset_y = y[manifest["subsets"][part]]
        chosen_quantum = [r for r in records if match(r, selection["selected"]["quantum"])]
        measured, ideal = [], []
        for row in records:
            if row["model"] != "quantum":
                continue
            directory = experiment.output / "shots" / row["id"] / part
            row.setdefault("measurements", {})[part] = load_json(directory / "evaluation.json")
            if row in chosen_quantum:
                p = pd.read_csv(directory / "predictions.csv")
                measured.append((p.measured_prediction.to_numpy(), p.measured_score.to_numpy()))
                ideal.append((p.ideal_prediction.to_numpy(), p.ideal_score.to_numpy()))
        ci = paired_intervals(
            subset_y, measured, ideal, experiment.config["bootstrap_resamples"], 2026
        )
        ci["difference_measured_minus_ideal"] = ci.pop("difference_vs_majority")
        shot_uncertainty[part] = ci
        print(f"report: {part} predictions and paired comparisons complete", flush=True)
    chosen = [r for r in records if match(r, selection["selected"]["classical"])]
    target_met = all(
        r["test"]["recall"] >= experiment.config["target_recall"]
        and r["test"]["false_positive_rate"] <= experiment.config["max_fpr"]
        for r in chosen
    )
    schema = {
        "schema_version": 1,
        "phase": "benchmark-v2-text-qiskit-local",
        "run_id": experiment.run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "identity": experiment.check(),
        "partitions": {
            k: v for k, v in manifest.items() if k not in {"training", "subsets", "files"}
        },
        "selection": selection["selected"],
        "selection_sha256": core.digest(experiment.output / "selection.json"),
        "target": {
            "recall": experiment.config["target_recall"],
            "max_fpr": experiment.config["max_fpr"],
        },
        "conclusion": "target_met_on_internal_continuity" if target_met else "target_not_met",
        "runs": records,
        "test_summary": group_summary(records, "test"),
        "external_summary": group_summary(records, "external"),
        "calibration_summary": group_summary(records, "calibration"),
        "comparisons": uncertainty,
        "measurement_intervals": shot_uncertainty,
        "search_budget": {
            name: len(features.candidates(name, experiment.config))
            for name in (*features.MODELS, "quantum")
        },
        "uncertainty_scope": "Paired row bootstrap conditional on fitted training seeds and realized shot samples; seed dispersion is separate. Shot noise variance is not estimated from repeated independent measurement runs.",
        "limitations": [
            "Both evaluation datasets have informed previous research choices; these are continuity evaluations, not untouched tests.",
            "Only host text is used; pages and URL paths are not fetched or analyzed.",
            "Four-component SVD can discard discriminative information; explained variance alone does not establish class separation.",
            "Aer uses an ideal local simulator with finite shots, not a device noise model or IBM hardware.",
            "Training and threshold calibration use ideal kernels; measured kernels evaluate the same frozen SVM and threshold.",
            "Forty-row balanced measurement subsets have wide uncertainty and differ in prevalence from full evaluation sets.",
            "No quantum advantage, operational readiness, hackathon submission or cloud execution is claimed.",
        ],
        "report_seconds": time.perf_counter() - started,
    }
    charts = render_charts(experiment, schema)
    schema["charts"] = charts
    serialized = json.dumps(schema, allow_nan=False)
    if '"host"' in serialized or '"vocabulary"' in serialized:
        raise ValueError("Published report contains private row-level text.")
    core.write_json(experiment.report_path, schema)
    core.write_json(
        experiment.output / "publication.json",
        {"report_sha256": core.digest(experiment.report_path), "charts": charts},
    )
    print(
        f"report: {schema['conclusion']} | {experiment.report_path.relative_to(experiment.root)}",
        flush=True,
    )
    return schema


def render_charts(experiment, result):
    caption = (
        f"Source: {experiment.run_id} | PhiUSIIL continuity | Mean across fitted training seeds"
    )
    names = ("numeric_rbf", "full_linear", "reduced_rbf", "quantum")
    series = []
    for name in names:
        rows = [r for r in result["test_summary"] if r["model"] == name]
        series.append(
            {
                "name": name,
                "quantum": name == "quantum",
                "points": [
                    (r["train_size"], r["metrics"]["average_precision"]["mean"]) for r in rows
                ],
            }
        )
    ticks = experiment.config["train_sizes"]
    files = []
    path = experiment.charts / f"{experiment.run_id}-representations.svg"
    svg_plot(
        path,
        "Text, reduction and quantum kernels: average precision",
        "Training examples (count)",
        "Internal continuity average precision (%)",
        series,
        max(ticks),
        ticks,
        caption,
    )
    files.append(path)
    series = []
    for name in names:
        rows = [r for r in result["external_summary"] if r["model"] == name]
        series.append(
            {
                "name": name,
                "quantum": name == "quantum",
                "points": [
                    (
                        r["metrics"]["false_positive_rate"]["mean"] * 100,
                        r["metrics"]["recall"]["mean"],
                    )
                    for r in rows
                ],
            }
        )
    path = experiment.charts / f"{experiment.run_id}-external.svg"
    svg_plot(
        path,
        "External continuity: recall and false positives",
        "External false positive rate (%)",
        "External recall (%)",
        series,
        100,
        [0, 10, 25, 50, 75, 100],
        caption,
    )
    files.append(path)
    selected_size = result["selection"]["quantum"]["train_size"]
    measured = [
        r for r in result["runs"] if r["model"] == "quantum" and r["train_size"] == selected_size
    ]
    series = [
        {
            "name": label,
            "quantum": index == 1,
            "points": [
                (i + 1, r["measurements"]["test"][source]["recall"]) for i, r in enumerate(measured)
            ],
        }
        for index, (label, source) in enumerate(
            (
                ("Qiskit ideal / fixed subset", "ideal"),
                ("Aer finite shots / same subset", "measured"),
            )
        )
    ]
    path = experiment.charts / f"{experiment.run_id}-measurements.svg"
    svg_plot(
        path,
        "Qiskit Statevector versus Aer: frozen SVM and threshold",
        "Training-seed index (ordered in configuration)",
        "40-row internal subset recall (%)",
        series,
        len(measured),
        list(range(1, len(measured) + 1)),
        f"Source: {experiment.run_id} | Same balanced subset | Conditional on one shot sample per fitted seed",
    )
    files.append(path)
    return {p.relative_to(experiment.root).as_posix(): core.digest(p) for p in files}


def publish(experiment):
    experiment.freeze()
    result = load_json(experiment.report_path)
    publication = load_json(experiment.output / "publication.json")
    if result["status"] != "complete" or publication["report_sha256"] != core.digest(
        experiment.report_path
    ):
        raise ValueError("Only a verified complete report can be published.")
    for filename, expected in publication["charts"].items():
        if core.digest(experiment.root / filename) != expected:
            raise ValueError("Chart/report identity mismatch.")
    selected = result["selection"]["classical"]
    internal = next(r for r in result["test_summary"] if match(r, selected))["metrics"]
    external = next(r for r in result["external_summary"] if match(r, selected))["metrics"]
    date = datetime.fromisoformat(result["generated_at_utc"]).date().isoformat()
    links = " | ".join(
        f"[{name.rsplit('-', 1)[-1].removesuffix('.svg')}]({name})" for name in result["charts"]
    )
    block = f"""{BEGIN}
## Benchmark v2 expansion: hostname text and local Qiskit

**Portfolio research, post-hackathon.** Run `{experiment.run_id}`, generated {date} (UTC): **{result["conclusion"]}**.

The selected classical path is `{selected["model"]}` with {selected["train_size"]} training examples. Mean internal continuity accuracy is **{internal["accuracy"]["mean"]:.2%}**, recall **{internal["recall"]["mean"]:.2%}**, and false positive rate **{internal["false_positive_rate"]["mean"]:.2%}**. External continuity accuracy is **{external["accuracy"]["mean"]:.2%}**, recall **{external["recall"]["mean"]:.2%}**, and false positive rate **{external["false_positive_rate"]["mean"]:.2%}**.

The study separates text representation, four-component reduction, ideal Qiskit kernels and local Aer sampling. These are previously inspected continuity datasets; no quantum advantage or production readiness is claimed. No IBM Quantum connection is used in this phase.

[Protocol and interpretation](docs/benchmark-v2-text.md) | [Run report]({experiment.report_path.relative_to(experiment.root).as_posix()}) | {links}
{END}"""
    path = experiment.root / "README.md"
    existing = path.read_text(encoding="utf-8")
    if existing.count(BEGIN) != existing.count(END) or existing.count(BEGIN) > 1:
        raise ValueError("Malformed README publication markers.")
    if BEGIN in existing:
        start, stop = existing.index(BEGIN), existing.index(END) + len(END)
        if stop <= start:
            raise ValueError("README publication markers are reversed.")
        updated = existing[:start] + block + existing[stop:]
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    if updated != existing:
        path.write_text(updated, encoding="utf-8")
    print("publish: README block verified and updated", flush=True)
