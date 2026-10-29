"""Leakage, threshold ties, exact kernels, page features, paired inference and offline run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.evaluation.benchmark_v2 import core, experiment, report

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def config():
    result = experiment.load_config(ROOT / "configs/benchmark-v2.json")
    result.update(
        {
            "seeds": [42, 43],
            "train_sizes": [6, 12],
            "quantum_train_sizes": [6, 12],
            "forest_trees": 3,
            "bootstrap_resamples": 20,
            "angle_maxima": [float(np.pi / 4)],
            "quantum_c": [1],
        }
    )
    result["grids"] = {
        "logistic_regression": {"C": [1]},
        "rbf_svm": {"C": [1], "gamma": ["scale"]},
        "random_forest": {"max_depth": [None], "min_samples_leaf": [1]},
    }
    return result


def raw_rows():
    rows = [
        {"URL": f"https://site{i}.com", "Domain": f"site{i}.com", "label": i % 2}
        for i in range(120)
    ]
    rows += [
        rows[0].copy(),
        {"URL": "https://sub.site1.com", "Domain": "sub.site1.com", "label": 1},
    ]
    rows += [
        {"URL": "https://conflict.com", "Domain": "conflict.com", "label": 0},
        {"URL": "https://sub.conflict.com", "Domain": "sub.conflict.com", "label": 1},
    ]
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(7)
    for name in core.PAGE_CANDIDATES:
        # Legitimate pages are larger, like PhiUSIIL homepages; the rest is noise.
        signal = name in core.PAGE_FEATURES
        table[name] = rng.poisson(5, len(table)) + signal * 50 * (1 - table.label)
    return table


def test_disjoint_reproducible_partitions_and_nested_samples():
    excluded = {core.group_hash("dns:site1.com")}
    first, exclusions = core.partitions(raw_rows(), excluded, 42)
    second, _ = core.partitions(raw_rows(), excluded, 42)
    assert exclusions["duplicate_url"] == 1
    assert exclusions["label_conflicting_groups"] == 1
    assert exclusions["historical_test_groups"] == 1
    seen = set()
    for name, table in first.items():
        pd.testing.assert_frame_equal(table, second[name])
        assert not seen.intersection(table.group_hash)
        assert not excluded.intersection(table.group_hash)
        seen.update(table.group_hash)
    samples = core.nested_samples(first["development"], [6, 12, 24], 42)
    for smaller, larger in ((6, 12), (12, 24)):
        assert set(samples[smaller]) < set(samples[larger])
    assert first["development"].iloc[samples[24]].is_phishing.sum() == 12


@pytest.mark.parametrize(
    "y,s,expected",
    [
        ([0, 0, 1, 1], [0.9, 0.1, 0.9, 0.8], 0.0),
        ([0, 0, 1, 1], [0.2, 0.1, 0.9, 0.8], 1.0),
        ([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], 0.0),
    ],
)
def test_thresholds_are_tie_safe_and_obey_fpr(y, s, expected):
    y, s = np.array(y), np.array(s)
    point = core.operating_point(y, s, 0.1)
    measured = core.metrics(y, core.predict(s, point["threshold"]), s)
    assert measured["false_positive_rate"] <= 0.1
    assert measured["recall"] == expected
    if expected == 0:
        assert point["threshold"] is None


def test_threshold_prefers_fewer_false_positives_for_same_recall():
    point = core.operating_point(np.array([1, 0, 0, 1]), np.array([4.0, 3.0, 2.0, 1.0]), 0.5)
    assert point["threshold"] == 4


@pytest.mark.parametrize("encoding", ["z", "zz"])
def test_vectorized_states_and_cross_kernels_match_qiskit(encoding):
    x = np.random.default_rng(42).uniform(0, np.pi / 2, (9, 4))
    circuit = (
        z_feature_map(4, reps=1)
        if encoding == "z"
        else zz_feature_map(4, reps=1, entanglement="linear")
    )
    reference = np.array([Statevector(circuit.assign_parameters(row)).data for row in x])
    actual = core.states(x, encoding)
    np.testing.assert_allclose(actual, reference, atol=1e-13)
    expected = np.abs(reference[:3].conj() @ reference[[7, 3, 8, 2]].T) ** 2
    np.testing.assert_allclose(core.kernel(actual[:3], actual[[7, 3, 8, 2]]), expected, atol=1e-13)


def test_angle_scaler_clipping_constant_feature_shift_and_train_only():
    train = np.array([[0.0, 3, 1, 0], [2.0, 3, 2, 1]])
    held = np.array([[8.0, 9, 0, 0]])
    low, high = train.min(axis=0), train.max(axis=0)
    a, clipping = core.angles(held, low, high, np.pi / 4)
    np.testing.assert_allclose(a, [[np.pi / 4, 0, 0, 0]])
    assert clipping["rows_with_any_clipped"] == 1
    assert clipping["constant_feature_shift_rows"] == 1
    np.testing.assert_array_equal(high, train.max(axis=0))


def test_search_fits_standard_scaler_on_fold_training_only(monkeypatch):
    observed = []
    original = StandardScaler.fit

    def spy(self, x, *args, **kwargs):
        observed.append(np.asarray(x).copy())
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy)
    x = np.arange(48).reshape(12, 4).astype(float)
    y = np.tile([0, 1], 6)
    experiment.search_classical("logistic_regression", x, y, 42, config())
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(x, y))
    for seen, (train, _) in zip(observed[:3], folds, strict=True):
        np.testing.assert_array_equal(seen, x[train])
    np.testing.assert_array_equal(observed[-1], x)


def test_weighted_bootstrap_ap_matches_sklearn_with_ties():
    y = np.array([0, 1, 0, 1, 1, 0])
    score = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 0.1])
    weights = np.array([2, 0, 1, 3, 2, 1])
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y) - 1]
    values = report.weighted_metrics(y, score >= 0.5, score, weights, order, ends)
    assert values[-1] == pytest.approx(average_precision_score(y, score, sample_weight=weights))
    pair = (score >= 0.5, score)
    ci = report.paired_intervals(y, [pair], [pair], 30, 42)
    assert ci["difference_vs_majority"]["recall"] == [0, 0]


def test_choice_uses_calibration_not_test():
    rows = []
    for model, recall in (("rbf_svm", 0.7), ("logistic_regression", 0.6)):
        metric = {m: 0.5 for m in report.METRICS}
        metric.update(recall=recall, false_positive_rate=0.1)
        rows.append(
            {
                "family": "classical",
                "model": model,
                "features": "host",
                "train_size": 24,
                "seed": 42,
                "calibration": metric,
                "test": {"recall": 1 - recall},
            }
        )
    assert report.choose(rows, "classical", 0.1)["model"] == "rbf_svm"
    rows[0]["features"] = "page"
    assert report.choose(rows, "classical", 0.1, "host")["model"] == "logistic_regression"
    for row in rows:
        row["calibration"]["false_positive_rate"] = 0.2
    with pytest.raises(ValueError, match="No configuration"):
        report.choose(rows, "classical", 0.1)


@pytest.fixture
def synthetic_experiment(tmp_path, monkeypatch):
    cfg = config()
    run = experiment.Experiment(cfg, tmp_path)
    identity = {"config": cfg, "data": {"synthetic": True}}
    monkeypatch.setattr(run, "identity", lambda: copy.deepcopy(identity))
    parts, _ = core.partitions(raw_rows(), set(), 42)
    parts["external"] = parts["test"].copy()
    run.output.mkdir(parents=True)
    manifests = {"files": {}, "partitions": {}}
    raw = raw_rows()
    for name, rows in parts.items():
        page = raw.iloc[rows.row_index][list(core.PAGE_FEATURES)].to_numpy()
        rows[list(core.PAGE_FEATURES)] = page
        page_columns = [] if name == "external" else list(core.PAGE_FEATURES)
        table = rows[
            ["row_index", "group_hash", "is_phishing", *core.CANDIDATE_FEATURES, *page_columns]
        ]
        table.to_csv(run.output / f"{name}.csv", index=False)
        manifests["files"][f"{name}.csv"] = core.digest(run.output / f"{name}.csv")
        manifests["partitions"][name] = {"rows": len(table)}
    core.write_json(run.output / "identity.json", identity)
    core.write_json(run.output / "partitions.json", manifests)
    core.write_json(tmp_path / "reports/external-validation.json", {"models": []})
    return run


def test_reduced_synthetic_pipeline_freezes_before_test_and_publishes_aggregates(
    synthetic_experiment, monkeypatch
):
    run = synthetic_experiment
    original = run.table

    def guarded(name):
        if name in {"test", "external"}:
            assert (run.output / "frozen-selection.json").exists()
        return original(name)

    monkeypatch.setattr(run, "table", guarded)
    run.train("classical")
    run.train("quantum")
    # Identical completed runs must be reused without fitting again.
    monkeypatch.setattr(experiment, "search_classical", lambda *a: pytest.fail("Unexpected refit"))
    run.train("classical")
    result = report.build_report(run)
    assert result["status"] == "complete"
    assert len(result["runs"]) == 40
    assert result["conclusion"] in {
        "target_met",
        "partial_improvement",
        "no_consistent_improvement",
    }
    assert result["quantum_conclusion"] in {"target_met", "target_not_met"}
    assert result["selection"]["quantum"]["features"] == "page"
    assert result["selection"]["classical_host"]["features"] == "host"
    # Page models cannot run on the external dataset; only host models are evaluated there.
    assert {r["features"] for r in result["runs"] if "external" in r} == {"host"}
    serialized = json.dumps(result)
    assert '"host":' not in serialized and '"training_group_hashes"' not in serialized
    for chart in (run.root / "docs").glob("benchmark-v2-*.svg"):
        root = ElementTree.parse(chart).getroot()
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
    assert len(list((run.root / "docs").glob("*.svg"))) == 3
    # Test results cannot alter selection; reporting can be repeated safely.
    before = (run.output / "frozen-selection.json").read_bytes()
    report.build_report(run)
    assert (run.output / "frozen-selection.json").read_bytes() == before


def test_resume_refuses_changed_partition(synthetic_experiment):
    run = synthetic_experiment
    with (run.output / "development.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Prepared partition changed"):
        run.check()


def test_resume_refuses_changed_configuration(synthetic_experiment):
    run = synthetic_experiment
    identity = run.identity()
    identity["config"]["target_recall"] = 0.9
    run.identity = lambda: identity
    with pytest.raises(ValueError, match="Run identity changed"):
        run.check()


def test_report_requires_all_experiments_before_test_access(synthetic_experiment, monkeypatch):
    run = synthetic_experiment
    monkeypatch.setattr(
        run, "table", lambda name: pytest.fail("Test opened before completed search")
    )
    with pytest.raises(ValueError, match="Missing completed experiment"):
        report.build_report(run)


def test_output_cannot_overwrite_original_benchmark(tmp_path):
    cfg = config()
    cfg["paths"]["output"] = "artifacts/benchmark"
    with pytest.raises(ValueError, match="historical artifacts"):
        experiment.Experiment(cfg, tmp_path)


def test_prepare_stage_records_partitions_before_any_training(tmp_path, monkeypatch):
    cfg = config()
    raw_path = tmp_path / cfg["paths"]["phiusiil"]
    raw_path.parent.mkdir(parents=True)
    raw_rows().to_csv(raw_path, index=False)
    ext_path = tmp_path / cfg["paths"]["external"]
    ext_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "url": [f"https://external{i}.net" for i in range(20)],
            "status": ["phishing", "legitimate"] * 10,
        }
    ).to_csv(ext_path, index=False)
    core.write_json(tmp_path / "configs/benchmark-split.json", {"test": []})
    core.write_json(
        tmp_path / "configs/external-validation.json",
        {"dataset": {"label_map": {"phishing": 1, "legitimate": 0}}},
    )
    run = experiment.Experiment(cfg, tmp_path)
    monkeypatch.setattr(run, "identity", lambda: {"synthetic": True})
    run.prepare()
    first = (run.output / "partitions.json").read_bytes()
    run.prepare()
    assert (run.output / "partitions.json").read_bytes() == first
    assert len(run.table("external")) == 20
    assert not set(core.PAGE_FEATURES) & set(run.table("external").columns)
    assert set(core.PAGE_FEATURES) <= set(run.table("test").columns)
    manifest = experiment.load_json(run.output / "partitions.json")
    ranked = [row["feature"] for row in manifest["page_feature_ranking"]]
    assert sorted(ranked) == sorted(core.PAGE_CANDIDATES)
    assert not (run.output / "runs").exists()


def test_resume_refuses_corrupted_model(synthetic_experiment):
    run = synthetic_experiment
    run.train("classical")
    model = next((run.output / "runs").glob("*.pkl"))
    model.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Run artifact changed"):
        run.train("classical")


@pytest.mark.parametrize(
    "changes",
    [
        {"train_sizes": [7]},
        {"quantum_train_sizes": []},
        {"max_fpr": 1},
        {"seeds": [42, 42]},
        {"n_jobs": 0},
        {"quantum_c": [0]},
        {"target_accuracy": 0},
        {"target_accuracy": 1.5},
    ],
)
def test_configuration_rejects_invalid_search_before_running(tmp_path, changes):
    cfg = config()
    cfg.update(changes)
    path = tmp_path / "invalid.json"
    core.write_json(path, cfg)
    with pytest.raises(ValueError):
        experiment.load_config(path)


def test_report_cannot_overwrite_historical_publication(tmp_path):
    cfg = config()
    cfg["paths"]["report"] = "reports/benchmark-results.json"
    with pytest.raises(ValueError, match="benchmark-v2"):
        experiment.Experiment(cfg, tmp_path)


def test_page_matrix_uses_log_scale_and_rejects_missing_values():
    rows = pd.DataFrame({name: [0.0, np.e - 1] for name in core.PAGE_FEATURES})
    np.testing.assert_allclose(core.matrix(rows, "page"), [[0] * 4, [1] * 4])
    host = pd.DataFrame({name: [2.0] for name in core.CANDIDATE_FEATURES})
    np.testing.assert_array_equal(core.matrix(host, "host"), [[2.0] * 8])
    rows.loc[0, "NoOfImage"] = np.nan
    with pytest.raises(ValueError, match="finite page"):
        core.matrix(rows, "page")


def test_page_candidates_exclude_url_derived_and_label_columns():
    assert set(core.PAGE_FEATURES) <= set(core.PAGE_CANDIDATES)
    leaky = {"URL", "Domain", "TLD", "URLSimilarityIndex", "IsHTTPS", "URLLength", "label"}
    assert not leaky & set(core.PAGE_CANDIDATES)
    assert core.FEATURE_SETS == {"host": core.CANDIDATE_FEATURES, "page": core.PAGE_FEATURES}


def test_page_ranking_is_deterministic_and_finds_informative_columns():
    rng = np.random.default_rng(3)
    labels = np.tile([0, 1], 400)
    rows = pd.DataFrame({"is_phishing": labels})
    for name in core.PAGE_CANDIDATES:
        # Each informative column separates only part of the legitimate pages.
        partial = rng.binomial(1, 0.6, len(labels)) * (name in core.PAGE_FEATURES)
        rows[name] = rng.poisson(5, len(labels)) + 40 * partial * (1 - labels)
    first = experiment.rank_page_features(rows, 42)
    assert first == experiment.rank_page_features(rows, 42)
    assert {row["feature"] for row in first[:4]} == set(core.PAGE_FEATURES)


def test_target_requires_recall_fpr_and_accuracy_for_every_seed():
    cfg = {"target_recall": 0.95, "max_fpr": 0.01, "target_accuracy": 0.98}
    good = {"test": {"recall": 0.96, "false_positive_rate": 0.01, "accuracy": 0.985}}
    assert report.meets_target([good, good], cfg)
    for field, value in (("recall", 0.9), ("false_positive_rate", 0.02), ("accuracy", 0.97)):
        bad = {"test": {**good["test"], field: value}}
        assert not report.meets_target([good, bad], cfg)


def test_published_v2_report_is_complete_and_quotes_match_measured_results():
    published = experiment.load_json(ROOT / "reports/benchmark-v2.json")
    cfg = experiment.load_config(ROOT / "configs/benchmark-v2.json")
    assert published["identity"]["config"] == cfg
    assert published["status"] == "complete"
    assert len(published["runs"]) == 114
    assert len({r["id"] for r in published["runs"]}) == 114
    ranking = published["partitions"]["page_feature_ranking"]
    assert {row["feature"] for row in ranking[:4]} == set(core.PAGE_FEATURES)
    for row in published["runs"]:
        assert row["status"] == "complete"
        assert "training_group_hashes" not in row
        point = row["test"]
        (tn, fp), (fn, tp) = point["confusion_matrix"]
        assert tn + fp + fn + tp == published["partitions"]["partitions"]["test"]["rows"]
        assert point["recall"] == pytest.approx(tp / (tp + fn))
        assert point["false_positive_rate"] == pytest.approx(fp / (fp + tn))
    chosen = published["selection"]["classical"]
    selected = [r for r in published["runs"] if report.matches(r, chosen)]
    assert {r["seed"] for r in selected} == set(cfg["seeds"])
    assert (published["conclusion"] == "target_met") == report.meets_target(selected, cfg)
    quantum = [r for r in published["runs"] if report.matches(r, published["selection"]["quantum"])]
    assert {r["seed"] for r in quantum} == set(cfg["seeds"])
    assert (published["quantum_conclusion"] == "target_met") == report.meets_target(quantum, cfg)
    host = [
        r for r in published["runs"] if report.matches(r, published["selection"]["classical_host"])
    ]
    assert all("external" in r for r in host)
    assert not any("external" in r for r in published["runs"] if r["features"] == "page")
    guide = (ROOT / "docs/benchmark-v2.md").read_text(encoding="utf-8")
    for rows, partition in ((selected, "test"), (quantum, "test"), (host, "external")):
        for metric in ("recall", "false_positive_rate", "accuracy"):
            mean = np.mean([r[partition][metric] for r in rows])
            assert f"{mean:.2%}" in guide
    assert (
        published["selected_confidence_intervals"]["resamples_used"] == cfg["bootstrap_resamples"]
    )
