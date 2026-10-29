"""Offline text/Qiskit study: leakage, real local circuits, integrity and publication."""

from __future__ import annotations

import ast
import copy
import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.evaluation.benchmark_v2 import core
from src.evaluation.benchmark_v2_text import experiment, features, quantum, report
from src.features.host_features import host_features

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def config():
    c = experiment.load_config(ROOT / "configs/benchmark-v2-text.json")
    c.update(
        train_sizes=[12],
        seeds=[42],
        c_values=[1],
        gamma_values=["scale"],
        angle_maxima=[float(np.pi / 4)],
        shots=32,
        batch_size=8,
        subset_per_class=2,
        bootstrap_resamples=8,
        max_features=500,
    )
    return c


def rows(n=60, prefix="sample"):
    hosts = [f"{prefix}{i}-{'verify-secure' if i % 2 else 'example-shop'}.com" for i in range(n)]
    table = pd.DataFrame(
        {
            "host": hosts,
            "row_index": np.arange(n),
            "is_phishing": np.arange(n) % 2,
            "group_hash": [core.group_hash(f"dns:{h}") for h in hosts],
        }
    )
    return pd.concat([table, pd.DataFrame([host_features(h) for h in hosts])], axis=1)


def test_representation_is_fitted_only_on_training_and_has_four_components():
    train, held = rows(24), rows(12, "unseenuniquetoken")
    rep = features.Representation(42).fit(train)
    vocabulary = dict(rep.vectorizer.vocabulary_)
    means, low, high = rep.scaler.mean_.copy(), rep.low.copy(), rep.high.copy()
    views, clipping = rep.transform(held)
    assert views["reduced"].shape == (12, 4)
    assert views["full"].shape[1] == len(vocabulary) + 8
    assert rep.vectorizer.vocabulary_ == vocabulary
    np.testing.assert_allclose(rep.scaler.mean_, means)
    np.testing.assert_allclose(rep.low, low)
    np.testing.assert_allclose(rep.high, high)
    assert views["reduced"].min() >= 0 and views["reduced"].max() <= 1
    assert len(clipping["per_feature"]) == 4
    assert len(rep.diagnostics()["explained_variance_ratio"]) == 4


@pytest.mark.parametrize("hosts", [["", " ", "", "", "", ""], ["ab"] * 6])
def test_empty_or_insufficient_text_fails_without_fallback(hosts):
    table = rows(len(hosts))
    table["host"] = hosts
    with pytest.raises(ValueError):
        features.Representation(42).fit(table)


def test_nonfinite_features_are_rejected():
    table = rows(12)
    table.loc[0, "host_entropy"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        features.Representation(42).fit(table)


def test_recovery_rejects_label_row_and_group_changes():
    table = rows(12)
    expected = {"rows": 12, "phishing": 6, "membership_sha256": experiment.membership(table)}
    experiment.verify_recovery(table, expected, table.copy())
    modified = table.copy()
    modified.loc[0, "group_hash"] = "changed"
    with pytest.raises(ValueError, match="membership"):
        experiment.verify_recovery(modified, expected)
    modified = table.copy()
    modified.loc[0, "row_index"] = 900
    with pytest.raises(ValueError, match="row_index"):
        experiment.verify_recovery(modified, expected, table)
    modified = table.copy()
    modified.loc[0, "is_phishing"] = 1
    with pytest.raises(ValueError, match="labels"):
        experiment.verify_recovery(modified, expected)


@pytest.mark.parametrize("encoding", ["z", "zz"])
def test_actual_qiskit_states_match_analytic_reference_and_order(encoding, tmp_path):
    x = np.random.default_rng(42).uniform(0, np.pi / 2, (7, 4))
    actual = quantum.states(x, encoding, "transform-a", tmp_path)
    expected = core.states(x, encoding)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(
        core.kernel(actual[:3], actual[[5, 2, 4]]),
        core.kernel(expected[:3], expected[[5, 2, 4]]),
        atol=1e-13,
    )
    first_file = next(tmp_path.glob("*.npy"))
    quantum.states(x, encoding, "transform-b", tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 2
    np.testing.assert_array_equal(quantum.states(x, encoding, "transform-a", tmp_path), actual)
    first_file.write_bytes(b"bad")
    with pytest.raises(ValueError, match="cache"):
        quantum.states(x, encoding, "transform-a", tmp_path)


@pytest.mark.parametrize("encoding", ["z", "zz"])
def test_aer_pairs_counts_seed_and_resume(tmp_path, encoding):
    x = np.random.default_rng(7).uniform(0, np.pi / 2, (3, 4))
    a = quantum.measured_kernel(x, x, encoding, 128, 42, 4, tmp_path / "a")
    b = quantum.measured_kernel(x, x, encoding, 128, 42, 4, tmp_path / "b")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.diag(a), 1)
    np.testing.assert_array_equal(
        quantum.measured_kernel(x, x, encoding, 128, 42, 4, tmp_path / "a"), a
    )
    counts = json.loads((tmp_path / "a/counts-000000.json").read_text())["pairs"]
    assert [(r["left"], r["right"]) for r in counts] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert all(sum(r["counts"].values()) == 128 for r in counts)
    with pytest.raises(ValueError, match="identity"):
        quantum.measured_kernel(x, x, encoding, 129, 42, 4, tmp_path / "a")


def test_aer_rejects_modified_raw_counts(tmp_path):
    x = np.zeros((2, 4))
    quantum.measured_kernel(x, x, "z", 32, 42, 4, tmp_path)
    (tmp_path / "counts-000000.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        quantum.measured_kernel(x, x, "z", 32, 42, 4, tmp_path)


@pytest.fixture
def study(tmp_path, monkeypatch):
    cfg = config()
    run = experiment.Experiment(cfg, "synthetic", tmp_path)
    identity = {"config": cfg, "synthetic": True}
    monkeypatch.setattr(run, "identity", lambda: copy.deepcopy(identity))
    run.output.mkdir(parents=True)
    manifest = {"files": {}, "partitions": {}, "training": {"42": {}}, "subsets": {}}
    for name, n in (("development", 36), ("calibration", 12), ("test", 12), ("external", 12)):
        table = rows(n, name)
        path = run.output / f"{name}.csv"
        table.to_csv(path, index=False)
        manifest["files"][path.name] = core.digest(path)
        manifest["partitions"][name] = {
            "rows": n,
            "phishing": n // 2,
            "membership_sha256": experiment.membership(table),
        }
        if name == "development":
            manifest["training"]["42"] = {
                str(n): v.tolist() for n, v in core.nested_samples(table, [12], 42).items()
            }
        elif name in {"test", "external"}:
            manifest["subsets"][name] = [0, 2, 1, 3]
    core.write_json(run.output / "identity.json", identity)
    core.write_json(run.output / "partitions.json", manifest)
    (tmp_path / "README.md").write_text("# Historical project\n\nKeep this section.\n")
    return run


def test_fold_transforms_see_only_fold_training(study, monkeypatch):
    observed = []
    original = features.Representation.fit

    def fit(self, table):
        observed.append(table.group_hash.tolist())
        return original(self, table)

    monkeypatch.setattr(features.Representation, "fit", fit)
    table = study.training_rows(12, 42)
    study.folds(12, 42)
    folds = StratifiedKFold(3, shuffle=True, random_state=42).split(table, table.is_phishing)
    for actual, (train, _) in zip(observed, folds, strict=True):
        assert actual == table.iloc[train].group_hash.tolist()


def test_full_local_synthetic_workflow_never_connects_and_publishes_only_block(study, monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("Network access is forbidden in the local study")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    original_table = study.table

    def guard(name):
        if name in {"test", "external"}:
            assert (study.output / "selection.json").exists()
            assert "comparisons" in json.loads((study.output / "selection.json").read_text())
        return original_table(name)

    monkeypatch.setattr(study, "table", guard)
    study.train("classical")
    study.train("statevector")
    for record in (study.record(i) for i in study.run_ids()):
        trials = record["search"]
        assert record["selected_trial"] == max(
            range(len(trials)), key=lambda i: trials[i]["mean_ap"]
        )
    study.shots()
    result = report.report(study)
    assert result["status"] == "complete" and len(result["runs"]) == 7
    assert len(result["charts"]) == 3
    assert '"host"' not in json.dumps(result)
    before = (study.root / "README.md").read_text()
    report.publish(study)
    published = (study.root / "README.md").read_text()
    assert published.startswith(before.rstrip())
    assert published.count(report.BEGIN) == published.count(report.END) == 1
    report.publish(study)
    assert published == (study.root / "README.md").read_text()
    (study.root / "README.md").write_text(published + report.BEGIN)
    with pytest.raises(ValueError, match="Malformed"):
        report.publish(study)
    (study.root / "README.md").write_text(published)
    study.train("classical")
    # A corrupted chart cannot be advertised as belonging to this report.
    (study.root / next(iter(result["charts"]))).write_text("modified")
    with pytest.raises(ValueError, match="Chart/report"):
        report.publish(study)


def test_freeze_requires_completed_classical_and_quantum_searches(study):
    with pytest.raises(OSError):
        study.freeze()
    assert not (study.output / "selection.json").exists()


def test_identity_and_partition_modifications_refuse_resume(study):
    original = study.identity
    study.identity = lambda: {**original(), "changed": True}
    with pytest.raises(ValueError, match="identity"):
        study.check()
    study.identity = original
    with (study.output / "development.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="modified"):
        study.check()


def test_local_modules_do_not_import_cloud_credential_or_network_clients():
    forbidden = {"qiskit_ibm_runtime", "dotenv", "requests", "urllib", "boto3", "src.config"}
    for path in (ROOT / "src/evaluation/benchmark_v2_text").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(any(a.name.startswith(f) for f in forbidden) for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not any((node.module or "").startswith(f) for f in forbidden)


def test_run_id_and_configuration_validation(tmp_path):
    for bad in ("../escape", "UPPER", "bad id", "x" * 81):
        with pytest.raises(ValueError, match="Run ID"):
            experiment.Experiment(config(), bad, tmp_path)
    c = config()
    c["components"] = 8
    core.write_json(tmp_path / "config.json", c)
    with pytest.raises(ValueError, match="four components"):
        experiment.load_config(tmp_path / "config.json")


def test_interrupted_aer_batch_without_checksum_is_recomputed(tmp_path):
    x = np.zeros((2, 4))
    expected = quantum.measured_kernel(x, x, "z", 32, 42, 4, tmp_path)
    (tmp_path / "counts-000000.sha256").unlink()
    (tmp_path / "counts-000000.json").write_text("unfinished")
    np.testing.assert_array_equal(quantum.measured_kernel(x, x, "z", 32, 42, 4, tmp_path), expected)


def test_published_text_study_is_complete_and_documentation_uses_measured_values():
    study = json.loads(
        (ROOT / "reports/benchmark-v2-text/portfolio-text-20261028-verified.json").read_text()
    )
    assert study["status"] == "complete"
    assert len(study["runs"]) == 42
    assert len({r["id"] for r in study["runs"]}) == 42
    total = 0
    for row in study["runs"]:
        for part in ("test", "external"):
            metric = row[part]
            (tn, fp), (fn, tp) = metric["confusion_matrix"]
            assert metric["rows"] == tn + fp + fn + tp
            assert metric["recall"] == pytest.approx(tp / (tp + fn))
            assert metric["false_positive_rate"] == pytest.approx(fp / (fp + tn))
        for measurement in row.get("measurements", {}).values():
            assert measurement["shots"] == 1024
            assert measurement["rows"] == 40
            total += measurement["pairs"]
    assert total == 29760
    assert study["search_budget"]["quantum"] == study["search_budget"]["reduced_rbf"] == 12
    guide = (ROOT / "docs/benchmark-v2-text.md").read_text(encoding="utf-8")
    selected = study["selection"]["classical"]
    for part in ("test_summary", "external_summary"):
        summary = next(r for r in study[part] if report.match(r, selected))
        for name in ("accuracy", "recall", "false_positive_rate"):
            assert f"{summary['metrics'][name]['mean']:.2%}" in guide
    for filename, expected in study["charts"].items():
        assert core.digest(ROOT / filename) == expected
