"""Frozen external preparation, featurization, and bootstrap on synthetic rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.evaluation import external
from src.features.scaling import AngleScaler

pytestmark = pytest.mark.unit
LABELS = {"phishing": 1, "legitimate": 0}


def frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["url", "status"])


def test_steps_run_in_order_and_count_each_exclusion() -> None:
    raw = frame(
        [
            ("not a url", "phishing"),  # invalid host
            ("https://www.mixed.example.com/a", "phishing"),  # conflicting group
            ("https://login.mixed.example.com/b", "legitimate"),
            ("https://seen.example.org/", "phishing"),  # in PhiUSIIL
            ("https://keep.example.net/x", "phishing"),  # first of its group
            ("http://other.keep.example.net/y", "phishing"),  # duplicate group
            ("https://www.solo.example.io:8080/", "legitimate"),
        ]
    )
    benchmark = external.phiusiil_groups(pd.Series(["WWW.Seen.Example.org."]))
    prepared = external.prepare(raw, LABELS, benchmark)
    assert [(e["step"], e["rows"]) for e in prepared.exclusions] == [
        ("invalid_host", 1),
        ("conflicting_labels", 2),
        ("phiusiil_overlap", 1),
        ("duplicate_group", 1),
    ]
    assert prepared.exclusions[1]["groups"] == 1 and prepared.exclusions[2]["groups"] == 1
    assert prepared.input_rows == 7
    assert prepared.rows["row_index"].tolist() == [4, 6]
    assert prepared.rows["host"].tolist() == ["keep.example.net", "solo.example.io"]
    assert prepared.class_counts == {"legitimate": 1, "phishing": 1}


def test_overlap_uses_registrable_domain_of_any_phiusiil_row() -> None:
    groups = external.phiusiil_groups(pd.Series(["a.shop.example.co.uk", "", "10.0.0.1"]))
    assert groups == {"dns:example.co.uk", "ip:10.0.0.1"}
    raw = frame([("https://other.example.co.uk/", "phishing"), ("http://10.0.0.1/", "phishing")])
    prepared = external.prepare(raw, LABELS, groups)
    assert prepared.exclusions[2]["rows"] == 2 and prepared.rows.empty


def test_host_without_registrable_domain_counts_as_invalid() -> None:
    raw = frame(
        [("https://host.notarealsuffix/", "phishing"), ("https://ok.example.com/", "phishing")]
    )
    prepared = external.prepare(raw, LABELS, set())
    assert prepared.exclusions[0] == {
        "step": "invalid_host",
        "rows": 1,
        "valid_host_without_registrable_domain": 1,
    }


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(external.ExternalError, match="Unknown status"):
        external.prepare(frame([("https://a.example/", "suspicious")]), LABELS, set())


def test_featurize_uses_scaler_order_and_counts_clipping() -> None:
    scaler = AngleScaler(("host_label_count", "host_hyphen_count"), (2.0, 0.0), (3.0, 1.0))
    raw, angles, clipped = external.featurize(
        pd.Series(["a.example", "a-b-c.d.example", "x.y.z.example"]), scaler
    )
    assert raw.tolist() == [[2.0, 0.0], [3.0, 2.0], [4.0, 0.0]]
    assert angles.max() == pytest.approx(np.pi / 2)
    assert clipped == {
        "per_feature": {"host_label_count": 1, "host_hyphen_count": 1},
        "rows_with_any_clipped": 2,
    }


def test_point_metrics_match_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    predictions = rng.integers(0, 2, 200)
    scores = rng.normal(size=200)
    metrics = external.point_metrics(labels, predictions, scores)
    assert metrics["accuracy"] == pytest.approx(accuracy_score(labels, predictions))
    assert metrics["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(labels, predictions)
    )
    assert metrics["precision"] == pytest.approx(precision_score(labels, predictions))
    assert metrics["recall"] == pytest.approx(recall_score(labels, predictions))
    assert metrics["f1"] == pytest.approx(f1_score(labels, predictions))
    assert metrics["average_precision"] == pytest.approx(average_precision_score(labels, scores))
    never = external.point_metrics(labels, np.zeros(200, dtype=int), None)
    assert never["precision"] == never["f1"] == never["recall"] == 0.0
    assert never["average_precision"] is None


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 300)
    predictions = np.where(rng.random(300) < 0.8, labels, 1 - labels)
    scores = predictions + rng.normal(scale=0.5, size=300)
    indices = external.resample_indices(300, 500, 42)
    assert np.array_equal(indices, external.resample_indices(300, 500, 42))
    intervals, used = external.bootstrap_intervals(labels, predictions, scores, indices, 0.95)
    point = external.point_metrics(labels, predictions, scores)
    assert used == 500
    for name in external.METRICS:
        low, high = intervals[name]
        assert low <= point[name] <= high


def test_bootstrap_skips_single_class_resamples_and_nulls_missing_scores() -> None:
    labels = np.array([0, 1])
    predictions = np.array([0, 1])
    indices = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    intervals, used = external.bootstrap_intervals(labels, predictions, None, indices, 0.95)
    assert used == 2
    assert intervals["accuracy"] == [1.0, 1.0]
    assert intervals["average_precision"] is None
    none, used = external.bootstrap_intervals(labels, predictions, None, indices[:2], 0.95)
    assert used == 0 and all(value is None for value in none.values())


def test_protocol_hash_ignores_line_endings(tmp_path) -> None:
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert external.protocol_sha256(lf) == external.protocol_sha256(crlf)


def test_ipv6_hosts_group_across_port_forms() -> None:
    same = frame(
        [("http://[2001:db8::1]:8080/a", "phishing"), ("http://[2001:db8::1]/b", "phishing")]
    )
    prepared = external.prepare(same, LABELS, set())
    assert prepared.rows["group"].tolist() == ["ip:2001:db8::1"]
    assert prepared.exclusions[3]["rows"] == 1
    mixed = frame(
        [("http://[2001:db8::1]:8080/a", "phishing"), ("http://[2001:db8::1]/b", "legitimate")]
    )
    prepared = external.prepare(mixed, LABELS, set())
    assert prepared.exclusions[1]["rows"] == 2 and prepared.rows.empty


def test_constant_training_feature_is_never_counted_as_clipped() -> None:
    scaler = AngleScaler(("host_label_count", "host_hyphen_count"), (2.0, 0.0), (2.0, 1.0))
    _, angles, clipped = external.featurize(pd.Series(["a.b.example"]), scaler)
    assert angles[0, 0] == 0.0
    assert clipped == {
        "per_feature": {"host_label_count": 0, "host_hyphen_count": 0},
        "rows_with_any_clipped": 0,
    }
