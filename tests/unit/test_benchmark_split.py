"""Host features, grouping, and the deterministic split on synthetic rows; no network or data."""

from __future__ import annotations

import math
import random

import pandas as pd
import pytest

from src.data import benchmark_split as split
from src.features.host_features import (
    CANDIDATE_FEATURES,
    host_features,
    normalize_host,
    shannon_entropy,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("domain", "host"),
    [
        ("WWW.Example.COM", "example.com"),
        ("www.example.com.", "example.com"),
        ("example.com:8080", "example.com"),
        ("www.www.example.com", "www.example.com"),
        ("[2001:db8::1]:443", "2001:db8::1"),
        ("2001:db8::1", "2001:db8::1"),
    ],
)
def test_normalize_host(domain: str, host: str) -> None:
    assert normalize_host(domain) == host


def test_host_features_values() -> None:
    features = host_features("a1-b.example.co.uk")
    assert set(features) == set(CANDIDATE_FEATURES)
    assert features["host_length"] == 18
    assert features["host_digit_ratio"] == pytest.approx(1 / 18)
    assert features["host_hyphen_count"] == 1
    assert features["host_label_count"] == 4
    assert features["longest_label_length"] == 7
    assert features["tld_length"] == 2
    assert features["host_is_ip"] == 0


def test_ip_host_features() -> None:
    features = host_features("10.0.0.1")
    assert features["host_is_ip"] == 1
    assert features["tld_length"] == 0


def test_entropy() -> None:
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("abcd") == pytest.approx(2.0)
    assert shannon_entropy("") == 0
    with pytest.raises(ValueError):
        host_features("")


@pytest.mark.parametrize(
    ("host", "key"),
    [
        ("login.example.co.uk", "dns:example.co.uk"),
        ("example.co.uk", "dns:example.co.uk"),
        ("victim-1.web.app", "dns:victim-1.web.app"),
        ("cloudflare-ipfs.com", "dns:cloudflare-ipfs.com"),
        ("10.0.0.1", "ip:10.0.0.1"),
        ("localhost", None),
        ("", None),
    ],
)
def test_group_key_uses_registrable_domain_with_private_suffixes(
    host: str, key: str | None
) -> None:
    assert split.group_key(host) == key


def test_group_hash_is_stable_and_hides_the_domain() -> None:
    assert split.group_hash("dns:example.com") == split.group_hash("dns:example.com")
    assert "example" not in split.group_hash("dns:example.com")


def synthetic_rows(groups_per_class: int = 30, rows_per_group: int = 2) -> pd.DataFrame:
    """Varied hosts so the eight candidate features are not all mutually redundant."""
    rng = random.Random(7)
    rows = []
    for label in (1, 0):
        for group in range(groups_per_class):
            letters = "".join(
                rng.choice("abcdefghijklmnop"[: rng.randint(3, 16)])
                for _ in range(rng.randint(4, 14))
            )
            digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 4)))
            hyphen = "-x" * rng.randint(0, 2)
            sub = "login." * rng.randint(0, 2)
            tld = rng.choice(["com", "org", "top", "info", "co.uk"])
            registrable = f"g{group}{'l' if label else 'p'}{letters}{digits}{hyphen}.{tld}"
            for copy in range(rows_per_group):
                host = f"{sub}{registrable}"
                rows.append(
                    {"URL": f"https://www.{host}/{copy}", "Domain": f"www.{host}", "label": label}
                )
    return pd.DataFrame(rows)


def test_prepare_counts_exclusions() -> None:
    raw = pd.DataFrame(
        {
            "URL": ["https://a.com", "https://a.com", "https://b.com/x", "https://b.com/y", "x"],
            "Domain": ["a.com", "a.com", "b.com", "www.b.com", "localhost"],
            "label": [1, 1, 1, 0, 1],
        }
    )
    prepared = split.prepare(raw)
    assert prepared.exclusions == {
        "duplicate_url": 1,
        "unusable_host": 1,
        "label_conflicting_group_rows": 2,
        "label_conflicting_groups": 1,
    }
    assert prepared.rows["row_index"].tolist() == [0]
    assert prepared.rows["is_phishing"].tolist() == [0]


def test_build_is_balanced_domain_disjoint_and_deterministic() -> None:
    raw = synthetic_rows()
    first = split.build(raw)
    second = split.build(raw)
    train, test = first["train"], first["test"]

    assert len(train) == 24 and len(test) == 12
    assert train["is_phishing"].value_counts().to_dict() == {0: 12, 1: 12}
    assert test["is_phishing"].value_counts().to_dict() == {0: 6, 1: 6}
    assert train["group"].is_unique and test["group"].is_unique
    assert not set(train["group"]) & set(test["group"])
    assert train["row_index"].tolist() == second["train"]["row_index"].tolist()
    assert test["row_index"].tolist() == second["test"]["row_index"].tolist()
    assert len(first["selection"]["selected"]) == split.FEATURE_COUNT
    assert first["pool_rows"] == first["prepared_rows"] - 12 * 2


def test_feature_selection_skips_redundant_features() -> None:
    selection = split.build(synthetic_rows())["selection"]
    for name, reason in selection["skipped_as_redundant"].items():
        assert name not in selection["selected"]
        assert "Spearman" in reason
    scores = selection["mutual_information"]
    assert all(math.isfinite(value) for value in scores.values())


def test_too_few_groups_fail_explicitly() -> None:
    with pytest.raises(split.SplitError, match="groups available"):
        split.build(synthetic_rows(groups_per_class=5))
