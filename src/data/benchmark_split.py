"""Deterministic, domain-disjoint benchmark sample and training-side feature selection.

Order of operations (see docs/benchmark-protocol.md):
1. Drop exact duplicate URLs and rows without a usable host.
2. Group rows by registrable domain (IP hosts by address); drop label-conflicting groups.
3. Draw the test rows first, one per group, balanced by class.
4. Select the features on a training-side pool that excludes every test group.
5. Draw the training rows from that pool, one per group, balanced by class.
The test rows never influence feature selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import tldextract
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

from src.features.host_features import CANDIDATE_FEATURES, host_features, is_ip, normalize_host

SEED = 42
TRAIN_PER_CLASS = 12
TEST_PER_CLASS = 6
FEATURE_COUNT = 4
SELECTION_SAMPLE_PER_CLASS = 10_000
MAX_ABS_SPEARMAN = 0.7

# Bundled Public Suffix List snapshot only: no network access. Private suffixes such as
# firebaseapp.com are honored, so each customer subdomain there is its own group.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


class SplitError(Exception):
    """The requested deterministic sample cannot be built from the data."""


def group_key(host: str) -> str | None:
    """Registrable domain (private PSL suffixes included), or the IP address itself."""
    if not host:
        return None
    if is_ip(host):
        return f"ip:{host}"
    parts = _EXTRACT(host)
    if parts.domain and parts.suffix:
        return f"dns:{parts.domain}.{parts.suffix}"
    if parts.suffix and not parts.subdomain:
        # The host is itself a public suffix, e.g. a private entry such as
        # cloudflare-ipfs.com. Nothing smaller is registrable, so the host is its own group.
        return f"dns:{host}"
    return None


def group_hash(key: str) -> str:
    """Stable identifier that can be published without listing phishing domains."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class PreparedData:
    rows: pd.DataFrame
    exclusions: dict[str, int] = field(default_factory=dict)


def prepare(raw: pd.DataFrame) -> PreparedData:
    """Clean, group, and featurize; `raw` needs URL, Domain, and label columns."""
    exclusions: dict[str, int] = {}
    data = raw[["URL", "Domain", "label"]].copy()
    data["row_index"] = raw.index.to_numpy()

    duplicated = data["URL"].duplicated(keep="first")
    exclusions["duplicate_url"] = int(duplicated.sum())
    data = data[~duplicated]

    data["host"] = data["Domain"].astype(str).map(normalize_host)
    data["group"] = data["host"].map(group_key)
    no_group = data["group"].isna()
    exclusions["unusable_host"] = int(no_group.sum())
    data = data[~no_group]

    conflicting = data.groupby("group")["label"].transform("nunique") > 1
    exclusions["label_conflicting_group_rows"] = int(conflicting.sum())
    exclusions["label_conflicting_groups"] = int(data.loc[conflicting, "group"].nunique())
    data = data[~conflicting]

    data["is_phishing"] = 1 - data["label"].astype(int)
    features = pd.DataFrame([host_features(host) for host in data["host"]], index=data.index)
    data = pd.concat([data, features], axis=1)
    return PreparedData(rows=data.reset_index(drop=True), exclusions=exclusions)


def _one_row_per_group(
    rows: pd.DataFrame, per_class: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Pick `per_class` distinct groups per class, then one row inside each group."""
    chosen = []
    for target in (0, 1):
        groups = np.sort(rows.loc[rows["is_phishing"] == target, "group"].unique())
        if len(groups) < per_class:
            raise SplitError(f"Only {len(groups)} groups available for class {target}.")
        for group in rng.choice(groups, size=per_class, replace=False):
            members = rows[rows["group"] == group].sort_values("row_index")
            chosen.append(members.iloc[int(rng.integers(len(members)))])
    return pd.DataFrame(chosen).reset_index(drop=True)


def select_features(pool: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Rank candidates by mutual information on a balanced pool sample; skip redundant ones."""
    sample = pd.concat(
        [
            pool[pool["is_phishing"] == target].sample(
                n=min(SELECTION_SAMPLE_PER_CLASS, int((pool["is_phishing"] == target).sum())),
                random_state=int(rng.integers(2**31 - 1)),
            )
            for target in (0, 1)
        ]
    )
    candidates = [name for name in CANDIDATE_FEATURES if sample[name].nunique() > 1]
    scores = mutual_info_classif(
        sample[candidates].to_numpy(),
        sample["is_phishing"].to_numpy(),
        discrete_features=[name in {"host_is_ip"} for name in candidates],
        random_state=SEED,
    )
    ranking = sorted(zip(candidates, scores, strict=True), key=lambda item: -item[1])
    selected: list[str] = []
    skipped: dict[str, str] = {}
    for name, _score in ranking:
        if len(selected) == FEATURE_COUNT:
            break
        correlated = [
            other
            for other in selected
            if abs(spearmanr(sample[name], sample[other]).statistic) >= MAX_ABS_SPEARMAN
        ]
        if correlated:
            skipped[name] = f"|Spearman| >= {MAX_ABS_SPEARMAN} with {correlated[0]}"
            continue
        selected.append(name)
    if len(selected) < FEATURE_COUNT:
        raise SplitError(f"Only {len(selected)} non-redundant features available.")
    return {
        "selected": selected,
        "mutual_information": {name: round(float(score), 6) for name, score in ranking},
        "skipped_as_redundant": skipped,
        "selection_rows": int(len(sample)),
    }


def build(raw: pd.DataFrame, seed: int = SEED) -> dict:
    """Return the test rows, the selected features, and the training rows."""
    prepared = prepare(raw)
    rows = prepared.rows
    rng = np.random.default_rng(seed)

    test = _one_row_per_group(rows, TEST_PER_CLASS, rng)
    pool = rows[~rows["group"].isin(set(test["group"]))]
    selection = select_features(pool, rng)
    train = _one_row_per_group(pool, TRAIN_PER_CLASS, rng)

    if set(train["group"]) & set(test["group"]):
        raise SplitError("Training and test groups overlap.")
    return {
        "prepared_rows": int(len(rows)),
        "exclusions": prepared.exclusions,
        "pool_rows": int(len(pool)),
        "selection": selection,
        "train": train,
        "test": test,
    }
