"""Training-only text/numeric representations and deterministic classical searches."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.evaluation.benchmark_v2 import core
from src.features.host_features import CANDIDATE_FEATURES

MODELS = (
    "majority",
    "numeric_rbf",
    "full_logistic",
    "full_linear",
    "reduced_linear",
    "reduced_rbf",
)


def numeric(rows):
    values = rows[list(CANDIDATE_FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Numeric host features must be finite.")
    return values


@dataclass
class Representation:
    seed: int
    max_features: int = 20000

    def fit(self, rows):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            max_features=self.max_features,
            sublinear_tf=True,
            norm="l2",
            lowercase=False,
        )
        text = self.vectorizer.fit_transform(rows.host.tolist())
        self.scaler = StandardScaler().fit(numeric(rows))
        full = self._combine(text, self.scaler.transform(numeric(rows)))
        if min(full.shape) <= 4 or text.shape[1] < 4:
            raise ValueError(
                "At least five training rows and four text features are required for four components."
            )
        self.svd = TruncatedSVD(n_components=4, random_state=self.seed).fit(full)
        reduced = self.svd.transform(full)
        if (
            not np.isfinite(reduced).all()
            or not np.isfinite(self.svd.explained_variance_ratio_).all()
        ):
            raise ValueError("Degenerate or non-finite SVD representation.")
        self.low, self.high = reduced.min(axis=0), reduced.max(axis=0)
        return self

    @staticmethod
    def _combine(text, values):
        number_block = Normalizer(norm="l2").transform(values)
        return sparse.hstack(
            [text * np.sqrt(0.5), sparse.csr_matrix(number_block * np.sqrt(0.5))], format="csr"
        )

    def transform(self, rows):
        values = self.scaler.transform(numeric(rows))
        full = self._combine(self.vectorizer.transform(rows.host.tolist()), values)
        components = self.svd.transform(full)
        unit, clipping = core.angles(components, self.low, self.high, 1.0)
        return {"numeric": values, "full": full, "reduced": unit}, clipping

    def diagnostics(self):
        return {
            "text_features": len(self.vectorizer.vocabulary_),
            "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
            "component_minima": self.low.tolist(),
            "component_maxima": self.high.tolist(),
        }


def view(name):
    return (
        "numeric"
        if name in {"majority", "numeric_rbf"}
        else "full"
        if name.startswith("full_")
        else "reduced"
    )


def candidates(name, config):
    if name == "majority":
        return [{}]
    if name.endswith("rbf"):
        return [
            {"C": c, "gamma": gamma} for c in config["c_values"] for gamma in config["gamma_values"]
        ]
    if name == "quantum":
        return [
            {"encoding": encoding, "angle_max": maximum, "C": c}
            for encoding in config["maps"]
            for maximum in config["angle_maxima"]
            for c in config["c_values"]
        ]
    return [{"C": c} for c in config["c_values"]]


def estimator(name, params, seed):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name.endswith("rbf"):
        return SVC(kernel="rbf", **params)
    if name == "quantum":
        return SVC(kernel="precomputed", C=params["C"])
    if name == "full_logistic":
        return LogisticRegression(C=params["C"], max_iter=10000, random_state=seed)
    return LinearSVC(C=params["C"], max_iter=100000, random_state=seed, dual="auto")


def fit_model(model, values, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return model.fit(values, labels)


def scores(model, values):
    return (
        model.decision_function(values)
        if hasattr(model, "decision_function")
        else model.predict_proba(values)[:, 1]
    )
