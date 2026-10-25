"""The four fixed benchmark models (docs/benchmark-protocol.md, "Fixed model configurations").

No hyperparameter search, cross-validation selection, class weights, or threshold tuning.
SVMs keep the default probability=False; scikit-learn 1.9 deprecates passing it explicitly.
All models receive the same scaled features; the quantum-kernel SVM receives the
precomputed kernel built from those features instead.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

MODEL_NAMES = ("majority", "logistic_regression", "rbf_svm", "quantum_kernel_svm")


def build_models() -> dict:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        "rbf_svm": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "quantum_kernel_svm": SVC(kernel="precomputed", C=1.0),
    }


def settings(models: dict) -> dict:
    """Explicit, JSON-serializable settings of each model, recorded with the run."""
    keep = {"strategy", "C", "solver", "max_iter", "kernel", "gamma", "class_weight"}
    return {
        name: {key: value for key, value in model.get_params().items() if key in keep}
        for name, model in models.items()
    }


def fit_all(angles: np.ndarray, kernel: np.ndarray, labels: np.ndarray) -> dict:
    """Fit every model on the training rows. A convergence warning is a run failure."""
    if len(angles) != len(labels) or kernel.shape != (len(labels), len(labels)):
        raise ValueError("Training features, kernel, and labels must describe the same rows.")
    models = build_models()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, model in models.items():
            model.fit(kernel if name == "quantum_kernel_svm" else angles, labels)
    return models
