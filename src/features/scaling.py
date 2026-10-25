"""Training-only min-max scaling of features to rotation angles in [0, pi/2].

The circuit applies a phase of 2 * x to each qubit. With x up to pi/2 the phase stays in
[0, pi], so the smallest and largest training values remain distinguishable; with x up to
pi they would both give a phase equivalent to zero.

Fit on training rows only. Test and inference values outside the training range are
clipped to [0, pi/2] and counted. A feature that is constant in training maps to 0 for
every partition, because it carries no information the circuit could encode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = 1
ANGLE_MAX = math.pi / 2


class ScalingError(ValueError):
    """Input values or a saved scaler are unusable."""


def _as_matrix(values: object, width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ScalingError(f"Expected a 2-D array with {width} feature columns.")
    if not np.isfinite(matrix).all():
        raise ScalingError("Missing or non-finite feature values are not accepted.")
    return matrix


@dataclass(frozen=True)
class AngleScaler:
    features: tuple[str, ...]
    minima: tuple[float, ...]
    maxima: tuple[float, ...]

    @classmethod
    def fit(cls, features: list[str] | tuple[str, ...], train: object) -> AngleScaler:
        names = tuple(features)
        if not names or len(set(names)) != len(names):
            raise ScalingError("Feature names must be unique and non-empty.")
        matrix = _as_matrix(train, len(names))
        if len(matrix) == 0:
            raise ScalingError("Cannot fit on zero training rows.")
        return cls(
            names,
            tuple(float(v) for v in matrix.min(axis=0)),
            tuple(float(v) for v in matrix.max(axis=0)),
        )

    @property
    def constant(self) -> tuple[bool, ...]:
        return tuple(low == high for low, high in zip(self.minima, self.maxima, strict=True))

    def transform(self, values: object) -> tuple[np.ndarray, list[int]]:
        """Return angles and, per feature, how many values were clipped."""
        matrix = _as_matrix(values, len(self.features))
        low = np.array(self.minima)
        span = np.array(self.maxima) - low
        constant = np.array(self.constant)
        scaled = np.zeros_like(matrix)
        varying = ~constant
        scaled[:, varying] = (matrix[:, varying] - low[varying]) / span[varying] * ANGLE_MAX
        outside = (scaled < 0) | (scaled > ANGLE_MAX)
        return np.clip(scaled, 0.0, ANGLE_MAX), [int(n) for n in outside.sum(axis=0)]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "method": "training-only min-max to [0, angle_max]; clip outside; constant feature -> 0",
            "angle_max": ANGLE_MAX,
            "features": list(self.features),
            "minima": list(self.minima),
            "maxima": list(self.maxima),
            "constant": list(self.constant),
        }

    @classmethod
    def from_dict(cls, data: dict) -> AngleScaler:
        try:
            if data["schema_version"] != SCHEMA_VERSION:
                raise ScalingError("Unsupported scaler schema version.")
            scaler = cls(
                tuple(data["features"]),
                tuple(float(v) for v in data["minima"]),
                tuple(float(v) for v in data["maxima"]),
            )
        except (KeyError, TypeError) as error:
            raise ScalingError(f"Invalid saved scaler: {error}") from error
        width = len(scaler.features)
        if width == 0 or len(scaler.minima) != width or len(scaler.maxima) != width:
            raise ScalingError("Saved scaler arrays do not match its feature list.")
        if any(low > high for low, high in zip(scaler.minima, scaler.maxima, strict=True)):
            raise ScalingError("Saved scaler has a minimum above its maximum.")
        if data.get("angle_max") != ANGLE_MAX:
            raise ScalingError("Saved scaler uses a different angle range.")
        if list(scaler.constant) != list(data.get("constant", scaler.constant)):
            raise ScalingError("Saved constant-feature flags are inconsistent.")
        return scaler
