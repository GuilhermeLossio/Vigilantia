"""Training-only angle scaling; synthetic values only."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.features.scaling import ANGLE_MAX, AngleScaler, ScalingError

pytestmark = pytest.mark.unit
FEATURES = ["a", "b", "c"]
TRAIN = [[0.0, 10.0, 5.0], [2.0, 20.0, 5.0], [4.0, 30.0, 5.0]]


def test_training_range_maps_to_zero_and_angle_max() -> None:
    scaler = AngleScaler.fit(FEATURES, TRAIN)
    angles, clipped = scaler.transform(TRAIN)
    assert angles[:, 0] == pytest.approx([0, ANGLE_MAX / 2, ANGLE_MAX])
    assert angles[:, 1] == pytest.approx([0, ANGLE_MAX / 2, ANGLE_MAX])
    assert clipped == [0, 0, 0]


def test_constant_training_feature_maps_to_zero_everywhere() -> None:
    scaler = AngleScaler.fit(FEATURES, TRAIN)
    assert scaler.constant == (False, False, True)
    angles, clipped = scaler.transform([[1.0, 15.0, 99.0]])
    assert angles[0, 2] == 0
    assert clipped == [0, 0, 0]


def test_values_outside_training_range_are_clipped_and_counted() -> None:
    scaler = AngleScaler.fit(FEATURES, TRAIN)
    angles, clipped = scaler.transform([[-1.0, 40.0, 5.0], [5.0, 10.0, 5.0]])
    assert angles[0, 0] == 0 and angles[1, 0] == pytest.approx(ANGLE_MAX)
    assert angles[0, 1] == pytest.approx(ANGLE_MAX)
    assert clipped == [2, 1, 0]
    assert ((angles >= 0) & (angles <= ANGLE_MAX)).all()


@pytest.mark.parametrize(
    "values",
    [[[np.nan, 1.0, 1.0]], [[np.inf, 1.0, 1.0]], [[1.0, 1.0]], [1.0, 1.0, 1.0]],
)
def test_invalid_inputs_are_rejected(values: object) -> None:
    scaler = AngleScaler.fit(FEATURES, TRAIN)
    with pytest.raises(ScalingError):
        scaler.transform(values)


def test_fit_rejects_bad_feature_lists_and_empty_training() -> None:
    with pytest.raises(ScalingError):
        AngleScaler.fit(["a", "a", "b"], TRAIN)
    with pytest.raises(ScalingError):
        AngleScaler.fit(FEATURES, np.empty((0, 3)))


def test_saved_scaler_round_trips_and_is_validated() -> None:
    scaler = AngleScaler.fit(FEATURES, TRAIN)
    data = scaler.to_dict()
    assert AngleScaler.from_dict(data) == scaler
    with pytest.raises(ScalingError):
        AngleScaler.from_dict({**data, "schema_version": 2})
    with pytest.raises(ScalingError):
        AngleScaler.from_dict({**data, "minima": [0.0, 10.0]})
    with pytest.raises(ScalingError):
        AngleScaler.from_dict({**data, "minima": [9.0, 10.0, 5.0]})
    with pytest.raises(ScalingError):
        AngleScaler.from_dict({**data, "constant": [True, True, True]})
    with pytest.raises(ScalingError):
        AngleScaler.from_dict({"schema_version": 1})
    with pytest.raises(ScalingError, match="angle range"):
        AngleScaler.from_dict({**data, "angle_max": math.pi})
