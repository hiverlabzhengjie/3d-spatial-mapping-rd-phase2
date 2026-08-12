import math

import pytest

from spatial_mapping_phase2.capture_time import TimestampTransform


def test_timestamp_transform_defaults_to_identity_mapping() -> None:
    assert TimestampTransform().apply(12.5) == 12.5


def test_timestamp_transform_applies_a_finite_affine_mapping() -> None:
    transform = TimestampTransform(scale=1.001, offset_seconds=0.25)

    assert transform.apply(10.0) == pytest.approx(10.26)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf])
def test_timestamp_transform_rejects_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        TimestampTransform(scale=scale)


@pytest.mark.parametrize("offset_seconds", [math.nan, math.inf, -math.inf])
def test_timestamp_transform_rejects_non_finite_offsets(offset_seconds: float) -> None:
    with pytest.raises(ValueError, match="offset"):
        TimestampTransform(offset_seconds=offset_seconds)


@pytest.mark.parametrize("source_timestamp_seconds", [math.nan, math.inf, -math.inf])
def test_timestamp_transform_rejects_non_finite_source_time(
    source_timestamp_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="source timestamp"):
        TimestampTransform().apply(source_timestamp_seconds)


def test_timestamp_transform_rejects_a_negative_capture_time() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TimestampTransform(offset_seconds=-1.0).apply(0.5)
