from __future__ import annotations

import pytest

from spatial_mapping_phase2.p04_intrinsic_fleet import (
    CameraIntrinsicEstimate,
    IntrinsicFleetError,
    build_fleet_profiles,
    huber_location,
    summarize_between_camera_variation,
)


def _estimate(camera_id: str, focal: float, distortion: float) -> CameraIntrinsicEstimate:
    return CameraIntrinsicEstimate(
        camera_id,
        "stream-profile-v1",
        "simple_radial",
        1920,
        1080,
        focal,
        focal,
        960.0,
        540.0,
        (distortion,),
        0.01,
    )


def test_leave_one_out_profiles_give_each_camera_one_vote() -> None:
    estimates = (
        _estimate("office-cam-01", 1200.0, -0.20),
        _estimate("office-cam-02", 1400.0, -0.28),
        _estimate("office-cam-03", 9000.0, 2.0),
        _estimate("office-cam-04", 1410.0, -0.29),
    )

    mean, component_median, huber = build_fleet_profiles(
        estimates, exclude_camera_id="office-cam-03"
    )

    assert mean.included_camera_ids == (
        "office-cam-01",
        "office-cam-02",
        "office-cam-04",
    )
    assert mean.fx_pixels == pytest.approx((1200 + 1400 + 1410) / 3)
    assert component_median.fx_pixels == pytest.approx(1400.0)
    assert 1390.0 < huber.fx_pixels < 1400.0
    assert all(profile.fx_pixels < 2000 for profile in (mean, component_median, huber))


def test_variation_separates_between_and_within_camera_spread() -> None:
    result = summarize_between_camera_variation(
        (
            _estimate("office-cam-01", 1300.0, -0.25),
            _estimate("office-cam-02", 1400.0, -0.28),
            _estimate("office-cam-03", 1350.0, -0.26),
        )
    )
    assert result["focal_range_pixels"] == pytest.approx(100.0)
    assert result["maximum_within_camera_focal_cv"] == pytest.approx(0.01)


def test_rejects_incompatible_profiles_duplicate_cameras_and_too_few_inputs() -> None:
    estimates = [
        _estimate("office-cam-01", 1300.0, -0.25),
        _estimate("office-cam-02", 1400.0, -0.28),
        _estimate("office-cam-04", 1410.0, -0.29),
    ]
    incompatible = CameraIntrinsicEstimate(
        "office-cam-05",
        "stream-profile-v2",
        "simple_radial",
        1920,
        1080,
        1400,
        1400,
        960,
        540,
        (-0.28,),
        0.01,
    )
    with pytest.raises(IntrinsicFleetError, match="one profile"):
        build_fleet_profiles((*estimates, incompatible), exclude_camera_id="none")
    with pytest.raises(IntrinsicFleetError, match="exactly one"):
        build_fleet_profiles((*estimates, estimates[0]), exclude_camera_id="none")
    with pytest.raises(IntrinsicFleetError, match="at least three"):
        build_fleet_profiles(estimates[:2], exclude_camera_id="none")


def test_huber_location_rejects_invalid_input() -> None:
    assert huber_location((1.0, 1.0, 1.0)) == pytest.approx(1.0)
    with pytest.raises(IntrinsicFleetError, match="finite"):
        huber_location(())
    with pytest.raises(IntrinsicFleetError, match="positive"):
        huber_location((1.0, 2.0), tuning=0)
