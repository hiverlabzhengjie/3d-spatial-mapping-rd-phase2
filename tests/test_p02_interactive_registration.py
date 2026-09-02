from dataclasses import replace

import pytest

from spatial_mapping_phase2.p01_observability import CAMERA_IDS
from spatial_mapping_phase2.p02_interactive_registration import (
    INTERACTIVE_REGISTRATION_SCHEMA_VERSION,
    CameraPlacement,
    FramePlacement,
    InteractiveRegistrationError,
    InteractiveRegistrationState,
    PixelPoint,
    PlanMetadata,
    ScaleAxis,
    ScaleControl,
    ScaleSourceKind,
    build_interactive_export,
    empty_registration,
)


def _plan() -> PlanMetadata:
    return PlanMetadata("a" * 64, "office.pdf", 1, 1000, 1200)


def _control(
    control_id: str = "known-width",
    point_b: PixelPoint | None = None,
    axis: ScaleAxis = ScaleAxis.HORIZONTAL,
) -> ScaleControl:
    resolved_point_b = point_b or (
        PixelPoint(100.0, 0.0) if axis is ScaleAxis.HORIZONTAL else PixelPoint(0.0, 100.0)
    )
    return ScaleControl(
        control_id,
        "known permanent endpoints",
        PixelPoint(0.0, 0.0),
        resolved_point_b,
        1.0,
        0.02,
        ScaleSourceKind.PRINTED_DIMENSION,
        axis,
    )


def _state() -> InteractiveRegistrationState:
    empty = empty_registration(_plan())
    cameras = list(empty.cameras)
    cameras[0] = CameraPlacement(
        CAMERA_IDS[0],
        "red-pillar-camera",
        PixelPoint(0.0, 200.0),
        3.0,
        0.1,
        "camera bracket centre",
        PixelPoint(50.0, 250.0),
    )
    return InteractiveRegistrationState(
        INTERACTIVE_REGISTRATION_SCHEMA_VERSION,
        3,
        _plan(),
        (_control(), _control("known-height", axis=ScaleAxis.VERTICAL)),
        FramePlacement(
            PixelPoint(100.0, 100.0),
            PixelPoint(0.0, 100.0),
            "exact corner of permanent pillar",
        ),
        tuple(cameras),
    )


def test_interactive_frame_derives_right_handed_plan_left_x_and_plan_down_y() -> None:
    state = _state()

    assert state.frame is not None
    assert state.frame.x_axis_pixel_unit == pytest.approx((-1.0, 0.0))
    assert state.frame.y_axis_pixel_unit == pytest.approx((0.0, 1.0))
    assert state.world_xy_from_pixel(PixelPoint(0.0, 200.0)) == pytest.approx((1.0, 1.0))


def test_scale_controls_report_disagreement_without_inventing_acceptance_tolerance() -> None:
    state = replace(
        _state(),
        scale_controls=(
            _control(),
            _control("known-height", PixelPoint(0.0, 110.0), ScaleAxis.VERTICAL),
        ),
    )

    assert state.pixels_per_metre == pytest.approx(105.0)
    assert state.scale_spread_fraction == pytest.approx(10.0 / 105.0)


def test_biaxial_scale_requires_one_control_per_direction() -> None:
    incomplete = replace(_state(), scale_controls=(_control(),))

    assert incomplete.pixels_per_metre is None
    assert incomplete.missing_scale_axes == (ScaleAxis.VERTICAL,)
    with pytest.raises(InteractiveRegistrationError, match="horizontal and vertical"):
        incomplete.world_xy_from_pixel(PixelPoint(20.0, 20.0))
    with pytest.raises(InteractiveRegistrationError, match="horizontal and vertical"):
        build_interactive_export(incomplete, {})

    with pytest.raises(InteractiveRegistrationError, match="exactly one horizontal"):
        replace(
            _state(),
            scale_controls=(
                _control(),
                _control("second-width", PixelPoint(120.0, 0.0)),
            ),
        )


def test_explicit_scale_axis_is_preserved_and_legacy_axis_is_inferred() -> None:
    assert _control("vertical-diagonal", PixelPoint(100.0, 90.0), ScaleAxis.VERTICAL).axis is (
        ScaleAxis.VERTICAL
    )
    payload = _control().to_dict()
    payload.pop("axis")
    assert ScaleControl.from_dict(payload).axis is ScaleAxis.HORIZONTAL


def test_export_contains_mount_prior_and_never_contains_rtsp_value() -> None:
    payload = build_interactive_export(_state(), {CAMERA_IDS[0]: True})
    camera = payload["camera_mounting_priors"][0]

    assert camera["C_world_mount_prior"] == pytest.approx(
        {"x_metres": 1.0, "y_metres": 1.0, "z_metres": 3.0}
    )
    assert camera["status"] == "ready-for-calibration"
    assert camera["horizontal_uncertainty_metres"] is None
    assert "optical centre" in camera["authority_note"]
    assert "rtsp://" not in str(payload).lower()
    calibration = payload["scale_calibration"]
    assert calibration["horizontal_pixels_per_metre"] == pytest.approx(100.0)
    assert calibration["vertical_pixels_per_metre"] == pytest.approx(100.0)
    assert calibration["pixels_per_metre"] == pytest.approx(100.0)
    assert calibration["aggregation"] == "arithmetic-mean-of-horizontal-and-vertical"


def test_registration_rejects_ambiguous_or_out_of_plan_inputs() -> None:
    state = _state()
    duplicate = replace(state.cameras[1], physical_label=state.cameras[0].physical_label)
    with pytest.raises(InteractiveRegistrationError, match="labels must be unique"):
        replace(state, cameras=(state.cameras[0], duplicate, *state.cameras[2:]))
    with pytest.raises(InteractiveRegistrationError, match="inside the rendered plan"):
        replace(
            state,
            frame=FramePlacement(
                PixelPoint(100.0, 100.0),
                PixelPoint(1001.0, 100.0),
                "permanent corner",
            ),
        )
    with pytest.raises(InteractiveRegistrationError, match="supplied together"):
        CameraPlacement(CAMERA_IDS[0], "label", PixelPoint(1, 1), 3.0, None, "bracket", None)


def test_state_round_trip_preserves_only_editable_observations() -> None:
    state = _state()

    loaded = InteractiveRegistrationState.from_dict(state.to_dict())

    assert loaded == state
    assert "C_world_mount_prior" not in str(state.to_dict())
