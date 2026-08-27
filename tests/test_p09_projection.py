from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_projection import (
    CameraProjectionCalibration,
    FloorRectangle,
    LiveFrameRectifier,
    P09ProjectionError,
    project_detection_to_floor,
)
from spatial_mapping_phase2.p09_tracking_domain import (
    FootpointKind,
    LiveFrameIdentity,
    PersonDetection,
    ProjectionStatus,
)


def _calibration(
    T_world_from_camera: NDArray[np.float64] | None = None,
) -> CameraProjectionCalibration:
    transform = np.eye(4) if T_world_from_camera is None else T_world_from_camera
    return CameraProjectionCalibration(
        camera_id="office-cam-01",
        K_native=np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
        simple_radial_k1=-0.2,
        K_processed=np.array([[100.0, 0.0, 252.0], [0.0, 100.0, 140.0], [0.0, 0.0, 1.0]]),
        T_world_from_camera=transform,
    )


def _detection(u: float = 252.0, v: float = 140.0) -> PersonDetection:
    frame = LiveFrameIdentity(
        "office-cam-01",
        "frame-1",
        1_000_000_000,
        "2026-08-20T00:00:00Z",
        1,
        "1/90000",
        504,
        280,
    )
    return PersonDetection(
        frame,
        0,
        0.9,
        (200.0, 50.0, 300.0, 200.0),
        (u, v),
        FootpointKind.BBOX_BOTTOM_CENTER,
    )


def test_projection_uses_T_world_from_camera_and_forward_z_zero_intersection() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [2.0, 3.0, 4.0]
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    result = project_detection_to_floor(
        _detection(),
        _calibration(transform),
        FloorRectangle((0.0, 0.0), (10.0, 10.0)),
        1_100_000_000,
    )
    assert result.status is ProjectionStatus.VALID
    assert result.observation is not None
    np.testing.assert_allclose(result.observation.xy_metres, [2.0, 3.0])
    assert result.observation.ray_parameter_metres == pytest.approx(4.0)
    assert result.observation.frame_age_ms == pytest.approx(100.0)


def test_projection_rejects_parallel_behind_outside_and_wrong_convention() -> None:
    floor = FloorRectangle((0.0, 0.0), (10.0, 10.0))
    parallel = np.eye(4)
    parallel[:3, 3] = [2.0, 3.0, 4.0]
    parallel[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert (
        project_detection_to_floor(
            _detection(), _calibration(parallel), floor, 1_000_000_000
        ).status
        is ProjectionStatus.PARALLEL_RAY
    )

    behind = np.eye(4)
    behind[:3, 3] = [2.0, 3.0, 4.0]
    assert (
        project_detection_to_floor(_detection(), _calibration(behind), floor, 1_000_000_000).status
        is ProjectionStatus.BEHIND_CAMERA
    )

    outside = np.eye(4)
    outside[:3, 3] = [20.0, 20.0, 4.0]
    outside[:3, :3] = np.diag([1.0, -1.0, -1.0])
    assert (
        project_detection_to_floor(
            _detection(), _calibration(outside), floor, 1_000_000_000
        ).status
        is ProjectionStatus.OUTSIDE_FLOOR
    )

    wrong_frame = _detection()
    bad_identity = LiveFrameIdentity(
        wrong_frame.frame.camera_id,
        "wrong-size",
        wrong_frame.frame.acquisition_monotonic_ns,
        wrong_frame.frame.observed_at_utc,
        None,
        None,
        1920,
        1080,
    )
    wrong_detection = PersonDetection(
        bad_identity,
        0,
        0.9,
        (1.0, 1.0, 100.0, 100.0),
        (50.0, 99.0),
        FootpointKind.BBOX_BOTTOM_CENTER,
    )
    assert (
        project_detection_to_floor(
            wrong_detection, _calibration(outside), floor, 1_000_000_000
        ).status
        is ProjectionStatus.INVALID_PIXEL
    )


def test_rectifier_requires_native_shape_and_produces_exact_calibrated_shape() -> None:
    rectifier = LiveFrameRectifier(_calibration())
    native = np.zeros((1080, 1920, 3), dtype=np.uint8)
    processed = rectifier.rectify(native)
    assert processed.shape == (280, 504, 3)
    assert processed.dtype == np.uint8
    assert processed.flags.c_contiguous
    with pytest.raises(P09ProjectionError, match="1920x1080"):
        rectifier.rectify(np.zeros((280, 504, 3), dtype=np.uint8))


def test_calibration_rejects_improper_transform_and_is_immutable() -> None:
    calibration = _calibration()
    assert not calibration.K_native.flags.writeable
    assert not calibration.K_processed.flags.writeable
    assert not calibration.T_world_from_camera.flags.writeable
    improper = np.eye(4)
    improper[0, 0] = -1.0
    with pytest.raises(P09ProjectionError, match="proper"):
        _calibration(improper)
