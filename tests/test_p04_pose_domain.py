from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolveError,
    angular_difference_degrees,
    project_world_points,
    solve_camera_pose,
)

FloatArray = npt.NDArray[np.float64]

WORLD: FloatArray = np.array(
    [
        [-2.0, 4.0, 0.0],
        [1.0, 5.0, 0.4],
        [4.0, 7.0, 1.2],
        [-1.0, 10.0, 2.5],
        [3.0, 12.0, 0.0],
        [6.0, 9.0, 3.2],
        [0.0, 14.0, 1.0],
        [7.0, 15.0, 2.0],
        [-3.0, 8.0, 3.5],
        [5.0, 18.0, 0.7],
    ],
    dtype=np.float64,
)


def _pose() -> tuple[FloatArray, FloatArray, FloatArray]:
    position: FloatArray = np.asarray([2.0, -4.0, 3.3], dtype=np.float64)
    forward = np.asarray([1.0, 10.0, 0.8], dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, -1.0], dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation_world_from_camera = np.column_stack((right, down, forward))
    rotation_camera_from_world = rotation_world_from_camera.T
    raw_rvec, _ = cv2.Rodrigues(rotation_camera_from_world)
    rvec: FloatArray = np.asarray(raw_rvec, dtype=np.float64).reshape(3)
    tvec: FloatArray = np.asarray(-rotation_camera_from_world @ position, dtype=np.float64)
    return position, rvec, tvec


@pytest.mark.parametrize(
    ("model", "distortion"),
    [
        ("pinhole", ()),
        ("simple_radial", (-0.12,)),
        ("simple_divisional", (-0.25,)),
        ("radial", (-0.12, 0.03)),
    ],
)
def test_recovers_synthetic_pose_and_keeps_held_out_separate(
    model: str, distortion: tuple[float, ...]
) -> None:
    intrinsics = CameraIntrinsics(model, 1920, 1080, 1250.0, 1250.0, 960.0, 540.0, distortion)
    position, rvec, tvec = _pose()
    image, _ = project_world_points(intrinsics, WORLD, rvec, tvec)
    image[:8] += np.random.default_rng(17).normal(0.0, 0.25, size=(8, 2))
    image[3] += np.array([35.0, -28.0])

    solution = solve_camera_pose(
        intrinsics, WORLD[:8].tolist(), image[:8].tolist(), WORLD[8:].tolist(), image[8:].tolist()
    )

    assert np.linalg.norm(np.array(solution.camera_position_world_metres) - position) < 0.2
    assert len(solution.ransac_inlier_indices) < 8
    assert max(solution.held_out_reprojection_errors_pixels) < 8.0
    assert len(solution.solve_reprojection_errors_pixels) == 8
    assert len(solution.held_out_reprojection_errors_pixels) == 2
    assert all(depth > 0 for depth in solution.point_depths_camera_metres)


def test_exposes_explicit_world_from_camera_transform() -> None:
    intrinsics = CameraIntrinsics("pinhole", 1920, 1080, 1300.0, 1300.0, 960.0, 540.0)
    position, rvec, tvec = _pose()
    image, _ = project_world_points(intrinsics, WORLD, rvec, tvec)

    solution = solve_camera_pose(
        intrinsics, WORLD[:8].tolist(), image[:8].tolist(), WORLD[8:].tolist(), image[8:].tolist()
    )
    transform = np.array(solution.T_world_from_camera)

    np.testing.assert_allclose(transform[:3, 3], position, atol=1e-5)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8)
    assert math.isclose(np.linalg.det(transform[:3, :3]), 1.0, abs_tol=1e-8)


def test_rejects_degenerate_or_insufficient_reference_sets() -> None:
    intrinsics = CameraIntrinsics("pinhole", 1000, 800, 700.0, 700.0, 500.0, 400.0)
    with pytest.raises(PoseSolveError, match="at least four"):
        solve_camera_pose(intrinsics, WORLD[:3].tolist(), np.ones((3, 2)).tolist())
    planar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    with pytest.raises(PoseSolveError, match="span three dimensions"):
        solve_camera_pose(intrinsics, planar.tolist(), np.ones((4, 2)).tolist())


def test_rejects_bad_intrinsics_and_mismatched_held_out_points() -> None:
    with pytest.raises(PoseSolveError, match="requires 1"):
        CameraIntrinsics("simple_radial", 1920, 1080, 1000, 1000, 960, 540)
    intrinsics = CameraIntrinsics("pinhole", 1920, 1080, 1000, 1000, 960, 540)
    _, rvec, tvec = _pose()
    image, _ = project_world_points(intrinsics, WORLD, rvec, tvec)
    with pytest.raises(PoseSolveError, match="counts must match"):
        solve_camera_pose(
            intrinsics,
            WORLD[:8].tolist(),
            image[:8].tolist(),
            WORLD[8:].tolist(),
            image[8:9].tolist(),
        )


def test_vector_angle_is_stable_and_rejects_zero_vector() -> None:
    assert angular_difference_degrees((0, 0, -1), (0, 0, -2)) == pytest.approx(0.0)
    assert angular_difference_degrees((1, 0, 0), (0, 1, 0)) == pytest.approx(90.0)
    with pytest.raises(PoseSolveError, match="non-zero"):
        angular_difference_degrees((0, 0, 0), (0, 1, 0))
