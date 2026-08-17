from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from spatial_mapping_phase2.p04_pose_closeout import (
    application_projection_envelope,
    build_pose_envelope,
    derive_provisional_application_tolerances,
)
from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolveError,
    project_world_points,
)

FloatArray = npt.NDArray[np.float64]

WORLD: FloatArray = np.asarray(
    [
        [-2.0, 4.0, 0.0],
        [1.0, 5.0, 0.4],
        [4.0, 7.0, 1.2],
        [-1.0, 10.0, 2.5],
        [3.0, 12.0, 0.0],
        [6.0, 9.0, 3.2],
        [0.0, 14.0, 1.0],
        [7.0, 15.0, 2.0],
    ],
    dtype=np.float64,
)


def _synthetic() -> tuple[CameraIntrinsics, FloatArray, FloatArray]:
    intrinsics = CameraIntrinsics(
        "simple_radial", 1920, 1080, 1350.0, 1350.0, 960.0, 540.0, (-0.22,)
    )
    position = np.asarray([2.0, -4.0, 3.3])
    forward = np.asarray([1.0, 10.0, 0.8]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, -1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation_camera_from_world = np.column_stack((right, down, forward)).T
    rvec, _ = cv2.Rodrigues(rotation_camera_from_world)
    tvec = -rotation_camera_from_world @ position
    image, _ = project_world_points(intrinsics, WORLD, rvec.reshape(3), tvec)
    return intrinsics, position, image


def test_all_solve_envelope_recovers_pose_and_retains_held_out() -> None:
    intrinsics, expected_position, image = _synthetic()
    noisy = image.copy()
    noisy[:6] += np.random.default_rng(29).normal(0, 0.4, size=(6, 2))
    noisy[5] += np.asarray([18.0, -14.0])

    envelope = build_pose_envelope(
        intrinsics,
        WORLD[:6].tolist(),
        noisy[:6].tolist(),
        WORLD[6:].tolist(),
        image[6:].tolist(),
    )
    summary = envelope.summary()

    assert cast(int, summary["successful_case_count"]) >= 27
    assert summary["active_bound_case_count"] == 0
    assert np.linalg.norm(
        np.asarray(
            cast(Sequence[float], summary["median_camera_position_world_metres"]),
            dtype=np.float64,
        )
        - expected_position
    ) < 0.2
    assert cast(float, summary["maximum_camera_position_distance_from_median_metres"]) < 0.2
    assert all(
        len(case.bounded_solution.held_out_reprojection_errors_pixels) == 2
        for case in envelope.cases
    )


def test_projection_envelope_converts_pixels_to_ray_and_lateral_bounds() -> None:
    intrinsics, _, image = _synthetic()
    envelope = build_pose_envelope(
        intrinsics,
        WORLD[:6].tolist(),
        image[:6].tolist(),
        WORLD[6:].tolist(),
        image[6:].tolist(),
    )
    result = application_projection_envelope((envelope,), minimum_focal_pixels=1350.0)
    assert result["conservative_ray_angle_degrees"] >= 0
    assert result["lateral_envelope_at_10_metres"] == pytest.approx(
        2 * result["lateral_envelope_at_5_metres"]
    )


def test_provisional_tolerances_round_measured_envelope_outward() -> None:
    result = derive_provisional_application_tolerances(
        {
            "maximum_position_distance_from_combined_median_metres": 0.6056,
            "maximum_pairwise_optical_axis_angle_degrees": 6.9971,
        },
        {
            "maximum_observed_held_out_error_pixels": 91.1103,
            "conservative_ray_angle_degrees": 3.9195,
            "lateral_envelope_at_5_metres": 0.3426,
            "lateral_envelope_at_10_metres": 0.6852,
        },
    )

    assert result["held_out_reprojection_max_pixels"] == pytest.approx(95.0)
    assert result["ray_angle_degrees"] == pytest.approx(4.0)
    assert result["lateral_at_5_metres"] == pytest.approx(0.35)
    assert result["lateral_at_10_metres"] == pytest.approx(0.70)
    assert result["camera_centre_change_from_envelope_median_metres"] == pytest.approx(0.65)
    assert result["optical_axis_change_degrees"] == pytest.approx(7.0)
    assert result["height_and_mounting_prior"] == "diagnostic-only"


def test_closeout_rejects_degenerate_geometry_and_invalid_projection_input() -> None:
    intrinsics, _, image = _synthetic()
    planar = WORLD[:6].copy()
    planar[:, 2] = 0
    with pytest.raises(PoseSolveError, match="spanning three dimensions"):
        build_pose_envelope(
            intrinsics,
            planar.tolist(),
            image[:6].tolist(),
            WORLD[6:].tolist(),
            image[6:].tolist(),
        )
    with pytest.raises(PoseSolveError, match="positive focal"):
        application_projection_envelope((), minimum_focal_pixels=0)
