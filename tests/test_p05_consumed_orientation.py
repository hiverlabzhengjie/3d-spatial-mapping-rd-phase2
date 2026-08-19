from __future__ import annotations

import inspect
from typing import Any

import cv2
import numpy as np
import pytest

from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolveError,
    project_world_points,
)
from spatial_mapping_phase2.p05_consumed_orientation import (
    ConsumedOrientationConfig,
    solve_consumed_eight_orientation,
)
from spatial_mapping_phase2.p05_fixed_center_orientation import rotation_difference_degrees


def _case() -> tuple[CameraIntrinsics, Any, Any, Any, Any]:
    intrinsics = CameraIntrinsics("pinhole", 1920, 1080, 1200, 1200, 960, 540)
    center = np.array([4.6, 10.4, 3.0])
    rvec = np.array([0.32, -0.58, 1.08])
    rotation_camera_from_world, _ = cv2.Rodrigues(rvec)
    translation = -rotation_camera_from_world @ center
    camera_points = np.array(
        [
            [-2.5, -1.5, 7.0],
            [2.0, -1.2, 8.0],
            [2.2, 1.8, 9.0],
            [-2.0, 1.5, 6.0],
            [0.2, -2.0, 10.0],
            [2.8, 0.5, 11.0],
            [-2.7, 0.4, 9.5],
            [0.0, 2.3, 12.0],
        ]
    )
    world = (rotation_camera_from_world.T @ (camera_points - translation).T).T
    pixels, _ = project_world_points(intrinsics, world, rvec, translation)
    return intrinsics, center, rotation_camera_from_world.T, world, pixels


def test_consumed_orientation_recovers_exact_center_rotation_and_every_subset() -> None:
    intrinsics, center, expected_rotation, world, pixels = _case()
    solution = solve_consumed_eight_orientation(
        intrinsics,
        center.tolist(),
        [f"p{index}" for index in range(8)],
        world.tolist(),
        pixels.tolist(),
    )
    transform = np.asarray(solution.T_world_from_camera)
    assert np.array_equal(transform[:3, 3], center)
    assert rotation_difference_degrees(expected_rotation, transform[:3, :3]) < 1e-5
    assert solution.inlier_indices == tuple(range(8))
    assert solution.evidence_strength == "strong"
    assert len(solution.subset_results) == 56
    assert solution.maximum_perturbation_rotation_degrees < 0.2


def test_consumed_orientation_recovers_four_point_weak_consensus() -> None:
    intrinsics, center, expected_rotation, world, pixels = _case()
    pixels[4:] += np.array(
        [[420, -260], [-350, 310], [500, 180], [-410, -290]], dtype=np.float64
    )
    solution = solve_consumed_eight_orientation(
        intrinsics,
        center.tolist(),
        [f"p{index}" for index in range(8)],
        world.tolist(),
        pixels.tolist(),
        assess_perturbations=False,
    )
    assert solution.inlier_indices == (0, 1, 2, 3)
    assert solution.rejected_landmark_ids == ("p4", "p5", "p6", "p7")
    assert solution.evidence_strength == "weak"
    assert rotation_difference_degrees(
        expected_rotation, np.asarray(solution.T_world_from_camera)[:3, :3]
    ) < 1e-5


def test_consumed_orientation_rejects_missing_four_point_consensus() -> None:
    intrinsics, center, _, world, _ = _case()
    incoherent_pixels = np.array(
        [
            [60, 60],
            [960, 80],
            [1860, 70],
            [80, 540],
            [1840, 520],
            [70, 1010],
            [970, 1000],
            [1850, 1020],
        ],
        dtype=np.float64,
    )
    with pytest.raises(PoseSolveError, match="four-point consensus"):
        solve_consumed_eight_orientation(
            intrinsics,
            center.tolist(),
            [f"p{index}" for index in range(8)],
            world.tolist(),
            incoherent_pixels.tolist(),
            assess_perturbations=False,
        )


def test_consumed_orientation_has_no_validation_input() -> None:
    signature = str(inspect.signature(solve_consumed_eight_orientation))
    assert "held" not in signature
    assert "validation" not in signature


def test_consumed_orientation_rejects_invalid_config() -> None:
    with pytest.raises(PoseSolveError, match="between 3 and 8"):
        ConsumedOrientationConfig(minimum_consensus_count=9)
