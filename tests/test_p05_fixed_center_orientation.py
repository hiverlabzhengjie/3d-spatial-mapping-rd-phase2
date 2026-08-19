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
from spatial_mapping_phase2.p05_fixed_center_orientation import (
    D034SolverConfig,
    evaluate_d034_validation,
    rotation_difference_degrees,
    select_d034_solve_set,
    solve_d034_orientation,
    solve_orientation_at_fixed_center,
)


def _synthetic_case() -> tuple[
    CameraIntrinsics, Any, Any, Any, Any
]:
    intrinsics = CameraIntrinsics("pinhole", 1920, 1080, 1200, 1200, 960, 540)
    center = np.array([8.7, 5.4, 3.1])
    rvec = np.array([0.35, -0.6, 1.1])
    rotation_camera_from_world, _ = cv2.Rodrigues(rvec)
    translation = -rotation_camera_from_world @ center
    camera_points = np.array(
        [[-2.5, -1.5, 7], [2.0, -1.2, 8], [2.2, 1.8, 9], [-2.0, 1.5, 6]]
    )
    world = (rotation_camera_from_world.T @ (camera_points - translation).T).T
    pixels, _ = project_world_points(intrinsics, world, rvec, translation)
    return intrinsics, center, rotation_camera_from_world.T, world, pixels


def test_fixed_center_orientation_recovers_synthetic_rotation() -> None:
    intrinsics = CameraIntrinsics("pinhole", 1920, 1080, 1200, 1200, 960, 540)
    center = np.array([8.7, 5.4, 3.1])
    rotation_camera_from_world, _ = cv2.Rodrigues(np.array([0.35, -0.6, 1.1]))
    translation = -rotation_camera_from_world @ center
    camera_points = np.array(
        [
            [-2, -1, 7],
            [1, -1.5, 8],
            [2, 1, 9],
            [-1.5, 1.5, 6],
            [0, 0.5, 10],
            [2, -2, 12],
            [0.5, 2, 11],
        ]
    )
    world = (rotation_camera_from_world.T @ (camera_points - translation).T).T
    pixels, _ = project_world_points(
        intrinsics, world, np.array([0.35, -0.6, 1.1]), translation
    )
    solution = solve_orientation_at_fixed_center(
        intrinsics, center.tolist(), world[:5], pixels[:5], world[5:], pixels[5:]
    )
    recovered = np.asarray(solution.T_world_from_camera)[:3, :3]
    assert solution.camera_position_world_metres == pytest.approx(center)
    assert rotation_difference_degrees(rotation_camera_from_world.T, recovered) < 1e-5
    assert max(solution.solve_reprojection_errors_pixels) < 1e-6
    assert max(solution.held_out_reprojection_errors_pixels) < 1e-6


def test_d034_recovers_proper_wahba_rotation_and_exact_center() -> None:
    intrinsics, center, expected_rotation, world, pixels = _synthetic_case()
    solution = solve_d034_orientation(
        intrinsics,
        center.tolist(),
        ["p1", "p2", "p3", "p4"],
        world.tolist(),
        pixels.tolist(),
    )
    transform = np.asarray(solution.T_world_from_camera)
    assert np.array_equal(transform[:3, 3], center)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)
    assert rotation_difference_degrees(expected_rotation, transform[:3, :3]) < 1e-5
    assert solution.rejected_solve_landmark_id is None
    assert solution.refinement_active_bound is False


def test_d034_enumerates_every_three_of_four_subset() -> None:
    intrinsics, center, _, world, pixels = _synthetic_case()
    solution = solve_d034_orientation(
        intrinsics,
        center.tolist(),
        ["p1", "p2", "p3", "p4"],
        world.tolist(),
        pixels.tolist(),
        assess_perturbations=False,
    )
    assert {value.subset_indices for value in solution.subset_results} == {
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    }


def test_d034_recovers_from_one_solve_outlier() -> None:
    intrinsics, center, expected_rotation, world, pixels = _synthetic_case()
    pixels[3] += [180, 120]
    solution = solve_d034_orientation(
        intrinsics,
        center.tolist(),
        ["p1", "p2", "p3", "bad"],
        world.tolist(),
        pixels.tolist(),
    )
    assert solution.rejected_solve_landmark_id == "bad"
    assert solution.inlier_indices == (0, 1, 2)
    assert rotation_difference_degrees(
        expected_rotation, np.asarray(solution.T_world_from_camera)[:3, :3]
    ) < 1e-5


def test_d034_rejects_two_inconsistent_solve_points() -> None:
    intrinsics, center, _, world, pixels = _synthetic_case()
    pixels[2] += [280, -190]
    pixels[3] += [-240, 170]
    with pytest.raises(PoseSolveError, match="consensus|ambiguous|sensitivity"):
        solve_d034_orientation(
            intrinsics,
            center.tolist(),
            ["p1", "p2", "bad1", "bad2"],
            world.tolist(),
            pixels.tolist(),
            assess_perturbations=False,
        )


def test_d034_rejects_poor_directional_conditioning() -> None:
    intrinsics = CameraIntrinsics("pinhole", 1920, 1080, 1200, 1200, 960, 540)
    camera_points = np.array(
        [[0, 0, 10], [0.001, 0, 10], [0, 0.001, 10], [0.001, 0.001, 10]]
    )
    pixels = np.column_stack(
        (1200 * camera_points[:, 0] / 10 + 960, 1200 * camera_points[:, 1] / 10 + 540)
    )
    with pytest.raises(PoseSolveError, match="poorly conditioned"):
        solve_d034_orientation(
            intrinsics,
            [0, 0, 0],
            ["p1", "p2", "p3", "p4"],
            camera_points.tolist(),
            pixels.tolist(),
            assess_perturbations=False,
        )


def test_d034_rejects_ambiguous_subset_rotations() -> None:
    intrinsics, center, _, world, pixels = _synthetic_case()
    pixels[0] += [1.0, -0.5]
    with pytest.raises(PoseSolveError, match="ambiguous"):
        solve_d034_orientation(
            intrinsics,
            center.tolist(),
            ["p1", "p2", "p3", "p4"],
            world.tolist(),
            pixels.tolist(),
            config=D034SolverConfig(ambiguity_rotation_degrees=1e-7),
            assess_perturbations=False,
        )


def test_d034_rejects_geometry_with_a_point_behind_candidate_camera() -> None:
    intrinsics, center, _, world, pixels = _synthetic_case()
    world[3] = 2 * center - world[3]
    with pytest.raises(PoseSolveError, match="consensus"):
        solve_d034_orientation(
            intrinsics,
            center.tolist(),
            ["p1", "p2", "p3", "behind"],
            world.tolist(),
            pixels.tolist(),
            assess_perturbations=False,
        )


def test_d034_rejects_excessive_click_perturbation_sensitivity() -> None:
    intrinsics, center, _, world, pixels = _synthetic_case()
    with pytest.raises(PoseSolveError, match="perturbation sensitivity"):
        solve_d034_orientation(
            intrinsics,
            center.tolist(),
            ["p1", "p2", "p3", "p4"],
            world.tolist(),
            pixels.tolist(),
            config=D034SolverConfig(maximum_perturbation_rotation_degrees=1e-8),
        )


def test_d034_validation_is_separate_and_individual() -> None:
    intrinsics, center, _, world, pixels = _synthetic_case()
    solution = solve_d034_orientation(
        intrinsics,
        center.tolist(),
        ["p1", "p2", "p3", "p4"],
        world.tolist(),
        pixels.tolist(),
        assess_perturbations=False,
    )
    assert "held" not in str(inspect.signature(solve_d034_orientation))
    result = evaluate_d034_validation(
        intrinsics,
        solution,
        ["v1", "v2"],
        world[:2].tolist(),
        (pixels[:2] + [[0, 0], [40, 0]]).tolist(),
    )
    assert result.individual_pass == (True, False)
    assert result.status == "rejected"


def test_d034_selection_uses_distribution_and_conditioning() -> None:
    intrinsics, center, _, world, pixels = _synthetic_case()
    extra_world = np.vstack((world, (world[0] + world[1]) / 2))
    extra_pixels = np.vstack((pixels, (pixels[0] + pixels[1]) / 2))
    selection = select_d034_solve_set(
        ["p1", "p2", "p3", "p4", "middle"],
        extra_world.tolist(),
        extra_pixels.tolist(),
        center.tolist(),
        intrinsics.width_pixels,
        intrinsics.height_pixels,
    )
    assert selection.selected_landmark_ids == ("p1", "p2", "p3", "p4")
    assert len(selection.ranking) == 5


def test_fixed_center_orientation_rejects_too_few_points() -> None:
    intrinsics = CameraIntrinsics("pinhole", 100, 100, 80, 80, 50, 50)
    with pytest.raises(PoseSolveError, match="at least three"):
        solve_orientation_at_fixed_center(
            intrinsics, [0, 0, 0], [[0, 0, 2], [1, 0, 2]], [[50, 50], [60, 50]]
        )


def test_fixed_center_orientation_rejects_point_at_center() -> None:
    intrinsics = CameraIntrinsics("pinhole", 100, 100, 80, 80, 50, 50)
    with pytest.raises(PoseSolveError, match="coincides"):
        solve_orientation_at_fixed_center(
            intrinsics,
            [0, 0, 0],
            [[0, 0, 0], [1, 0, 2], [0, 1, 2]],
            [[50, 50], [60, 50], [50, 60]],
        )
