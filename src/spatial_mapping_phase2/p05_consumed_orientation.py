"""Best-available fixed-centre orientation from eight already-consumed observations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolveError,
    project_world_points,
    undistort_image_points,
)
from spatial_mapping_phase2.p05_fixed_center_orientation import rotation_difference_degrees


@dataclass(frozen=True, slots=True)
class ConsumedOrientationConfig:
    """Frozen limits for the explicitly non-validated eight-point fallback."""

    inlier_threshold_pixels: float = 45.0
    minimum_consensus_count: int = 4
    huber_scale_pixels: float = 45.0
    minimum_information_ratio: float = 0.01
    competitive_rmse_margin_pixels: float = 2.0
    refinement_bound_rotation_radians: float = 0.5
    active_bound_margin_radians: float = 1e-6
    perturbation_pixels: float = 2.0
    maximum_refinement_passes: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.inlier_threshold_pixels,
            self.huber_scale_pixels,
            self.minimum_information_ratio,
            self.competitive_rmse_margin_pixels,
            self.refinement_bound_rotation_radians,
            self.active_bound_margin_radians,
            self.perturbation_pixels,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise PoseSolveError("consumed-orientation limits must be finite and positive")
        if not 3 <= self.minimum_consensus_count <= 8:
            raise PoseSolveError("consumed-orientation consensus count must be between 3 and 8")
        if self.maximum_refinement_passes < 1:
            raise PoseSolveError("consumed-orientation refinement needs at least one pass")

    def to_dict(self) -> dict[str, float | int]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ConsumedSubsetResult:
    subset_indices: tuple[int, int, int]
    subset_landmark_ids: tuple[str, str, str]
    consensus_indices: tuple[int, ...]
    inlier_rmse_pixels: float
    maximum_inlier_error_pixels: float
    reprojection_errors_pixels: tuple[float, ...]
    rotation_world_from_camera: tuple[tuple[float, float, float], ...]
    positive_depths: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "subset_indices": list(self.subset_indices),
            "subset_landmark_ids": list(self.subset_landmark_ids),
            "consensus_indices": list(self.consensus_indices),
            "inlier_rmse_pixels": self.inlier_rmse_pixels,
            "maximum_inlier_error_pixels": self.maximum_inlier_error_pixels,
            "reprojection_errors_pixels": list(self.reprojection_errors_pixels),
            "rotation_world_from_camera": [
                list(row) for row in self.rotation_world_from_camera
            ],
            "positive_depths": self.positive_depths,
        }


@dataclass(frozen=True, slots=True)
class ConsumedOrientationSolution:
    T_world_from_camera: tuple[tuple[float, ...], ...]
    fixed_camera_center_world_metres: tuple[float, float, float]
    optical_axis_world: tuple[float, float, float]
    landmark_ids: tuple[str, ...]
    selected_subset_indices: tuple[int, int, int]
    selected_subset_landmark_ids: tuple[str, str, str]
    inlier_indices: tuple[int, ...]
    inlier_landmark_ids: tuple[str, ...]
    rejected_landmark_ids: tuple[str, ...]
    projected_pixels: tuple[tuple[float, float], ...]
    reprojection_errors_pixels: tuple[float, ...]
    point_depths_camera_metres: tuple[float, ...]
    evidence_strength: str
    camera_information_ratio: float
    world_information_ratio: float
    maximum_consensus_candidate_count: int
    competitive_rotation_spread_degrees: float
    maximum_consensus_rotation_spread_degrees: float
    maximum_perturbation_rotation_degrees: float
    subset_results: tuple[ConsumedSubsetResult, ...]
    refinement_passes: int
    refinement_active_bound: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "T_world_from_camera": [list(row) for row in self.T_world_from_camera],
            "fixed_camera_center_world_metres": list(
                self.fixed_camera_center_world_metres
            ),
            "translation_optimization": "prohibited; P02 centre copied exactly",
            "optical_axis_world": list(self.optical_axis_world),
            "landmark_ids": list(self.landmark_ids),
            "selected_subset_indices": list(self.selected_subset_indices),
            "selected_subset_landmark_ids": list(self.selected_subset_landmark_ids),
            "inlier_indices": list(self.inlier_indices),
            "inlier_landmark_ids": list(self.inlier_landmark_ids),
            "rejected_landmark_ids": list(self.rejected_landmark_ids),
            "projected_pixels": [list(point) for point in self.projected_pixels],
            "reprojection_errors_pixels": list(self.reprojection_errors_pixels),
            "inlier_reprojection_rmse_pixels": _rmse(
                [self.reprojection_errors_pixels[index] for index in self.inlier_indices]
            ),
            "point_depths_camera_metres": list(self.point_depths_camera_metres),
            "consensus_count": len(self.inlier_indices),
            "evidence_strength": self.evidence_strength,
            "camera_information_ratio": self.camera_information_ratio,
            "world_information_ratio": self.world_information_ratio,
            "maximum_consensus_candidate_count": self.maximum_consensus_candidate_count,
            "competitive_rotation_spread_degrees": (
                self.competitive_rotation_spread_degrees
            ),
            "maximum_consensus_rotation_spread_degrees": (
                self.maximum_consensus_rotation_spread_degrees
            ),
            "maximum_perturbation_rotation_degrees": (
                self.maximum_perturbation_rotation_degrees
            ),
            "refinement_passes": self.refinement_passes,
            "refinement_active_bound": self.refinement_active_bound,
            "subset_results": [result.to_dict() for result in self.subset_results],
            "validation": "none; all eight observations were consumed by estimation",
            "authority": "provisional consumed-evidence hypothesis only",
        }


def solve_consumed_eight_orientation(
    intrinsics: CameraIntrinsics,
    camera_center_world_metres: Sequence[float],
    landmark_ids: Sequence[str],
    world_points: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    *,
    config: ConsumedOrientationConfig | None = None,
    assess_perturbations: bool = True,
) -> ConsumedOrientationSolution:
    """Estimate a provisional rotation from eight consumed observations."""

    selected_config = config or ConsumedOrientationConfig()
    solution = _solve_once(
        intrinsics,
        camera_center_world_metres,
        landmark_ids,
        world_points,
        image_points,
        selected_config,
    )
    if not assess_perturbations:
        return solution
    base_rotation = np.asarray(solution.T_world_from_camera)[:3, :3]
    image = _points(image_points, 2, "consumed image")
    maximum = 0.0
    for point_index in range(8):
        for axis in range(2):
            for sign in (-1.0, 1.0):
                perturbed = image.copy()
                perturbed[point_index, axis] += sign * selected_config.perturbation_pixels
                candidate = _solve_once(
                    intrinsics,
                    camera_center_world_metres,
                    landmark_ids,
                    world_points,
                    perturbed.tolist(),
                    selected_config,
                )
                maximum = max(
                    maximum,
                    rotation_difference_degrees(
                        base_rotation,
                        np.asarray(candidate.T_world_from_camera)[:3, :3],
                    ),
                )
    return replace(solution, maximum_perturbation_rotation_degrees=maximum)


def _solve_once(
    intrinsics: CameraIntrinsics,
    camera_center_world_metres: Sequence[float],
    landmark_ids: Sequence[str],
    world_points: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    config: ConsumedOrientationConfig,
) -> ConsumedOrientationSolution:
    center = _vector(camera_center_world_metres, 3, "camera centre")
    world = _points(world_points, 3, "consumed world")
    image = _points(image_points, 2, "consumed image")
    ids = tuple(str(value) for value in landmark_ids)
    if world.shape != (8, 3) or image.shape != (8, 2) or len(ids) != 8:
        raise PoseSolveError("consumed-orientation fallback requires exactly eight points")
    if len(set(ids)) != 8:
        raise PoseSolveError("consumed-orientation landmark identities must be unique")
    world_directions = _unit_directions(world - center, "world")
    normalized = undistort_image_points(intrinsics, image)
    camera_directions = _unit_directions(
        np.column_stack((normalized, np.ones(8))), "camera"
    )
    world_ratio = _information_ratio(world_directions)
    camera_ratio = _information_ratio(camera_directions)
    if min(world_ratio, camera_ratio) < config.minimum_information_ratio:
        raise PoseSolveError(
            "consumed-orientation directional geometry is poorly conditioned: "
            f"camera={camera_ratio:.6f}, world={world_ratio:.6f}"
        )

    subset_results: list[ConsumedSubsetResult] = []
    rotations: list[Any] = []
    for subset in combinations(range(8), 3):
        selected = list(subset)
        rotation = _wahba_rotation(
            camera_directions[selected], world_directions[selected]
        )
        candidate_projected, candidate_depths = _project(
            intrinsics, world, center, rotation
        )
        candidate_errors = np.linalg.norm(candidate_projected - image, axis=1)
        inliers = tuple(
            int(index)
            for index, error in enumerate(candidate_errors)
            if error <= config.inlier_threshold_pixels
        )
        inlier_errors = [float(candidate_errors[index]) for index in inliers]
        subset_results.append(
            ConsumedSubsetResult(
                (subset[0], subset[1], subset[2]),
                (ids[subset[0]], ids[subset[1]], ids[subset[2]]),
                inliers,
                math.inf if not inlier_errors else _rmse(inlier_errors),
                math.inf if not inlier_errors else max(inlier_errors),
                tuple(float(value) for value in candidate_errors),
                tuple(
                    (float(row[0]), float(row[1]), float(row[2])) for row in rotation
                ),
                bool(np.all(candidate_depths > 0)),
            )
        )
        rotations.append(rotation)
    eligible = [
        index
        for index, result in enumerate(subset_results)
        if result.positive_depths
        and len(result.consensus_indices) >= config.minimum_consensus_count
    ]
    if not eligible:
        raise PoseSolveError("consumed-orientation failed to form four-point consensus")
    maximum_consensus = max(
        len(subset_results[index].consensus_indices) for index in eligible
    )
    maximum_indices = [
        index
        for index in eligible
        if len(subset_results[index].consensus_indices) == maximum_consensus
    ]
    maximum_indices.sort(
        key=lambda index: (
            subset_results[index].inlier_rmse_pixels,
            subset_results[index].maximum_inlier_error_pixels,
            subset_results[index].subset_indices,
        )
    )
    best_index = maximum_indices[0]
    best = subset_results[best_index]
    competitive = [
        index
        for index in maximum_indices
        if subset_results[index].inlier_rmse_pixels
        <= best.inlier_rmse_pixels + config.competitive_rmse_margin_pixels
    ]
    competitive_spread = _rotation_spread([rotations[index] for index in competitive])
    maximum_spread = _rotation_spread([rotations[index] for index in maximum_indices])

    current_inliers = tuple(best.consensus_indices)
    rotation = rotations[best_index]
    passes = 0
    active_bound = False
    final_projected: Any = None
    final_depths: Any = None
    final_errors: Any = None
    for pass_number in range(1, config.maximum_refinement_passes + 1):
        passes = pass_number
        initial_rvec, _ = cv2.Rodrigues(np.asarray(rotation).T)
        initial = initial_rvec.reshape(3)
        refinement_world = world[list(current_inliers)]
        refinement_image = image[list(current_inliers)]

        def residuals(
            rvec: Any,
            selected_world: Any = refinement_world,
            selected_image: Any = refinement_image,
        ) -> Any:
            rotation_camera_from_world, _ = cv2.Rodrigues(
                np.asarray(rvec, dtype=np.float64).reshape(3)
            )
            translation = -rotation_camera_from_world @ center
            candidate_projected, _ = project_world_points(
                intrinsics, selected_world, rvec, translation
            )
            return (candidate_projected - selected_image).reshape(-1)

        lower = initial - config.refinement_bound_rotation_radians
        upper = initial + config.refinement_bound_rotation_radians
        refined = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            loss="huber",
            f_scale=config.huber_scale_pixels,
            max_nfev=500,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        if not refined.success or not np.all(np.isfinite(refined.x)):
            raise PoseSolveError("consumed-orientation rotation refinement failed")
        active_bound = bool(
            np.any(refined.x - lower <= config.active_bound_margin_radians)
            or np.any(upper - refined.x <= config.active_bound_margin_radians)
        )
        if active_bound:
            raise PoseSolveError("consumed-orientation refinement reached an active bound")
        rotation_camera_from_world, _ = cv2.Rodrigues(refined.x)
        rotation = rotation_camera_from_world.T
        final_projected, final_depths = _project(intrinsics, world, center, rotation)
        final_errors = np.linalg.norm(final_projected - image, axis=1)
        updated_inliers = tuple(
            int(index)
            for index, error in enumerate(final_errors)
            if error <= config.inlier_threshold_pixels
        )
        if len(updated_inliers) < config.minimum_consensus_count:
            raise PoseSolveError(
                "consumed-orientation refinement lost four-point consensus"
            )
        if updated_inliers == current_inliers:
            break
        current_inliers = updated_inliers
    if final_projected is None or final_depths is None or final_errors is None:
        raise PoseSolveError("consumed-orientation refinement produced no result")
    if np.any(final_depths <= 0):
        raise PoseSolveError("consumed-orientation has a point behind the camera")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-8:
        raise PoseSolveError("consumed-orientation refinement produced an improper rotation")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    inlier_ids = tuple(ids[index] for index in current_inliers)
    rejected_ids = tuple(ids[index] for index in range(8) if index not in current_inliers)
    strength = (
        "strong"
        if len(current_inliers) >= 6
        else "moderate"
        if len(current_inliers) == 5
        else "weak"
    )
    return ConsumedOrientationSolution(
        tuple(tuple(float(value) for value in row) for row in transform),
        (float(center[0]), float(center[1]), float(center[2])),
        tuple(float(value) for value in rotation[:, 2]),  # type: ignore[arg-type]
        ids,
        best.subset_indices,
        best.subset_landmark_ids,
        current_inliers,
        inlier_ids,
        rejected_ids,
        tuple(
            (float(point[0]), float(point[1])) for point in final_projected
        ),
        tuple(float(value) for value in final_errors),
        tuple(float(value) for value in final_depths),
        strength,
        camera_ratio,
        world_ratio,
        maximum_consensus,
        competitive_spread,
        maximum_spread,
        0.0,
        tuple(subset_results),
        passes,
        active_bound,
    )


def _project(
    intrinsics: CameraIntrinsics,
    world: Any,
    center: Any,
    rotation_world_from_camera: Any,
) -> tuple[Any, Any]:
    rotation_camera_from_world = np.asarray(rotation_world_from_camera).T
    rvec, _ = cv2.Rodrigues(rotation_camera_from_world)
    translation = -rotation_camera_from_world @ center
    return project_world_points(intrinsics, world, rvec, translation)


def _wahba_rotation(camera_directions: Any, world_directions: Any) -> Any:
    left, _, right_transpose = np.linalg.svd(
        np.asarray(world_directions).T @ np.asarray(camera_directions)
    )
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(left @ right_transpose) >= 0 else -1.0
    rotation = left @ correction @ right_transpose
    if not np.all(np.isfinite(rotation)) or abs(float(np.linalg.det(rotation)) - 1) > 1e-8:
        raise PoseSolveError("consumed-orientation Wahba rotation is not proper")
    return rotation


def _rotation_spread(rotations: Sequence[Any]) -> float:
    return max(
        (
            rotation_difference_degrees(left, right)
            for left, right in combinations(rotations, 2)
        ),
        default=0.0,
    )


def _information_ratio(directions: Any) -> float:
    information = sum(
        (np.eye(3) - np.outer(direction, direction) for direction in directions),
        start=np.zeros((3, 3)),
    )
    values = np.linalg.eigvalsh(information)
    return float(values[0] / values[-1]) if values[-1] > 1e-12 else 0.0


def _unit_directions(values: Any, label: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-9) or not np.all(np.isfinite(norms)):
        raise PoseSolveError(f"a consumed {label} direction is zero or non-finite")
    return array / norms[:, None]


def _vector(values: Sequence[float], width: int, label: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise PoseSolveError(f"{label} must contain {width} finite values")
    return array


def _points(values: Sequence[Sequence[float]], width: int, label: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.all(np.isfinite(array)):
        raise PoseSolveError(f"{label} points must be finite Nx{width} values")
    return array


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))
