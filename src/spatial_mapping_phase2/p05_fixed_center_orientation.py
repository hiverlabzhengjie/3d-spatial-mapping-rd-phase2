"""D034 fixed-centre operational orientation estimation and sealed validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import cv2
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolution,
    PoseSolveError,
    project_world_points,
    undistort_image_points,
)


@dataclass(frozen=True, slots=True)
class D034SolverConfig:
    solve_inlier_threshold_pixels: float = 30.0
    huber_scale_pixels: float = 6.0
    minimum_information_ratio: float = 0.01
    ambiguity_trimmed_rmse_pixels: float = 2.0
    ambiguity_rotation_degrees: float = 2.0
    maximum_subset_spread_degrees: float = 7.0
    perturbation_pixels: float = 2.0
    maximum_perturbation_rotation_degrees: float = 0.5
    refinement_bound_rotation_radians: float = 0.5
    active_bound_margin_radians: float = 1e-6

    def __post_init__(self) -> None:
        positive = (
            self.solve_inlier_threshold_pixels,
            self.huber_scale_pixels,
            self.minimum_information_ratio,
            self.ambiguity_trimmed_rmse_pixels,
            self.ambiguity_rotation_degrees,
            self.maximum_subset_spread_degrees,
            self.perturbation_pixels,
            self.maximum_perturbation_rotation_degrees,
            self.refinement_bound_rotation_radians,
            self.active_bound_margin_radians,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise PoseSolveError("all D034 solver limits must be finite and positive")
        if self.minimum_information_ratio >= 1:
            raise PoseSolveError("D034 information ratio must be smaller than one")

    def to_dict(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class D034SubsetResult:
    subset_indices: tuple[int, int, int]
    subset_landmark_ids: tuple[str, str, str]
    rotation_world_from_camera: tuple[tuple[float, float, float], ...]
    reprojection_errors_pixels: tuple[float, float, float, float]
    inlier_indices: tuple[int, ...]
    trimmed_three_rmse_pixels: float
    proper_rotation: bool
    positive_depths: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "subset_indices": list(self.subset_indices),
            "subset_landmark_ids": list(self.subset_landmark_ids),
            "rotation_world_from_camera": [list(row) for row in self.rotation_world_from_camera],
            "reprojection_errors_pixels": list(self.reprojection_errors_pixels),
            "inlier_indices": list(self.inlier_indices),
            "trimmed_three_rmse_pixels": self.trimmed_three_rmse_pixels,
            "proper_rotation": self.proper_rotation,
            "positive_depths": self.positive_depths,
        }


@dataclass(frozen=True, slots=True)
class D034OrientationSolution:
    T_world_from_camera: tuple[tuple[float, ...], ...]
    fixed_camera_center_world_metres: tuple[float, float, float]
    optical_axis_world: tuple[float, float, float]
    solve_landmark_ids: tuple[str, str, str, str]
    selected_subset_indices: tuple[int, int, int]
    inlier_indices: tuple[int, ...]
    rejected_solve_landmark_id: str | None
    solve_projected_pixels: tuple[tuple[float, float], ...]
    solve_reprojection_errors_pixels: tuple[float, float, float, float]
    solve_point_depths_camera_metres: tuple[float, float, float, float]
    camera_information_ratio: float
    world_information_ratio: float
    subset_spread_degrees: float
    maximum_perturbation_rotation_degrees: float
    subset_results: tuple[D034SubsetResult, ...]
    refinement_active_bound: bool

    def to_dict(self) -> dict[str, object]:
        errors = self.solve_reprojection_errors_pixels
        return {
            "T_world_from_camera": [list(row) for row in self.T_world_from_camera],
            "fixed_camera_center_world_metres": list(self.fixed_camera_center_world_metres),
            "translation_optimization": "prohibited; exact D034 fixed-centre authority",
            "optical_axis_world": list(self.optical_axis_world),
            "solve_landmark_ids": list(self.solve_landmark_ids),
            "selected_subset_indices": list(self.selected_subset_indices),
            "inlier_indices": list(self.inlier_indices),
            "rejected_solve_landmark_id": self.rejected_solve_landmark_id,
            "solve_projected_pixels": [list(point) for point in self.solve_projected_pixels],
            "solve_reprojection_errors_pixels": list(errors),
            "solve_reprojection_rmse_pixels": _rmse(errors),
            "solve_reprojection_max_pixels": max(errors),
            "solve_point_depths_camera_metres": list(self.solve_point_depths_camera_metres),
            "camera_information_ratio": self.camera_information_ratio,
            "world_information_ratio": self.world_information_ratio,
            "subset_spread_degrees": self.subset_spread_degrees,
            "maximum_perturbation_rotation_degrees": (
                self.maximum_perturbation_rotation_degrees
            ),
            "refinement_active_bound": self.refinement_active_bound,
            "subset_results": [value.to_dict() for value in self.subset_results],
            "held_out_evidence": "not loaded by the D034 solve",
        }


@dataclass(frozen=True, slots=True)
class D034ValidationResult:
    validation_landmark_ids: tuple[str, str]
    projected_pixels: tuple[tuple[float, float], tuple[float, float]]
    individual_reprojection_errors_pixels: tuple[float, float]
    individual_pass: tuple[bool, bool]
    threshold_pixels: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "validation_landmark_ids": list(self.validation_landmark_ids),
            "projected_pixels": [list(point) for point in self.projected_pixels],
            "individual_reprojection_errors_pixels": list(
                self.individual_reprojection_errors_pixels
            ),
            "individual_pass": list(self.individual_pass),
            "threshold_pixels": self.threshold_pixels,
            "descriptive_rmse_pixels": _rmse(self.individual_reprojection_errors_pixels),
            "status": self.status,
            "influence": "evaluated once after solve freeze; never used for tuning or selection",
        }


@dataclass(frozen=True, slots=True)
class D034SolveSelection:
    selected_indices: tuple[int, int, int, int]
    selected_landmark_ids: tuple[str, str, str, str]
    image_hull_fraction: float
    minimum_image_separation_fraction: float
    world_information_ratio: float
    minimum_world_direction_separation_degrees: float
    ranking: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_indices": list(self.selected_indices),
            "selected_landmark_ids": list(self.selected_landmark_ids),
            "image_hull_fraction": self.image_hull_fraction,
            "minimum_image_separation_fraction": self.minimum_image_separation_fraction,
            "world_information_ratio": self.world_information_ratio,
            "minimum_world_direction_separation_degrees": (
                self.minimum_world_direction_separation_degrees
            ),
            "ranking": list(self.ranking),
            "selection_influence": "identity, distribution and conditioning only",
        }


def select_d034_solve_set(
    landmark_ids: Sequence[str],
    world_points: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    camera_center_world_metres: Sequence[float],
    image_width_pixels: int,
    image_height_pixels: int,
    *,
    minimum_information_ratio: float = 0.01,
) -> D034SolveSelection:
    """Freeze four points using distribution and conditioning, never residual evidence."""

    world = _points(world_points, 3, "selection world")
    image = _points(image_points, 2, "selection image")
    center = _vector(camera_center_world_metres, 3, "camera centre")
    ids = tuple(str(value) for value in landmark_ids)
    if len(ids) != len(world) or len(world) != len(image) or len(ids) < 4:
        raise PoseSolveError("D034 selection requires matching inputs with at least four points")
    if len(set(ids)) != len(ids):
        raise PoseSolveError("D034 selection landmark identities must be unique")
    if min(image_width_pixels, image_height_pixels) <= 0:
        raise PoseSolveError("D034 selection image dimensions must be positive")
    diagonal = math.hypot(image_width_pixels, image_height_pixels)
    records: list[tuple[tuple[float, float, float, float], dict[str, object]]] = []
    for indices in combinations(range(len(ids)), 4):
        subset_world = world[list(indices)]
        directions = _unit_directions(subset_world - center, "selection world")
        ratio = _information_ratio(directions)
        if ratio < minimum_information_ratio:
            continue
        subset_image = image[list(indices)]
        hull = cv2.convexHull(subset_image.astype(np.float32))
        hull_fraction = float(cv2.contourArea(hull)) / (
            image_width_pixels * image_height_pixels
        )
        image_separation = _minimum_pairwise_distance(subset_image) / diagonal
        world_separation = _minimum_direction_separation_degrees(directions)
        record: dict[str, object] = {
            "indices": list(indices),
            "landmark_ids": [ids[index] for index in indices],
            "image_hull_fraction": hull_fraction,
            "minimum_image_separation_fraction": image_separation,
            "world_information_ratio": ratio,
            "minimum_world_direction_separation_degrees": world_separation,
        }
        score = (hull_fraction, image_separation, ratio, world_separation)
        records.append((score, record))
    if not records:
        raise PoseSolveError("no four-point set passes D034 directional conditioning")
    records.sort(key=lambda value: value[0], reverse=True)
    selected = records[0][1]
    indices_value = tuple(int(value) for value in cast(list[int], selected["indices"]))
    ids_value = tuple(str(value) for value in cast(list[str], selected["landmark_ids"]))
    return D034SolveSelection(
        indices_value,  # type: ignore[arg-type]
        ids_value,  # type: ignore[arg-type]
        cast(float, selected["image_hull_fraction"]),
        cast(float, selected["minimum_image_separation_fraction"]),
        cast(float, selected["world_information_ratio"]),
        cast(float, selected["minimum_world_direction_separation_degrees"]),
        tuple(record for _, record in records),
    )


def solve_d034_orientation(
    intrinsics: CameraIntrinsics,
    camera_center_world_metres: Sequence[float],
    solve_landmark_ids: Sequence[str],
    solve_world_points: Sequence[Sequence[float]],
    solve_image_points: Sequence[Sequence[float]],
    *,
    config: D034SolverConfig | None = None,
    assess_perturbations: bool = True,
) -> D034OrientationSolution:
    """Solve D034 rotation from exactly four solve points without accepting validation data."""

    selected_config = config or D034SolverConfig()
    solution = _solve_d034_once(
        intrinsics,
        camera_center_world_metres,
        solve_landmark_ids,
        solve_world_points,
        solve_image_points,
        selected_config,
    )
    if not assess_perturbations:
        return solution
    base_rotation = np.asarray(solution.T_world_from_camera)[:3, :3]
    image = _points(solve_image_points, 2, "solve image")
    maximum = 0.0
    for point_index in range(4):
        for axis in range(2):
            for sign in (-1.0, 1.0):
                perturbed = image.copy()
                perturbed[point_index, axis] += sign * selected_config.perturbation_pixels
                try:
                    candidate = _solve_d034_once(
                        intrinsics,
                        camera_center_world_metres,
                        solve_landmark_ids,
                        solve_world_points,
                        perturbed,
                        selected_config,
                    )
                except PoseSolveError as error:
                    raise PoseSolveError(
                        "D034 click perturbation caused an unstable solve: " + str(error)
                    ) from error
                angle = rotation_difference_degrees(
                    base_rotation, np.asarray(candidate.T_world_from_camera)[:3, :3]
                )
                maximum = max(maximum, angle)
    if maximum > selected_config.maximum_perturbation_rotation_degrees:
        raise PoseSolveError(
            "D034 perturbation sensitivity exceeds "
            f"{selected_config.maximum_perturbation_rotation_degrees:.3f} degrees: {maximum:.3f}"
        )
    return D034OrientationSolution(
        solution.T_world_from_camera,
        solution.fixed_camera_center_world_metres,
        solution.optical_axis_world,
        solution.solve_landmark_ids,
        solution.selected_subset_indices,
        solution.inlier_indices,
        solution.rejected_solve_landmark_id,
        solution.solve_projected_pixels,
        solution.solve_reprojection_errors_pixels,
        solution.solve_point_depths_camera_metres,
        solution.camera_information_ratio,
        solution.world_information_ratio,
        solution.subset_spread_degrees,
        maximum,
        solution.subset_results,
        solution.refinement_active_bound,
    )


def evaluate_d034_validation(
    intrinsics: CameraIntrinsics,
    frozen_solution: D034OrientationSolution,
    validation_landmark_ids: Sequence[str],
    validation_world_points: Sequence[Sequence[float]],
    validation_image_points: Sequence[Sequence[float]],
    *,
    threshold_pixels: float = 30.0,
) -> D034ValidationResult:
    """Evaluate exactly two sealed validation points against an already-frozen orientation."""

    return evaluate_d034_frozen_validation(
        intrinsics,
        frozen_solution.T_world_from_camera,
        frozen_solution.solve_landmark_ids,
        validation_landmark_ids,
        validation_world_points,
        validation_image_points,
        threshold_pixels=threshold_pixels,
    )


def evaluate_d034_frozen_validation(
    intrinsics: CameraIntrinsics,
    T_world_from_camera: Sequence[Sequence[float]],
    solve_landmark_ids: Sequence[str],
    validation_landmark_ids: Sequence[str],
    validation_world_points: Sequence[Sequence[float]],
    validation_image_points: Sequence[Sequence[float]],
    *,
    threshold_pixels: float = 30.0,
) -> D034ValidationResult:
    """Evaluate a sealed pair from a serialized, already-frozen D034 transform."""

    world = _points(validation_world_points, 3, "validation world")
    image = _points(validation_image_points, 2, "validation image")
    ids = tuple(str(value) for value in validation_landmark_ids)
    if world.shape != (2, 3) or image.shape != (2, 2) or len(ids) != 2:
        raise PoseSolveError("D034 validation requires exactly two matching points")
    if len(set(ids)) != 2 or set(ids) & {str(value) for value in solve_landmark_ids}:
        raise PoseSolveError("D034 validation identities must be unique and separate from solve")
    if not math.isfinite(threshold_pixels) or threshold_pixels <= 0:
        raise PoseSolveError("D034 validation threshold must be finite and positive")
    transform = np.asarray(T_world_from_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise PoseSolveError("frozen D034 transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise PoseSolveError("frozen D034 transform has an invalid homogeneous row")
    if abs(float(np.linalg.det(transform[:3, :3])) - 1) > 1e-8:
        raise PoseSolveError("frozen D034 transform contains an improper rotation")
    rotation_camera_from_world = transform[:3, :3].T
    rvec, _ = cv2.Rodrigues(rotation_camera_from_world)
    center = transform[:3, 3]
    translation = -rotation_camera_from_world @ center
    projected, depths = project_world_points(intrinsics, world, rvec, translation)
    if np.any(depths <= 0):
        raise PoseSolveError("D034 validation point is behind the frozen camera")
    errors = np.linalg.norm(projected - image, axis=1)
    passed = tuple(bool(value <= threshold_pixels) for value in errors)
    return D034ValidationResult(
        (ids[0], ids[1]),
        ((float(projected[0, 0]), float(projected[0, 1])),
         (float(projected[1, 0]), float(projected[1, 1]))),
        (float(errors[0]), float(errors[1])),
        (passed[0], passed[1]),
        threshold_pixels,
        "accepted" if all(passed) else "rejected",
    )


def _solve_d034_once(
    intrinsics: CameraIntrinsics,
    camera_center_world_metres: Sequence[float],
    solve_landmark_ids: Sequence[str],
    solve_world_points: Sequence[Sequence[float]],
    solve_image_points: Sequence[Sequence[float]],
    config: D034SolverConfig,
) -> D034OrientationSolution:
    center = _vector(camera_center_world_metres, 3, "camera centre")
    world = _points(solve_world_points, 3, "solve world")
    image = _points(solve_image_points, 2, "solve image")
    ids = tuple(str(value) for value in solve_landmark_ids)
    if world.shape != (4, 3) or image.shape != (4, 2) or len(ids) != 4:
        raise PoseSolveError("D034 orientation requires exactly four matching solve points")
    if len(set(ids)) != 4:
        raise PoseSolveError("D034 solve landmark identities must be unique")
    world_directions = _unit_directions(world - center, "solve world")
    normalized = undistort_image_points(intrinsics, image)
    camera_directions = _unit_directions(
        np.column_stack((normalized, np.ones(4))), "solve camera"
    )
    world_ratio = _information_ratio(world_directions)
    camera_ratio = _information_ratio(camera_directions)
    if min(world_ratio, camera_ratio) < config.minimum_information_ratio:
        raise PoseSolveError(
            "D034 directional geometry is poorly conditioned: "
            f"camera={camera_ratio:.6f}, world={world_ratio:.6f}"
        )

    subset_results: list[D034SubsetResult] = []
    rotations: list[np.ndarray[Any, Any]] = []
    for subset in combinations(range(4), 3):
        indices = list(subset)
        rotation = _wahba_rotation(
            camera_directions[indices], world_directions[indices]
        )
        proper = bool(
            np.all(np.isfinite(rotation)) and abs(float(np.linalg.det(rotation)) - 1) <= 1e-8
        )
        projected, depths = _project_from_fixed_center(intrinsics, world, center, rotation)
        errors = np.linalg.norm(projected - image, axis=1)
        inliers = tuple(
            int(index)
            for index, error in enumerate(errors)
            if error <= config.solve_inlier_threshold_pixels
        )
        trimmed = np.sort(errors)[:3]
        subset_results.append(
            D034SubsetResult(
                (subset[0], subset[1], subset[2]),
                (ids[subset[0]], ids[subset[1]], ids[subset[2]]),
                tuple(
                    (float(row[0]), float(row[1]), float(row[2])) for row in rotation
                ),
                (float(errors[0]), float(errors[1]), float(errors[2]), float(errors[3])),
                inliers,
                float(math.sqrt(float(np.mean(trimmed**2)))),
                proper,
                bool(np.all(depths > 0)),
            )
        )
        rotations.append(rotation)
    eligible = [
        index
        for index, value in enumerate(subset_results)
        if value.proper_rotation and value.positive_depths and len(value.inlier_indices) >= 3
    ]
    if not eligible:
        raise PoseSolveError("D034 failed to form three-point solve consensus")
    maximum_inliers = max(len(subset_results[index].inlier_indices) for index in eligible)
    eligible = [
        index for index in eligible
        if len(subset_results[index].inlier_indices) == maximum_inliers
    ]
    eligible.sort(
        key=lambda index: (
            subset_results[index].trimmed_three_rmse_pixels,
            max(
                subset_results[index].reprojection_errors_pixels[value]
                for value in subset_results[index].inlier_indices
            ),
            subset_results[index].subset_indices,
        )
    )
    best_index = eligible[0]
    best = subset_results[best_index]
    competitive = [
        index for index in eligible
        if subset_results[index].trimmed_three_rmse_pixels
        <= best.trimmed_three_rmse_pixels + config.ambiguity_trimmed_rmse_pixels
    ]
    competitive_spread = _rotation_spread([rotations[index] for index in competitive])
    if competitive_spread > config.ambiguity_rotation_degrees:
        raise PoseSolveError(
            f"D034 has ambiguous competitive rotations: {competitive_spread:.3f} degrees"
        )
    consensus_spread = _rotation_spread([rotations[index] for index in eligible])
    if consensus_spread > config.maximum_subset_spread_degrees:
        raise PoseSolveError(
            f"D034 subset sensitivity exceeds limit: {consensus_spread:.3f} degrees"
        )

    initial_rotation = rotations[best_index]
    initial_rvec, _ = cv2.Rodrigues(initial_rotation.T)
    initial = initial_rvec.reshape(3)
    inlier_indices = list(best.inlier_indices)

    def residuals(rvec: Any) -> Any:
        rotation_camera_from_world, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
        translation = -rotation_camera_from_world @ center
        projected, _ = project_world_points(
            intrinsics, world[inlier_indices], rvec, translation
        )
        return (projected - image[inlier_indices]).reshape(-1)

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
        raise PoseSolveError("D034 bounded robust rotation refinement failed")
    active_bound = bool(
        np.any(refined.x - lower <= config.active_bound_margin_radians)
        or np.any(upper - refined.x <= config.active_bound_margin_radians)
    )
    if active_bound:
        raise PoseSolveError("D034 rotation refinement reached an active bound")
    rotation_camera_from_world, _ = cv2.Rodrigues(refined.x)
    rotation_world_from_camera = rotation_camera_from_world.T
    if abs(float(np.linalg.det(rotation_world_from_camera)) - 1) > 1e-8:
        raise PoseSolveError("D034 refinement produced an improper rotation")
    translation = -rotation_camera_from_world @ center
    projected, depths = project_world_points(intrinsics, world, refined.x, translation)
    if np.any(depths <= 0):
        raise PoseSolveError("D034 solve point is behind the camera")
    errors = np.linalg.norm(projected - image, axis=1)
    final_inliers = tuple(
        int(index)
        for index, error in enumerate(errors)
        if error <= config.solve_inlier_threshold_pixels
    )
    if len(final_inliers) < 3 or 4 - len(final_inliers) > 1:
        raise PoseSolveError("D034 refinement does not retain three-point consensus")
    rejected = next((ids[index] for index in range(4) if index not in final_inliers), None)
    transform = np.eye(4)
    transform[:3, :3] = rotation_world_from_camera
    transform[:3, 3] = center
    return D034OrientationSolution(
        tuple(tuple(float(value) for value in row) for row in transform),
        (float(center[0]), float(center[1]), float(center[2])),
        tuple(float(value) for value in rotation_world_from_camera[:, 2]),  # type: ignore[arg-type]
        (ids[0], ids[1], ids[2], ids[3]),
        best.subset_indices,
        final_inliers,
        rejected,
        tuple((float(point[0]), float(point[1])) for point in projected),
        (float(errors[0]), float(errors[1]), float(errors[2]), float(errors[3])),
        (float(depths[0]), float(depths[1]), float(depths[2]), float(depths[3])),
        camera_ratio,
        world_ratio,
        consensus_spread,
        0.0,
        tuple(subset_results),
        active_bound,
    )


def solve_orientation_at_fixed_center(
    intrinsics: CameraIntrinsics,
    camera_center_world_metres: Sequence[float],
    solve_world_points: Sequence[Sequence[float]],
    solve_image_points: Sequence[Sequence[float]],
    held_out_world_points: Sequence[Sequence[float]] = (),
    held_out_image_points: Sequence[Sequence[float]] = (),
    *,
    huber_scale_pixels: float = 6.0,
    refinement_bound_rotation_radians: float = 0.5,
) -> PoseSolution:
    """Legacy bounded-trial helper; not an operational D034 entry point."""

    center = _vector(camera_center_world_metres, 3, "camera centre")
    world = _points(solve_world_points, 3, "solve world")
    image = _points(solve_image_points, 2, "solve image")
    held_world = _points(held_out_world_points, 3, "held-out world", allow_empty=True)
    held_image = _points(held_out_image_points, 2, "held-out image", allow_empty=True)
    if len(world) != len(image) or len(held_world) != len(held_image) or len(world) < 3:
        raise PoseSolveError("legacy trial requires at least three matching solve points")
    camera = _unit_directions(
        np.column_stack((undistort_image_points(intrinsics, image), np.ones(len(image)))),
        "solve camera",
    )
    try:
        directions = _unit_directions(world - center, "solve world")
    except PoseSolveError as error:
        raise PoseSolveError(
            "a solve world point coincides with the fixed camera centre"
        ) from error
    rotation = _wahba_rotation(camera, directions)
    initial, _ = cv2.Rodrigues(rotation.T)

    def residuals(rvec: Any) -> Any:
        rotation_camera_from_world, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3))
        translation = -rotation_camera_from_world @ center
        projected, _ = project_world_points(intrinsics, world, rvec, translation)
        return (projected - image).reshape(-1)

    initial_vector = initial.reshape(3)
    refined = least_squares(
        residuals,
        initial_vector,
        bounds=(initial_vector - refinement_bound_rotation_radians,
                initial_vector + refinement_bound_rotation_radians),
        loss="huber",
        f_scale=huber_scale_pixels,
        max_nfev=500,
    )
    rotation_camera_from_world, _ = cv2.Rodrigues(refined.x)
    translation = -rotation_camera_from_world @ center
    solve_projected, solve_depths = project_world_points(
        intrinsics, world, refined.x, translation
    )
    held_projected, held_depths = project_world_points(
        intrinsics, held_world, refined.x, translation
    )
    depths = np.concatenate((solve_depths, held_depths))
    if np.any(depths <= 0):
        raise PoseSolveError("legacy trial fails cheirality")
    rotation_world_from_camera = rotation_camera_from_world.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_world_from_camera
    transform[:3, 3] = center
    return PoseSolution(
        tuple(tuple(float(value) for value in row) for row in transform),
        (float(center[0]), float(center[1]), float(center[2])),
        tuple(float(value) for value in rotation_world_from_camera[:, 2]),  # type: ignore[arg-type]
        tuple(range(len(world))),
        tuple((float(value[0]), float(value[1])) for value in solve_projected),
        tuple((float(value[0]), float(value[1])) for value in held_projected),
        tuple(float(value) for value in np.linalg.norm(solve_projected - image, axis=1)),
        tuple(float(value) for value in np.linalg.norm(held_projected - held_image, axis=1)),
        tuple(float(value) for value in depths),
    )


def rotation_difference_degrees(left: Any, right: Any) -> float:
    relative = np.asarray(left, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        right, dtype=np.float64
    ).reshape(3, 3)
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _wahba_rotation(camera_directions: Any, world_directions: Any) -> Any:
    cross_covariance = np.asarray(world_directions).T @ np.asarray(camera_directions)
    left, _, right_transpose = np.linalg.svd(cross_covariance)
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(left @ right_transpose) >= 0 else -1.0
    rotation = left @ correction @ right_transpose
    if not np.all(np.isfinite(rotation)) or abs(float(np.linalg.det(rotation)) - 1) > 1e-8:
        raise PoseSolveError("Wahba/SVD did not produce a proper finite rotation")
    return rotation


def _project_from_fixed_center(
    intrinsics: CameraIntrinsics, world: Any, center: Any, rotation_world_from_camera: Any
) -> tuple[Any, Any]:
    rotation_camera_from_world = np.asarray(rotation_world_from_camera).T
    rvec, _ = cv2.Rodrigues(rotation_camera_from_world)
    translation = -rotation_camera_from_world @ center
    return project_world_points(intrinsics, world, rvec, translation)


def _unit_directions(values: Any, label: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-9) or not np.all(np.isfinite(norms)):
        raise PoseSolveError(f"a {label} direction is zero or non-finite")
    return array / norms[:, None]


def _information_ratio(directions: Any) -> float:
    information = sum(
        (np.eye(3) - np.outer(direction, direction) for direction in directions),
        start=np.zeros((3, 3)),
    )
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[-1] <= 1e-12:
        return 0.0
    return float(eigenvalues[0] / eigenvalues[-1])


def _rotation_spread(rotations: Sequence[Any]) -> float:
    if len(rotations) < 2:
        return 0.0
    return max(
        rotation_difference_degrees(rotations[first], rotations[second])
        for first, second in combinations(range(len(rotations)), 2)
    )


def _minimum_pairwise_distance(points: Any) -> float:
    return min(
        float(np.linalg.norm(points[first] - points[second]))
        for first, second in combinations(range(len(points)), 2)
    )


def _minimum_direction_separation_degrees(directions: Any) -> float:
    return min(
        math.degrees(
            math.acos(float(np.clip(np.dot(directions[first], directions[second]), -1, 1)))
        )
        for first, second in combinations(range(len(directions)), 2)
    )


def _vector(values: Sequence[float], width: int, label: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise PoseSolveError(f"{label} must contain {width} finite values")
    return array


def _points(values: Any, width: int, label: str, *, allow_empty: bool = False) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if allow_empty and array.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.all(np.isfinite(array)):
        raise PoseSolveError(f"{label} points must be finite Nx{width} values")
    return array


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))
