"""All-solve robust refinement and solver-envelope evidence for P04 closeout."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, cast

import cv2
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolution,
    PoseSolveError,
    project_world_points,
    solve_camera_pose,
    undistort_image_points,
)

INITIALIZERS = ("ransac-epnp", "all-epnp", "all-sqpnp", "all-iterative")
ROBUST_LOSSES = ("huber", "soft_l1", "cauchy")
ROBUST_SCALES_PIXELS = (3.0, 6.0, 12.0)


@dataclass(frozen=True, slots=True)
class PoseEnvelopeCase:
    initializer: str
    robust_loss: str
    robust_scale_pixels: float
    bounded_solution: PoseSolution
    refinement_cost: float
    bound_was_active: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "initializer": self.initializer,
            "robust_loss": self.robust_loss,
            "robust_scale_pixels": self.robust_scale_pixels,
            "refinement_cost": self.refinement_cost,
            "bound_was_active": self.bound_was_active,
            "pose": self.bounded_solution.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PoseEnvelope:
    cases: tuple[PoseEnvelopeCase, ...]
    failed_initializers: tuple[tuple[str, str], ...]

    def summary(self) -> dict[str, object]:
        if not self.cases:
            raise PoseSolveError("pose envelope contains no successful cases")
        positions = np.asarray(
            [case.bounded_solution.camera_position_world_metres for case in self.cases]
        )
        centre = np.asarray([median(tuple(column)) for column in positions.T])
        distances = np.linalg.norm(positions - centre, axis=1)
        held_rmse = np.asarray(
            [
                _rmse(case.bounded_solution.held_out_reprojection_errors_pixels)
                for case in self.cases
            ]
        )
        solve_rmse = np.asarray(
            [_rmse(case.bounded_solution.solve_reprojection_errors_pixels) for case in self.cases]
        )
        optical_axes = tuple(case.bounded_solution.optical_axis_world for case in self.cases)
        return {
            "successful_case_count": len(self.cases),
            "failed_initializers": [
                {"initializer": name, "reason": reason}
                for name, reason in self.failed_initializers
            ],
            "median_camera_position_world_metres": centre.tolist(),
            "camera_position_min_world_metres": np.min(positions, axis=0).tolist(),
            "camera_position_max_world_metres": np.max(positions, axis=0).tolist(),
            "maximum_camera_position_distance_from_median_metres": float(max(distances)),
            "maximum_pairwise_optical_axis_angle_degrees": _maximum_pairwise_angle(optical_axes),
            "solve_rmse_min_pixels": float(min(solve_rmse)),
            "solve_rmse_median_pixels": float(median(tuple(solve_rmse))),
            "solve_rmse_max_pixels": float(max(solve_rmse)),
            "held_out_rmse_min_pixels": float(min(held_rmse)),
            "held_out_rmse_median_pixels": float(median(tuple(held_rmse))),
            "held_out_rmse_max_pixels": float(max(held_rmse)),
            "held_out_error_max_pixels": max(
                max(case.bounded_solution.held_out_reprojection_errors_pixels)
                for case in self.cases
            ),
            "camera_height_min_metres": float(min(positions[:, 2])),
            "camera_height_median_metres": float(median(tuple(positions[:, 2]))),
            "camera_height_max_metres": float(max(positions[:, 2])),
            "active_bound_case_count": sum(case.bound_was_active for case in self.cases),
        }


def build_pose_envelope(
    intrinsics: CameraIntrinsics,
    solve_world_points: Sequence[Sequence[float]],
    solve_image_points: Sequence[Sequence[float]],
    held_out_world_points: Sequence[Sequence[float]],
    held_out_image_points: Sequence[Sequence[float]],
) -> PoseEnvelope:
    """Refine all solve points from fixed initializers and a fixed robust-loss grid."""

    world = _points(solve_world_points, 3, "solve world")
    image = _points(solve_image_points, 2, "solve image")
    held_world = _points(held_out_world_points, 3, "held-out world")
    held_image = _points(held_out_image_points, 2, "held-out image")
    if len(world) != len(image) or len(held_world) != len(held_image):
        raise PoseSolveError("world and image point counts must match")
    if len(world) < 6 or np.linalg.matrix_rank(world - np.mean(world, axis=0), tol=1e-8) < 3:
        raise PoseSolveError(
            "closeout requires at least six solve points spanning three dimensions"
        )
    initializers: list[tuple[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for name in INITIALIZERS:
        try:
            initializers.append((name, _initialize(name, intrinsics, world, image)))
        except PoseSolveError as error:
            failures.append((name, str(error)))
    cases: list[PoseEnvelopeCase] = []
    for initializer, initial in initializers:
        for loss in ROBUST_LOSSES:
            for scale in ROBUST_SCALES_PIXELS:
                cases.append(
                    _refine_case(
                        initializer,
                        loss,
                        scale,
                        intrinsics,
                        world,
                        image,
                        held_world,
                        held_image,
                        initial,
                    )
                )
    return PoseEnvelope(tuple(cases), tuple(failures))


def application_projection_envelope(
    envelopes: Sequence[PoseEnvelope], *, minimum_focal_pixels: float
) -> dict[str, float]:
    """Convert the measured worst held-out pixel residual into a conservative ray envelope."""

    if not envelopes or minimum_focal_pixels <= 0:
        raise PoseSolveError("projection envelope requires poses and a positive focal length")
    maximum_pixels = max(
        cast(float, envelope.summary()["held_out_error_max_pixels"]) for envelope in envelopes
    )
    angular_degrees = math.degrees(math.atan(maximum_pixels / minimum_focal_pixels))
    return {
        "maximum_observed_held_out_error_pixels": maximum_pixels,
        "conservative_ray_angle_degrees": angular_degrees,
        "lateral_envelope_at_5_metres": 5.0 * math.tan(math.radians(angular_degrees)),
        "lateral_envelope_at_10_metres": 10.0 * math.tan(math.radians(angular_degrees)),
    }


def derive_provisional_application_tolerances(
    combined_summary: dict[str, object],
    projection_envelope: dict[str, float],
) -> dict[str, object]:
    """Round the measured envelope outward into provisional operating guardrails.

    These are change-detection and coarse-projection limits for the office pilot. They are not
    ground-truth XYZ accuracy because the P02 horizontal control uncertainty remains unknown.
    """
    return {
        "held_out_reprojection_max_pixels": _round_up(
            projection_envelope["maximum_observed_held_out_error_pixels"], 5.0
        ),
        "ray_angle_degrees": _round_up(
            projection_envelope["conservative_ray_angle_degrees"], 0.5
        ),
        "lateral_at_5_metres": _round_up(
            projection_envelope["lateral_envelope_at_5_metres"], 0.05
        ),
        "lateral_at_10_metres": _round_up(
            projection_envelope["lateral_envelope_at_10_metres"], 0.05
        ),
        "camera_centre_change_from_envelope_median_metres": _round_up(
            cast(
                float,
                combined_summary["maximum_position_distance_from_combined_median_metres"],
            ),
            0.05,
        ),
        "optical_axis_change_degrees": _round_up(
            cast(float, combined_summary["maximum_pairwise_optical_axis_angle_degrees"]), 0.5
        ),
        "height_and_mounting_prior": "diagnostic-only",
        "authority": (
            "provisional application envelope; not full XYZ accuracy and not a substitute for "
            "camera-specific held-out review"
        ),
    }


def _round_up(value: float, increment: float) -> float:
    if not math.isfinite(value) or value < 0.0 or increment <= 0.0:
        raise ValueError("round-up inputs must be finite and non-negative with positive increment")
    return math.ceil((value - 1e-12) / increment) * increment


def _initialize(name: str, intrinsics: CameraIntrinsics, world: Any, image: Any) -> Any:
    if name == "ransac-epnp":
        solution = solve_camera_pose(
            intrinsics, world, image, ransac_threshold_pixels=30.0
        )
        transform = np.asarray(solution.T_world_from_camera, dtype=np.float64)
        rotation_camera_from_world = transform[:3, :3].T
        rvec, _ = cv2.Rodrigues(rotation_camera_from_world)
        tvec = -rotation_camera_from_world @ transform[:3, 3]
        return np.concatenate((rvec.reshape(3), tvec.reshape(3)))
    flag = {
        "all-epnp": cv2.SOLVEPNP_EPNP,
        "all-sqpnp": cv2.SOLVEPNP_SQPNP,
        "all-iterative": cv2.SOLVEPNP_ITERATIVE,
    }.get(name)
    if flag is None:
        raise PoseSolveError(f"unsupported initializer: {name}")
    normalized = undistort_image_points(intrinsics, image)
    success, rvec, tvec = cv2.solvePnP(
        world,
        normalized,
        np.eye(3, dtype=np.float64),
        np.zeros((0, 1), dtype=np.float64),
        flags=flag,
    )
    if not success:
        raise PoseSolveError(f"{name} failed")
    return np.concatenate((np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)))


def _refine_case(
    initializer: str,
    loss: str,
    scale: float,
    intrinsics: CameraIntrinsics,
    world: Any,
    image: Any,
    held_world: Any,
    held_image: Any,
    initial: Any,
) -> PoseEnvelopeCase:
    lower = initial - np.asarray([0.5, 0.5, 0.5, 5.0, 5.0, 5.0])
    upper = initial + np.asarray([0.5, 0.5, 0.5, 5.0, 5.0, 5.0])

    def residuals(parameters: Any) -> Any:
        projected, _ = project_world_points(intrinsics, world, parameters[:3], parameters[3:])
        return (projected - image).reshape(-1)

    result = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss=loss,
        f_scale=scale,
        max_nfev=500,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise PoseSolveError("bounded all-solve refinement failed")
    solution = _solution(intrinsics, world, image, held_world, held_image, result.x)
    return PoseEnvelopeCase(
        initializer,
        loss,
        scale,
        solution,
        float(result.cost),
        bool(np.any(result.active_mask)),
    )


def _solution(
    intrinsics: CameraIntrinsics,
    world: Any,
    image: Any,
    held_world: Any,
    held_image: Any,
    parameters: Any,
) -> PoseSolution:
    solve_projected, solve_depth = project_world_points(
        intrinsics, world, parameters[:3], parameters[3:]
    )
    held_projected, held_depth = project_world_points(
        intrinsics, held_world, parameters[:3], parameters[3:]
    )
    depths = np.concatenate((solve_depth, held_depth))
    if np.any(depths <= 0):
        raise PoseSolveError("refined closeout case fails cheirality")
    rotation_camera_from_world, _ = cv2.Rodrigues(parameters[:3])
    rotation_world_from_camera = rotation_camera_from_world.T
    position = -rotation_world_from_camera @ parameters[3:]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_world_from_camera
    transform[:3, 3] = position
    optical = rotation_world_from_camera[:, 2]
    solve_error = np.linalg.norm(solve_projected - image, axis=1)
    held_error = np.linalg.norm(held_projected - held_image, axis=1)
    return PoseSolution(
        tuple(tuple(float(value) for value in row) for row in transform),
        (float(position[0]), float(position[1]), float(position[2])),
        (float(optical[0]), float(optical[1]), float(optical[2])),
        tuple(range(len(world))),
        tuple((float(point[0]), float(point[1])) for point in solve_projected),
        tuple((float(point[0]), float(point[1])) for point in held_projected),
        tuple(float(value) for value in solve_error),
        tuple(float(value) for value in held_error),
        tuple(float(value) for value in depths),
    )


def _points(values: Sequence[Sequence[float]], width: int, label: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.all(np.isfinite(array)):
        raise PoseSolveError(f"{label} points must be finite Nx{width} values")
    return array


def _maximum_pairwise_angle(vectors: Sequence[Sequence[float]]) -> float:
    maximum = 0.0
    for left_index, left_values in enumerate(vectors):
        left = np.asarray(left_values, dtype=np.float64)
        for right_values in vectors[left_index + 1 :]:
            right = np.asarray(right_values, dtype=np.float64)
            dot = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            maximum = max(maximum, math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0)))))
    return maximum


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))
