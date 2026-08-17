"""Independent P04 camera-pose solving and physical diagnostics.

Coordinate convention: world points are metres in the provisional P02 facility frame
(+X plan-left, +Y plan-down, +Z up). OpenCV estimates ``X_camera =
R_camera_from_world X_world + t_camera_from_world``. Public results expose its explicit
inverse, ``T_world_from_camera``; the camera optical axis is camera +Z.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]


class PoseSolveError(ValueError):
    """Raised when pose inputs or a candidate solution are unusable."""


CAMERA_MODELS = {"pinhole", "simple_radial", "simple_divisional", "radial"}


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    model: str
    width_pixels: int
    height_pixels: int
    fx_pixels: float
    fy_pixels: float
    cx_pixels: float
    cy_pixels: float
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.model not in CAMERA_MODELS:
            raise PoseSolveError(f"unsupported camera model: {self.model}")
        values = (
            self.width_pixels,
            self.height_pixels,
            self.fx_pixels,
            self.fy_pixels,
            self.cx_pixels,
            self.cy_pixels,
            *self.distortion,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise PoseSolveError("intrinsic parameters must be finite")
        if min(self.width_pixels, self.height_pixels, self.fx_pixels, self.fy_pixels) <= 0:
            raise PoseSolveError("image dimensions and focal lengths must be positive")
        expected = {"pinhole": 0, "simple_radial": 1, "simple_divisional": 1, "radial": 2}
        if len(self.distortion) != expected[self.model]:
            raise PoseSolveError(f"{self.model} requires {expected[self.model]} distortion values")

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "fx_pixels": self.fx_pixels,
            "fy_pixels": self.fy_pixels,
            "cx_pixels": self.cx_pixels,
            "cy_pixels": self.cy_pixels,
            "distortion": list(self.distortion),
        }


@dataclass(frozen=True, slots=True)
class PoseSolution:
    T_world_from_camera: tuple[tuple[float, ...], ...]
    camera_position_world_metres: tuple[float, float, float]
    optical_axis_world: tuple[float, float, float]
    ransac_inlier_indices: tuple[int, ...]
    solve_projected_pixels: tuple[tuple[float, float], ...]
    held_out_projected_pixels: tuple[tuple[float, float], ...]
    solve_reprojection_errors_pixels: tuple[float, ...]
    held_out_reprojection_errors_pixels: tuple[float, ...]
    point_depths_camera_metres: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        solve = self.solve_reprojection_errors_pixels
        held_out = self.held_out_reprojection_errors_pixels
        return {
            "T_world_from_camera": [list(row) for row in self.T_world_from_camera],
            "camera_position_world_metres": list(self.camera_position_world_metres),
            "optical_axis_world": list(self.optical_axis_world),
            "ransac_inlier_indices": list(self.ransac_inlier_indices),
            "solve_projected_pixels": [list(point) for point in self.solve_projected_pixels],
            "held_out_projected_pixels": [
                list(point) for point in self.held_out_projected_pixels
            ],
            "solve_reprojection_errors_pixels": list(solve),
            "held_out_reprojection_errors_pixels": list(held_out),
            "solve_reprojection_rmse_pixels": _rmse(solve),
            "solve_reprojection_max_pixels": max(solve),
            "held_out_reprojection_rmse_pixels": _rmse(held_out) if held_out else None,
            "held_out_reprojection_max_pixels": max(held_out) if held_out else None,
            "point_depths_camera_metres": list(self.point_depths_camera_metres),
        }


def solve_camera_pose(
    intrinsics: CameraIntrinsics,
    solve_world_points: Sequence[Sequence[float]],
    solve_image_points: Sequence[Sequence[float]],
    held_out_world_points: Sequence[Sequence[float]] = (),
    held_out_image_points: Sequence[Sequence[float]] = (),
    *,
    ransac_threshold_pixels: float = 8.0,
    refinement_bound_rotation_radians: float = 0.5,
    refinement_bound_translation_metres: float = 5.0,
) -> PoseSolution:
    """Solve independently from any mounting prior and evaluate untouched held-out points."""

    world = _points(solve_world_points, 3, "solve world")
    image = _points(solve_image_points, 2, "solve image")
    held_world = _points(held_out_world_points, 3, "held-out world", allow_empty=True)
    held_image = _points(held_out_image_points, 2, "held-out image", allow_empty=True)
    if len(world) != len(image) or len(held_world) != len(held_image):
        raise PoseSolveError("world and image point counts must match")
    if len(world) < 4:
        raise PoseSolveError("at least four solve correspondences are required")
    if np.linalg.matrix_rank(world - np.mean(world, axis=0), tol=1e-8) < 3:
        raise PoseSolveError("solve world points must span three dimensions")
    if ransac_threshold_pixels <= 0:
        raise PoseSolveError("RANSAC threshold must be positive")

    normalized = undistort_image_points(intrinsics, image)
    cv2.setRNGSeed(20260814)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        world,
        normalized,
        np.eye(3, dtype=np.float64),
        np.zeros((0, 1), dtype=np.float64),
        iterationsCount=1000,
        reprojectionError=(
            ransac_threshold_pixels / ((intrinsics.fx_pixels + intrinsics.fy_pixels) / 2)
        ),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or rvec is None or tvec is None or inliers is None or len(inliers) < 4:
        raise PoseSolveError("OpenCV PnP/RANSAC did not produce at least four inliers")
    inlier_indices = tuple(int(value) for value in inliers.reshape(-1))
    initial = np.concatenate((rvec.reshape(3), tvec.reshape(3)))
    lower = initial - np.array(
        [refinement_bound_rotation_radians] * 3 + [refinement_bound_translation_metres] * 3
    )
    upper = initial + np.array(
        [refinement_bound_rotation_radians] * 3 + [refinement_bound_translation_metres] * 3
    )

    def residuals(parameters: Any) -> Any:
        projected, _ = project_world_points(
            intrinsics, world[list(inlier_indices)], parameters[:3], parameters[3:]
        )
        return (projected - image[list(inlier_indices)]).reshape(-1)

    refined = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss="huber",
        f_scale=4.0,
        max_nfev=300,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    if not refined.success or not np.all(np.isfinite(refined.x)):
        raise PoseSolveError("bounded robust pose refinement failed")
    refined_rvec = refined.x[:3]
    refined_tvec = refined.x[3:]
    solve_projected, solve_depths = project_world_points(
        intrinsics, world, refined_rvec, refined_tvec
    )
    held_projected, held_depths = project_world_points(
        intrinsics, held_world, refined_rvec, refined_tvec
    )
    depths = np.concatenate((solve_depths, held_depths))
    if np.any(depths <= 0):
        raise PoseSolveError("candidate fails cheirality: at least one reference is behind camera")
    rotation_camera_from_world, _ = cv2.Rodrigues(refined_rvec)
    rotation_world_from_camera = rotation_camera_from_world.T
    camera_position = -rotation_world_from_camera @ refined_tvec
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_world_from_camera
    transform[:3, 3] = camera_position
    optical_axis = rotation_world_from_camera[:, 2]
    solve_errors = np.linalg.norm(solve_projected - image, axis=1)
    held_errors = np.linalg.norm(held_projected - held_image, axis=1)
    return PoseSolution(
        tuple(tuple(float(value) for value in row) for row in transform),
        (
            float(camera_position[0]),
            float(camera_position[1]),
            float(camera_position[2]),
        ),
        (float(optical_axis[0]), float(optical_axis[1]), float(optical_axis[2])),
        inlier_indices,
        tuple((float(point[0]), float(point[1])) for point in solve_projected),
        tuple((float(point[0]), float(point[1])) for point in held_projected),
        tuple(float(value) for value in solve_errors),
        tuple(float(value) for value in held_errors),
        tuple(float(value) for value in depths),
    )


def project_world_points(
    intrinsics: CameraIntrinsics,
    world_points: Sequence[Sequence[float]] | Any,
    rvec_camera_from_world: Sequence[float] | Any,
    tvec_camera_from_world: Sequence[float] | Any,
) -> tuple[Any, Any]:
    """Project world XYZ through a GeoCalib camera model; return pixels and camera Z."""

    world = np.asarray(world_points, dtype=np.float64).reshape((-1, 3))
    if not len(world):
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    rotation, _ = cv2.Rodrigues(np.asarray(rvec_camera_from_world, dtype=np.float64).reshape(3))
    translation = np.asarray(tvec_camera_from_world, dtype=np.float64).reshape(3)
    camera = (rotation @ world.T).T + translation
    normalized = camera[:, :2] / camera[:, 2:3]
    distorted = _distort_normalized(intrinsics, normalized)
    pixels = np.column_stack(
        (
            intrinsics.fx_pixels * distorted[:, 0] + intrinsics.cx_pixels,
            intrinsics.fy_pixels * distorted[:, 1] + intrinsics.cy_pixels,
        )
    )
    return pixels, camera[:, 2]


def undistort_image_points(intrinsics: CameraIntrinsics, image_points: Any) -> Any:
    """Apply GeoCalib's declared normalized-image inverse for the candidate model."""

    pixels = np.asarray(image_points, dtype=np.float64).reshape((-1, 2))
    distorted = np.column_stack(
        (
            (pixels[:, 0] - intrinsics.cx_pixels) / intrinsics.fx_pixels,
            (pixels[:, 1] - intrinsics.cy_pixels) / intrinsics.fy_pixels,
        )
    )
    radius2 = np.sum(distorted**2, axis=1, keepdims=True)
    if intrinsics.model == "pinhole":
        scale = np.ones_like(radius2)
    elif intrinsics.model == "simple_radial":
        scale = 1 - intrinsics.distortion[0] * radius2
    elif intrinsics.model == "radial":
        k1, k2 = intrinsics.distortion
        scale = 1 - k1 * radius2 + (3 * k1**2 - k2) * radius2**2
    else:
        denominator = 1 + intrinsics.distortion[0] * radius2
        if np.any(np.abs(denominator) < 1e-9):
            raise PoseSolveError("simple-divisional undistortion is singular")
        scale = 1 / denominator
    normalized = distorted * scale
    if not np.all(np.isfinite(normalized)):
        raise PoseSolveError("undistorted image points are not finite")
    return normalized


def angular_difference_degrees(left: Sequence[float], right: Sequence[float]) -> float:
    left_unit = _unit(left)
    right_unit = _unit(right)
    return math.degrees(math.acos(float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))))


def _distort_normalized(intrinsics: CameraIntrinsics, normalized: Any) -> Any:
    radius2 = np.sum(normalized**2, axis=1, keepdims=True)
    if intrinsics.model == "pinhole":
        scale = np.ones_like(radius2)
    elif intrinsics.model == "simple_radial":
        scale = 1 + intrinsics.distortion[0] * radius2
    elif intrinsics.model == "radial":
        k1, k2 = intrinsics.distortion
        scale = 1 + k1 * radius2 + k2 * radius2**2
    else:
        k1 = intrinsics.distortion[0]
        discriminant = 1 - 4 * k1 * radius2
        if np.any(discriminant < -1e-10):
            raise PoseSolveError("simple-divisional projection is outside its valid domain")
        numerator = 1 - np.sqrt(np.maximum(discriminant, 0))
        denominator = 2 * k1 * radius2
        scale = np.ones_like(radius2)
        np.divide(numerator, denominator, out=scale, where=np.abs(denominator) > 1e-12)
    result = normalized * scale
    if not np.all(np.isfinite(result)):
        raise PoseSolveError("distorted image points are not finite")
    return result


def _points(
    values: Sequence[Sequence[float]], width: int, label: str, *, allow_empty: bool = False
) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if allow_empty and array.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.all(np.isfinite(array)):
        raise PoseSolveError(f"{label} points must be finite Nx{width} values")
    return array


def _unit(values: Sequence[float]) -> Any:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise PoseSolveError("vectors must contain three finite values")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise PoseSolveError("vectors must be non-zero")
    return vector / norm


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))
