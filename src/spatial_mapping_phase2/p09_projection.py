"""Exact P09 live-image convention and analytic camera-ray/floor projection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_tracking_domain import (
    CALIBRATED_FRAME,
    CAMERA_IDS,
    FACILITY_FRAME,
    PersonDetection,
    ProjectionResult,
    ProjectionStatus,
    WorldFloorObservation,
)

Array = NDArray[Any]
NATIVE_RESOLUTION_XY = (1920, 1080)
PROCESSED_RESOLUTION_XY = (504, 280)


class P09ProjectionError(ValueError):
    """Raised when frozen projection inputs or image conventions are invalid."""


@dataclass(frozen=True, slots=True)
class FloorRectangle:
    minimum_xy_metres: tuple[float, float]
    maximum_xy_metres: tuple[float, float]
    z_metres: float = 0.0
    frame_id: str = FACILITY_FRAME

    def __post_init__(self) -> None:
        values = self.minimum_xy_metres + self.maximum_xy_metres + (self.z_metres,)
        if not all(math.isfinite(value) for value in values):
            raise P09ProjectionError("floor rectangle values must be finite")
        if self.frame_id != FACILITY_FRAME or self.z_metres != 0.0:
            raise P09ProjectionError("P09 requires the accepted facility-frame Z=0 floor")
        if not all(
            high > low
            for low, high in zip(self.minimum_xy_metres, self.maximum_xy_metres, strict=True)
        ):
            raise P09ProjectionError("floor rectangle must have positive XY area")

    def contains(self, xy_metres: tuple[float, float]) -> bool:
        return all(
            low <= value <= high
            for value, low, high in zip(
                xy_metres,
                self.minimum_xy_metres,
                self.maximum_xy_metres,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class CameraProjectionCalibration:
    """Frozen native rectification, processed pinhole and working camera transform."""

    camera_id: str
    K_native: Array
    simple_radial_k1: float
    K_processed: Array
    T_world_from_camera: Array
    native_resolution_xy: tuple[int, int] = NATIVE_RESOLUTION_XY
    processed_resolution_xy: tuple[int, int] = PROCESSED_RESOLUTION_XY
    processed_frame_id: str = CALIBRATED_FRAME

    def __post_init__(self) -> None:
        if self.camera_id not in CAMERA_IDS:
            raise P09ProjectionError("camera is absent from the P09 roster")
        if self.native_resolution_xy != NATIVE_RESOLUTION_XY:
            raise P09ProjectionError("native stream must remain 1920x1080")
        if self.processed_resolution_xy != PROCESSED_RESOLUTION_XY:
            raise P09ProjectionError("processed convention must remain 504x280")
        if self.processed_frame_id != CALIBRATED_FRAME:
            raise P09ProjectionError("processed frame convention changed")
        if not math.isfinite(self.simple_radial_k1):
            raise P09ProjectionError("simple-radial coefficient must be finite")
        K_native = _immutable_matrix(self.K_native, (3, 3), "native intrinsic")
        K_processed = _immutable_matrix(self.K_processed, (3, 3), "processed intrinsic")
        transform = _immutable_matrix(self.T_world_from_camera, (4, 4), "T_world_from_camera")
        for intrinsic in (K_native, K_processed):
            if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or intrinsic[2, 2] != 1:
                raise P09ProjectionError(
                    "intrinsic focal lengths and homogeneous scale are invalid"
                )
        rotation = transform[:3, :3]
        if not np.array_equal(transform[3], [0.0, 0.0, 0.0, 1.0]):
            raise P09ProjectionError("T_world_from_camera has an invalid homogeneous row")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0):
            raise P09ProjectionError("T_world_from_camera rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0):
            raise P09ProjectionError("T_world_from_camera rotation is not proper")
        object.__setattr__(self, "K_native", K_native)
        object.__setattr__(self, "K_processed", K_processed)
        object.__setattr__(self, "T_world_from_camera", transform)


class LiveFrameRectifier:
    """Precomputed native simple-radial undistortion followed by exact 504x280 resize."""

    def __init__(self, calibration: CameraProjectionCalibration) -> None:
        self.calibration = calibration
        coefficients = np.asarray(
            [calibration.simple_radial_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            calibration.K_native,
            coefficients,
            np.eye(3, dtype=np.float64),
            calibration.K_native,
            calibration.native_resolution_xy,
            cv2.CV_32FC1,
        )

    def rectify(self, native_frame_bgr: Array) -> Array:
        frame = np.asarray(native_frame_bgr)
        expected_shape = (
            self.calibration.native_resolution_xy[1],
            self.calibration.native_resolution_xy[0],
            3,
        )
        if frame.dtype != np.uint8 or frame.shape != expected_shape:
            raise P09ProjectionError("native frame must be uint8 1920x1080 BGR")
        pinhole = cv2.remap(
            frame,
            self._map_x,
            self._map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        processed = cv2.resize(
            pinhole,
            self.calibration.processed_resolution_xy,
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(processed)


def project_detection_to_floor(
    detection: PersonDetection,
    calibration: CameraProjectionCalibration,
    floor: FloorRectangle,
    evaluated_monotonic_ns: int,
    parallel_epsilon: float = 1e-8,
) -> ProjectionResult:
    """Intersect the calibrated pixel bearing with the accepted mathematical floor."""

    if detection.frame.camera_id != calibration.camera_id:
        raise P09ProjectionError("detection and calibration camera IDs disagree")
    if (detection.frame.width_pixels, detection.frame.height_pixels) != (
        calibration.processed_resolution_xy
    ):
        return ProjectionResult(
            ProjectionStatus.INVALID_PIXEL,
            "detection frame is not the frozen calibrated 504x280 convention",
            None,
        )
    if evaluated_monotonic_ns < detection.frame.acquisition_monotonic_ns:
        raise P09ProjectionError("evaluation time cannot precede frame acquisition")
    if not math.isfinite(parallel_epsilon) or parallel_epsilon <= 0:
        raise P09ProjectionError("parallel epsilon must be finite and positive")

    u, v = detection.image_point_uv
    homogeneous = np.asarray([u, v, 1.0], dtype=np.float64)
    bearing_camera = np.linalg.solve(calibration.K_processed, homogeneous)
    norm = float(np.linalg.norm(bearing_camera))
    if not math.isfinite(norm) or norm <= 0:
        return ProjectionResult(ProjectionStatus.INVALID_PIXEL, "invalid camera bearing", None)
    bearing_camera /= norm
    rotation = calibration.T_world_from_camera[:3, :3]
    origin_world = calibration.T_world_from_camera[:3, 3]
    bearing_world = rotation @ bearing_camera
    z_direction = float(bearing_world[2])
    if abs(z_direction) <= parallel_epsilon:
        return ProjectionResult(
            ProjectionStatus.PARALLEL_RAY,
            "camera bearing is parallel or near-parallel to the floor",
            None,
        )
    ray_parameter = float((floor.z_metres - origin_world[2]) / z_direction)
    if ray_parameter <= 0:
        return ProjectionResult(
            ProjectionStatus.BEHIND_CAMERA,
            "floor intersection lies behind the camera",
            None,
        )
    point_world = origin_world + ray_parameter * bearing_world
    xy_metres = (float(point_world[0]), float(point_world[1]))
    if not floor.contains(xy_metres):
        return ProjectionResult(
            ProjectionStatus.OUTSIDE_FLOOR,
            "floor intersection is outside the accepted P08 rectangle",
            None,
        )
    observation = WorldFloorObservation(
        detection=detection,
        xy_metres=xy_metres,
        ray_parameter_metres=ray_parameter,
        ray_floor_incidence=abs(z_direction),
        frame_age_ms=(evaluated_monotonic_ns - detection.frame.acquisition_monotonic_ns)
        / 1_000_000.0,
    )
    return ProjectionResult(
        ProjectionStatus.VALID,
        "forward ray intersects accepted floor",
        observation,
    )


@dataclass(frozen=True, slots=True)
class FrozenProjectionInputs:
    p06_input_manifest: Path
    p06_input_manifest_sha256: str
    p07_frustum_manifest: Path
    p07_frustum_manifest_sha256: str
    p08_floor_manifest: Path
    p08_floor_manifest_sha256: str
    p08_floor_plane: Path
    p08_floor_plane_sha256: str


def load_frozen_projection_inputs(
    inputs: FrozenProjectionInputs,
) -> tuple[dict[str, CameraProjectionCalibration], FloorRectangle]:
    """Load and cross-check exact P06/P07/P08 inputs without changing authority."""

    for path, expected_hash in (
        (inputs.p06_input_manifest, inputs.p06_input_manifest_sha256),
        (inputs.p07_frustum_manifest, inputs.p07_frustum_manifest_sha256),
        (inputs.p08_floor_manifest, inputs.p08_floor_manifest_sha256),
        (inputs.p08_floor_plane, inputs.p08_floor_plane_sha256),
    ):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_hash:
            raise P09ProjectionError(f"frozen input identity changed: {path.name}")
    p06 = _json_object(inputs.p06_input_manifest)
    p07 = _json_object(inputs.p07_frustum_manifest)
    p08 = _json_object(inputs.p08_floor_manifest)
    camera_inputs = _camera_records(p06, "cameras")
    camera_context = _camera_records(p07, "cameras")
    if tuple(camera_inputs) != CAMERA_IDS or tuple(camera_context) != CAMERA_IDS:
        raise P09ProjectionError("frozen camera order must remain Camera 1/2/3/4")

    calibrations: dict[str, CameraProjectionCalibration] = {}
    for camera_id in CAMERA_IDS:
        native = camera_inputs[camera_id]
        context = camera_context[camera_id]
        intrinsic = _mapping(native, "intrinsics")
        distortion = intrinsic.get("distortion")
        if not isinstance(distortion, list) or len(distortion) != 1:
            raise P09ProjectionError("P06 simple-radial distortion record is malformed")
        calibrations[camera_id] = CameraProjectionCalibration(
            camera_id=camera_id,
            K_native=np.asarray(intrinsic.get("K_pinhole"), dtype=np.float64),
            simple_radial_k1=float(distortion[0]),
            K_processed=np.asarray(context.get("processed_intrinsics"), dtype=np.float64),
            T_world_from_camera=np.asarray(context.get("T_world_from_camera"), dtype=np.float64),
        )

    summary = _mapping(p08, "summary")
    if summary.get("floor_plane_z_metres") != 0.0 or not p08.get("floor_surface_authoritative"):
        raise P09ProjectionError("P08 floor authority is not the selected exact Z=0 plane")
    bounds = summary.get("floor_plane_bounds_xy_metres")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise P09ProjectionError("P08 floor bounds are malformed")
    with np.load(inputs.p08_floor_plane, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices_xyz_metres"], dtype=np.float64)
    if vertices.shape != (4, 3) or not np.all(vertices[:, 2] == 0.0):
        raise P09ProjectionError("stored P08 floor plane is not the exact four-vertex Z=0 plane")
    minimum_xy = (float(bounds[0][0]), float(bounds[0][1]))
    maximum_xy = (float(bounds[1][0]), float(bounds[1][1]))
    floor = FloorRectangle(minimum_xy, maximum_xy)
    expected_corners = {
        (floor.minimum_xy_metres[0], floor.minimum_xy_metres[1]),
        (floor.minimum_xy_metres[0], floor.maximum_xy_metres[1]),
        (floor.maximum_xy_metres[0], floor.minimum_xy_metres[1]),
        (floor.maximum_xy_metres[0], floor.maximum_xy_metres[1]),
    }
    if {(float(row[0]), float(row[1])) for row in vertices} != expected_corners:
        raise P09ProjectionError("P08 floor manifest and stored plane bounds disagree")
    return calibrations, floor


def _immutable_matrix(value: Array, shape: tuple[int, int], label: str) -> Array:
    matrix = np.asarray(value, dtype=np.float64).copy()
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise P09ProjectionError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
    matrix.setflags(write=False)
    return matrix


def _json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise P09ProjectionError(f"{path.name} must contain a JSON object")
    return loaded


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, dict):
        raise P09ProjectionError(f"{key} record is malformed")
    return selected


def _camera_records(value: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    records = value.get(key)
    if not isinstance(records, list):
        raise P09ProjectionError(f"{key} camera list is malformed")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("camera_id"), str):
            raise P09ProjectionError(f"{key} camera record is malformed")
        result[str(record["camera_id"])] = record
    if len(result) != len(records):
        raise P09ProjectionError(f"{key} camera IDs are duplicated")
    return result
