"""Evidence summaries for P04 multi-frame intrinsic candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev


class IntrinsicEvidenceError(ValueError):
    """Raised when intrinsic evidence is malformed or not comparable."""


@dataclass(frozen=True, slots=True)
class IntrinsicStabilitySummary:
    frame_count: int
    focal_mean_pixels: float
    focal_std_pixels: float
    focal_range_pixels: float
    focal_cv: float
    shared_focal_pixels: float
    max_gravity_separation_degrees: float
    distortion_mean: tuple[float, ...]
    distortion_std: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "focal_mean_pixels": self.focal_mean_pixels,
            "focal_std_pixels": self.focal_std_pixels,
            "focal_range_pixels": self.focal_range_pixels,
            "focal_cv": self.focal_cv,
            "shared_focal_pixels": self.shared_focal_pixels,
            "max_gravity_separation_degrees": self.max_gravity_separation_degrees,
            "distortion_mean": list(self.distortion_mean),
            "distortion_std": list(self.distortion_std),
        }


def summarize_intrinsic_candidate(
    individual_camera_parameters: Sequence[Sequence[float]],
    shared_camera_parameters: Sequence[Sequence[float]],
    gravity_vectors: Sequence[Sequence[float]],
    distortion_parameter_count: int | None = None,
) -> IntrinsicStabilitySummary:
    """Summarize independent-frame spread and one shared-intrinsic estimate.

    Camera rows follow GeoCalib's explicit order
    ``width, height, fx, fy, cx, cy, k1, k2``. Gravity is expressed in each
    Camera 3 optical frame; the fixed camera assumption makes cross-frame
    angular separation a useful stability diagnostic.
    """

    count = len(individual_camera_parameters)
    if count < 2:
        raise IntrinsicEvidenceError("at least two frames are required")
    if len(shared_camera_parameters) != count or len(gravity_vectors) != count:
        raise IntrinsicEvidenceError("camera and gravity evidence must have equal frame counts")
    individual = tuple(_camera_row(row) for row in individual_camera_parameters)
    shared = tuple(_camera_row(row) for row in shared_camera_parameters)
    gravity = tuple(_unit_vector(row) for row in gravity_vectors)
    sizes = {(row[0], row[1]) for row in individual + shared}
    if len(sizes) != 1:
        raise IntrinsicEvidenceError("all intrinsic frames must use one image size")
    focals = tuple((row[2] + row[3]) / 2.0 for row in individual)
    focal_mean = fmean(focals)
    shared_focals = tuple((row[2] + row[3]) / 2.0 for row in shared)
    if max(shared_focals) - min(shared_focals) > 1e-3:
        raise IntrinsicEvidenceError("shared-intrinsic result does not share one focal value")
    distortion_count = (
        _distortion_parameter_count(individual)
        if distortion_parameter_count is None
        else distortion_parameter_count
    )
    if distortion_count not in {0, 1, 2}:
        raise IntrinsicEvidenceError("distortion parameter count must be zero, one or two")
    distortion_columns = tuple(
        tuple(row[6 + index] for row in individual) for index in range(distortion_count)
    )
    max_gravity = max(
        _vector_angle_degrees(gravity[left], gravity[right])
        for left in range(count)
        for right in range(left + 1, count)
    )
    return IntrinsicStabilitySummary(
        count,
        focal_mean,
        pstdev(focals),
        max(focals) - min(focals),
        pstdev(focals) / focal_mean,
        fmean(shared_focals),
        max_gravity,
        tuple(fmean(column) for column in distortion_columns),
        tuple(pstdev(column) for column in distortion_columns),
    )


def looks_like_geocalib_initialization(
    camera_rows: Sequence[Sequence[float]], gravity_rows: Sequence[Sequence[float]]
) -> bool:
    """Detect GeoCalib's exact default values after a stalled optimizer."""

    if not camera_rows or len(camera_rows) != len(gravity_rows):
        raise IntrinsicEvidenceError("stalled-result check requires equal non-empty evidence")
    cameras = tuple(_camera_row(row) for row in camera_rows)
    gravity = tuple(_unit_vector(row) for row in gravity_rows)
    focals = tuple((row[2] + row[3]) / 2.0 for row in cameras)
    return max(focals) - min(focals) < 1e-6 and all(
        abs(vector[0]) < 1e-8
        and abs(vector[1] + 1.0) < 1e-8
        and abs(vector[2]) < 1e-8
        for vector in gravity
    )


def _camera_row(values: Sequence[float]) -> tuple[float, ...]:
    row = tuple(float(value) for value in values)
    if len(row) != 8 or not all(math.isfinite(value) for value in row):
        raise IntrinsicEvidenceError("GeoCalib camera rows must contain eight finite values")
    if row[0] <= 0 or row[1] <= 0 or row[2] <= 0 or row[3] <= 0:
        raise IntrinsicEvidenceError("image dimensions and focal lengths must be positive")
    return row


def _unit_vector(values: Sequence[float]) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise IntrinsicEvidenceError("gravity vectors must contain three finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        raise IntrinsicEvidenceError("gravity vectors must be non-zero")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _vector_angle_degrees(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))
    return math.degrees(math.acos(dot))


def _distortion_parameter_count(rows: Sequence[tuple[float, ...]]) -> int:
    if any(abs(row[7]) > 1e-12 for row in rows):
        return 2
    if any(abs(row[6]) > 1e-12 for row in rows):
        return 1
    return 0
