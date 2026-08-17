"""Equal-camera robust summaries for the D027 intrinsic-fleet study."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, median, pstdev


class IntrinsicFleetError(ValueError):
    """Raised when intrinsic estimates cannot be pooled without hidden assumptions."""


@dataclass(frozen=True, slots=True)
class CameraIntrinsicEstimate:
    camera_id: str
    profile_version: str
    model: str
    width_pixels: int
    height_pixels: int
    fx_pixels: float
    fy_pixels: float
    cx_pixels: float
    cy_pixels: float
    distortion: tuple[float, ...]
    within_camera_focal_cv: float

    def __post_init__(self) -> None:
        if (
            not self.camera_id.strip()
            or not self.profile_version.strip()
            or not self.model.strip()
        ):
            raise IntrinsicFleetError("camera, profile and model identities must be non-blank")
        numeric = (
            self.width_pixels,
            self.height_pixels,
            self.fx_pixels,
            self.fy_pixels,
            self.cx_pixels,
            self.cy_pixels,
            self.within_camera_focal_cv,
            *self.distortion,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise IntrinsicFleetError("intrinsic estimates must be finite")
        if min(self.width_pixels, self.height_pixels, self.fx_pixels, self.fy_pixels) <= 0:
            raise IntrinsicFleetError("image dimensions and focal lengths must be positive")
        if self.within_camera_focal_cv < 0:
            raise IntrinsicFleetError("within-camera focal CV cannot be negative")

    @property
    def focal_pixels(self) -> float:
        return (self.fx_pixels + self.fy_pixels) / 2.0

    def normalized_parameters(self) -> tuple[float, ...]:
        return (
            self.fx_pixels / self.width_pixels,
            self.fy_pixels / self.height_pixels,
            self.cx_pixels / self.width_pixels,
            self.cy_pixels / self.height_pixels,
            *self.distortion,
        )


@dataclass(frozen=True, slots=True)
class FleetIntrinsicProfile:
    method: str
    included_camera_ids: tuple[str, ...]
    profile_version: str
    model: str
    width_pixels: int
    height_pixels: int
    fx_pixels: float
    fy_pixels: float
    cx_pixels: float
    cy_pixels: float
    distortion: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "included_camera_ids": list(self.included_camera_ids),
            "profile_version": self.profile_version,
            "model": self.model,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "fx_pixels": self.fx_pixels,
            "fy_pixels": self.fy_pixels,
            "cx_pixels": self.cx_pixels,
            "cy_pixels": self.cy_pixels,
            "distortion": list(self.distortion),
        }


def build_fleet_profiles(
    estimates: Sequence[CameraIntrinsicEstimate], *, exclude_camera_id: str
) -> tuple[FleetIntrinsicProfile, ...]:
    """Build predetermined mean, median and Huber profiles with one vote per camera."""

    retained = tuple(item for item in estimates if item.camera_id != exclude_camera_id)
    if len(retained) < 3:
        raise IntrinsicFleetError("at least three cameras must remain after exclusion")
    if len({item.camera_id for item in retained}) != len(retained):
        raise IntrinsicFleetError("each included camera must contribute exactly one estimate")
    profile, model, width, height, parameter_count = _compatible(retained)
    rows = tuple(item.normalized_parameters() for item in retained)
    methods = (
        ("equal-camera-arithmetic-mean", fmean),
        ("equal-camera-componentwise-median", median),
        ("equal-camera-componentwise-huber", huber_location),
    )
    output: list[FleetIntrinsicProfile] = []
    for method, estimator in methods:
        pooled = tuple(
            estimator(tuple(row[index] for row in rows)) for index in range(parameter_count)
        )
        output.append(
            FleetIntrinsicProfile(
                method,
                tuple(sorted(item.camera_id for item in retained)),
                profile,
                model,
                width,
                height,
                pooled[0] * width,
                pooled[1] * height,
                pooled[2] * width,
                pooled[3] * height,
                pooled[4:],
            )
        )
    return tuple(output)


def summarize_between_camera_variation(
    estimates: Sequence[CameraIntrinsicEstimate], *, exclude_camera_id: str | None = None
) -> dict[str, object]:
    retained = tuple(
        item
        for item in estimates
        if exclude_camera_id is None or item.camera_id != exclude_camera_id
    )
    if len(retained) < 2:
        raise IntrinsicFleetError("at least two cameras are required for variation")
    _compatible(retained)
    focals = tuple(item.focal_pixels for item in retained)
    distortion_count = len(retained[0].distortion)
    distortion_columns = tuple(
        tuple(item.distortion[index] for item in retained) for index in range(distortion_count)
    )
    return {
        "camera_ids": [item.camera_id for item in retained],
        "focal_mean_pixels": fmean(focals),
        "focal_std_pixels": pstdev(focals),
        "focal_range_pixels": max(focals) - min(focals),
        "focal_cv": pstdev(focals) / fmean(focals),
        "distortion_mean": [fmean(column) for column in distortion_columns],
        "distortion_std": [pstdev(column) for column in distortion_columns],
        "maximum_within_camera_focal_cv": max(
            item.within_camera_focal_cv for item in retained
        ),
    }


def huber_location(values: Sequence[float], *, tuning: float = 1.345) -> float:
    """Robust one-dimensional location with a MAD scale and equal input weights."""

    data = tuple(float(value) for value in values)
    if not data or not all(math.isfinite(value) for value in data):
        raise IntrinsicFleetError("Huber location requires finite values")
    if tuning <= 0:
        raise IntrinsicFleetError("Huber tuning must be positive")
    location = median(data)
    mad = median(tuple(abs(value - location) for value in data))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return fmean(data)
    for _ in range(100):
        weights = tuple(
            min(1.0, tuning * scale / max(abs(value - location), 1e-15)) for value in data
        )
        updated = sum(weight * value for weight, value in zip(weights, data, strict=True)) / sum(
            weights
        )
        if abs(updated - location) <= 1e-12 * max(1.0, abs(location)):
            return updated
        location = updated
    return location


def _compatible(
    estimates: Sequence[CameraIntrinsicEstimate],
) -> tuple[str, str, int, int, int]:
    profiles = {item.profile_version for item in estimates}
    models = {item.model for item in estimates}
    sizes = {(item.width_pixels, item.height_pixels) for item in estimates}
    distortion_counts = {len(item.distortion) for item in estimates}
    if len(profiles) != 1 or len(models) != 1 or len(sizes) != 1 or len(distortion_counts) != 1:
        raise IntrinsicFleetError(
            "fleet pooling requires one profile, model, image size and parameterization"
        )
    profile = next(iter(profiles))
    model = next(iter(models))
    width, height = next(iter(sizes))
    return profile, model, width, height, 4 + next(iter(distortion_counts))
