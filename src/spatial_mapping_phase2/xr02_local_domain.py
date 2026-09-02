"""Typed, scene-scoped contracts for XR02 camera-local tracking evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum

from spatial_mapping_phase2.p09_tracking_domain import FACILITY_FRAME

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class XR02ContractError(ValueError):
    """Raised when XR02 local evidence violates its public contract."""


class CameraAvailability(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    MISSING = "missing"
    FAILED = "failed"


class FootpointSource(StrEnum):
    MASK_BOTTOM = "mask_bottom"
    BBOX_BOTTOM_CENTER = "bbox_bottom_center"
    TORSO_PROXY = "torso_proxy"


class EmbeddingStatus(StrEnum):
    AVAILABLE = "available"
    NOT_DUE = "not_due"
    LOW_QUALITY = "low_quality"
    MODEL_UNAVAILABLE = "model_unavailable"
    EXTRACTION_FAILED = "extraction_failed"


class WorldProjectionStatus(StrEnum):
    VALID = "valid"
    INVALID_PIXEL = "invalid_pixel"
    PARALLEL_RAY = "parallel_ray"
    BEHIND_CAMERA = "behind_camera"
    OUTSIDE_FLOOR = "outside_floor"


@dataclass(frozen=True, slots=True)
class SceneContextKey:
    """Immutable identity of one scene epoch and its accepted spatial authority."""

    scene_id: str
    scene_epoch_id: str
    geometry_sha256: str
    floor_sha256: str
    calibration_sha256: str
    facility_frame: str = FACILITY_FRAME
    camera_policy_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.scene_id, "scene_id")
        _require_identifier(self.scene_epoch_id, "scene_epoch_id")
        for value, label in (
            (self.geometry_sha256, "geometry_sha256"),
            (self.floor_sha256, "floor_sha256"),
            (self.calibration_sha256, "calibration_sha256"),
        ):
            if not _SHA256.fullmatch(value):
                raise XR02ContractError(f"{label} must be a lowercase SHA-256")
        if self.camera_policy_sha256 is not None and not _SHA256.fullmatch(
            self.camera_policy_sha256
        ):
            raise XR02ContractError("camera_policy_sha256 must be a lowercase SHA-256")
        if self.facility_frame != FACILITY_FRAME:
            raise XR02ContractError("XR02 WP2 must preserve the accepted facility frame")

    @property
    def context_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "scene_id": self.scene_id,
            "scene_epoch_id": self.scene_epoch_id,
            "geometry_sha256": self.geometry_sha256,
            "floor_sha256": self.floor_sha256,
            "calibration_sha256": self.calibration_sha256,
            "facility_frame": self.facility_frame,
        }
        if self.camera_policy_sha256 is not None:
            value["camera_policy_sha256"] = self.camera_policy_sha256
        return value


@dataclass(frozen=True, slots=True)
class FrameKey:
    """One frame identity without assuming a four-camera deployment."""

    scene: SceneContextKey
    camera_id: str
    frame_id: str
    frame_sequence: int
    acquisition_monotonic_ns: int
    observed_at_utc: str
    width_pixels: int
    height_pixels: int

    def __post_init__(self) -> None:
        _require_identifier(self.camera_id, "camera_id")
        _require_identifier(self.frame_id, "frame_id")
        if self.frame_sequence < 0 or self.acquisition_monotonic_ns < 0:
            raise XR02ContractError("frame sequence and acquisition time must be non-negative")
        if not self.observed_at_utc:
            raise XR02ContractError("observed_at_utc is required")
        if self.width_pixels <= 0 or self.height_pixels <= 0:
            raise XR02ContractError("frame dimensions must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_context_sha256": self.scene.context_sha256,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "frame_sequence": self.frame_sequence,
            "acquisition_monotonic_ns": self.acquisition_monotonic_ns,
            "observed_at_utc": self.observed_at_utc,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
        }


@dataclass(frozen=True, slots=True)
class LocalTrackKey:
    """A camera-local identity namespaced to prevent accidental global-ID use."""

    scene_context_sha256: str
    camera_id: str
    tracker_profile: str
    local_track_id: int

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.scene_context_sha256):
            raise XR02ContractError("scene context identity must be a lowercase SHA-256")
        _require_identifier(self.camera_id, "camera_id")
        _require_identifier(self.tracker_profile, "tracker_profile")
        if self.local_track_id < 0:
            raise XR02ContractError("local track ID must be non-negative")

    @property
    def stable_id(self) -> str:
        return (
            f"{self.scene_context_sha256[:12]}:{self.camera_id}:"
            f"{self.tracker_profile}:{self.local_track_id}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_context_sha256": self.scene_context_sha256,
            "camera_id": self.camera_id,
            "tracker_profile": self.tracker_profile,
            "local_track_id": self.local_track_id,
            "stable_id": self.stable_id,
        }


@dataclass(frozen=True, slots=True)
class CropQuality:
    visible_fraction: float
    area_pixels: float
    aspect_ratio: float
    clipped_sides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (self.visible_fraction, self.area_pixels, self.aspect_ratio)
        if not all(math.isfinite(value) for value in values):
            raise XR02ContractError("crop quality values must be finite")
        if not 0.0 <= self.visible_fraction <= 1.0:
            raise XR02ContractError("visible fraction must be within [0, 1]")
        if self.area_pixels <= 0 or self.aspect_ratio <= 0:
            raise XR02ContractError("crop area and aspect ratio must be positive")
        if len(set(self.clipped_sides)) != len(self.clipped_sides):
            raise XR02ContractError("clipped crop sides must be unique")

    def as_dict(self) -> dict[str, object]:
        return {
            "visible_fraction": self.visible_fraction,
            "area_pixels": self.area_pixels,
            "aspect_ratio": self.aspect_ratio,
            "clipped_sides": list(self.clipped_sides),
        }


@dataclass(frozen=True, slots=True)
class EmbeddingReference:
    sha256: str
    model_id: str
    dimension: int
    relative_path: str

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.sha256):
            raise XR02ContractError("embedding reference requires a lowercase SHA-256")
        _require_identifier(self.model_id, "model_id")
        if self.dimension <= 0:
            raise XR02ContractError("embedding dimension must be positive")
        if (
            not self.relative_path
            or "\\" in self.relative_path
            or self.relative_path.startswith("/")
        ):
            raise XR02ContractError("embedding path must be a relative POSIX path")

    def as_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True, slots=True)
class LocalTrackObservation:
    """One append-only camera-local observation; never a global identity decision."""

    frame: FrameKey
    track: LocalTrackKey
    detection_index: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    footpoint_uv: tuple[float, float]
    footpoint_source: FootpointSource
    crop_quality: CropQuality
    embedding_status: EmbeddingStatus
    embedding: EmbeddingReference | None
    projection_status: WorldProjectionStatus
    world_xy_metres: tuple[float, float] | None
    projection_reason: str

    def __post_init__(self) -> None:
        if self.track.scene_context_sha256 != self.frame.scene.context_sha256:
            raise XR02ContractError("track and frame scene contexts disagree")
        if self.track.camera_id != self.frame.camera_id:
            raise XR02ContractError("track and frame cameras disagree")
        if self.detection_index < 0:
            raise XR02ContractError("detection index must be non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise XR02ContractError("confidence must be within [0, 1]")
        if not all(math.isfinite(value) for value in self.bbox_xyxy + self.footpoint_uv):
            raise XR02ContractError("observation coordinates must be finite")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise XR02ContractError("observation box must have positive area")
        u, v = self.footpoint_uv
        if not 0 <= u < self.frame.width_pixels or not 0 <= v < self.frame.height_pixels:
            raise XR02ContractError("footpoint must lie inside its frame")
        if (self.embedding_status is EmbeddingStatus.AVAILABLE) != (self.embedding is not None):
            raise XR02ContractError("only available embeddings may carry a reference")
        valid_projection = self.projection_status is WorldProjectionStatus.VALID
        if valid_projection != (self.world_xy_metres is not None):
            raise XR02ContractError("only valid projections may contain world XY")
        if self.world_xy_metres is not None and not all(
            math.isfinite(value) for value in self.world_xy_metres
        ):
            raise XR02ContractError("world XY must be finite")
        if not self.projection_reason:
            raise XR02ContractError("projection reason is required")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "xr02.local_track_observation.v1",
            "frame": self.frame.as_dict(),
            "track": self.track.as_dict(),
            "detection_index": self.detection_index,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "footpoint_uv": list(self.footpoint_uv),
            "footpoint_source": self.footpoint_source.value,
            "crop_quality": self.crop_quality.as_dict(),
            "embedding_status": self.embedding_status.value,
            "embedding": None if self.embedding is None else self.embedding.as_dict(),
            "projection_status": self.projection_status.value,
            "world_xy_metres": (
                None if self.world_xy_metres is None else list(self.world_xy_metres)
            ),
            "projection_reason": self.projection_reason,
        }


def _require_identifier(value: str, label: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise XR02ContractError(f"{label} is invalid")


def _canonical_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
