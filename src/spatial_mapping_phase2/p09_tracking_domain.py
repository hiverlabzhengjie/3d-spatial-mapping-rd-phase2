"""Typed observational contracts for the bounded P09 anonymous tracker."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

CAMERA_IDS = tuple(f"office-cam-0{index}" for index in range(1, 5))
FACILITY_FRAME = "facility-world-x-plan-left-y-plan-down-z-up"
CALIBRATED_FRAME = "p09-pinhole-upper-bound-resize-504x280"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class P09ContractError(ValueError):
    """Raised when P09 observational evidence violates its frozen contract."""


class FootpointKind(StrEnum):
    MASK_BOTTOM = "mask_bottom"
    BBOX_BOTTOM_CENTER = "bbox_bottom_center"
    TORSO_PROXY = "torso_proxy"


class ProjectionStatus(StrEnum):
    VALID = "valid"
    INVALID_PIXEL = "invalid_pixel"
    PARALLEL_RAY = "parallel_ray"
    BEHIND_CAMERA = "behind_camera"
    OUTSIDE_FLOOR = "outside_floor"


class TrackingState(StrEnum):
    TRACKED_FUSED = "tracked_fused"
    TRACKED_SINGLE_CAMERA = "tracked_single_camera"
    AMBIGUOUS = "ambiguous"
    MULTI_PERSON_UNSUPPORTED = "multi_person_unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, repr=False)
class LiveFrameIdentity:
    """One decoded frame in the P03 host-monotonic acquisition domain."""

    camera_id: str
    frame_id: str
    acquisition_monotonic_ns: int
    observed_at_utc: str
    source_pts: int | None
    source_time_base: str | None
    width_pixels: int
    height_pixels: int

    def __post_init__(self) -> None:
        if self.camera_id not in CAMERA_IDS:
            raise P09ContractError("frame camera_id is not in the P09 roster")
        if not _IDENTIFIER.fullmatch(self.frame_id):
            raise P09ContractError("frame_id is invalid")
        if self.acquisition_monotonic_ns < 0:
            raise P09ContractError("frame acquisition time must be non-negative")
        if not self.observed_at_utc:
            raise P09ContractError("frame UTC observation time is required")
        if (self.source_pts is None) != (self.source_time_base is None):
            raise P09ContractError("source PTS and time base must be present together")
        if self.width_pixels <= 0 or self.height_pixels <= 0:
            raise P09ContractError("frame dimensions must be positive")


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """One current person detection in the calibrated 504x280 image convention."""

    frame: LiveFrameIdentity
    detection_index: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    image_point_uv: tuple[float, float]
    footpoint_kind: FootpointKind
    clipped_at_image_bottom: bool = False

    def __post_init__(self) -> None:
        if self.detection_index < 0:
            raise P09ContractError("detection_index must be non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise P09ContractError("detection confidence must be within [0, 1]")
        if not all(math.isfinite(value) for value in self.bbox_xyxy + self.image_point_uv):
            raise P09ContractError("detection coordinates must be finite")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise P09ContractError("detection bounding box must have positive area")
        width, height = self.frame.width_pixels, self.frame.height_pixels
        u, v = self.image_point_uv
        if not 0.0 <= u < width or not 0.0 <= v < height:
            raise P09ContractError("detection image point is outside its frame")
        if self.footpoint_kind is FootpointKind.TORSO_PROXY and not self.clipped_at_image_bottom:
            raise P09ContractError("torso_proxy must retain explicit bottom-clipping evidence")


@dataclass(frozen=True, slots=True)
class WorldFloorObservation:
    """One valid current camera-ray intersection with the accepted P08 floor."""

    detection: PersonDetection
    xy_metres: tuple[float, float]
    ray_parameter_metres: float
    ray_floor_incidence: float
    frame_age_ms: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.xy_metres):
            raise P09ContractError("world XY must be finite")
        if not math.isfinite(self.ray_parameter_metres) or self.ray_parameter_metres <= 0:
            raise P09ContractError("world observation must lie on a forward ray")
        if not math.isfinite(self.ray_floor_incidence) or not 0 < self.ray_floor_incidence <= 1:
            raise P09ContractError("ray-floor incidence must be within (0, 1]")
        if not math.isfinite(self.frame_age_ms) or self.frame_age_ms < 0:
            raise P09ContractError("frame age must be finite and non-negative")

    @property
    def camera_id(self) -> str:
        return self.detection.frame.camera_id

    @property
    def acquisition_monotonic_ns(self) -> int:
        return self.detection.frame.acquisition_monotonic_ns

    @property
    def quality_weight(self) -> float:
        """Return a transparent bounded fusion weight, never an accuracy probability."""

        kind_weight = {
            FootpointKind.MASK_BOTTOM: 1.0,
            FootpointKind.BBOX_BOTTOM_CENTER: 0.75,
            FootpointKind.TORSO_PROXY: 0.30,
        }[self.detection.footpoint_kind]
        age_weight = math.exp(-self.frame_age_ms / 500.0)
        return max(
            1e-6,
            self.detection.confidence * kind_weight * self.ray_floor_incidence * age_weight,
        )


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    status: ProjectionStatus
    reason: str
    observation: WorldFloorObservation | None

    def __post_init__(self) -> None:
        if not self.reason:
            raise P09ContractError("projection result requires a reason")
        if (self.status is ProjectionStatus.VALID) != (self.observation is not None):
            raise P09ContractError("only valid projections may contain an observation")


@dataclass(frozen=True, slots=True)
class TrackingEstimate:
    """One current anonymous tracker state; last-known is always separate."""

    state: TrackingState
    evaluated_monotonic_ns: int
    current_xy_metres: tuple[float, float] | None
    contributing_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    reason: str
    last_known_xy_metres: tuple[float, float] | None = None
    last_known_age_ms: float | None = None

    def __post_init__(self) -> None:
        if self.evaluated_monotonic_ns < 0 or not self.reason:
            raise P09ContractError("tracking estimate time and reason are required")
        current_state = self.state in {
            TrackingState.TRACKED_FUSED,
            TrackingState.TRACKED_SINGLE_CAMERA,
        }
        if current_state != (self.current_xy_metres is not None):
            raise P09ContractError("only tracked states may contain a current XY")
        if self.state is TrackingState.TRACKED_FUSED and len(self.contributing_camera_ids) < 2:
            raise P09ContractError("tracked_fused requires at least two contributing cameras")
        if (
            self.state is TrackingState.TRACKED_SINGLE_CAMERA
            and len(self.contributing_camera_ids) != 1
        ):
            raise P09ContractError("tracked_single_camera requires exactly one camera")
        if len(set(self.contributing_camera_ids)) != len(self.contributing_camera_ids):
            raise P09ContractError("contributing cameras must be unique")
        if len(set(self.rejected_camera_ids)) != len(self.rejected_camera_ids):
            raise P09ContractError("rejected cameras must be unique")
        if set(self.contributing_camera_ids) & set(self.rejected_camera_ids):
            raise P09ContractError("a camera cannot both contribute and be rejected")
        if (self.last_known_xy_metres is None) != (self.last_known_age_ms is None):
            raise P09ContractError("last-known XY and age must be present together")
        if self.last_known_age_ms is not None and (
            not math.isfinite(self.last_known_age_ms) or self.last_known_age_ms < 0
        ):
            raise P09ContractError("last-known age must be finite and non-negative")
