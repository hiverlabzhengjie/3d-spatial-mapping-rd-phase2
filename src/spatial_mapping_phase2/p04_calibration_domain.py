"""Typed P04 contracts for linked camera-pixel and facility-world landmarks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

P04_SCHEMA_VERSION = "p04-calibration-workspace-v1"
P04_EXPORT_SCHEMA_VERSION = "p04-linked-correspondence-export-v1"
D034_VALIDATION_SCHEMA_VERSION = "p05-d034-validation-seal-v1"
PILOT_CAMERA_ID = "office-cam-03"
CALIBRATION_CAMERA_IDS = tuple(f"office-cam-0{index}" for index in range(1, 5))
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


class P04CalibrationError(ValueError):
    """Raised when P04 input is incomplete, inconsistent, or unsafe to interpret."""


class FrameReviewStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class LandmarkRole(StrEnum):
    SOLVE = "solve"
    HELD_OUT = "held-out"
    D034_VALIDATION = "d034-validation"


@dataclass(frozen=True, slots=True)
class PixelPoint:
    """A point in a full-resolution image with top-left origin, u right, v down."""

    u: float
    v: float

    def __post_init__(self) -> None:
        _require_non_negative(self.u, "pixel u")
        _require_non_negative(self.v, "pixel v")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], parent: str) -> PixelPoint:
        return cls(_number(payload, "u", parent), _number(payload, "v", parent))

    def to_dict(self) -> dict[str, float]:
        return {"u": self.u, "v": self.v}


@dataclass(frozen=True, slots=True)
class WorldPoint:
    """A point in the right-handed facility frame, expressed in metres."""

    x_metres: float
    y_metres: float
    z_metres: float

    def __post_init__(self) -> None:
        _require_finite(self.x_metres, "world x_metres")
        _require_finite(self.y_metres, "world y_metres")
        _require_non_negative(self.z_metres, "world z_metres")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorldPoint:
        return cls(
            _number(payload, "x_metres", "world_point"),
            _number(payload, "y_metres", "world_point"),
            _number(payload, "z_metres", "world_point"),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x_metres": self.x_metres,
            "y_metres": self.y_metres,
            "z_metres": self.z_metres,
        }


@dataclass(frozen=True, slots=True)
class FacilityReference:
    """Hash-bound P02 reference and display transform used to derive provisional XY."""

    export_sha256: str
    source_revision: int
    plan_source_sha256: str
    plan_image_sha256: str
    plan_image_width_pixels: int
    plan_image_height_pixels: int
    frame_id: str
    world_from_plan_pixel: tuple[tuple[float, float, float], ...]
    authority_note: str

    def __post_init__(self) -> None:
        for identity, field in (
            (self.export_sha256, "facility export SHA-256"),
            (self.plan_source_sha256, "plan source SHA-256"),
            (self.plan_image_sha256, "plan image SHA-256"),
        ):
            _require_sha256(identity, field)
        if self.source_revision < 0:
            raise P04CalibrationError("facility source_revision must be non-negative")
        if self.plan_image_width_pixels <= 0 or self.plan_image_height_pixels <= 0:
            raise P04CalibrationError("plan image dimensions must be positive")
        _require_non_blank(self.frame_id, "facility frame_id")
        _require_non_blank(self.authority_note, "facility authority_note")
        if len(self.world_from_plan_pixel) != 3 or any(
            len(row) != 3 for row in self.world_from_plan_pixel
        ):
            raise P04CalibrationError("world-from-plan-pixel transform must be 3x3")
        for row in self.world_from_plan_pixel:
            for matrix_value in row:
                _require_finite(matrix_value, "world-from-plan-pixel transform value")
        if not math.isclose(self.world_from_plan_pixel[2][2], 1.0, abs_tol=1e-12):
            raise P04CalibrationError("world-from-plan-pixel transform must be affine")

    def world_xy(self, point: PixelPoint) -> tuple[float, float]:
        if point.u > self.plan_image_width_pixels or point.v > self.plan_image_height_pixels:
            raise P04CalibrationError("plan point must lie inside the rendered plan")
        matrix = self.world_from_plan_pixel
        return (
            matrix[0][0] * point.u + matrix[0][1] * point.v + matrix[0][2],
            matrix[1][0] * point.u + matrix[1][1] * point.v + matrix[1][2],
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FacilityReference:
        matrix = _array(payload, "world_from_plan_pixel", "facility_reference")
        rows: list[tuple[float, float, float]] = []
        for index, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != 3:
                raise P04CalibrationError(
                    f"facility_reference.world_from_plan_pixel[{index}] must have 3 numbers"
                )
            rows.append(
                (
                    _finite_value(row[0], "transform value"),
                    _finite_value(row[1], "transform value"),
                    _finite_value(row[2], "transform value"),
                )
            )
        return cls(
            _string(payload, "export_sha256", "facility_reference"),
            _integer(payload, "source_revision", "facility_reference"),
            _string(payload, "plan_source_sha256", "facility_reference"),
            _string(payload, "plan_image_sha256", "facility_reference"),
            _integer(payload, "plan_image_width_pixels", "facility_reference"),
            _integer(payload, "plan_image_height_pixels", "facility_reference"),
            _string(payload, "frame_id", "facility_reference"),
            tuple(rows),
            _string(payload, "authority_note", "facility_reference"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_sha256": self.export_sha256,
            "source_revision": self.source_revision,
            "plan_source_sha256": self.plan_source_sha256,
            "plan_image_sha256": self.plan_image_sha256,
            "plan_image_width_pixels": self.plan_image_width_pixels,
            "plan_image_height_pixels": self.plan_image_height_pixels,
            "frame_id": self.frame_id,
            "world_from_plan_pixel": [list(row) for row in self.world_from_plan_pixel],
            "authority_note": self.authority_note,
        }


@dataclass(frozen=True, slots=True)
class CalibrationFrame:
    """Immutable candidate frame bound to one camera and stream profile."""

    frame_id: str
    camera_id: str
    profile_version: str
    sha256: str
    byte_count: int
    image_width_pixels: int
    image_height_pixels: int
    relative_path: str
    status: FrameReviewStatus
    review_note: str | None
    capture_kind: str = "imported-local"
    observed_at_utc: str | None = None
    source_pts: int | None = None
    source_time_base: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.frame_id, "frame_id")
        _require_calibration_camera_id(self.camera_id)
        _require_non_blank(self.profile_version, "profile_version")
        _require_sha256(self.sha256, "frame SHA-256")
        if self.byte_count <= 0:
            raise P04CalibrationError("frame byte_count must be positive")
        if self.image_width_pixels <= 0 or self.image_height_pixels <= 0:
            raise P04CalibrationError("frame dimensions must be positive")
        _require_non_blank(self.relative_path, "frame relative_path")
        if self.review_note is not None:
            _require_non_blank(self.review_note, "frame review_note")
        if self.capture_kind not in {"imported-local", "user-timed-live"}:
            raise P04CalibrationError("frame capture_kind is unsupported")
        if self.observed_at_utc is not None:
            _require_non_blank(self.observed_at_utc, "frame observed_at_utc")
        if self.source_pts is not None and not isinstance(self.source_pts, int):
            raise P04CalibrationError("frame source_pts must be an integer or null")
        if self.source_time_base is not None:
            _require_non_blank(self.source_time_base, "frame source_time_base")
        if self.source_pts is None and self.source_time_base is not None:
            raise P04CalibrationError("frame source_time_base requires source_pts")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationFrame:
        return cls(
            _string(payload, "frame_id", "frame"),
            _string(payload, "camera_id", "frame"),
            _string(payload, "profile_version", "frame"),
            _string(payload, "sha256", "frame"),
            _integer(payload, "byte_count", "frame"),
            _integer(payload, "image_width_pixels", "frame"),
            _integer(payload, "image_height_pixels", "frame"),
            _string(payload, "relative_path", "frame"),
            FrameReviewStatus(_string(payload, "status", "frame")),
            _optional_string(payload, "review_note", "frame"),
            _optional_string(payload, "capture_kind", "frame") or "imported-local",
            _optional_string(payload, "observed_at_utc", "frame"),
            _optional_integer(payload, "source_pts", "frame"),
            _optional_string(payload, "source_time_base", "frame"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "profile_version": self.profile_version,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "image_width_pixels": self.image_width_pixels,
            "image_height_pixels": self.image_height_pixels,
            "relative_path": self.relative_path,
            "status": self.status.value,
            "review_note": self.review_note,
            "capture_kind": self.capture_kind,
            "observed_at_utc": self.observed_at_utc,
            "source_pts": self.source_pts,
            "source_time_base": self.source_time_base,
        }


@dataclass(frozen=True, slots=True)
class LinkedLandmark:
    """One semantic physical point linked across camera pixels and facility XYZ."""

    landmark_id: str
    name: str
    physical_meaning: str
    frame_id: str
    image_point: PixelPoint
    plan_point: PixelPoint
    world_point: WorldPoint
    z_source: str | None
    z_uncertainty_metres: float | None
    role: LandmarkRole

    def __post_init__(self) -> None:
        _require_id(self.landmark_id, "landmark_id")
        _require_non_blank(self.name, "landmark name")
        _require_non_blank(self.physical_meaning, "landmark physical_meaning")
        _require_id(self.frame_id, "landmark frame_id")
        if self.z_source is not None:
            _require_non_blank(self.z_source, "landmark z_source")
        if self.z_uncertainty_metres is not None:
            _require_non_negative(
                self.z_uncertainty_metres, "landmark z_uncertainty_metres"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinkedLandmark:
        return cls(
            _string(payload, "landmark_id", "landmark"),
            _string(payload, "name", "landmark"),
            _string(payload, "physical_meaning", "landmark"),
            _string(payload, "frame_id", "landmark"),
            PixelPoint.from_dict(_object(payload, "image_point", "landmark"), "image_point"),
            PixelPoint.from_dict(_object(payload, "plan_point", "landmark"), "plan_point"),
            WorldPoint.from_dict(_object(payload, "world_point", "landmark")),
            _optional_string(payload, "z_source", "landmark"),
            _optional_number(payload, "z_uncertainty_metres", "landmark"),
            LandmarkRole(_string(payload, "role", "landmark")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "name": self.name,
            "physical_meaning": self.physical_meaning,
            "frame_id": self.frame_id,
            "image_point": self.image_point.to_dict(),
            "plan_point": self.plan_point.to_dict(),
            "world_point": self.world_point.to_dict(),
            "z_source": self.z_source,
            "z_uncertainty_metres": self.z_uncertainty_metres,
            "role": self.role.value,
        }


@dataclass(frozen=True, slots=True)
class CalibrationWorkspace:
    """Versioned P04 frame-review and linked-landmark workspace."""

    schema_version: str
    revision: int
    camera_id: str
    facility_reference: FacilityReference
    frames: tuple[CalibrationFrame, ...]
    landmarks: tuple[LinkedLandmark, ...]

    def __post_init__(self) -> None:
        if self.schema_version != P04_SCHEMA_VERSION:
            raise P04CalibrationError("unsupported P04 calibration workspace schema")
        if self.revision < 0:
            raise P04CalibrationError("workspace revision must be non-negative")
        _require_calibration_camera_id(self.camera_id)
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise P04CalibrationError("frame IDs must be unique")
        approved = [frame for frame in self.frames if frame.status is FrameReviewStatus.APPROVED]
        if len(approved) > 1:
            raise P04CalibrationError("only one primary pose-annotation frame may be approved")
        landmark_ids = [landmark.landmark_id for landmark in self.landmarks]
        if len(landmark_ids) != len(set(landmark_ids)):
            raise P04CalibrationError("landmark IDs must be unique")
        landmark_names = [landmark.name.casefold() for landmark in self.landmarks]
        if len(landmark_names) != len(set(landmark_names)):
            raise P04CalibrationError("landmark names must be unique")
        frames = {frame.frame_id: frame for frame in self.frames}
        for landmark in self.landmarks:
            frame = frames.get(landmark.frame_id)
            if frame is None:
                raise P04CalibrationError("landmark references an unknown frame")
            if (
                landmark.image_point.u > frame.image_width_pixels
                or landmark.image_point.v > frame.image_height_pixels
            ):
                raise P04CalibrationError("landmark image point must lie inside its frame")
            expected_x, expected_y = self.facility_reference.world_xy(landmark.plan_point)
            if not math.isclose(
                expected_x, landmark.world_point.x_metres, abs_tol=1e-9
            ) or not math.isclose(
                expected_y,
                landmark.world_point.y_metres,
                abs_tol=1e-9,
            ):
                raise P04CalibrationError("landmark world XY must be derived from its plan point")

    @property
    def approved_frame(self) -> CalibrationFrame | None:
        return next(
            (frame for frame in self.frames if frame.status is FrameReviewStatus.APPROVED),
            None,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationWorkspace:
        return cls(
            _string(payload, "schema_version", "workspace"),
            _integer(payload, "revision", "workspace"),
            _string(payload, "camera_id", "workspace"),
            FacilityReference.from_dict(_object(payload, "facility_reference", "workspace")),
            tuple(
                CalibrationFrame.from_dict(_typed_object(item, "frame"))
                for item in _array(payload, "frames", "workspace")
            ),
            tuple(
                LinkedLandmark.from_dict(_typed_object(item, "landmark"))
                for item in _array(payload, "landmarks", "workspace")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "camera_id": self.camera_id,
            "facility_reference": self.facility_reference.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "landmarks": [landmark.to_dict() for landmark in self.landmarks],
        }

    def with_frame_review(
        self, frame_id: str, status: FrameReviewStatus, note: str | None
    ) -> CalibrationWorkspace:
        if status not in {FrameReviewStatus.APPROVED, FrameReviewStatus.REJECTED}:
            raise P04CalibrationError("review status must be approved or rejected")
        if not any(frame.frame_id == frame_id for frame in self.frames):
            raise P04CalibrationError("unknown frame_id")
        frames: list[CalibrationFrame] = []
        for frame in self.frames:
            next_status = frame.status
            next_note = frame.review_note
            if frame.frame_id == frame_id:
                next_status = status
                next_note = note
            elif (
                status is FrameReviewStatus.APPROVED
                and frame.status is FrameReviewStatus.APPROVED
            ):
                next_status = FrameReviewStatus.SUPERSEDED
                next_note = "superseded by a later approved primary frame"
            frames.append(replace(frame, status=next_status, review_note=next_note))
        return replace(self, frames=tuple(frames), revision=self.revision + 1)


def build_p04_export(workspace: CalibrationWorkspace) -> dict[str, Any]:
    """Build the credential-free linked-correspondence snapshot used by later PnP work."""

    approved = workspace.approved_frame
    operational = tuple(
        landmark
        for landmark in workspace.landmarks
        if landmark.role is not LandmarkRole.D034_VALIDATION
    )
    return {
        "schema_version": P04_EXPORT_SCHEMA_VERSION,
        "source_revision": workspace.revision,
        "camera_id": workspace.camera_id,
        "status": "ready-for-pose-input-review" if approved and operational else "draft",
        "approved_frame": None if approved is None else approved.to_dict(),
        "facility_reference": workspace.facility_reference.to_dict(),
        "landmarks": [landmark.to_dict() for landmark in operational],
        "excluded_d034_validation_landmark_ids": [
            landmark.landmark_id
            for landmark in workspace.landmarks
            if landmark.role is LandmarkRole.D034_VALIDATION
        ],
        "role_counts": {
            role.value: sum(landmark.role is role for landmark in operational)
            for role in (LandmarkRole.SOLVE, LandmarkRole.HELD_OUT)
        },
        "authority_note": (
            "world XY is derived from provisional P02 revision 3; Z is operator-entered from "
            "physical evidence; no camera pose or accuracy acceptance is implied"
        ),
    }


def build_d034_validation_seal(workspace: CalibrationWorkspace) -> dict[str, Any]:
    """Export exactly two D034 validation points separately from every solve artifact."""

    approved = workspace.approved_frame
    validation = tuple(
        landmark
        for landmark in workspace.landmarks
        if landmark.role is LandmarkRole.D034_VALIDATION
    )
    if approved is None:
        raise P04CalibrationError("approve a primary frame before sealing D034 validation")
    if len(validation) != 2:
        raise P04CalibrationError("D034 validation seal requires exactly two validation points")
    if any(landmark.frame_id != approved.frame_id for landmark in validation):
        raise P04CalibrationError("D034 validation points must use the approved primary frame")
    return {
        "schema_version": D034_VALIDATION_SCHEMA_VERSION,
        "source_revision": workspace.revision,
        "camera_id": workspace.camera_id,
        "status": "sealed-unconsumed",
        "approved_frame_id": approved.frame_id,
        "approved_frame_sha256": approved.sha256,
        "facility_reference": workspace.facility_reference.to_dict(),
        "validation_landmarks": [landmark.to_dict() for landmark in validation],
        "solve_data_included": False,
        "authority_note": (
            "D034 validation-only seal; do not open until the intrinsic, four-point solve set, "
            "algorithm, thresholds and orientation manifest are frozen"
        ),
    }


def _require_calibration_camera_id(camera_id: str) -> None:
    if camera_id not in CALIBRATION_CAMERA_IDS:
        raise P04CalibrationError("camera_id must be one of office-cam-01 through office-cam-04")


def _typed_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise P04CalibrationError(f"{field_name} must be an object")
    return value


def _object(payload: dict[str, Any], key: str, parent: str) -> dict[str, Any]:
    return _typed_object(payload.get(key), f"{parent}.{key}")


def _array(payload: dict[str, Any], key: str, parent: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise P04CalibrationError(f"{parent}.{key} must be an array")
    return value


def _string(payload: dict[str, Any], key: str, parent: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise P04CalibrationError(f"{parent}.{key} must be a non-blank string")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str, parent: str) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise P04CalibrationError(f"{parent}.{key} must be a string or null")
    return value.strip()


def _number(payload: dict[str, Any], key: str, parent: str) -> float:
    return _finite_value(payload.get(key), f"{parent}.{key}")


def _optional_number(payload: dict[str, Any], key: str, parent: str) -> float | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    return _finite_value(value, f"{parent}.{key}")


def _finite_value(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise P04CalibrationError(f"{field_name} must be a finite number")
    return float(value)


def _integer(payload: dict[str, Any], key: str, parent: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise P04CalibrationError(f"{parent}.{key} must be an integer")
    return value


def _optional_integer(payload: dict[str, Any], key: str, parent: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise P04CalibrationError(f"{parent}.{key} must be an integer or null")
    return value


def _require_id(value: str, field_name: str) -> None:
    if not _ID_PATTERN.fullmatch(value):
        raise P04CalibrationError(f"{field_name} must be a lowercase hyphenated ID")


def _require_sha256(value: str, field_name: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise P04CalibrationError(f"{field_name} must be a lowercase SHA-256")


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise P04CalibrationError(f"{field_name} must be non-blank")


def _require_finite(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise P04CalibrationError(f"{field_name} must be finite")


def _require_non_negative(value: float, field_name: str) -> None:
    _require_finite(value, field_name)
    if value < 0:
        raise P04CalibrationError(f"{field_name} must be non-negative")
