"""Typed contracts for interactive P02 plan and camera mounting-reference registration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from spatial_mapping_phase2.p01_observability import CAMERA_ENDPOINT_KEYS, CAMERA_IDS

INTERACTIVE_REGISTRATION_SCHEMA_VERSION = "p02-interactive-registration-v1"
INTERACTIVE_EXPORT_SCHEMA_VERSION = "p02-interactive-export-v1"
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


class InteractiveRegistrationError(ValueError):
    """Raised when interactive registration input is incomplete or unsafe to interpret."""


class ScaleSourceKind(StrEnum):
    PRINTED_DIMENSION = "printed-dimension"
    PHYSICAL_CHECK = "physical-check"


class CameraRegistrationStatus(StrEnum):
    UNPLACED = "unplaced"
    PLACED = "placed"
    MOUNT_PRIOR_COMPLETE = "mount-prior-complete"
    READY_FOR_CALIBRATION = "ready-for-calibration"


@dataclass(frozen=True, slots=True)
class PixelPoint:
    """A point in the rendered plan's full-resolution pixel frame."""

    u: float
    v: float

    def __post_init__(self) -> None:
        _require_finite_non_negative(self.u, "pixel u")
        _require_finite_non_negative(self.v, "pixel v")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], field_name: str) -> PixelPoint:
        return cls(_number(payload, "u", field_name), _number(payload, "v", field_name))

    def to_dict(self) -> dict[str, float]:
        return {"u": self.u, "v": self.v}


@dataclass(frozen=True, slots=True)
class PlanMetadata:
    """Hash-bound source and its display derivative dimensions; paths remain local."""

    source_sha256: str
    original_filename: str
    page_number: int
    image_width_pixels: int
    image_height_pixels: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise InteractiveRegistrationError("plan source_sha256 must be a lowercase SHA-256")
        _require_non_blank(self.original_filename, "plan original_filename")
        if self.page_number != 1:
            raise InteractiveRegistrationError("the first interactive release supports PDF page 1")
        if self.image_width_pixels <= 0 or self.image_height_pixels <= 0:
            raise InteractiveRegistrationError("plan display dimensions must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanMetadata:
        return cls(
            _string(payload, "source_sha256", "plan"),
            _string(payload, "original_filename", "plan"),
            _integer(payload, "page_number", "plan"),
            _integer(payload, "image_width_pixels", "plan"),
            _integer(payload, "image_height_pixels", "plan"),
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "source_sha256": self.source_sha256,
            "original_filename": self.original_filename,
            "page_number": self.page_number,
            "image_width_pixels": self.image_width_pixels,
            "image_height_pixels": self.image_height_pixels,
        }


@dataclass(frozen=True, slots=True)
class ScaleControl:
    """A user-selected pixel segment tied to a named metric observation."""

    control_id: str
    meaning: str
    point_a: PixelPoint
    point_b: PixelPoint
    distance_metres: float
    distance_uncertainty_metres: float
    source_kind: ScaleSourceKind

    def __post_init__(self) -> None:
        _require_id(self.control_id, "scale control_id")
        _require_non_blank(self.meaning, "scale meaning")
        _require_positive(self.distance_metres, "scale distance_metres")
        _require_finite_non_negative(
            self.distance_uncertainty_metres, "scale distance_uncertainty_metres"
        )
        if self.pixel_length <= 1.0:
            raise InteractiveRegistrationError("scale control endpoints must be distinct")

    @property
    def pixel_length(self) -> float:
        return math.hypot(self.point_b.u - self.point_a.u, self.point_b.v - self.point_a.v)

    @property
    def pixels_per_metre(self) -> float:
        return self.pixel_length / self.distance_metres

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScaleControl:
        return cls(
            _string(payload, "control_id", "scale control"),
            _string(payload, "meaning", "scale control"),
            PixelPoint.from_dict(_object(payload, "point_a", "scale control"), "point_a"),
            PixelPoint.from_dict(_object(payload, "point_b", "scale control"), "point_b"),
            _number(payload, "distance_metres", "scale control"),
            _number(payload, "distance_uncertainty_metres", "scale control"),
            ScaleSourceKind(_string(payload, "source_kind", "scale control")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "meaning": self.meaning,
            "point_a": self.point_a.to_dict(),
            "point_b": self.point_b.to_dict(),
            "distance_metres": self.distance_metres,
            "distance_uncertainty_metres": self.distance_uncertainty_metres,
            "source_kind": self.source_kind.value,
        }


@dataclass(frozen=True, slots=True)
class FramePlacement:
    """Origin plus a user-chosen +X handle; +Y and +Z are derived right-handed directions."""

    origin: PixelPoint
    positive_x_handle: PixelPoint
    origin_feature_meaning: str

    def __post_init__(self) -> None:
        _require_non_blank(self.origin_feature_meaning, "origin feature meaning")
        if (
            math.hypot(
                self.positive_x_handle.u - self.origin.u,
                self.positive_x_handle.v - self.origin.v,
            )
            <= 10.0
        ):
            raise InteractiveRegistrationError("+X handle must be at least 10 pixels from origin")

    @property
    def x_axis_pixel_unit(self) -> tuple[float, float]:
        delta_u = self.positive_x_handle.u - self.origin.u
        delta_v = self.positive_x_handle.v - self.origin.v
        length = math.hypot(delta_u, delta_v)
        return delta_u / length, delta_v / length

    @property
    def y_axis_pixel_unit(self) -> tuple[float, float]:
        x_u, x_v = self.x_axis_pixel_unit
        return x_v, -x_u

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FramePlacement:
        return cls(
            PixelPoint.from_dict(_object(payload, "origin", "frame"), "origin"),
            PixelPoint.from_dict(
                _object(payload, "positive_x_handle", "frame"), "positive_x_handle"
            ),
            _string(payload, "origin_feature_meaning", "frame"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin.to_dict(),
            "positive_x_handle": self.positive_x_handle.to_dict(),
            "origin_feature_meaning": self.origin_feature_meaning,
        }


@dataclass(frozen=True, slots=True)
class CameraPlacement:
    """A direct physical mounting-reference mark, explicitly not an optical-centre pose."""

    camera_id: str
    physical_label: str
    marker: PixelPoint | None
    mounting_height_metres: float | None
    height_uncertainty_metres: float | None
    reference_meaning: str
    rough_pan_endpoint: PixelPoint | None

    def __post_init__(self) -> None:
        if self.camera_id not in CAMERA_IDS:
            raise InteractiveRegistrationError("camera_id must identify one fixed office camera")
        if self.physical_label:
            _require_non_blank(self.physical_label, "physical label")
        if self.reference_meaning:
            _require_non_blank(self.reference_meaning, "camera reference meaning")
        if (self.mounting_height_metres is None) != (self.height_uncertainty_metres is None):
            raise InteractiveRegistrationError(
                "mounting height and height uncertainty must be supplied together"
            )
        if self.mounting_height_metres is not None:
            _require_positive(self.mounting_height_metres, "mounting height")
            assert self.height_uncertainty_metres is not None
            _require_finite_non_negative(self.height_uncertainty_metres, "height uncertainty")
        if self.rough_pan_endpoint is not None and self.marker is None:
            raise InteractiveRegistrationError("rough pan requires a placed camera marker")

    def status(self, endpoint_configured: bool) -> CameraRegistrationStatus:
        if self.marker is None:
            return CameraRegistrationStatus.UNPLACED
        if not self.physical_label or self.mounting_height_metres is None:
            return CameraRegistrationStatus.PLACED
        if endpoint_configured:
            return CameraRegistrationStatus.READY_FOR_CALIBRATION
        return CameraRegistrationStatus.MOUNT_PRIOR_COMPLETE

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CameraPlacement:
        marker_payload = payload.get("marker")
        pan_payload = payload.get("rough_pan_endpoint")
        return cls(
            _string(payload, "camera_id", "camera"),
            _optional_string(payload, "physical_label", "camera") or "",
            None
            if marker_payload is None
            else PixelPoint.from_dict(_typed_object(marker_payload, "camera marker"), "marker"),
            _optional_number(payload, "mounting_height_metres", "camera"),
            _optional_number(payload, "height_uncertainty_metres", "camera"),
            _optional_string(payload, "reference_meaning", "camera") or "",
            None
            if pan_payload is None
            else PixelPoint.from_dict(
                _typed_object(pan_payload, "rough pan endpoint"), "rough_pan_endpoint"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "physical_label": self.physical_label,
            "marker": None if self.marker is None else self.marker.to_dict(),
            "mounting_height_metres": self.mounting_height_metres,
            "height_uncertainty_metres": self.height_uncertainty_metres,
            "reference_meaning": self.reference_meaning,
            "rough_pan_endpoint": (
                None if self.rough_pan_endpoint is None else self.rough_pan_endpoint.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class InteractiveRegistrationState:
    """Editable, versioned registration state; derived metric coordinates are not persisted."""

    schema_version: str
    revision: int
    plan: PlanMetadata
    scale_controls: tuple[ScaleControl, ...]
    frame: FramePlacement | None
    cameras: tuple[CameraPlacement, ...]

    def __post_init__(self) -> None:
        if self.schema_version != INTERACTIVE_REGISTRATION_SCHEMA_VERSION:
            raise InteractiveRegistrationError("unsupported interactive registration schema")
        if self.revision < 0:
            raise InteractiveRegistrationError("revision must be non-negative")
        if tuple(camera.camera_id for camera in self.cameras) != CAMERA_IDS:
            raise InteractiveRegistrationError(
                "registration must contain the four cameras in order"
            )
        control_ids = [control.control_id for control in self.scale_controls]
        if len(control_ids) != len(set(control_ids)):
            raise InteractiveRegistrationError("scale control IDs must be unique")
        labels = [camera.physical_label for camera in self.cameras if camera.physical_label]
        if len(labels) != len(set(labels)):
            raise InteractiveRegistrationError("non-empty physical camera labels must be unique")
        for point, meaning in self._all_points():
            if point.u > self.plan.image_width_pixels or point.v > self.plan.image_height_pixels:
                raise InteractiveRegistrationError(f"{meaning} must lie inside the rendered plan")

    def _all_points(self) -> tuple[tuple[PixelPoint, str], ...]:
        points: list[tuple[PixelPoint, str]] = []
        for control in self.scale_controls:
            points.extend(((control.point_a, "scale point"), (control.point_b, "scale point")))
        if self.frame is not None:
            points.extend(
                ((self.frame.origin, "origin"), (self.frame.positive_x_handle, "+X handle"))
            )
        for camera in self.cameras:
            if camera.marker is not None:
                points.append((camera.marker, f"{camera.camera_id} marker"))
            if camera.rough_pan_endpoint is not None:
                points.append((camera.rough_pan_endpoint, f"{camera.camera_id} pan endpoint"))
        return tuple(points)

    @property
    def pixels_per_metre(self) -> float | None:
        if not self.scale_controls:
            return None
        return sum(control.pixels_per_metre for control in self.scale_controls) / len(
            self.scale_controls
        )

    @property
    def scale_spread_fraction(self) -> float | None:
        if len(self.scale_controls) < 2:
            return None
        values = [control.pixels_per_metre for control in self.scale_controls]
        average = sum(values) / len(values)
        return (max(values) - min(values)) / average

    def world_xy_from_pixel(self, point: PixelPoint) -> tuple[float, float]:
        pixels_per_metre = self.pixels_per_metre
        if pixels_per_metre is None or self.frame is None:
            raise InteractiveRegistrationError(
                "metric XY requires at least one scale control and a placed facility frame"
            )
        delta_u = point.u - self.frame.origin.u
        delta_v = point.v - self.frame.origin.v
        x_u, x_v = self.frame.x_axis_pixel_unit
        y_u, y_v = self.frame.y_axis_pixel_unit
        return (
            (delta_u * x_u + delta_v * x_v) / pixels_per_metre,
            (delta_u * y_u + delta_v * y_v) / pixels_per_metre,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InteractiveRegistrationState:
        plan = PlanMetadata.from_dict(_object(payload, "plan", "registration"))
        controls_payload = _array(payload, "scale_controls", "registration")
        frame_payload = payload.get("frame")
        cameras_payload = _array(payload, "cameras", "registration")
        return cls(
            _string(payload, "schema_version", "registration"),
            _integer(payload, "revision", "registration"),
            plan,
            tuple(
                ScaleControl.from_dict(_typed_object(item, "scale control"))
                for item in controls_payload
            ),
            None
            if frame_payload is None
            else FramePlacement.from_dict(_typed_object(frame_payload, "frame")),
            tuple(
                CameraPlacement.from_dict(_typed_object(item, "camera"))
                for item in cameras_payload
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "plan": self.plan.to_dict(),
            "scale_controls": [control.to_dict() for control in self.scale_controls],
            "frame": None if self.frame is None else self.frame.to_dict(),
            "cameras": [camera.to_dict() for camera in self.cameras],
        }


def empty_registration(plan: PlanMetadata) -> InteractiveRegistrationState:
    """Create a clean editable state for a newly uploaded immutable source plan."""

    cameras = tuple(
        CameraPlacement(camera_id, "", None, None, None, "", None) for camera_id in CAMERA_IDS
    )
    return InteractiveRegistrationState(
        INTERACTIVE_REGISTRATION_SCHEMA_VERSION,
        0,
        plan,
        (),
        None,
        cameras,
    )


def build_interactive_export(
    state: InteractiveRegistrationState, endpoint_configured: dict[str, bool]
) -> dict[str, Any]:
    """Build a credential-free facility-frame and provisional mounting-prior snapshot."""

    if state.frame is None or state.pixels_per_metre is None:
        raise InteractiveRegistrationError(
            "export requires scale calibration and a facility frame"
        )
    frame = state.frame
    pixels_per_metre = state.pixels_per_metre
    x_u, x_v = frame.x_axis_pixel_unit
    y_u, y_v = frame.y_axis_pixel_unit
    origin_u = frame.origin.u
    origin_v = frame.origin.v
    world_from_pixel = [
        [
            x_u / pixels_per_metre,
            x_v / pixels_per_metre,
            -(origin_u * x_u + origin_v * x_v) / pixels_per_metre,
        ],
        [
            y_u / pixels_per_metre,
            y_v / pixels_per_metre,
            -(origin_u * y_u + origin_v * y_v) / pixels_per_metre,
        ],
        [0.0, 0.0, 1.0],
    ]
    pixel_from_world = [
        [pixels_per_metre * x_u, pixels_per_metre * y_u, origin_u],
        [pixels_per_metre * x_v, pixels_per_metre * y_v, origin_v],
        [0.0, 0.0, 1.0],
    ]
    cameras: list[dict[str, Any]] = []
    for camera in state.cameras:
        configured = endpoint_configured.get(camera.camera_id, False)
        world_xy = None if camera.marker is None else state.world_xy_from_pixel(camera.marker)
        pan_vector = None
        if camera.marker is not None and camera.rough_pan_endpoint is not None:
            pan_end = state.world_xy_from_pixel(camera.rough_pan_endpoint)
            assert world_xy is not None
            pan_vector = [pan_end[0] - world_xy[0], pan_end[1] - world_xy[1]]
        cameras.append(
            {
                "camera_id": camera.camera_id,
                "physical_label": camera.physical_label or None,
                "status": camera.status(configured).value,
                "reference_meaning": camera.reference_meaning or None,
                "C_world_mount_prior": (
                    None
                    if world_xy is None or camera.mounting_height_metres is None
                    else {
                        "x_metres": world_xy[0],
                        "y_metres": world_xy[1],
                        "z_metres": camera.mounting_height_metres,
                    }
                ),
                "height_uncertainty_metres": camera.height_uncertainty_metres,
                "horizontal_uncertainty_metres": None,
                "rough_pan_vector_world_xy": pan_vector,
                "rough_pan_authority": "display-only owner estimate; not a pose constraint",
                "endpoint_environment_key": CAMERA_ENDPOINT_KEYS[camera.camera_id],
                "endpoint_configured": configured,
                "authority_note": "physical mounting reference prior; not optical centre or pose",
            }
        )
    residuals = [
        {
            "control_id": control.control_id,
            "implied_distance_metres": control.pixel_length / pixels_per_metre,
            "stated_distance_metres": control.distance_metres,
            "residual_metres": control.pixel_length / pixels_per_metre - control.distance_metres,
        }
        for control in state.scale_controls
    ]
    return {
        "schema_version": INTERACTIVE_EXPORT_SCHEMA_VERSION,
        "source_revision": state.revision,
        "plan": state.plan.to_dict(),
        "facility_frame": {
            "frame_id": "facility-world-interactive-v1",
            "origin_feature_meaning": frame.origin_feature_meaning,
            "units": "metres",
            "floor_reference": "finished floor Z=0",
            "handedness": "+X cross +Y = +Z",
            "z_direction": "upward",
            "T_world_from_plan_display_pixel": world_from_pixel,
            "T_plan_display_pixel_from_world": pixel_from_world,
        },
        "scale_calibration": {
            "pixels_per_metre": pixels_per_metre,
            "control_count": len(state.scale_controls),
            "scale_spread_fraction": state.scale_spread_fraction,
            "controls": [control.to_dict() for control in state.scale_controls],
            "residuals": residuals,
            "authority_note": (
                "explicit user-selected metric controls; raster pixels alone are not metric "
                "authority"
            ),
        },
        "camera_mounting_priors": cameras,
    }


def _typed_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InteractiveRegistrationError(f"{field_name} must be an object")
    return value


def _object(payload: dict[str, Any], key: str, parent: str) -> dict[str, Any]:
    return _typed_object(payload.get(key), f"{parent}.{key}")


def _array(payload: dict[str, Any], key: str, parent: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise InteractiveRegistrationError(f"{parent}.{key} must be an array")
    return value


def _string(payload: dict[str, Any], key: str, parent: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InteractiveRegistrationError(f"{parent}.{key} must be a non-blank string")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str, parent: str) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise InteractiveRegistrationError(f"{parent}.{key} must be a string or null")
    return value.strip()


def _number(payload: dict[str, Any], key: str, parent: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise InteractiveRegistrationError(f"{parent}.{key} must be a finite number")
    return float(value)


def _optional_number(payload: dict[str, Any], key: str, parent: str) -> float | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    return _number(payload, key, parent)


def _integer(payload: dict[str, Any], key: str, parent: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InteractiveRegistrationError(f"{parent}.{key} must be an integer")
    return value


def _require_id(value: str, field_name: str) -> None:
    if not _ID_PATTERN.fullmatch(value):
        raise InteractiveRegistrationError(f"{field_name} must be a lowercase hyphenated ID")


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InteractiveRegistrationError(f"{field_name} must be non-blank")


def _require_positive(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise InteractiveRegistrationError(f"{field_name} must be finite and positive")


def _require_finite_non_negative(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise InteractiveRegistrationError(f"{field_name} must be finite and non-negative")
