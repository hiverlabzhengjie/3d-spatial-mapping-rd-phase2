"""Credential-safe P01 input and read-only observability contracts.

P01 records what a camera source can support; it does not calibrate a camera, derive a facility
frame, or retain raw RTSP URLs. Runtime decoding and reconnect logic remain P03 responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

CAMERA_IDS: tuple[str, ...] = (
    "office-cam-01",
    "office-cam-02",
    "office-cam-03",
    "office-cam-04",
)
CAMERA_ENDPOINT_KEYS: dict[str, str] = {
    camera_id: f"PHASE2_RTSP_CAMERA_{index}"
    for index, camera_id in enumerate(CAMERA_IDS, start=1)
}
OWNER_INPUT_SCHEMA_VERSION = "p01-owner-input-v1"
STREAM_PROFILE_SCHEMA_VERSION = "p01-stream-profile-v1"
CAPTURE_MANIFEST_SCHEMA_VERSION = "p01-diagnostic-capture-v1"
MAX_DIAGNOSTIC_DURATION_SECONDS = 15.0
_CAMERA_ID_PATTERN = re.compile(r"^office-cam-0[1-4]$")
_PROFILE_VERSION_PATTERN = re.compile(r"^stream-profile-v[1-9][0-9]*$")


class P01ContractError(ValueError):
    """Raised for unsafe or incomplete P01 records without leaking endpoint secrets."""


class EndpointConfigurationError(P01ContractError):
    """Raised for missing or malformed local RTSP configuration."""


class StreamPreflightError(P01ContractError):
    """Raised when a read-only stream preflight cannot complete safely."""


class DiagnosticCaptureError(P01ContractError):
    """Raised when a bounded diagnostic capture cannot complete safely."""


def _require_non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P01ContractError(f"{field_name} must be a non-blank string")
    return value.strip()


def _require_finite_non_negative(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise P01ContractError(f"{field_name} must be finite and non-negative")
    return float(value)


def _parse_utc(timestamp: str, field_name: str) -> datetime:
    value = _require_non_blank(timestamp, field_name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise P01ContractError(f"{field_name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise P01ContractError(f"{field_name} must include a UTC offset")
    return parsed.astimezone(UTC)


def redact_rtsp_url(endpoint_url: str) -> str:
    """Return a credential-safe URL description for non-persistent diagnostic messages.

    Query, fragment, path and credentials are intentionally removed. The function is only for
    terminal-safe messages; tracked P01 manifests use an environment-key reference instead.
    """

    try:
        parsed = urlsplit(endpoint_url)
        port = parsed.port
    except ValueError:
        return "<invalid-rtsp-endpoint>"
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if scheme not in {"rtsp", "rtsps"} or hostname is None:
        return "<invalid-rtsp-endpoint>"
    authority = hostname
    if port is not None:
        authority = f"{authority}:{port}"
    return urlunsplit((scheme, authority, "", "", ""))


def _validate_rtsp_url(endpoint_url: str, endpoint_key: str) -> str:
    if not isinstance(endpoint_url, str) or not endpoint_url.strip():
        raise EndpointConfigurationError(f"{endpoint_key} is missing")
    if endpoint_url != endpoint_url.strip() or any(
        character.isspace() for character in endpoint_url
    ):
        raise EndpointConfigurationError(f"{endpoint_key} is malformed")
    try:
        parsed = urlsplit(endpoint_url)
        _ = parsed.port
    except ValueError as error:
        raise EndpointConfigurationError(f"{endpoint_key} is malformed") from error
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or parsed.hostname is None:
        raise EndpointConfigurationError(f"{endpoint_key} is malformed")
    return endpoint_url


@dataclass(frozen=True, slots=True, repr=False)
class LocalRtspEndpoint:
    """A local-only credential-bearing endpoint, deliberately hidden from representation."""

    camera_id: str
    environment_key: str
    _url: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_camera_id(self.camera_id)
        expected_key = CAMERA_ENDPOINT_KEYS[self.camera_id]
        if self.environment_key != expected_key:
            raise EndpointConfigurationError(
                f"{self.camera_id} must use local key {expected_key}"
            )
        _validate_rtsp_url(self._url, self.environment_key)

    def for_read_only_adapter(self) -> str:
        """Return the endpoint only to a local adapter; callers must not persist or log it."""

        return self._url


def load_local_rtsp_endpoints(
    environment: Mapping[str, str] | None = None,
) -> tuple[LocalRtspEndpoint, ...]:
    """Load all four configured endpoints without including their values in errors."""

    source = os.environ if environment is None else environment
    endpoints: list[LocalRtspEndpoint] = []
    for camera_id in CAMERA_IDS:
        endpoint_key = CAMERA_ENDPOINT_KEYS[camera_id]
        endpoint_url = source.get(endpoint_key, "")
        endpoints.append(LocalRtspEndpoint(camera_id, endpoint_key, endpoint_url))
    return tuple(endpoints)


def _validate_camera_id(camera_id: str) -> None:
    if not isinstance(camera_id, str) or not _CAMERA_ID_PATTERN.fullmatch(camera_id):
        raise P01ContractError("camera_id must be one of the four fixed office camera IDs")


def _validate_profile_version(profile_version: str) -> None:
    if (
        not isinstance(profile_version, str)
        or not _PROFILE_VERSION_PATTERN.fullmatch(profile_version)
    ):
        raise P01ContractError("profile_version must use stream-profile-vN")


@dataclass(frozen=True, slots=True)
class ApproximatePlanObservation:
    """Owner-supplied plan mark, explicitly not a verified world coordinate."""

    description: str
    uncertainty_note: str

    def __post_init__(self) -> None:
        _require_non_blank(self.description, "plan_position.description")
        _require_non_blank(self.uncertainty_note, "plan_position.uncertainty_note")


@dataclass(frozen=True, slots=True)
class MountingObservation:
    """Owner-supplied mounting height in metres, pending later physical verification."""

    height_metres: float
    uncertainty_metres: float
    measured_by: str

    def __post_init__(self) -> None:
        _require_finite_non_negative(self.height_metres, "mounting.height_metres")
        _require_finite_non_negative(self.uncertainty_metres, "mounting.uncertainty_metres")
        _require_non_blank(self.measured_by, "mounting.measured_by")


@dataclass(frozen=True, slots=True)
class CameraIdentity:
    """Stable physical-camera binding that cannot be silently retargeted."""

    camera_id: str
    physical_label: str
    endpoint_environment_key: str
    stream_profile_version: str

    def __post_init__(self) -> None:
        _validate_camera_id(self.camera_id)
        _require_non_blank(self.physical_label, "physical_label")
        if self.endpoint_environment_key != CAMERA_ENDPOINT_KEYS[self.camera_id]:
            raise P01ContractError("endpoint_environment_key does not match camera_id")
        _validate_profile_version(self.stream_profile_version)


@dataclass(frozen=True, slots=True)
class CameraOwnerInput:
    """Non-secret owner input associated with exactly one stable camera ID."""

    identity: CameraIdentity
    approximate_plan_position: ApproximatePlanObservation | None
    mounting: MountingObservation | None
    camera_model: str | None
    lens_details: str | None
    stream_alteration_confirmation: str | None

    def __post_init__(self) -> None:
        for optional_name in (
            "camera_model",
            "lens_details",
            "stream_alteration_confirmation",
        ):
            optional_value = getattr(self, optional_name)
            if optional_value is not None:
                _require_non_blank(optional_value, optional_name)


@dataclass(frozen=True, slots=True)
class OwnerInputManifest:
    """Local manifest for owner observations; does not contain RTSP URLs or credentials."""

    schema_version: str
    created_at: str
    cameras: tuple[CameraOwnerInput, ...]

    def __post_init__(self) -> None:
        if self.schema_version != OWNER_INPUT_SCHEMA_VERSION:
            raise P01ContractError("unsupported owner-input schema_version")
        _parse_utc(self.created_at, "created_at")
        camera_ids = tuple(camera.identity.camera_id for camera in self.cameras)
        if camera_ids != CAMERA_IDS:
            raise P01ContractError(
                "owner manifest must contain each camera once in fixed ID order"
            )

    @classmethod
    def from_json_file(cls, path: str | Path) -> OwnerInputManifest:
        """Load and validate a local non-secret owner-input JSON file."""

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise P01ContractError("owner-input manifest could not be read as JSON") from error
        if not isinstance(payload, dict):
            raise P01ContractError("owner-input manifest root must be an object")
        cameras_payload = payload.get("cameras")
        if not isinstance(cameras_payload, list):
            raise P01ContractError("owner-input manifest cameras must be an array")
        cameras = tuple(_parse_camera_owner_input(item) for item in cameras_payload)
        return cls(
            schema_version=_payload_string(payload, "schema_version"),
            created_at=_payload_string(payload, "created_at"),
            cameras=cameras,
        )


def _payload_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise P01ContractError(f"{key} must be a string")
    return value


def _payload_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise P01ContractError(f"{key} must be a number")
    return float(value)


def _payload_optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise P01ContractError(f"{key} must be a string or null")
    return value


def _parse_camera_owner_input(payload: object) -> CameraOwnerInput:
    if not isinstance(payload, dict):
        raise P01ContractError("each owner-input camera must be an object")
    identity_payload = payload.get("identity")
    if not isinstance(identity_payload, dict):
        raise P01ContractError("camera identity must be an object")
    identity = CameraIdentity(
        camera_id=_payload_string(identity_payload, "camera_id"),
        physical_label=_payload_string(identity_payload, "physical_label"),
        endpoint_environment_key=_payload_string(identity_payload, "endpoint_environment_key"),
        stream_profile_version=_payload_string(identity_payload, "stream_profile_version"),
    )
    plan_payload = payload.get("approximate_plan_position")
    mounting_payload = payload.get("mounting")
    if plan_payload is not None and not isinstance(plan_payload, dict):
        raise P01ContractError("approximate_plan_position must be an object or null")
    if mounting_payload is not None and not isinstance(mounting_payload, dict):
        raise P01ContractError("mounting must be an object or null")
    return CameraOwnerInput(
        identity=identity,
        approximate_plan_position=(
            None
            if plan_payload is None
            else ApproximatePlanObservation(
                description=_payload_string(plan_payload, "description"),
                uncertainty_note=_payload_string(plan_payload, "uncertainty_note"),
            )
        ),
        mounting=(
            None
            if mounting_payload is None
            else MountingObservation(
                height_metres=_payload_float(mounting_payload, "height_metres"),
                uncertainty_metres=_payload_float(mounting_payload, "uncertainty_metres"),
                measured_by=_payload_string(mounting_payload, "measured_by"),
            )
        ),
        camera_model=_payload_optional_string(payload, "camera_model"),
        lens_details=_payload_optional_string(payload, "lens_details"),
        stream_alteration_confirmation=_payload_optional_string(
            payload, "stream_alteration_confirmation"
        ),
    )


def assert_immutable_camera_identities(
    prior: OwnerInputManifest, replacement: OwnerInputManifest
) -> None:
    """Reject a replacement manifest that changes a stable physical-camera binding."""

    prior_bindings = tuple(camera.identity for camera in prior.cameras)
    replacement_bindings = tuple(camera.identity for camera in replacement.cameras)
    if prior_bindings != replacement_bindings:
        raise P01ContractError(
            "camera identity bindings are immutable; create a new reviewed profile version"
        )


@dataclass(frozen=True, slots=True)
class StreamProbeResult:
    """Observed stream characteristics from one bounded read-only preflight."""

    observed_at: str
    width_pixels: int
    height_pixels: int
    codec: str
    nominal_fps: float | None
    observed_fps: float | None
    time_base: str | None
    rotation_degrees: int | None
    crop_description: str | None
    overlay_description: str | None
    dewarping_indicator: str | None
    keyframe_behavior: str | None
    stability_note: str

    def __post_init__(self) -> None:
        _parse_utc(self.observed_at, "stream.observed_at")
        if not isinstance(self.width_pixels, int) or self.width_pixels <= 0:
            raise P01ContractError("stream.width_pixels must be a positive integer")
        if not isinstance(self.height_pixels, int) or self.height_pixels <= 0:
            raise P01ContractError("stream.height_pixels must be a positive integer")
        _require_non_blank(self.codec, "stream.codec")
        for value, name in (
            (self.nominal_fps, "stream.nominal_fps"),
            (self.observed_fps, "stream.observed_fps"),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise P01ContractError(f"{name} must be finite and positive when observed")
        if self.rotation_degrees is not None and self.rotation_degrees not in {0, 90, 180, 270}:
            raise P01ContractError(
                "stream.rotation_degrees must be 0, 90, 180 or 270 when observed"
            )
        _require_non_blank(self.stability_note, "stream.stability_note")


@dataclass(frozen=True, slots=True)
class StreamProfile:
    """Credential-free, immutable stream observation bound to one profile version."""

    schema_version: str
    camera_id: str
    profile_version: str
    endpoint_environment_key: str
    observation: StreamProbeResult

    def __post_init__(self) -> None:
        if self.schema_version != STREAM_PROFILE_SCHEMA_VERSION:
            raise P01ContractError("unsupported stream-profile schema_version")
        _validate_camera_id(self.camera_id)
        _validate_profile_version(self.profile_version)
        if self.endpoint_environment_key != CAMERA_ENDPOINT_KEYS[self.camera_id]:
            raise P01ContractError("stream profile endpoint key does not match camera_id")


@dataclass(frozen=True, slots=True)
class DiagnosticCaptureRequest:
    """Bounded capture request; the adapter must not alter the source stream."""

    duration_seconds: float
    connect_timeout_seconds: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
            or self.duration_seconds > MAX_DIAGNOSTIC_DURATION_SECONDS
        ):
            raise DiagnosticCaptureError(
                "duration_seconds must be finite, positive and at most "
                f"{MAX_DIAGNOSTIC_DURATION_SECONDS:g}"
            )
        if not math.isfinite(self.connect_timeout_seconds) or self.connect_timeout_seconds <= 0:
            raise DiagnosticCaptureError("connect_timeout_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class CapturedDiagnosticArtifact:
    """Local adapter output without source credentials or artifact-store absolute paths."""

    relative_artifact_path: str
    sha256: str
    source_pts_start_seconds: float | None
    source_pts_end_seconds: float | None
    acquisition_started_at: str
    acquisition_finished_at: str

    def __post_init__(self) -> None:
        _validate_capture_relative_path(self.relative_artifact_path)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise DiagnosticCaptureError("sha256 must be a lowercase SHA-256 digest")
        for value, name in (
            (self.source_pts_start_seconds, "source_pts_start_seconds"),
            (self.source_pts_end_seconds, "source_pts_end_seconds"),
        ):
            if value is not None:
                _require_finite_non_negative(value, name)
        if (
            self.source_pts_start_seconds is not None
            and self.source_pts_end_seconds is not None
            and self.source_pts_end_seconds < self.source_pts_start_seconds
        ):
            raise DiagnosticCaptureError("source PTS end must not precede source PTS start")
        if _parse_utc(self.acquisition_finished_at, "acquisition_finished_at") < _parse_utc(
            self.acquisition_started_at, "acquisition_started_at"
        ):
            raise DiagnosticCaptureError("acquisition finished before it started")


def _validate_capture_relative_path(relative_artifact_path: str) -> None:
    value = _require_non_blank(relative_artifact_path, "relative_artifact_path")
    normalized = value.replace("/", "\\")
    path = PureWindowsPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "captures":
        raise DiagnosticCaptureError("relative_artifact_path must remain below captures")


@dataclass(frozen=True, slots=True)
class DiagnosticCaptureManifest:
    """Credential-free capture provenance for an artifact retained outside Git."""

    schema_version: str
    capture_id: str
    camera_id: str
    stream_profile_version: str
    request: DiagnosticCaptureRequest
    artifact: CapturedDiagnosticArtifact

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_MANIFEST_SCHEMA_VERSION:
            raise DiagnosticCaptureError("unsupported diagnostic-capture schema_version")
        _require_non_blank(self.capture_id, "capture_id")
        _validate_camera_id(self.camera_id)
        _validate_profile_version(self.stream_profile_version)

    def to_sanitized_json(self) -> str:
        """Serialize only credential-free capture provenance for artifact-side manifests."""

        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


class ReadOnlyStreamAdapter(Protocol):
    """Boundary used by P01 tests and future P03 adapters; never changes source settings."""

    def probe(self, endpoint_url: str, timeout_seconds: float) -> StreamProbeResult:
        """Read metadata/frames within a bounded timeout without changing the source."""

    def capture_diagnostic(
        self, endpoint_url: str, request: DiagnosticCaptureRequest
    ) -> CapturedDiagnosticArtifact:
        """Retain one bounded local diagnostic artifact without changing the source."""


class ReadOnlyObservabilityService:
    """Safe orchestration of local endpoint use with credential-free exception messages."""

    def __init__(self, adapter: ReadOnlyStreamAdapter) -> None:
        self._adapter = adapter

    def preflight(self, endpoint: LocalRtspEndpoint, timeout_seconds: float) -> StreamProbeResult:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise StreamPreflightError("timeout_seconds must be finite and positive")
        try:
            return self._adapter.probe(endpoint.for_read_only_adapter(), timeout_seconds)
        except TimeoutError:
            failure_message = f"{endpoint.environment_key} preflight timed out"
        except ConnectionError:
            failure_message = f"{endpoint.environment_key} preflight disconnected"
        except Exception:
            failure_message = f"{endpoint.environment_key} preflight failed"
        raise StreamPreflightError(failure_message)

    def capture(
        self, endpoint: LocalRtspEndpoint, request: DiagnosticCaptureRequest
    ) -> CapturedDiagnosticArtifact:
        try:
            return self._adapter.capture_diagnostic(endpoint.for_read_only_adapter(), request)
        except TimeoutError:
            failure_message = f"{endpoint.environment_key} diagnostic capture timed out"
        except ConnectionError:
            failure_message = f"{endpoint.environment_key} diagnostic capture disconnected"
        except Exception:
            failure_message = f"{endpoint.environment_key} diagnostic capture failed"
        raise DiagnosticCaptureError(failure_message)


def fingerprint_diagnostic_bytes(content: bytes) -> str:
    """Return the SHA-256 required for an immutable artifact manifest."""

    return hashlib.sha256(content).hexdigest()


def validate_landmark_inventory(
    solving_candidates: Sequence[str], held_out_candidates: Sequence[str]
) -> None:
    """Validate P01 candidate separation without claiming that counts prove geometric viability."""

    solving = tuple(_require_non_blank(value, "solving landmark") for value in solving_candidates)
    held_out = tuple(
        _require_non_blank(value, "held-out landmark") for value in held_out_candidates
    )
    if len(set(solving)) != len(solving) or len(set(held_out)) != len(held_out):
        raise P01ContractError("landmark candidate labels must be unique within each role")
    if set(solving) & set(held_out):
        raise P01ContractError("solving and held-out landmark candidates must not overlap")
