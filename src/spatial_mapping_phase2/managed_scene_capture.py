"""Credential-safe, variable-roster capture for managed XR03 scenes.

This module deliberately does not reuse P03's fixed-office contracts.  It is a
separate scene-scoped capture boundary: camera IDs and secret-key bindings are
provided by the managed-scene registry, while endpoint values are loaded from
that scene's ``secrets.env`` immediately before each live operation.

All persisted manifests contain camera IDs, endpoint *key* references and
hashes, never RTSP URLs or credentials.  The implementation is intentionally
bounded: a capture owns one cancellation event, limits parallel work, and
persists immutable session and bundle manifests below its supplied artifact
root.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import import_module
from io import BytesIO
from pathlib import Path, PurePosixPath
from time import monotonic, monotonic_ns
from typing import TYPE_CHECKING, Any, Protocol

from spatial_mapping_phase2.p01_observability import P01ContractError, validate_scene_rtsp_endpoint

if TYPE_CHECKING:
    from fastapi import FastAPI

SCENE_CAPTURE_PROFILE_SCHEMA = "managed-scene-stream-profile-v1"
SCENE_CAPTURE_SESSION_SCHEMA = "managed-scene-capture-session-v1"
SCENE_CAPTURE_BUNDLE_SCHEMA = "managed-scene-selected-bundle-v1"
SCENE_CAPTURE_SELECTION_SCHEMA = "managed-scene-capture-selection-v1"
SCENE_CAPTURE_SOFTWARE_IDENTITY = "spatial-mapping-phase2-managed-scene-capture-v1"

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SCENE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SceneCaptureError(ValueError):
    """A safe validation error at the managed-scene capture boundary."""


class SceneCaptureConfigurationError(SceneCaptureError):
    """A scene's credential-free binding has no usable local endpoint."""


class SceneCaptureBusyError(SceneCaptureError):
    """Another bounded capture is already in progress for this scene."""


class SceneCaptureAdapterError(RuntimeError):
    """A credential-safe live adapter failure."""


class SceneCaptureConnectTimeoutError(SceneCaptureAdapterError):
    """A bounded RTSP connection did not establish in time."""


class SceneCaptureReadTimeoutError(SceneCaptureAdapterError):
    """A bounded RTSP read did not produce usable media in time."""


class SceneCaptureCancelledError(SceneCaptureAdapterError):
    """An operator cancelled a bounded capture."""


class SceneCaptureBackpressureError(SceneCaptureAdapterError):
    """A capture exceeded its configured bounded frame count."""


def _require_identifier(value: str, field: str, *, scene_camera: bool = False) -> str:
    pattern = _SCENE_IDENTIFIER if scene_camera else _IDENTIFIER
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise SceneCaptureError(f"{field} is invalid")
    return value


def _require_non_negative_integer(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SceneCaptureError(f"{field} must be a non-negative integer")
    return value


def _require_positive_integer(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SceneCaptureError(f"{field} must be a positive integer")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise SceneCaptureError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SceneCaptureError(f"{field} must include a UTC offset")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_capture_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "captures":
        raise SceneCaptureError("artifact path must be relative below captures")


@dataclass(frozen=True, slots=True)
class SceneCameraBinding:
    """One non-secret camera-to-local-environment-key binding."""

    camera_id: str
    endpoint_environment_key: str

    def __post_init__(self) -> None:
        _require_identifier(self.camera_id, "camera_id", scene_camera=True)
        if not isinstance(self.endpoint_environment_key, str) or not _ENVIRONMENT_KEY.fullmatch(
            self.endpoint_environment_key
        ):
            raise SceneCaptureError("camera endpoint environment key is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class SceneRtspEndpoint:
    """A local-only endpoint that must never enter a manifest or representation."""

    binding: SceneCameraBinding
    _url: str

    def __post_init__(self) -> None:
        try:
            validate_scene_rtsp_endpoint(
                self.binding.camera_id,
                self.binding.endpoint_environment_key,
                self._url,
            )
        except P01ContractError as error:
            raise SceneCaptureConfigurationError("camera endpoint is invalid") from error

    def for_read_only_adapter(self) -> str:
        """Expose the credential only to a local read-only adapter."""

        return self._url


@dataclass(frozen=True, slots=True)
class SceneEndpointResolution:
    """A safe per-camera endpoint resolution result for one current request."""

    binding: SceneCameraBinding
    endpoint: SceneRtspEndpoint | None
    configuration_state: str

    def __post_init__(self) -> None:
        if self.configuration_state not in {"configured", "missing", "invalid", "unavailable"}:
            raise SceneCaptureError("endpoint configuration state is invalid")
        if (self.endpoint is not None) == (self.configuration_state == "configured"):
            return
        raise SceneCaptureError("endpoint configuration state does not match endpoint")


class SceneEndpointLoader:
    """Read only a managed scene's local secret file for each service operation."""

    def __init__(self, camera_bindings: Mapping[str, str], secret_file: Path) -> None:
        bindings = tuple(
            SceneCameraBinding(camera_id, endpoint_key)
            for camera_id, endpoint_key in camera_bindings.items()
        )
        if not bindings:
            raise SceneCaptureError("at least one camera binding is required")
        if len(bindings) > 64:
            raise SceneCaptureError("a managed scene may contain at most 64 cameras")
        ids = tuple(binding.camera_id for binding in bindings)
        keys = tuple(binding.endpoint_environment_key for binding in bindings)
        if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
            raise SceneCaptureError("camera bindings must have unique IDs and endpoint keys")
        self._bindings = bindings
        self.secret_file = secret_file.resolve()

    @property
    def bindings(self) -> tuple[SceneCameraBinding, ...]:
        return self._bindings

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(binding.camera_id for binding in self._bindings)

    def resolve(self) -> tuple[SceneEndpointResolution, ...]:
        values, duplicate_keys, readable = self._read_secret_values()
        resolutions: list[SceneEndpointResolution] = []
        for binding in self._bindings:
            if not readable:
                resolutions.append(SceneEndpointResolution(binding, None, "unavailable"))
                continue
            if binding.endpoint_environment_key in duplicate_keys:
                resolutions.append(SceneEndpointResolution(binding, None, "invalid"))
                continue
            value = values.get(binding.endpoint_environment_key, "").strip()
            if not value:
                resolutions.append(SceneEndpointResolution(binding, None, "missing"))
                continue
            try:
                endpoint = SceneRtspEndpoint(binding, value)
            except SceneCaptureConfigurationError:
                resolutions.append(SceneEndpointResolution(binding, None, "invalid"))
            else:
                resolutions.append(SceneEndpointResolution(binding, endpoint, "configured"))
        return tuple(resolutions)

    def endpoint_for(self, camera_id: str) -> SceneRtspEndpoint:
        _require_identifier(camera_id, "camera_id", scene_camera=True)
        resolution = next(
            (item for item in self.resolve() if item.binding.camera_id == camera_id), None
        )
        if resolution is None:
            raise SceneCaptureError("unknown camera_id")
        if resolution.endpoint is None:
            raise SceneCaptureConfigurationError(
                _configuration_message(resolution.configuration_state)
            )
        return resolution.endpoint

    def _read_secret_values(self) -> tuple[dict[str, str], frozenset[str], bool]:
        if not self.secret_file.is_file():
            return {}, frozenset(), True
        try:
            lines = self.secret_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {}, frozenset(), False
        values: dict[str, str] = {}
        duplicate_keys: set[str] = set()
        relevant_keys = {binding.endpoint_environment_key for binding in self._bindings}
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key not in relevant_keys:
                continue
            if key in values:
                duplicate_keys.add(key)
            values[key] = value.strip()
        return values, frozenset(duplicate_keys), True


@dataclass(frozen=True, slots=True)
class SceneCapturePolicy:
    """Finite limits for one managed-scene live operation."""

    duration_seconds: float = 2.0
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    retry_limit: int = 2
    initial_backoff_seconds: float = 0.1
    queue_capacity: int = 512
    max_parallel_cameras: int = 8

    def __post_init__(self) -> None:
        for value, name, maximum in (
            (self.duration_seconds, "duration_seconds", 30.0),
            (self.connect_timeout_seconds, "connect_timeout_seconds", 30.0),
            (self.read_timeout_seconds, "read_timeout_seconds", 30.0),
            (self.initial_backoff_seconds, "initial_backoff_seconds", 10.0),
        ):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
                or value > maximum
            ):
                raise SceneCaptureError(f"{name} must be finite and between zero and {maximum}")
        if not isinstance(self.retry_limit, int) or isinstance(self.retry_limit, bool):
            raise SceneCaptureError("retry_limit must be an integer")
        if self.retry_limit < 0 or self.retry_limit > 5:
            raise SceneCaptureError("retry_limit must be between zero and five")
        _require_positive_integer(self.queue_capacity, "queue_capacity")
        _require_positive_integer(self.max_parallel_cameras, "max_parallel_cameras")
        if self.max_parallel_cameras > 16:
            raise SceneCaptureError("max_parallel_cameras must be at most 16")

    def safe_payload(self) -> dict[str, int | float]:
        return {
            "duration_seconds": float(self.duration_seconds),
            "connect_timeout_seconds": float(self.connect_timeout_seconds),
            "read_timeout_seconds": float(self.read_timeout_seconds),
            "retry_limit": self.retry_limit,
            "initial_backoff_seconds": float(self.initial_backoff_seconds),
            "queue_capacity": self.queue_capacity,
            "max_parallel_cameras": self.max_parallel_cameras,
        }


@dataclass(frozen=True, slots=True)
class SceneRationalTimeBase:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.numerator, "time_base.numerator")
        _require_positive_integer(self.denominator, "time_base.denominator")

    def to_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class SceneStreamProfile:
    schema_version: str
    camera_id: str
    profile_version: str
    endpoint_environment_key: str
    endpoint_binding_id: str
    width_pixels: int
    height_pixels: int
    codec: str
    time_base: SceneRationalTimeBase
    crop: str | None
    rotation_degrees: int | None
    observed_at_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_CAPTURE_PROFILE_SCHEMA:
            raise SceneCaptureError("unsupported stream profile schema")
        _require_identifier(self.camera_id, "camera_id", scene_camera=True)
        _require_identifier(self.profile_version, "profile_version")
        if not _ENVIRONMENT_KEY.fullmatch(self.endpoint_environment_key):
            raise SceneCaptureError("endpoint environment key is invalid")
        _require_identifier(self.endpoint_binding_id, "endpoint_binding_id")
        _require_positive_integer(self.width_pixels, "width_pixels")
        _require_positive_integer(self.height_pixels, "height_pixels")
        if not isinstance(self.codec, str) or not self.codec.strip():
            raise SceneCaptureError("codec must be non-blank")
        if self.rotation_degrees not in {None, 0, 90, 180, 270}:
            raise SceneCaptureError("rotation_degrees is invalid")
        _parse_utc(self.observed_at_utc, "observed_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "profile_version": self.profile_version,
            "endpoint_environment_key": self.endpoint_environment_key,
            "endpoint_binding_id": self.endpoint_binding_id,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "codec": self.codec,
            "time_base": self.time_base.to_dict(),
            "crop": self.crop,
            "rotation_degrees": self.rotation_degrees,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True, slots=True, repr=False)
class ScenePreviewFrame:
    """An in-memory JPEG preview, never a retained capture artifact."""

    camera_id: str
    media_type: str
    content: bytes
    observed_at_utc: str
    source_pts: int | None
    source_time_base: str | None

    def __post_init__(self) -> None:
        _require_identifier(self.camera_id, "camera_id", scene_camera=True)
        if self.media_type != "image/jpeg":
            raise SceneCaptureError("preview media type must be image/jpeg")
        if not self.content or len(self.content) > 10 * 1024 * 1024:
            raise SceneCaptureError("preview content is outside the bounded size")
        _parse_utc(self.observed_at_utc, "observed_at_utc")
        if self.source_pts is not None and (
            not isinstance(self.source_pts, int) or isinstance(self.source_pts, bool)
        ):
            raise SceneCaptureError("preview source_pts must be an integer or null")


@dataclass(frozen=True, slots=True)
class SceneSourceFrame:
    frame_id: str
    camera_id: str
    profile_version: str
    source_pts: int | None
    source_time_base: SceneRationalTimeBase | None
    acquisition_monotonic_ns: int
    observed_at_utc: str

    def __post_init__(self) -> None:
        _require_identifier(self.frame_id, "frame_id")
        _require_identifier(self.camera_id, "camera_id", scene_camera=True)
        _require_identifier(self.profile_version, "profile_version")
        if (self.source_pts is None) != (self.source_time_base is None):
            raise SceneCaptureError("source PTS and time base must be present together")
        if self.source_pts is not None and (
            not isinstance(self.source_pts, int) or isinstance(self.source_pts, bool)
        ):
            raise SceneCaptureError("source_pts must be an integer or null")
        _require_non_negative_integer(self.acquisition_monotonic_ns, "acquisition_monotonic_ns")
        _parse_utc(self.observed_at_utc, "observed_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "profile_version": self.profile_version,
            "source_pts": self.source_pts,
            "source_time_base": (
                self.source_time_base.to_dict() if self.source_time_base is not None else None
            ),
            "acquisition_monotonic_ns": self.acquisition_monotonic_ns,
            "observed_at_utc": self.observed_at_utc,
        }


class SceneStorageMode(StrEnum):
    PACKET_PRESERVING_MP4 = "packet-preserving-mp4"
    DECODED_FRAME_FALLBACK = "decoded-frame-fallback"


@dataclass(frozen=True, slots=True)
class SceneCaptureArtifact:
    relative_path: str
    sha256: str
    byte_count: int
    storage_mode: SceneStorageMode
    interrupted: bool
    fallback_reason: str | None

    def __post_init__(self) -> None:
        _relative_capture_path(self.relative_path)
        if not _SHA256.fullmatch(self.sha256):
            raise SceneCaptureError("artifact sha256 is invalid")
        _require_positive_integer(self.byte_count, "artifact byte_count")
        if (
            self.storage_mode is SceneStorageMode.DECODED_FRAME_FALLBACK
            and not self.fallback_reason
        ):
            raise SceneCaptureError("decoded fallback requires a reason")
        if self.storage_mode is SceneStorageMode.PACKET_PRESERVING_MP4 and self.fallback_reason:
            raise SceneCaptureError("packet-preserving artifact cannot include fallback reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "storage_mode": self.storage_mode.value,
            "interrupted": self.interrupted,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class SceneReconnectEvent:
    attempt: int
    state: str
    monotonic_ns: int
    observed_at_utc: str
    failure_class: str | None

    def __post_init__(self) -> None:
        _require_positive_integer(self.attempt, "reconnect attempt")
        if self.state not in {"disconnected", "backoff", "reconnected", "exhausted"}:
            raise SceneCaptureError("reconnect state is invalid")
        _require_non_negative_integer(self.monotonic_ns, "reconnect monotonic_ns")
        _parse_utc(self.observed_at_utc, "reconnect observed_at_utc")

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "state": self.state,
            "monotonic_ns": self.monotonic_ns,
            "observed_at_utc": self.observed_at_utc,
            "failure_class": self.failure_class,
        }


class SceneCaptureStatus(StrEnum):
    CAPTURED = "captured"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SceneCameraCaptureResult:
    camera_id: str
    profile: SceneStreamProfile
    status: SceneCaptureStatus
    frames: tuple[SceneSourceFrame, ...]
    artifact: SceneCaptureArtifact | None
    reconnect_events: tuple[SceneReconnectEvent, ...]
    failure_class: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        _require_identifier(self.camera_id, "camera_id", scene_camera=True)
        if self.profile.camera_id != self.camera_id:
            raise SceneCaptureError("capture profile camera does not match result")
        if any(frame.camera_id != self.camera_id for frame in self.frames):
            raise SceneCaptureError("capture frame camera does not match result")
        if self.status is SceneCaptureStatus.CAPTURED and (
            not self.frames or self.artifact is None
        ):
            raise SceneCaptureError("captured result requires frames and artifact")
        if self.status is SceneCaptureStatus.FAILED and not self.failure_class:
            raise SceneCaptureError("failed result requires a failure class")

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "profile": self.profile.to_dict(),
            "status": self.status.value,
            "frames": [frame.to_dict() for frame in self.frames],
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "reconnect_events": [event.to_dict() for event in self.reconnect_events],
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True, slots=True)
class SceneCaptureSessionManifest:
    schema_version: str
    scene_id: str
    session_id: str
    created_at_utc: str
    acquisition_started_monotonic_ns: int
    acquisition_finished_monotonic_ns: int
    software_identity: str
    configuration_identity_sha256: str
    camera_roster: tuple[SceneCameraBinding, ...]
    results: tuple[SceneCameraCaptureResult, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_CAPTURE_SESSION_SCHEMA:
            raise SceneCaptureError("unsupported capture session schema")
        _require_identifier(self.scene_id, "scene_id", scene_camera=True)
        _require_identifier(self.session_id, "session_id")
        _parse_utc(self.created_at_utc, "created_at_utc")
        _require_non_negative_integer(self.acquisition_started_monotonic_ns, "session start")
        _require_non_negative_integer(self.acquisition_finished_monotonic_ns, "session finish")
        if self.acquisition_finished_monotonic_ns < self.acquisition_started_monotonic_ns:
            raise SceneCaptureError("session finish cannot precede start")
        if not isinstance(self.software_identity, str) or not self.software_identity.strip():
            raise SceneCaptureError("software identity must be non-blank")
        if not _SHA256.fullmatch(self.configuration_identity_sha256):
            raise SceneCaptureError("configuration identity is invalid")
        roster_ids = tuple(binding.camera_id for binding in self.camera_roster)
        result_ids = tuple(result.camera_id for result in self.results)
        if not roster_ids or len(roster_ids) != len(set(roster_ids)):
            raise SceneCaptureError("camera roster must contain unique camera IDs")
        if result_ids != roster_ids:
            raise SceneCaptureError("session results must match camera roster order")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "session_id": self.session_id,
            "created_at_utc": self.created_at_utc,
            "acquisition_started_monotonic_ns": self.acquisition_started_monotonic_ns,
            "acquisition_finished_monotonic_ns": self.acquisition_finished_monotonic_ns,
            "software_identity": self.software_identity,
            "configuration_identity_sha256": self.configuration_identity_sha256,
            "camera_roster": [
                {
                    "camera_id": binding.camera_id,
                    "endpoint_environment_key": binding.endpoint_environment_key,
                }
                for binding in self.camera_roster
            ],
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class SceneSelectedFrame:
    camera_id: str
    frame_id: str
    acquisition_monotonic_ns: int
    profile_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "acquisition_monotonic_ns": self.acquisition_monotonic_ns,
            "profile_version": self.profile_version,
        }


@dataclass(frozen=True, slots=True)
class ScenePairwiseSkew:
    camera_a: str
    camera_b: str
    skew_ns: int

    def to_dict(self) -> dict[str, object]:
        return {"camera_a": self.camera_a, "camera_b": self.camera_b, "skew_ns": self.skew_ns}


class SceneBundleStatus(StrEnum):
    COMPLETE = "complete-roster"
    PARTIAL = "partial"
    INCOMPATIBLE_PROFILES = "stale-or-incompatible-profiles"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SceneSelectedBundleManifest:
    schema_version: str
    scene_id: str
    bundle_id: str
    source_session_id: str
    status: SceneBundleStatus
    selected_frames: tuple[SceneSelectedFrame, ...]
    missing_camera_ids: tuple[str, ...]
    pairwise_skew: tuple[ScenePairwiseSkew, ...]
    overall_skew_ns: int | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_CAPTURE_BUNDLE_SCHEMA:
            raise SceneCaptureError("unsupported selected bundle schema")
        _require_identifier(self.scene_id, "scene_id", scene_camera=True)
        _require_identifier(self.bundle_id, "bundle_id")
        _require_identifier(self.source_session_id, "source_session_id")
        if self.status is SceneBundleStatus.COMPLETE and self.missing_camera_ids:
            raise SceneCaptureError("complete bundle cannot omit cameras")
        if self.status is SceneBundleStatus.REJECTED and not self.rejection_reason:
            raise SceneCaptureError("rejected bundle requires a reason")
        if self.overall_skew_ns is not None:
            _require_non_negative_integer(self.overall_skew_ns, "overall_skew_ns")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "bundle_id": self.bundle_id,
            "source_session_id": self.source_session_id,
            "status": self.status.value,
            "selected_frames": [frame.to_dict() for frame in self.selected_frames],
            "missing_camera_ids": list(self.missing_camera_ids),
            "pairwise_skew": [pair.to_dict() for pair in self.pairwise_skew],
            "overall_skew_ns": self.overall_skew_ns,
            "rejection_reason": self.rejection_reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SceneCaptureAdapter(Protocol):
    """A bounded local transport, intentionally independent of P03's office types."""

    def profile(
        self, endpoint: SceneRtspEndpoint, policy: SceneCapturePolicy
    ) -> SceneStreamProfile:
        """Read a credential-safe observed stream profile."""

    def capture(
        self,
        endpoint: SceneRtspEndpoint,
        profile: SceneStreamProfile,
        output_path: Path,
        relative_path: str,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SceneSourceFrame, ...], SceneCaptureArtifact]:
        """Persist one bounded stream capture below the authorized session directory."""

    def preview(
        self,
        endpoint: SceneRtspEndpoint,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> ScenePreviewFrame:
        """Decode one in-memory JPEG without persisting capture evidence."""

    def close(self) -> None:
        """Release transport-owned resources."""


class SceneCaptureRepository:
    """Immutable capture storage contained under one managed scene's artifact root."""

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.resolve()
        self.sessions_root = self.root / "captures" / "managed-scene" / "sessions"
        self.selection_path = self.sessions_root.parent / "current-bundle.json"
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def session_directory(self, session_id: str) -> Path:
        _require_identifier(session_id, "session_id")
        return self.sessions_root / session_id

    def create_session_directory(self, session_id: str) -> Path:
        directory = self.session_directory(session_id)
        try:
            directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise SceneCaptureError("session identity already exists and is immutable") from error
        return directory

    def write_session(self, manifest: SceneCaptureSessionManifest) -> Path:
        directory = self.session_directory(manifest.session_id)
        if not directory.is_dir():
            raise SceneCaptureError("session artifact directory does not exist")
        path = directory / "session.json"
        self._exclusive_write(path, manifest.to_json())
        return path

    def write_bundle(self, manifest: SceneSelectedBundleManifest) -> Path:
        directory = self.session_directory(manifest.source_session_id) / "bundles"
        directory.mkdir(exist_ok=True)
        path = directory / f"{manifest.bundle_id}.json"
        self._exclusive_write(path, manifest.to_json())
        self._write_current_bundle_selection(path, manifest)
        return path

    def list_bundle_paths(self) -> tuple[Path, ...]:
        """Return immutable bundle manifests in deterministic storage order."""

        return tuple(
            sorted(
                (
                    path.resolve()
                    for path in self.sessions_root.glob("*/bundles/*.json")
                    if path.is_file()
                ),
                key=lambda path: (path.stat().st_mtime_ns, str(path).lower()),
            )
        )

    def read_bundle(self, path: Path) -> SceneSelectedBundleManifest:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.sessions_root) or not resolved.is_file():
            raise SceneCaptureError("selected bundle path is outside capture storage")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SceneCaptureError("selected bundle manifest is unreadable") from error
        if not isinstance(payload, dict):
            raise SceneCaptureError("selected bundle manifest must be an object")
        return scene_selected_bundle_from_payload(payload)

    def current_bundle(self) -> tuple[Path, SceneSelectedBundleManifest, str] | None:
        """Resolve the current operator selection without falling back past a bad pointer.

        Older D099 captures predate the explicit pointer.  For those only, the most recently
        written immutable bundle is inferred.  Every new selection writes a hash-bound pointer.
        """

        if self.selection_path.is_file():
            try:
                payload = json.loads(self.selection_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SceneCaptureError("current bundle selection is unreadable") from error
            if not isinstance(payload, dict) or payload.get("schema_version") != (
                SCENE_CAPTURE_SELECTION_SCHEMA
            ):
                raise SceneCaptureError("current bundle selection has an unsupported schema")
            relative_path = payload.get("bundle_relative_path")
            expected_sha256 = payload.get("bundle_sha256")
            if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
                raise SceneCaptureError("current bundle selection is malformed")
            path = (self.root / Path(relative_path)).resolve()
            if not path.is_relative_to(self.sessions_root) or not path.is_file():
                raise SceneCaptureError("current bundle selection points to a missing artifact")
            if _sha256_file(path) != expected_sha256:
                raise SceneCaptureError("current bundle selection identity changed")
            bundle = self.read_bundle(path)
            if (
                payload.get("bundle_id") != bundle.bundle_id
                or payload.get("source_session_id") != bundle.source_session_id
            ):
                raise SceneCaptureError("current bundle selection identity is inconsistent")
            return path, bundle, "explicit-pointer"
        paths = self.list_bundle_paths()
        if not paths:
            return None
        path = paths[-1]
        return path, self.read_bundle(path), "inferred-pre-pointer"

    def select_existing_bundle(
        self, path: Path, expected_sha256: str
    ) -> SceneSelectedBundleManifest:
        """Move only the mutable current pointer to a verified immutable bundle."""

        resolved = path.resolve()
        if _sha256_file(resolved) != expected_sha256:
            raise SceneCaptureError("selected bundle identity changed")
        manifest = self.read_bundle(resolved)
        self._write_current_bundle_selection(resolved, manifest)
        return manifest

    def list_sessions(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                path.name
                for path in self.sessions_root.iterdir()
                if path.is_dir() and (path / "session.json").is_file()
            )
        )

    def read_session_payload(self, session_id: str) -> dict[str, object]:
        path = self.session_directory(session_id) / "session.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SceneCaptureError("cached session manifest is unreadable") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCENE_CAPTURE_SESSION_SCHEMA
        ):
            raise SceneCaptureError("cached session manifest has an unsupported schema")
        return payload

    def read_session(self, session_id: str) -> SceneCaptureSessionManifest:
        return scene_capture_session_from_payload(self.read_session_payload(session_id))

    def _write_current_bundle_selection(
        self, path: Path, manifest: SceneSelectedBundleManifest
    ) -> None:
        relative_path = path.resolve().relative_to(self.root).as_posix()
        payload = {
            "schema_version": SCENE_CAPTURE_SELECTION_SCHEMA,
            "selected_at_utc": _utc_now(),
            "bundle_id": manifest.bundle_id,
            "source_session_id": manifest.source_session_id,
            "bundle_relative_path": relative_path,
            "bundle_sha256": _sha256_file(path),
            "bundle_status": manifest.status.value,
        }
        temporary = self.selection_path.with_name(f".{self.selection_path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.selection_path)

    @staticmethod
    def _exclusive_write(path: Path, content: str) -> None:
        try:
            with path.open("x", encoding="utf-8") as destination:
                destination.write(content)
        except FileExistsError as error:
            raise SceneCaptureError("immutable manifest already exists") from error


class SceneCaptureService:
    """Scene-scoped bounded capture lifecycle with reloadable local endpoint configuration."""

    def __init__(
        self,
        endpoint_loader: SceneEndpointLoader,
        adapter: SceneCaptureAdapter,
        repository: SceneCaptureRepository,
        scene_id: str,
        software_identity: str = SCENE_CAPTURE_SOFTWARE_IDENTITY,
    ) -> None:
        _require_identifier(scene_id, "scene_id", scene_camera=True)
        if not isinstance(software_identity, str) or not software_identity.strip():
            raise SceneCaptureError("software_identity must be non-blank")
        self._endpoint_loader = endpoint_loader
        self._adapter = adapter
        self.repository = repository
        self.scene_id = scene_id
        self.software_identity = software_identity
        self._state_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._closed = False

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._endpoint_loader.camera_ids

    def health(self, policy: SceneCapturePolicy | None = None) -> dict[str, dict[str, object]]:
        self._require_open()
        effective_policy = policy or SceneCapturePolicy()
        resolutions = self._endpoint_loader.resolve()
        output: dict[str, dict[str, object]] = {}
        configured = [resolution for resolution in resolutions if resolution.endpoint is not None]
        with ThreadPoolExecutor(
            max_workers=min(effective_policy.max_parallel_cameras, max(1, len(configured))),
            thread_name_prefix="scene-capture-health",
        ) as executor:
            futures = {
                executor.submit(
                    self._safe_profile, resolution.endpoint, effective_policy
                ): resolution
                for resolution in configured
                if resolution.endpoint is not None
            }
            for future in as_completed(futures):
                resolution = futures[future]
                profile, failure = future.result()
                output[resolution.binding.camera_id] = {
                    "state": "healthy" if profile is not None else "failed",
                    "configuration_state": resolution.configuration_state,
                    "profile": profile.to_dict() if profile is not None else None,
                    "failure_class": failure,
                }
        for resolution in resolutions:
            if resolution.endpoint is None:
                output[resolution.binding.camera_id] = {
                    "state": "not_configured",
                    "configuration_state": resolution.configuration_state,
                    "profile": None,
                    "failure_class": "SceneCaptureConfigurationError",
                }
        return {camera_id: output[camera_id] for camera_id in self.camera_ids}

    def preview(
        self, camera_id: str, policy: SceneCapturePolicy | None = None
    ) -> ScenePreviewFrame:
        self._require_open()
        endpoint = self._endpoint_loader.endpoint_for(camera_id)
        cancel = threading.Event()
        return self._adapter.preview(
            endpoint,
            policy or SceneCapturePolicy(duration_seconds=1.0),
            cancel,
        )

    def capture_session(
        self, session_id: str, policy: SceneCapturePolicy | None = None
    ) -> SceneCaptureSessionManifest:
        self._require_open()
        effective_policy = policy or SceneCapturePolicy()
        _require_identifier(session_id, "session_id")
        if not self._capture_lock.acquire(blocking=False):
            raise SceneCaptureBusyError("a capture is already active for this scene")
        cancel = threading.Event()
        try:
            with self._state_lock:
                self._require_open()
                self._active_cancel = cancel
            session_directory = self.repository.create_session_directory(session_id)
            started = monotonic_ns()
            resolutions = self._endpoint_loader.resolve()
            profiles = self._preflight_profiles(resolutions, effective_policy)
            results = self._capture_all(
                resolutions,
                profiles,
                session_directory,
                effective_policy,
                cancel,
            )
            manifest = SceneCaptureSessionManifest(
                SCENE_CAPTURE_SESSION_SCHEMA,
                self.scene_id,
                session_id,
                _utc_now(),
                started,
                monotonic_ns(),
                self.software_identity,
                _configuration_identity(self.scene_id, resolutions, effective_policy),
                tuple(item.binding for item in resolutions),
                tuple(results[camera_id] for camera_id in self.camera_ids),
            )
            self.repository.write_session(manifest)
            return manifest
        finally:
            with self._state_lock:
                if self._active_cancel is cancel:
                    self._active_cancel = None
            self._capture_lock.release()

    def select_bundle(
        self,
        session: SceneCaptureSessionManifest,
        bundle_id: str,
        target_monotonic_ns: int | None = None,
    ) -> SceneSelectedBundleManifest:
        self._require_open()
        if session.scene_id != self.scene_id:
            raise SceneCaptureError("capture session belongs to another scene")
        bundle = select_scene_capture_bundle(session, bundle_id, target_monotonic_ns)
        self.repository.write_bundle(bundle)
        return bundle

    def cancel(self) -> bool:
        with self._state_lock:
            if self._active_cancel is None:
                return False
            self._active_cancel.set()
            return True

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            if self._active_cancel is not None:
                self._active_cancel.set()
        self._adapter.close()

    def _preflight_profiles(
        self,
        resolutions: Sequence[SceneEndpointResolution],
        policy: SceneCapturePolicy,
    ) -> dict[str, tuple[SceneStreamProfile | None, str | None]]:
        output: dict[str, tuple[SceneStreamProfile | None, str | None]] = {}
        configured = [resolution for resolution in resolutions if resolution.endpoint is not None]
        with ThreadPoolExecutor(
            max_workers=min(policy.max_parallel_cameras, max(1, len(configured))),
            thread_name_prefix="scene-capture-preflight",
        ) as executor:
            futures = {
                executor.submit(self._safe_profile, resolution.endpoint, policy): resolution
                for resolution in configured
                if resolution.endpoint is not None
            }
            for future in as_completed(futures):
                resolution = futures[future]
                output[resolution.binding.camera_id] = future.result()
        for resolution in resolutions:
            if resolution.endpoint is None:
                output[resolution.binding.camera_id] = (
                    None,
                    "SceneCaptureConfigurationError",
                )
        return output

    def _capture_all(
        self,
        resolutions: Sequence[SceneEndpointResolution],
        profiles: Mapping[str, tuple[SceneStreamProfile | None, str | None]],
        session_directory: Path,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> dict[str, SceneCameraCaptureResult]:
        output: dict[str, SceneCameraCaptureResult] = {}
        with ThreadPoolExecutor(
            max_workers=min(policy.max_parallel_cameras, len(resolutions)),
            thread_name_prefix="scene-capture",
        ) as executor:
            futures = {
                executor.submit(
                    self._capture_camera,
                    resolution,
                    profiles[resolution.binding.camera_id],
                    session_directory,
                    policy,
                    cancel,
                ): resolution.binding.camera_id
                for resolution in resolutions
            }
            for future in as_completed(futures):
                output[futures[future]] = future.result()
        return output

    def _capture_camera(
        self,
        resolution: SceneEndpointResolution,
        profile_result: tuple[SceneStreamProfile | None, str | None],
        session_directory: Path,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> SceneCameraCaptureResult:
        binding = resolution.binding
        profile, profile_failure = profile_result
        if resolution.endpoint is None or profile is None:
            return _failed_capture_result(
                binding,
                profile_failure or "SceneCaptureConfigurationError",
                _configuration_message(resolution.configuration_state)
                if resolution.endpoint is None
                else "stream profile could not be read",
            )
        events: list[SceneReconnectEvent] = []
        output_path = session_directory / f"{binding.camera_id}.mp4"
        relative_path = (
            f"captures/managed-scene/sessions/{session_directory.name}/{binding.camera_id}.mp4"
        )
        last_error: Exception | None = None
        for attempt in range(1, policy.retry_limit + 2):
            if cancel.is_set():
                last_error = SceneCaptureCancelledError("capture cancelled")
                break
            try:
                frames, artifact = self._adapter.capture(
                    resolution.endpoint,
                    profile,
                    output_path,
                    relative_path,
                    policy,
                    cancel,
                )
                if attempt > 1:
                    events.append(_reconnect_event(attempt, "reconnected", None))
                return SceneCameraCaptureResult(
                    binding.camera_id,
                    profile,
                    SceneCaptureStatus.CAPTURED,
                    frames,
                    artifact,
                    tuple(events),
                    None,
                    None,
                )
            except Exception as error:
                last_error = error
                events.append(_reconnect_event(attempt, "disconnected", type(error).__name__))
                if attempt > policy.retry_limit:
                    events.append(_reconnect_event(attempt, "exhausted", type(error).__name__))
                    break
                events.append(_reconnect_event(attempt, "backoff", type(error).__name__))
                if cancel.wait(policy.initial_backoff_seconds * (2 ** (attempt - 1))):
                    last_error = SceneCaptureCancelledError("capture cancelled")
                    break
        return SceneCameraCaptureResult(
            binding.camera_id,
            profile,
            SceneCaptureStatus.FAILED,
            (),
            None,
            tuple(events),
            type(last_error).__name__ if last_error is not None else "SceneCaptureAdapterError",
            "bounded capture failed",
        )

    def _safe_profile(
        self,
        endpoint: SceneRtspEndpoint,
        policy: SceneCapturePolicy,
    ) -> tuple[SceneStreamProfile | None, str | None]:
        try:
            return self._adapter.profile(endpoint, policy), None
        except Exception as error:
            return None, type(error).__name__

    def _require_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise SceneCaptureError("managed-scene capture service is closed")


def _failed_capture_result(
    binding: SceneCameraBinding,
    failure_class: str,
    message: str,
) -> SceneCameraCaptureResult:
    profile = SceneStreamProfile(
        SCENE_CAPTURE_PROFILE_SCHEMA,
        binding.camera_id,
        "scene-stream-profile-v1",
        binding.endpoint_environment_key,
        f"unobserved-{binding.camera_id}",
        1,
        1,
        "unobserved",
        SceneRationalTimeBase(1, 1),
        None,
        None,
        _utc_now(),
    )
    return SceneCameraCaptureResult(
        binding.camera_id,
        profile,
        SceneCaptureStatus.FAILED,
        (),
        None,
        (),
        failure_class,
        message,
    )


def _configuration_message(state: str) -> str:
    return {
        "missing": "camera endpoint is not configured",
        "invalid": "camera endpoint configuration is invalid",
        "unavailable": "camera endpoint configuration cannot be read",
        "configured": "camera endpoint is configured",
    }[state]


def _reconnect_event(attempt: int, state: str, failure_class: str | None) -> SceneReconnectEvent:
    return SceneReconnectEvent(attempt, state, monotonic_ns(), _utc_now(), failure_class)


def _configuration_identity(
    scene_id: str,
    resolutions: Sequence[SceneEndpointResolution],
    policy: SceneCapturePolicy,
) -> str:
    payload = {
        "scene_id": scene_id,
        "camera_bindings": [
            {
                "camera_id": item.binding.camera_id,
                "endpoint_environment_key": item.binding.endpoint_environment_key,
            }
            for item in resolutions
        ],
        "policy": policy.safe_payload(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def select_scene_capture_bundle(
    session: SceneCaptureSessionManifest,
    bundle_id: str,
    target_monotonic_ns: int | None = None,
) -> SceneSelectedBundleManifest:
    """Choose one closest frame per captured camera without consulting current secrets."""

    _require_identifier(bundle_id, "bundle_id")
    if target_monotonic_ns is not None:
        _require_non_negative_integer(target_monotonic_ns, "target_monotonic_ns")
    usable = [
        result
        for result in session.results
        if (
            result.status is SceneCaptureStatus.CAPTURED
            and result.frames
            and result.artifact is not None
        )
    ]
    roster_ids = tuple(binding.camera_id for binding in session.camera_roster)
    missing = tuple(
        camera_id
        for camera_id in roster_ids
        if not any(result.camera_id == camera_id for result in usable)
    )
    if not usable:
        return SceneSelectedBundleManifest(
            SCENE_CAPTURE_BUNDLE_SCHEMA,
            session.scene_id,
            bundle_id,
            session.session_id,
            SceneBundleStatus.REJECTED,
            (),
            roster_ids,
            (),
            None,
            "session has no usable captured frames",
        )
    expected_versions = {result.camera_id: result.profile.profile_version for result in usable}
    if any(
        frame.profile_version != expected_versions[result.camera_id]
        for result in usable
        for frame in result.frames
    ):
        return SceneSelectedBundleManifest(
            SCENE_CAPTURE_BUNDLE_SCHEMA,
            session.scene_id,
            bundle_id,
            session.session_id,
            SceneBundleStatus.INCOMPATIBLE_PROFILES,
            (),
            missing,
            (),
            None,
            "frame profile does not match immutable session profile",
        )
    candidates = [frame for result in usable for frame in result.frames]
    target = target_monotonic_ns
    if target is None:
        ordered_times = sorted(frame.acquisition_monotonic_ns for frame in candidates)
        target = ordered_times[(len(ordered_times) - 1) // 2]
    selected: list[SceneSelectedFrame] = []
    usable_by_id = {result.camera_id: result for result in usable}
    for camera_id in roster_ids:
        result = usable_by_id.get(camera_id)
        if result is None:
            continue
        frame = min(
            result.frames,
            key=lambda item: (abs(item.acquisition_monotonic_ns - target), item.frame_id),
        )
        selected.append(
            SceneSelectedFrame(
                result.camera_id,
                frame.frame_id,
                frame.acquisition_monotonic_ns,
                frame.profile_version,
            )
        )
    pairwise: list[ScenePairwiseSkew] = []
    for index, first in enumerate(selected):
        for second in selected[index + 1 :]:
            pairwise.append(
                ScenePairwiseSkew(
                    first.camera_id,
                    second.camera_id,
                    abs(first.acquisition_monotonic_ns - second.acquisition_monotonic_ns),
                )
            )
    times = [frame.acquisition_monotonic_ns for frame in selected]
    return SceneSelectedBundleManifest(
        SCENE_CAPTURE_BUNDLE_SCHEMA,
        session.scene_id,
        bundle_id,
        session.session_id,
        SceneBundleStatus.COMPLETE if not missing else SceneBundleStatus.PARTIAL,
        tuple(selected),
        missing,
        tuple(pairwise),
        max(times) - min(times),
        None,
    )


class PyAvSceneCaptureAdapter:
    """A read-only, bounded PyAV transport for a variable managed-scene roster."""

    def __init__(self) -> None:
        self._closed = threading.Event()

    def profile(
        self, endpoint: SceneRtspEndpoint, policy: SceneCapturePolicy
    ) -> SceneStreamProfile:
        av = _av()
        try:
            with av.open(
                endpoint.for_read_only_adapter(),
                mode="r",
                options={"rtsp_transport": "tcp"},
                timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
            ) as container:
                stream = next(iter(container.streams.video), None)
                if stream is None:
                    raise SceneCaptureAdapterError("source has no video stream")
                time_base = stream.time_base
                if time_base is None:
                    raise SceneCaptureAdapterError("source video time base is unavailable")
                rotation = stream.metadata.get("rotate")
                binding = endpoint.binding
                return SceneStreamProfile(
                    SCENE_CAPTURE_PROFILE_SCHEMA,
                    binding.camera_id,
                    "scene-stream-profile-v1",
                    binding.endpoint_environment_key,
                    f"local-{binding.endpoint_environment_key.lower()}",
                    stream.codec_context.width,
                    stream.codec_context.height,
                    stream.codec_context.name,
                    SceneRationalTimeBase(time_base.numerator, time_base.denominator),
                    None,
                    int(rotation) if rotation and rotation.isdigit() else None,
                    _utc_now(),
                )
        except SceneCaptureAdapterError:
            raise
        except Exception as error:
            raise SceneCaptureConnectTimeoutError(
                "bounded RTSP profile connection failed"
            ) from error

    def capture(
        self,
        endpoint: SceneRtspEndpoint,
        profile: SceneStreamProfile,
        output_path: Path,
        relative_path: str,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SceneSourceFrame, ...], SceneCaptureArtifact]:
        try:
            return self._remux(endpoint, profile, output_path, relative_path, policy, cancel)
        except (SceneCaptureCancelledError, SceneCaptureReadTimeoutError):
            raise
        except Exception as remux_error:
            fallback_path = output_path.with_suffix(".jpg")
            fallback_relative = str(Path(relative_path).with_suffix(".jpg")).replace("\\", "/")
            return self._decoded_fallback(
                endpoint,
                profile,
                fallback_path,
                fallback_relative,
                policy,
                cancel,
                type(remux_error).__name__,
            )

    def preview(
        self,
        endpoint: SceneRtspEndpoint,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> ScenePreviewFrame:
        av = _av()
        try:
            with av.open(
                endpoint.for_read_only_adapter(),
                mode="r",
                options={"rtsp_transport": "tcp"},
                timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
            ) as source:
                stream = next(iter(source.streams.video), None)
                if stream is None:
                    raise SceneCaptureAdapterError("source has no video stream")
                if cancel.is_set() or self._closed.is_set():
                    raise SceneCaptureCancelledError("preview cancelled")
                frame = next(iter(source.decode(stream)), None)
                if frame is None:
                    raise SceneCaptureReadTimeoutError("source produced no preview frame")
                destination = BytesIO()
                frame.to_image().save(destination, format="JPEG", quality=80)
                return ScenePreviewFrame(
                    endpoint.binding.camera_id,
                    "image/jpeg",
                    destination.getvalue(),
                    _utc_now(),
                    frame.pts,
                    str(stream.time_base) if stream.time_base is not None else None,
                )
        except SceneCaptureAdapterError:
            raise
        except Exception as error:
            raise SceneCaptureReadTimeoutError("bounded RTSP preview failed") from error

    def _remux(
        self,
        endpoint: SceneRtspEndpoint,
        profile: SceneStreamProfile,
        output_path: Path,
        relative_path: str,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SceneSourceFrame, ...], SceneCaptureArtifact]:
        av = _av()
        frames: list[SceneSourceFrame] = []
        try:
            with av.open(
                endpoint.for_read_only_adapter(),
                mode="r",
                options={"rtsp_transport": "tcp"},
                timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
            ) as source:
                stream = next(iter(source.streams.video), None)
                if stream is None:
                    raise SceneCaptureAdapterError("source has no video stream")
                # The bounded PyAV open/read timeouts own connection and stall handling.
                # Start the requested evidence duration only after the stream is open so
                # connection scheduling pressure cannot consume the entire capture window.
                started = monotonic()
                with av.open(str(output_path), mode="w") as destination:
                    output_stream = destination.add_stream_from_template(stream)
                    for packet in source.demux(stream):
                        if cancel.is_set() or self._closed.is_set():
                            raise SceneCaptureCancelledError("capture cancelled")
                        elapsed = monotonic() - started
                        if elapsed >= policy.duration_seconds:
                            break
                        if packet.dts is None:
                            continue
                        frames.append(
                            SceneSourceFrame(
                                f"{endpoint.binding.camera_id}-packet-{len(frames):06d}",
                                endpoint.binding.camera_id,
                                profile.profile_version,
                                packet.pts,
                                profile.time_base if packet.pts is not None else None,
                                monotonic_ns(),
                                _utc_now(),
                            )
                        )
                        packet.stream = output_stream
                        destination.mux(packet)
                        if len(frames) > policy.queue_capacity:
                            raise SceneCaptureBackpressureError(
                                "bounded capture frame limit exceeded"
                            )
        except SceneCaptureAdapterError:
            raise
        except Exception as error:
            raise SceneCaptureAdapterError("bounded RTSP remux failed") from error
        if not frames or not output_path.is_file() or output_path.stat().st_size <= 0:
            raise SceneCaptureAdapterError("packet-preserving capture produced no media")
        return tuple(frames), SceneCaptureArtifact(
            relative_path,
            _sha256_file(output_path),
            output_path.stat().st_size,
            SceneStorageMode.PACKET_PRESERVING_MP4,
            False,
            None,
        )

    def _decoded_fallback(
        self,
        endpoint: SceneRtspEndpoint,
        profile: SceneStreamProfile,
        output_path: Path,
        relative_path: str,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
        reason: str,
    ) -> tuple[tuple[SceneSourceFrame, ...], SceneCaptureArtifact]:
        av = _av()
        try:
            with av.open(
                endpoint.for_read_only_adapter(),
                mode="r",
                options={"rtsp_transport": "tcp"},
                timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
            ) as source:
                stream = next(iter(source.streams.video), None)
                if stream is None:
                    raise SceneCaptureAdapterError("source has no video stream")
                if cancel.is_set() or self._closed.is_set():
                    raise SceneCaptureCancelledError("capture cancelled")
                frame = next(iter(source.decode(stream)), None)
                if frame is None:
                    raise SceneCaptureReadTimeoutError("source produced no decoded fallback frame")
                frame.to_image().save(output_path, format="JPEG", quality=95)
                source_frame = SceneSourceFrame(
                    f"{endpoint.binding.camera_id}-fallback-000000",
                    endpoint.binding.camera_id,
                    profile.profile_version,
                    frame.pts,
                    profile.time_base if frame.pts is not None else None,
                    monotonic_ns(),
                    _utc_now(),
                )
        except SceneCaptureAdapterError:
            raise
        except Exception as error:
            raise SceneCaptureReadTimeoutError("bounded RTSP decoded fallback failed") from error
        return (source_frame,), SceneCaptureArtifact(
            relative_path,
            _sha256_file(output_path),
            output_path.stat().st_size,
            SceneStorageMode.DECODED_FRAME_FALLBACK,
            False,
            reason,
        )

    def close(self) -> None:
        self._closed.set()


def _av() -> Any:
    try:
        return import_module("av")
    except ImportError as error:
        raise SceneCaptureAdapterError("PyAV runtime dependency is unavailable") from error


def scene_capture_session_from_payload(
    payload: Mapping[str, object],
) -> SceneCaptureSessionManifest:
    """Revalidate a persisted managed-scene session before bundle selection."""

    roster_raw = _object_list(payload, "camera_roster")
    roster = tuple(
        SceneCameraBinding(_string(item, "camera_id"), _string(item, "endpoint_environment_key"))
        for item in roster_raw
    )
    results = tuple(
        _capture_result_from_payload(item) for item in _object_list(payload, "results")
    )
    return SceneCaptureSessionManifest(
        _string(payload, "schema_version"),
        _string(payload, "scene_id"),
        _string(payload, "session_id"),
        _string(payload, "created_at_utc"),
        _integer(payload, "acquisition_started_monotonic_ns"),
        _integer(payload, "acquisition_finished_monotonic_ns"),
        _string(payload, "software_identity"),
        _string(payload, "configuration_identity_sha256"),
        roster,
        results,
    )


def scene_selected_bundle_from_payload(
    payload: Mapping[str, object],
) -> SceneSelectedBundleManifest:
    """Revalidate a persisted selected-bundle manifest at the storage boundary."""

    try:
        status = SceneBundleStatus(_string(payload, "status"))
    except ValueError as error:
        raise SceneCaptureError("selected bundle status is invalid") from error
    selected_frames = tuple(
        SceneSelectedFrame(
            _string(item, "camera_id"),
            _string(item, "frame_id"),
            _integer(item, "acquisition_monotonic_ns"),
            _string(item, "profile_version"),
        )
        for item in _object_list(payload, "selected_frames")
    )
    pairwise_skew = tuple(
        ScenePairwiseSkew(
            _string(item, "camera_a"),
            _string(item, "camera_b"),
            _integer(item, "skew_ns"),
        )
        for item in _object_list(payload, "pairwise_skew")
    )
    overall_skew = payload.get("overall_skew_ns")
    if overall_skew is not None and (
        not isinstance(overall_skew, int) or isinstance(overall_skew, bool)
    ):
        raise SceneCaptureError("overall_skew_ns must be an integer or null")
    missing = payload.get("missing_camera_ids")
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise SceneCaptureError("missing_camera_ids must be an array of strings")
    return SceneSelectedBundleManifest(
        _string(payload, "schema_version"),
        _string(payload, "scene_id"),
        _string(payload, "bundle_id"),
        _string(payload, "source_session_id"),
        status,
        selected_frames,
        tuple(missing),
        pairwise_skew,
        overall_skew,
        _optional_string(payload, "rejection_reason"),
    )


def _capture_result_from_payload(payload: Mapping[str, object]) -> SceneCameraCaptureResult:
    profile = _profile_from_payload(_object(payload, "profile"))
    frames = tuple(_source_frame_from_payload(item) for item in _object_list(payload, "frames"))
    artifact_value = payload.get("artifact")
    artifact = (
        None
        if artifact_value is None
        else _artifact_from_payload(_mapping(artifact_value, "artifact"))
    )
    events = tuple(_event_from_payload(item) for item in _object_list(payload, "reconnect_events"))
    failure_class = _optional_string(payload, "failure_class")
    failure_message = _optional_string(payload, "failure_message")
    try:
        status = SceneCaptureStatus(_string(payload, "status"))
    except ValueError as error:
        raise SceneCaptureError("capture status is invalid") from error
    return SceneCameraCaptureResult(
        _string(payload, "camera_id"),
        profile,
        status,
        frames,
        artifact,
        events,
        failure_class,
        failure_message,
    )


def _profile_from_payload(payload: Mapping[str, object]) -> SceneStreamProfile:
    time_base = _object(payload, "time_base")
    rotation_value = payload.get("rotation_degrees")
    if rotation_value is not None and (
        not isinstance(rotation_value, int) or isinstance(rotation_value, bool)
    ):
        raise SceneCaptureError("rotation_degrees must be an integer or null")
    return SceneStreamProfile(
        _string(payload, "schema_version"),
        _string(payload, "camera_id"),
        _string(payload, "profile_version"),
        _string(payload, "endpoint_environment_key"),
        _string(payload, "endpoint_binding_id"),
        _integer(payload, "width_pixels"),
        _integer(payload, "height_pixels"),
        _string(payload, "codec"),
        SceneRationalTimeBase(
            _integer(time_base, "numerator"), _integer(time_base, "denominator")
        ),
        _optional_string(payload, "crop"),
        rotation_value,
        _string(payload, "observed_at_utc"),
    )


def _source_frame_from_payload(payload: Mapping[str, object]) -> SceneSourceFrame:
    time_base_value = payload.get("source_time_base")
    time_base = (
        None
        if time_base_value is None
        else SceneRationalTimeBase(
            _integer(_mapping(time_base_value, "source_time_base"), "numerator"),
            _integer(_mapping(time_base_value, "source_time_base"), "denominator"),
        )
    )
    source_pts = payload.get("source_pts")
    if source_pts is not None and (
        not isinstance(source_pts, int) or isinstance(source_pts, bool)
    ):
        raise SceneCaptureError("source_pts must be an integer or null")
    return SceneSourceFrame(
        _string(payload, "frame_id"),
        _string(payload, "camera_id"),
        _string(payload, "profile_version"),
        source_pts,
        time_base,
        _integer(payload, "acquisition_monotonic_ns"),
        _string(payload, "observed_at_utc"),
    )


def _artifact_from_payload(payload: Mapping[str, object]) -> SceneCaptureArtifact:
    try:
        storage_mode = SceneStorageMode(_string(payload, "storage_mode"))
    except ValueError as error:
        raise SceneCaptureError("storage_mode is invalid") from error
    interrupted = payload.get("interrupted")
    if not isinstance(interrupted, bool):
        raise SceneCaptureError("artifact interrupted must be boolean")
    return SceneCaptureArtifact(
        _string(payload, "relative_path"),
        _string(payload, "sha256"),
        _integer(payload, "byte_count"),
        storage_mode,
        interrupted,
        _optional_string(payload, "fallback_reason"),
    )


def _event_from_payload(payload: Mapping[str, object]) -> SceneReconnectEvent:
    failure_class = _optional_string(payload, "failure_class")
    return SceneReconnectEvent(
        _integer(payload, "attempt"),
        _string(payload, "state"),
        _integer(payload, "monotonic_ns"),
        _string(payload, "observed_at_utc"),
        failure_class,
    )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SceneCaptureError(f"{field} must be an object")
    return value


def _object(payload: Mapping[str, object], field: str) -> Mapping[str, object]:
    return _mapping(payload.get(field), field)


def _object_list(payload: Mapping[str, object], field: str) -> list[Mapping[str, object]]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise SceneCaptureError(f"{field} must be an array")
    return [_mapping(item, field) for item in value]


def _string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise SceneCaptureError(f"{field} must be a string")
    return value


def _optional_string(payload: Mapping[str, object], field: str) -> str | None:
    value = payload.get(field)
    if value is not None and not isinstance(value, str):
        raise SceneCaptureError(f"{field} must be a string or null")
    return value


def _integer(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SceneCaptureError(f"{field} must be an integer")
    return value


def create_scene_capture_app(
    camera_bindings: Mapping[str, str],
    secret_file: Path,
    artifact_root: Path,
    scene_id: str,
    *,
    adapter: SceneCaptureAdapter | None = None,
) -> FastAPI:
    """Build the scene-scoped FastAPI capture tool with a reloadable secret file.

    The delayed import avoids making the service/domain layer depend on FastAPI at
    import time.  The returned application exposes its service as
    ``app.state.scene_capture_service`` so its owning scene runtime can close it.
    """

    from spatial_mapping_phase2.managed_scene_capture_app import (
        create_scene_capture_app as factory,
    )

    return factory(camera_bindings, secret_file, artifact_root, scene_id, adapter=adapter)
