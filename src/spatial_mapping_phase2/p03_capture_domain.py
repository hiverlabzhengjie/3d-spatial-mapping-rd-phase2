"""P03 immutable capture provenance and deterministic static-bundle selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from spatial_mapping_phase2.p01_observability import CAMERA_ENDPOINT_KEYS, CAMERA_IDS

PROFILE_SCHEMA = "p03-stream-profile-v1"
SESSION_SCHEMA = "p03-capture-session-v1"
BUNDLE_SCHEMA = "p03-selected-bundle-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_VERSION = re.compile(r"^stream-profile-v[1-9][0-9]*$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class P03ContractError(ValueError):
    """A safe P03 boundary-validation error."""


class CaptureStatus(StrEnum):
    CAPTURED = "captured"
    PARTIAL = "partial"
    FAILED = "failed"
    REJECTED = "rejected"


class StorageMode(StrEnum):
    PACKET_PRESERVING_MP4 = "packet-preserving-mp4"
    DECODED_FRAME_FALLBACK = "decoded-frame-fallback"


class BundleStatus(StrEnum):
    COMPLETE = "complete-four-camera"
    PARTIAL = "partial"
    INCOMPATIBLE_PROFILES = "stale-or-incompatible-profiles"
    REJECTED = "rejected"


def parse_utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise P03ContractError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise P03ContractError(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise P03ContractError(f"{field} must be a positive integer")


def _finite(value: float, field: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise P03ContractError(f"{field} must be finite")


def _identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise P03ContractError(f"{field} is invalid")


def _camera(camera_id: str) -> None:
    if camera_id not in CAMERA_IDS:
        raise P03ContractError("camera_id must be a fixed office camera ID")


def _relative_artifact(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "captures":
        raise P03ContractError("artifact path must be relative below captures")


@dataclass(frozen=True, slots=True)
class RationalTimeBase:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _positive(self.numerator, "time_base.numerator")
        _positive(self.denominator, "time_base.denominator")

    def seconds_for_pts(self, pts: int) -> float:
        if not isinstance(pts, int) or isinstance(pts, bool):
            raise P03ContractError("pts must be an integer")
        return float(Fraction(pts * self.numerator, self.denominator))


@dataclass(frozen=True, slots=True)
class StreamProfileIdentity:
    schema_version: str
    camera_id: str
    profile_version: str
    endpoint_environment_key: str
    endpoint_binding_id: str
    width_pixels: int
    height_pixels: int
    codec: str
    time_base: RationalTimeBase
    crop: str | None
    rotation_degrees: int | None
    observed_at_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA:
            raise P03ContractError("unsupported stream-profile schema")
        _camera(self.camera_id)
        if not _PROFILE_VERSION.fullmatch(self.profile_version):
            raise P03ContractError("profile_version must use stream-profile-vN")
        if self.endpoint_environment_key != CAMERA_ENDPOINT_KEYS[self.camera_id]:
            raise P03ContractError("endpoint key does not match camera_id")
        _identifier(self.endpoint_binding_id, "endpoint_binding_id")
        _positive(self.width_pixels, "width_pixels")
        _positive(self.height_pixels, "height_pixels")
        if not self.codec.strip():
            raise P03ContractError("codec must be non-blank")
        if self.rotation_degrees not in {None, 0, 90, 180, 270}:
            raise P03ContractError("rotation must be null, 0, 90, 180 or 270")
        parse_utc(self.observed_at_utc, "observed_at_utc")

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.camera_id,
            self.profile_version,
            self.endpoint_binding_id,
            self.width_pixels,
            self.height_pixels,
            self.codec.lower(),
            self.time_base,
            self.crop,
            self.rotation_degrees,
        )


@dataclass(frozen=True, slots=True)
class SourceFrame:
    frame_id: str
    camera_id: str
    profile_version: str
    source_pts: int | None
    source_time_base: RationalTimeBase | None
    acquisition_monotonic_ns: int
    observed_at_utc: str
    processing_started_monotonic_ns: int | None = None
    model_completed_at_utc: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.frame_id, "frame_id")
        _camera(self.camera_id)
        if not _PROFILE_VERSION.fullmatch(self.profile_version):
            raise P03ContractError("invalid frame profile_version")
        if (self.source_pts is None) != (self.source_time_base is None):
            raise P03ContractError("source PTS and time base must be present together")
        if not isinstance(self.acquisition_monotonic_ns, int) or self.acquisition_monotonic_ns < 0:
            raise P03ContractError("acquisition_monotonic_ns must be non-negative")
        parse_utc(self.observed_at_utc, "observed_at_utc")
        if self.processing_started_monotonic_ns is not None:
            if self.processing_started_monotonic_ns < self.acquisition_monotonic_ns:
                raise P03ContractError("processing cannot precede acquisition")
        if self.model_completed_at_utc is not None:
            parse_utc(self.model_completed_at_utc, "model_completed_at_utc")


@dataclass(frozen=True, slots=True)
class ReconnectEvent:
    attempt: int
    state: str
    monotonic_ns: int
    observed_at_utc: str
    failure_class: str | None

    def __post_init__(self) -> None:
        _positive(self.attempt, "reconnect attempt")
        if self.state not in {"disconnected", "backoff", "reconnected", "exhausted"}:
            raise P03ContractError("invalid reconnect state")
        if self.monotonic_ns < 0:
            raise P03ContractError("reconnect monotonic time must be non-negative")
        parse_utc(self.observed_at_utc, "reconnect observed_at_utc")


@dataclass(frozen=True, slots=True)
class CaptureArtifact:
    relative_path: str
    sha256: str
    byte_count: int
    storage_mode: StorageMode
    interrupted: bool
    fallback_reason: str | None

    def __post_init__(self) -> None:
        _relative_artifact(self.relative_path)
        if not _SHA256.fullmatch(self.sha256):
            raise P03ContractError("artifact sha256 must be lowercase SHA-256")
        if self.byte_count <= 0:
            raise P03ContractError("artifact byte_count must be positive")
        if self.storage_mode is StorageMode.DECODED_FRAME_FALLBACK and not self.fallback_reason:
            raise P03ContractError("decoded fallback requires a manifested reason")
        if self.storage_mode is StorageMode.PACKET_PRESERVING_MP4 and self.fallback_reason:
            raise P03ContractError("packet-preserving media cannot have a fallback reason")


@dataclass(frozen=True, slots=True)
class CameraCaptureResult:
    camera_id: str
    profile: StreamProfileIdentity
    status: CaptureStatus
    frames: tuple[SourceFrame, ...]
    artifact: CaptureArtifact | None
    reconnect_events: tuple[ReconnectEvent, ...]
    failure_class: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        _camera(self.camera_id)
        if self.profile.camera_id != self.camera_id:
            raise P03ContractError("capture profile camera mismatch")
        if any(frame.camera_id != self.camera_id for frame in self.frames):
            raise P03ContractError("capture contains a frame from another camera")
        if self.status is CaptureStatus.CAPTURED and (not self.frames or self.artifact is None):
            raise P03ContractError("captured result requires frames and artifact")
        if (
            self.status in {CaptureStatus.FAILED, CaptureStatus.REJECTED}
            and not self.failure_class
        ):
            raise P03ContractError("failed/rejected result requires failure_class")


@dataclass(frozen=True, slots=True)
class CaptureSessionManifest:
    schema_version: str
    session_id: str
    created_at_utc: str
    acquisition_started_monotonic_ns: int
    acquisition_finished_monotonic_ns: int
    software_identity: str
    configuration_identity_sha256: str
    results: tuple[CameraCaptureResult, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SESSION_SCHEMA:
            raise P03ContractError("unsupported capture-session schema")
        _identifier(self.session_id, "session_id")
        parse_utc(self.created_at_utc, "created_at_utc")
        if self.acquisition_started_monotonic_ns < 0:
            raise P03ContractError("session start monotonic time must be non-negative")
        if self.acquisition_finished_monotonic_ns < self.acquisition_started_monotonic_ns:
            raise P03ContractError("session finish cannot precede start")
        if not self.software_identity.strip():
            raise P03ContractError("software_identity must be non-blank")
        if not _SHA256.fullmatch(self.configuration_identity_sha256):
            raise P03ContractError("configuration identity must be SHA-256")
        ids = tuple(result.camera_id for result in self.results)
        if len(ids) != len(set(ids)) or any(camera_id not in CAMERA_IDS for camera_id in ids):
            raise P03ContractError("session results must have unique fixed camera IDs")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    camera_id: str
    frame_id: str
    acquisition_monotonic_ns: int
    profile_version: str


@dataclass(frozen=True, slots=True)
class PairwiseSkew:
    camera_a: str
    camera_b: str
    skew_ns: int


@dataclass(frozen=True, slots=True)
class SelectedBundleManifest:
    schema_version: str
    bundle_id: str
    source_session_id: str
    status: BundleStatus
    selected_frames: tuple[SelectedFrame, ...]
    missing_camera_ids: tuple[str, ...]
    pairwise_skew: tuple[PairwiseSkew, ...]
    overall_skew_ns: int | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_version != BUNDLE_SCHEMA:
            raise P03ContractError("unsupported selected-bundle schema")
        _identifier(self.bundle_id, "bundle_id")
        _identifier(self.source_session_id, "source_session_id")
        if self.status is BundleStatus.COMPLETE and self.missing_camera_ids:
            raise P03ContractError("complete bundle cannot have missing cameras")
        if self.status is BundleStatus.REJECTED and not self.rejection_reason:
            raise P03ContractError("rejected bundle requires a reason")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def select_closest_bundle(
    session: CaptureSessionManifest,
    bundle_id: str,
    target_monotonic_ns: int | None = None,
) -> SelectedBundleManifest:
    """Select each camera's closest compatible frame; ties use frame ID deterministically."""

    usable = [
        result for result in session.results if result.frames and result.artifact is not None
    ]
    missing = tuple(
        camera_id
        for camera_id in CAMERA_IDS
        if not any(result.camera_id == camera_id for result in usable)
    )
    if not usable:
        return SelectedBundleManifest(
            BUNDLE_SCHEMA,
            bundle_id,
            session.session_id,
            BundleStatus.REJECTED,
            (),
            CAMERA_IDS,
            (),
            None,
            "session has no usable captured frames",
        )
    profile_versions = {result.camera_id: result.profile.profile_version for result in usable}
    if any(
        frame.profile_version != profile_versions[result.camera_id]
        for result in usable
        for frame in result.frames
    ):
        return SelectedBundleManifest(
            BUNDLE_SCHEMA,
            bundle_id,
            session.session_id,
            BundleStatus.INCOMPATIBLE_PROFILES,
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
    selected = []
    for result in sorted(usable, key=lambda item: item.camera_id):
        frame = min(
            result.frames,
            key=lambda item: (abs(item.acquisition_monotonic_ns - target), item.frame_id),
        )
        selected.append(
            SelectedFrame(
                result.camera_id,
                frame.frame_id,
                frame.acquisition_monotonic_ns,
                frame.profile_version,
            )
        )
    pairwise = []
    for index, first in enumerate(selected):
        for second in selected[index + 1 :]:
            pairwise.append(
                PairwiseSkew(
                    first.camera_id,
                    second.camera_id,
                    abs(first.acquisition_monotonic_ns - second.acquisition_monotonic_ns),
                )
            )
    times = [item.acquisition_monotonic_ns for item in selected]
    status = BundleStatus.COMPLETE if not missing else BundleStatus.PARTIAL
    return SelectedBundleManifest(
        BUNDLE_SCHEMA,
        bundle_id,
        session.session_id,
        status,
        tuple(selected),
        missing,
        tuple(pairwise),
        max(times) - min(times),
        None,
    )


def manifest_from_json(text: str) -> dict[str, Any]:
    """Decode a JSON object for repository boundary validation."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise P03ContractError("manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise P03ContractError("manifest root must be an object")
    return payload


def capture_session_from_payload(payload: dict[str, Any]) -> CaptureSessionManifest:
    """Revalidate a cached capture session into immutable domain objects."""

    def time_base(value: object) -> RationalTimeBase | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise P03ContractError("source time base must be an object or null")
        return RationalTimeBase(int(value["numerator"]), int(value["denominator"]))

    results: list[CameraCaptureResult] = []
    for raw_result in payload["results"]:
        raw_profile = raw_result["profile"]
        profile_time_base = time_base(raw_profile["time_base"])
        if profile_time_base is None:
            raise P03ContractError("stream profile requires a time base")
        profile = StreamProfileIdentity(
            raw_profile["schema_version"],
            raw_profile["camera_id"],
            raw_profile["profile_version"],
            raw_profile["endpoint_environment_key"],
            raw_profile["endpoint_binding_id"],
            raw_profile["width_pixels"],
            raw_profile["height_pixels"],
            raw_profile["codec"],
            profile_time_base,
            raw_profile["crop"],
            raw_profile["rotation_degrees"],
            raw_profile["observed_at_utc"],
        )
        frames = tuple(
            SourceFrame(
                frame["frame_id"],
                frame["camera_id"],
                frame["profile_version"],
                frame["source_pts"],
                time_base(frame["source_time_base"]),
                frame["acquisition_monotonic_ns"],
                frame["observed_at_utc"],
                frame["processing_started_monotonic_ns"],
                frame["model_completed_at_utc"],
            )
            for frame in raw_result["frames"]
        )
        raw_artifact = raw_result["artifact"]
        artifact = (
            None
            if raw_artifact is None
            else CaptureArtifact(
                raw_artifact["relative_path"],
                raw_artifact["sha256"],
                raw_artifact["byte_count"],
                StorageMode(raw_artifact["storage_mode"]),
                raw_artifact["interrupted"],
                raw_artifact["fallback_reason"],
            )
        )
        events = tuple(
            ReconnectEvent(
                event["attempt"],
                event["state"],
                event["monotonic_ns"],
                event["observed_at_utc"],
                event["failure_class"],
            )
            for event in raw_result["reconnect_events"]
        )
        results.append(
            CameraCaptureResult(
                raw_result["camera_id"],
                profile,
                CaptureStatus(raw_result["status"]),
                frames,
                artifact,
                events,
                raw_result["failure_class"],
                raw_result["failure_message"],
            )
        )
    return CaptureSessionManifest(
        payload["schema_version"],
        payload["session_id"],
        payload["created_at_utc"],
        payload["acquisition_started_monotonic_ns"],
        payload["acquisition_finished_monotonic_ns"],
        payload["software_identity"],
        payload["configuration_identity_sha256"],
        tuple(results),
    )
