"""Shared P03 capture service, immutable repository, and bounded adapter orchestration."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns, sleep
from typing import Protocol

from spatial_mapping_phase2.p01_observability import CAMERA_IDS, LocalRtspEndpoint
from spatial_mapping_phase2.p03_capture_domain import (
    SESSION_SCHEMA,
    CameraCaptureResult,
    CaptureArtifact,
    CaptureSessionManifest,
    CaptureStatus,
    P03ContractError,
    ReconnectEvent,
    SelectedBundleManifest,
    SourceFrame,
    StorageMode,
    StreamProfileIdentity,
    capture_session_from_payload,
    select_closest_bundle,
    sha256_file,
)


class CaptureAdapterError(RuntimeError):
    """Credential-free adapter failure."""


class ConnectTimeoutError(CaptureAdapterError):
    pass


class ReadTimeoutError(CaptureAdapterError):
    pass


class CaptureCancelledError(CaptureAdapterError):
    pass


class BackpressureError(CaptureAdapterError):
    pass


@dataclass(frozen=True, slots=True)
class CapturePolicy:
    duration_seconds: float = 2.0
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    retry_limit: int = 2
    initial_backoff_seconds: float = 0.1
    queue_capacity: int = 512

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise P03ContractError("duration must be positive")
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise P03ContractError("timeouts must be positive")
        if self.retry_limit < 0 or self.queue_capacity <= 0:
            raise P03ContractError("retry limit and queue capacity are invalid")


@dataclass(frozen=True, slots=True, repr=False)
class PreviewFrame:
    """Ephemeral decoded preview that is never a capture-session artifact."""

    camera_id: str
    media_type: str
    content: bytes
    observed_at_utc: str
    source_pts: int | None
    source_time_base: str | None

    def __post_init__(self) -> None:
        if self.camera_id not in CAMERA_IDS:
            raise P03ContractError("preview camera_id is invalid")
        if self.media_type != "image/jpeg":
            raise P03ContractError("preview media type must be image/jpeg")
        if not self.content:
            raise P03ContractError("preview content must be non-empty")
        if len(self.content) > 10 * 1024 * 1024:
            raise P03ContractError("preview content exceeds the bounded size")


class CaptureAdapter(Protocol):
    def profile(self, endpoint: LocalRtspEndpoint, policy: CapturePolicy) -> StreamProfileIdentity:
        """Return a sanitized observed profile within policy bounds."""

    def capture(
        self,
        endpoint: LocalRtspEndpoint,
        profile: StreamProfileIdentity,
        output_path: Path,
        relative_path: str,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SourceFrame, ...], CaptureArtifact]:
        """Capture bounded media, raising only credential-free typed errors."""

    def preview(
        self,
        endpoint: LocalRtspEndpoint,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> PreviewFrame:
        """Decode one bounded ephemeral frame without persisting capture evidence."""

    def close(self) -> None:
        """Release adapter-owned resources."""


class CaptureRepository:
    """Immutable local session/bundle storage below the authorized capture root."""

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.resolve()
        self.sessions_root = self.root / "captures" / "p03" / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def session_directory(self, session_id: str) -> Path:
        if not session_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for char in session_id
        ):
            raise P03ContractError("session_id is invalid")
        return self.sessions_root / session_id

    def create_session_directory(self, session_id: str) -> Path:
        directory = self.session_directory(session_id)
        try:
            directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise P03ContractError("session identity already exists and is immutable") from error
        return directory

    def write_session(self, manifest: CaptureSessionManifest) -> Path:
        directory = self.session_directory(manifest.session_id)
        if not directory.is_dir():
            raise P03ContractError("session artifact directory does not exist")
        path = directory / "session.json"
        self._exclusive_write(path, manifest.to_json())
        return path

    def write_bundle(self, manifest: SelectedBundleManifest) -> Path:
        directory = self.session_directory(manifest.source_session_id) / "bundles"
        directory.mkdir(exist_ok=True)
        path = directory / f"{manifest.bundle_id}.json"
        self._exclusive_write(path, manifest.to_json())
        return path

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
            raise P03ContractError("cached session manifest is unreadable") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != SESSION_SCHEMA:
            raise P03ContractError("cached session manifest has an unsupported schema")
        return payload

    def read_session(self, session_id: str) -> CaptureSessionManifest:
        payload = self.read_session_payload(session_id)
        return capture_session_from_payload(payload)

    @staticmethod
    def _exclusive_write(path: Path, content: str) -> None:
        try:
            with path.open("x", encoding="utf-8") as destination:
                destination.write(content)
        except FileExistsError as error:
            raise P03ContractError("immutable manifest already exists") from error


class CaptureWorkflowService:
    """Own concurrent capture lifecycle for both UI and CLI callers."""

    def __init__(
        self,
        endpoints: tuple[LocalRtspEndpoint, ...],
        adapter: CaptureAdapter,
        repository: CaptureRepository,
        software_identity: str,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if tuple(endpoint.camera_id for endpoint in endpoints) != CAMERA_IDS:
            raise P03ContractError("workflow requires all four endpoints in fixed order")
        self._endpoints = endpoints
        self._adapter = adapter
        self.repository = repository
        self.software_identity = software_identity
        self._clock_ns = clock_ns
        self._closed = False
        self._cancel = threading.Event()

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Return the configured camera roster without exposing endpoint secrets."""
        return tuple(endpoint.camera_id for endpoint in self._endpoints)

    def health(self, policy: CapturePolicy) -> dict[str, dict[str, object]]:
        self._require_open()
        output: dict[str, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="p03-health") as executor:
            futures = {
                executor.submit(self._safe_profile, endpoint, policy): endpoint.camera_id
                for endpoint in self._endpoints
            }
            for future in as_completed(futures):
                camera_id = futures[future]
                profile, failure = future.result()
                output[camera_id] = {
                    "state": "healthy" if profile else "failed",
                    "profile": asdict(profile) if profile else None,
                    "failure_class": failure,
                }
        return {camera_id: output[camera_id] for camera_id in CAMERA_IDS}

    def preview(self, camera_id: str, policy: CapturePolicy) -> PreviewFrame:
        self._require_open()
        endpoint = next((item for item in self._endpoints if item.camera_id == camera_id), None)
        if endpoint is None:
            raise P03ContractError("preview camera_id must be a fixed office camera ID")
        preview_cancel = threading.Event()
        return self._adapter.preview(endpoint, policy, preview_cancel)

    def capture_session(self, session_id: str, policy: CapturePolicy) -> CaptureSessionManifest:
        self._require_open()
        self._cancel.clear()
        session_dir = self.repository.create_session_directory(session_id)
        started = self._clock_ns()
        configuration_hash = hashlib.sha256(
            json.dumps(asdict(policy), sort_keys=True).encode("utf-8")
        ).hexdigest()
        results: dict[str, CameraCaptureResult] = {}
        profiles: dict[str, tuple[StreamProfileIdentity | None, str | None]] = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="p03-preflight") as executor:
            profile_futures = {
                executor.submit(self._safe_profile, endpoint, policy): endpoint.camera_id
                for endpoint in self._endpoints
            }
            for profile_future in as_completed(profile_futures):
                profiles[profile_futures[profile_future]] = profile_future.result()
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="p03-capture") as executor:
            capture_futures = {
                executor.submit(
                    self._capture_camera,
                    endpoint,
                    profiles[endpoint.camera_id],
                    session_dir,
                    policy,
                ): endpoint.camera_id
                for endpoint in self._endpoints
            }
            for capture_future in as_completed(capture_futures):
                results[capture_futures[capture_future]] = capture_future.result()
        manifest = CaptureSessionManifest(
            SESSION_SCHEMA,
            session_id,
            _utc_now(),
            started,
            self._clock_ns(),
            self.software_identity,
            configuration_hash,
            tuple(results[camera_id] for camera_id in CAMERA_IDS),
        )
        self.repository.write_session(manifest)
        return manifest

    def select_bundle(
        self, session: CaptureSessionManifest, bundle_id: str, target_ns: int | None = None
    ) -> SelectedBundleManifest:
        bundle = select_closest_bundle(session, bundle_id, target_ns)
        self.repository.write_bundle(bundle)
        return bundle

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self._cancel.set()
        self._adapter.close()
        self._closed = True

    def _safe_profile(
        self, endpoint: LocalRtspEndpoint, policy: CapturePolicy
    ) -> tuple[StreamProfileIdentity | None, str | None]:
        try:
            return self._adapter.profile(endpoint, policy), None
        except Exception as error:
            return None, type(error).__name__

    def _capture_camera(
        self,
        endpoint: LocalRtspEndpoint,
        profile_result: tuple[StreamProfileIdentity | None, str | None],
        session_dir: Path,
        policy: CapturePolicy,
    ) -> CameraCaptureResult:
        events: list[ReconnectEvent] = []
        profile, failure = profile_result
        if profile is None:
            return _failed_result(endpoint.camera_id, failure or "ProfileError", "profile failed")
        relative = f"captures/p03/sessions/{session_dir.name}/{endpoint.camera_id}.mp4"
        output = session_dir / f"{endpoint.camera_id}.mp4"
        last_error: Exception | None = None
        for attempt in range(1, policy.retry_limit + 2):
            if self._cancel.is_set():
                last_error = CaptureCancelledError("capture cancelled")
                break
            try:
                frames, artifact = self._adapter.capture(
                    endpoint, profile, output, relative, policy, self._cancel
                )
                if attempt > 1:
                    events.append(_event(attempt, "reconnected", None, self._clock_ns()))
                return CameraCaptureResult(
                    endpoint.camera_id,
                    profile,
                    CaptureStatus.CAPTURED,
                    frames,
                    artifact,
                    tuple(events),
                    None,
                    None,
                )
            except Exception as error:
                last_error = error
                events.append(
                    _event(attempt, "disconnected", type(error).__name__, self._clock_ns())
                )
                if attempt > policy.retry_limit:
                    events.append(
                        _event(attempt, "exhausted", type(error).__name__, self._clock_ns())
                    )
                    break
                events.append(_event(attempt, "backoff", type(error).__name__, self._clock_ns()))
                sleep(policy.initial_backoff_seconds * (2 ** (attempt - 1)))
        failure_class = type(last_error).__name__ if last_error else "CaptureError"
        return CameraCaptureResult(
            endpoint.camera_id,
            profile,
            CaptureStatus.FAILED,
            (),
            None,
            tuple(events),
            failure_class,
            "bounded capture failed",
        )

    def _require_open(self) -> None:
        if self._closed:
            raise P03ContractError("capture workflow is closed")


def _failed_result(camera_id: str, failure_class: str, message: str) -> CameraCaptureResult:
    # A failed profile cannot truthfully construct a profile-bound result. Use a deterministic
    # credential-free placeholder profile to retain the per-camera failure in the session.
    from spatial_mapping_phase2.p01_observability import CAMERA_ENDPOINT_KEYS
    from spatial_mapping_phase2.p03_capture_domain import PROFILE_SCHEMA, RationalTimeBase

    profile = StreamProfileIdentity(
        PROFILE_SCHEMA,
        camera_id,
        "stream-profile-v1",
        CAMERA_ENDPOINT_KEYS[camera_id],
        f"unobserved-{camera_id}",
        1,
        1,
        "unobserved",
        RationalTimeBase(1, 1),
        None,
        None,
        _utc_now(),
    )
    return CameraCaptureResult(
        camera_id, profile, CaptureStatus.FAILED, (), None, (), failure_class, message
    )


def _event(attempt: int, state: str, failure: str | None, clock: int) -> ReconnectEvent:
    return ReconnectEvent(attempt, state, clock, _utc_now(), failure)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class SyntheticCaptureAdapter:
    """Deterministic four-camera fixture with injected interruption/backpressure behavior."""

    def __init__(self, failures_before_success: Mapping[str, int] | None = None) -> None:
        self.failures = dict(failures_before_success or {})
        self.attempts: dict[str, int] = {}
        self.closed = False

    def profile(self, endpoint: LocalRtspEndpoint, policy: CapturePolicy) -> StreamProfileIdentity:
        del policy
        from spatial_mapping_phase2.p03_capture_domain import PROFILE_SCHEMA, RationalTimeBase

        return StreamProfileIdentity(
            PROFILE_SCHEMA,
            endpoint.camera_id,
            "stream-profile-v1",
            endpoint.environment_key,
            f"synthetic-{endpoint.camera_id}",
            640,
            360,
            "h264",
            RationalTimeBase(1, 90000),
            None,
            None,
            "2026-08-14T00:00:00Z",
        )

    def capture(
        self,
        endpoint: LocalRtspEndpoint,
        profile: StreamProfileIdentity,
        output_path: Path,
        relative_path: str,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SourceFrame, ...], CaptureArtifact]:
        attempt = self.attempts.get(endpoint.camera_id, 0) + 1
        self.attempts[endpoint.camera_id] = attempt
        if cancel.is_set():
            raise CaptureCancelledError("capture cancelled")
        if attempt <= self.failures.get(endpoint.camera_id, 0):
            raise ReadTimeoutError("synthetic read timeout")
        if policy.queue_capacity < 3:
            raise BackpressureError("synthetic queue full")
        frame_count = 3
        payload = f"synthetic:{endpoint.camera_id}:{attempt}".encode()
        output_path.write_bytes(payload)
        base = int(endpoint.camera_id[-2:]) * 1_000_000
        frames = tuple(
            SourceFrame(
                f"{endpoint.camera_id}-f{index}",
                endpoint.camera_id,
                profile.profile_version,
                index * 3000,
                profile.time_base,
                base + index * 33_000_000,
                "2026-08-14T00:00:00Z",
            )
            for index in range(frame_count)
        )
        artifact = CaptureArtifact(
            relative_path,
            sha256_file(output_path),
            output_path.stat().st_size,
            StorageMode.PACKET_PRESERVING_MP4,
            False,
            None,
        )
        return frames, artifact

    def preview(
        self,
        endpoint: LocalRtspEndpoint,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> PreviewFrame:
        del policy
        if cancel.is_set():
            raise CaptureCancelledError("preview cancelled")
        return PreviewFrame(
            endpoint.camera_id,
            "image/jpeg",
            f"preview:{endpoint.camera_id}".encode(),
            "2026-08-14T00:00:00Z",
            0,
            "1/90000",
        )

    def close(self) -> None:
        self.closed = True
