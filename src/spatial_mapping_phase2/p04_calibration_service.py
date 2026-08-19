"""Local persistence and immutable artifact handling for the P04 calibration console."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Protocol

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p04_calibration_domain import (
    P04_SCHEMA_VERSION,
    CalibrationFrame,
    CalibrationWorkspace,
    FacilityReference,
    FrameReviewStatus,
    LandmarkRole,
    LinkedLandmark,
    P04CalibrationError,
    PixelPoint,
    WorldPoint,
    build_d034_validation_seal,
    build_p04_export,
)

MAX_IMAGE_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ImageDimensions:
    width_pixels: int
    height_pixels: int


class ImageInspector(Protocol):
    def inspect(self, path: Path) -> ImageDimensions:
        """Return decoded image dimensions or raise an actionable P04 error."""


class PyMuPDFImageInspector:
    """Inspect local raster images using the already selected PyMuPDF dependency."""

    def inspect(self, path: Path) -> ImageDimensions:
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError as error:
            raise P04CalibrationError(
                "PyMuPDF is required to inspect calibration images"
            ) from error
        try:
            pixmap = fitz.Pixmap(str(path))
        except Exception as error:
            raise P04CalibrationError("calibration artifact is not a readable image") from error
        try:
            return ImageDimensions(pixmap.width, pixmap.height)
        finally:
            pixmap = None


@dataclass(frozen=True, slots=True, repr=False)
class CapturedCandidate:
    """Credential-free result of one user-triggered bounded Camera 3 capture."""

    content: bytes
    observed_at_utc: str
    source_pts: int | None
    source_time_base: str | None
    profile_version: str = "stream-profile-v1"

    def __post_init__(self) -> None:
        if not self.content or len(self.content) > MAX_IMAGE_BYTES:
            raise P04CalibrationError("captured candidate image size is invalid")
        if not self.observed_at_utc.strip():
            raise P04CalibrationError("captured candidate observation time is required")
        if self.source_pts is None and self.source_time_base is not None:
            raise P04CalibrationError("captured source_time_base requires source_pts")


class CandidateCapturer(Protocol):
    @property
    def camera_id(self) -> str:
        """Stable camera identity served by this capturer."""

    def capture(self) -> CapturedCandidate:
        """Capture one bounded read-only camera frame."""


class P03PreviewCandidateCapturer:
    """Reuse P03's bounded PyAV preview adapter as P04's live candidate source."""

    def __init__(self, endpoint: LocalRtspEndpoint) -> None:
        self._endpoint = endpoint

    @property
    def camera_id(self) -> str:
        return self._endpoint.camera_id

    def capture(self) -> CapturedCandidate:
        from spatial_mapping_phase2.p03_capture_service import CapturePolicy
        from spatial_mapping_phase2.p03_pyav_adapter import PyAvCaptureAdapter

        adapter = PyAvCaptureAdapter({self._endpoint.camera_id: "stream-profile-v1"})
        try:
            frame = adapter.preview(
                self._endpoint,
                CapturePolicy(
                    duration_seconds=1.0,
                    connect_timeout_seconds=5.0,
                    read_timeout_seconds=5.0,
                    retry_limit=0,
                ),
                threading.Event(),
            )
            return CapturedCandidate(
                frame.content,
                frame.observed_at_utc,
                frame.source_pts,
                frame.source_time_base if frame.source_pts is not None else None,
            )
        finally:
            adapter.close()


class P04CalibrationService:
    """Versioned P04 workspace shared by CLI and localhost web callers."""

    def __init__(
        self,
        workspace: Path,
        inspector: ImageInspector | None = None,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.workspace = workspace.resolve()
        self.inspector = inspector or PyMuPDFImageInspector()
        self._sleeper = sleeper
        self._lock = threading.RLock()
        self._state_path = self.workspace / "state.json"
        self._sources_dir = self.workspace / "sources"
        self._frames_dir = self.workspace / "frames"
        self._history_dir = self.workspace / "history"
        self._exports_dir = self.workspace / "exports"
        self._validation_seals_dir = self.workspace / "validation_seals"
        for directory in (
            self.workspace,
            self._sources_dir,
            self._frames_dir,
            self._history_dir,
            self._exports_dir,
            self._validation_seals_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def has_state(self) -> bool:
        return self._state_path.is_file()

    def initialize(
        self,
        facility_export_path: Path,
        plan_image_path: Path,
        camera_id: str = "office-cam-03",
    ) -> CalibrationWorkspace:
        """Create a clean workspace from exact P02 export and rendered-plan artifacts."""

        with self._lock:
            if self.has_state():
                raise P04CalibrationError("P04 workspace is already initialized")
            export_bytes = self._read_bounded_file(facility_export_path, "facility export")
            try:
                export = json.loads(export_bytes)
            except json.JSONDecodeError as error:
                raise P04CalibrationError("facility export is not valid JSON") from error
            if not isinstance(export, dict):
                raise P04CalibrationError("facility export root must be an object")
            reference = self._build_facility_reference(export, export_bytes, plan_image_path)
            export_copy = self._sources_dir / f"facility-export-{reference.export_sha256}.json"
            plan_suffix = plan_image_path.suffix.lower()
            plan_copy = self._sources_dir / f"plan-{reference.plan_image_sha256}{plan_suffix}"
            self._atomic_write_bytes(export_copy, export_bytes)
            if not plan_copy.exists():
                shutil.copyfile(plan_image_path, plan_copy)
            state = CalibrationWorkspace(
                P04_SCHEMA_VERSION,
                0,
                camera_id,
                reference,
                (),
                (),
            )
            self._write_state(state)
            return state

    def load_state(self) -> CalibrationWorkspace:
        with self._lock:
            if not self._state_path.is_file():
                raise P04CalibrationError("initialize the P04 workspace first")
            try:
                payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise P04CalibrationError("saved P04 workspace is unreadable") from error
            if not isinstance(payload, dict):
                raise P04CalibrationError("saved P04 workspace must be an object")
            return CalibrationWorkspace.from_dict(payload)

    def add_frame(
        self,
        source_path: Path,
        frame_id: str,
        profile_version: str,
        expected_sha256: str | None = None,
    ) -> CalibrationWorkspace:
        """Import an immutable candidate frame without modifying its source."""

        suffix = source_path.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise P04CalibrationError("candidate frame must be JPEG or PNG")
        content = self._read_bounded_file(source_path, "candidate frame")
        sha256 = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and expected_sha256.lower() != sha256:
            raise P04CalibrationError("candidate frame SHA-256 does not match expected identity")
        dimensions = self.inspector.inspect(source_path)
        with self._lock:
            state = self.load_state()
            if any(frame.frame_id == frame_id for frame in state.frames):
                raise P04CalibrationError("frame_id already exists")
            relative_path = f"frames/{sha256}{suffix}"
            destination = self.workspace / relative_path
            if not destination.exists():
                self._atomic_write_bytes(destination, content)
            frame = CalibrationFrame(
                frame_id,
                state.camera_id,
                profile_version,
                sha256,
                len(content),
                dimensions.width_pixels,
                dimensions.height_pixels,
                relative_path,
                FrameReviewStatus.CANDIDATE,
                None,
            )
            next_state = replace(state, frames=(*state.frames, frame), revision=state.revision + 1)
            self._persist_next(state, next_state)
            return next_state

    def capture_candidate(
        self, delay_seconds: float, capturer: CandidateCapturer
    ) -> CalibrationWorkspace:
        """Wait for the operator-selected delay, then persist one immutable candidate."""

        if (
            isinstance(delay_seconds, bool)
            or not isinstance(delay_seconds, int | float)
            or not 0 <= delay_seconds <= 30
        ):
            raise P04CalibrationError("capture delay_seconds must be between 0 and 30")
        state = self.load_state()
        if capturer.camera_id != state.camera_id:
            raise P04CalibrationError(
                f"capture source {capturer.camera_id} does not match workspace {state.camera_id}"
            )
        self._sleeper(float(delay_seconds))
        try:
            candidate = capturer.capture()
        except P04CalibrationError:
            raise
        except Exception as error:
            raise P04CalibrationError(
                f"timed camera capture failed ({type(error).__name__})"
            ) from error
        camera_number = state.camera_id.rsplit("-", 1)[-1]
        frame_id = datetime.now(UTC).strftime(
            f"cam{camera_number}-live-%Y%m%dt%H%M%S%fz"
        ).lower()
        temporary = self.workspace / f".{frame_id}.capturing.jpg"
        temporary.write_bytes(candidate.content)
        try:
            dimensions = self.inspector.inspect(temporary)
        finally:
            temporary.unlink(missing_ok=True)
        sha256 = hashlib.sha256(candidate.content).hexdigest()
        relative_path = f"frames/{sha256}.jpg"
        destination = self.workspace / relative_path
        with self._lock:
            state = self.load_state()
            if any(frame.frame_id == frame_id for frame in state.frames):
                raise P04CalibrationError("generated frame_id already exists; retry capture")
            if not destination.exists():
                self._atomic_write_bytes(destination, candidate.content)
            frame = CalibrationFrame(
                frame_id,
                state.camera_id,
                candidate.profile_version,
                sha256,
                len(candidate.content),
                dimensions.width_pixels,
                dimensions.height_pixels,
                relative_path,
                FrameReviewStatus.CANDIDATE,
                None,
                "user-timed-live",
                candidate.observed_at_utc,
                candidate.source_pts,
                candidate.source_time_base,
            )
            next_state = replace(state, frames=(*state.frames, frame), revision=state.revision + 1)
            self._persist_next(state, next_state)
            return next_state

    def review_frame(
        self, frame_id: str, status: FrameReviewStatus, note: str | None
    ) -> CalibrationWorkspace:
        with self._lock:
            state = self.load_state()
            next_state = state.with_frame_review(frame_id, status, note)
            self._persist_next(state, next_state)
            return next_state

    def add_landmark(self, payload: dict[str, Any]) -> CalibrationWorkspace:
        """Link one camera pixel to plan-derived XY and operator-entered physical Z."""

        with self._lock:
            state = self.load_state()
            approved = state.approved_frame
            if approved is None:
                raise P04CalibrationError("approve a primary frame before adding landmarks")
            frame_id = _required_string(payload, "frame_id")
            if frame_id != approved.frame_id:
                raise P04CalibrationError("new landmarks must use the approved primary frame")
            image_point = PixelPoint.from_dict(
                _required_object(payload, "image_point"), "image_point"
            )
            plan_point = PixelPoint.from_dict(
                _required_object(payload, "plan_point"), "plan_point"
            )
            x_metres, y_metres = state.facility_reference.world_xy(plan_point)
            z_metres = _required_number(payload, "z_metres")
            z_uncertainty = _optional_number(payload.get("z_uncertainty_metres"))
            z_source = _optional_string(payload.get("z_source"))
            try:
                role = LandmarkRole(_required_string(payload, "role"))
            except ValueError as error:
                raise P04CalibrationError(
                    "role must be solve, held-out or d034-validation"
                ) from error
            landmark = LinkedLandmark(
                _required_string(payload, "landmark_id"),
                _required_string(payload, "name"),
                _required_string(payload, "physical_meaning"),
                frame_id,
                image_point,
                plan_point,
                WorldPoint(x_metres, y_metres, z_metres),
                z_source,
                z_uncertainty,
                role,
            )
            next_state = replace(
                state,
                landmarks=(*state.landmarks, landmark),
                revision=state.revision + 1,
            )
            self._persist_next(state, next_state)
            return next_state

    def remove_landmark(self, landmark_id: str) -> CalibrationWorkspace:
        with self._lock:
            state = self.load_state()
            retained = tuple(
                landmark for landmark in state.landmarks if landmark.landmark_id != landmark_id
            )
            if len(retained) == len(state.landmarks):
                raise P04CalibrationError("unknown landmark_id")
            next_state = replace(state, landmarks=retained, revision=state.revision + 1)
            self._persist_next(state, next_state)
            return next_state

    def set_missing_z_sources(self, z_source: str) -> CalibrationWorkspace:
        """Bind one operator-supplied measurement source to every currently blank Z source."""

        normalized_source = _optional_string(z_source)
        if normalized_source is None:
            raise P04CalibrationError("z_source must be a non-blank string")
        with self._lock:
            state = self.load_state()
            missing_count = sum(landmark.z_source is None for landmark in state.landmarks)
            if missing_count == 0:
                raise P04CalibrationError("no landmarks have a missing z_source")
            landmarks = tuple(
                replace(landmark, z_source=normalized_source)
                if landmark.z_source is None
                else landmark
                for landmark in state.landmarks
            )
            next_state = replace(state, landmarks=landmarks, revision=state.revision + 1)
            self._persist_next(state, next_state)
            return next_state

    def state_response(self) -> dict[str, Any]:
        state = self.load_state()
        payload = state.to_dict()
        payload["derived"] = {
            "approved_frame_id": (
                None if state.approved_frame is None else state.approved_frame.frame_id
            ),
            "solve_count": sum(
                landmark.role is LandmarkRole.SOLVE for landmark in state.landmarks
            ),
            "held_out_count": sum(
                landmark.role is LandmarkRole.HELD_OUT for landmark in state.landmarks
            ),
            "d034_validation_count": sum(
                landmark.role is LandmarkRole.D034_VALIDATION
                for landmark in state.landmarks
            ),
        }
        return payload

    def plan_image_path(self) -> Path:
        reference = self.load_state().facility_reference
        matches = tuple(self._sources_dir.glob(f"plan-{reference.plan_image_sha256}.*"))
        if len(matches) != 1:
            raise P04CalibrationError("hash-bound plan image is missing or ambiguous")
        return matches[0]

    def frame_image_path(self, frame_id: str) -> Path:
        state = self.load_state()
        frame = next((item for item in state.frames if item.frame_id == frame_id), None)
        if frame is None:
            raise P04CalibrationError("unknown frame_id")
        path = self.workspace / frame.relative_path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != frame.sha256:
            raise P04CalibrationError("frame artifact is missing or its identity changed")
        return path

    def export_snapshot(self) -> tuple[Path, dict[str, Any]]:
        with self._lock:
            state = self.load_state()
            payload = build_p04_export(state)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            filename = f"p04-linked-correspondences-r{state.revision}-{timestamp}.json"
            path = self._exports_dir / filename
            self._atomic_write_json(path, payload)
            return path, payload

    def export_d034_validation_seal(self) -> tuple[Path, dict[str, Any]]:
        """Write a separate immutable two-point validation seal that excludes solve data."""

        with self._lock:
            state = self.load_state()
            payload = build_d034_validation_seal(state)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            filename = f"p05-d034-validation-r{state.revision}-{timestamp}.json"
            path = self._validation_seals_dir / filename
            self._atomic_write_json(path, payload)
            return path, payload

    def _build_facility_reference(
        self, export: dict[str, Any], export_bytes: bytes, plan_image_path: Path
    ) -> FacilityReference:
        if export.get("schema_version") != "p02-interactive-export-v1":
            raise P04CalibrationError("facility export must use P02 interactive export v1")
        plan = _required_object(export, "plan")
        facility = _required_object(export, "facility_frame")
        image_bytes = self._read_bounded_file(plan_image_path, "rendered plan image")
        dimensions = self.inspector.inspect(plan_image_path)
        expected_width = _required_integer(plan, "image_width_pixels")
        expected_height = _required_integer(plan, "image_height_pixels")
        if (dimensions.width_pixels, dimensions.height_pixels) != (
            expected_width,
            expected_height,
        ):
            raise P04CalibrationError("rendered plan dimensions do not match facility export")
        matrix = facility.get("T_world_from_plan_display_pixel")
        if not isinstance(matrix, list):
            raise P04CalibrationError("facility export lacks world-from-plan-pixel transform")
        return FacilityReference.from_dict(
            {
                "export_sha256": hashlib.sha256(export_bytes).hexdigest(),
                "source_revision": _required_integer(export, "source_revision"),
                "plan_source_sha256": _required_string(plan, "source_sha256"),
                "plan_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "plan_image_width_pixels": expected_width,
                "plan_image_height_pixels": expected_height,
                "frame_id": _required_string(facility, "frame_id"),
                "world_from_plan_pixel": matrix,
                "authority_note": (
                    "P02 revision 3 single-control provisional XY; horizontal uncertainty unknown"
                ),
            }
        )

    def _persist_next(
        self, previous: CalibrationWorkspace, next_state: CalibrationWorkspace
    ) -> None:
        if next_state.revision != previous.revision + 1:
            raise P04CalibrationError("workspace revision must advance by exactly one")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        history_path = self._history_dir / f"state-r{previous.revision}-{timestamp}.json"
        self._atomic_write_json(history_path, previous.to_dict())
        self._write_state(next_state)

    def _write_state(self, state: CalibrationWorkspace) -> None:
        self._atomic_write_json(self._state_path, state.to_dict())

    @staticmethod
    def _read_bounded_file(path: Path, meaning: str) -> bytes:
        try:
            size = path.stat().st_size
        except OSError as error:
            raise P04CalibrationError(f"{meaning} is missing or unreadable") from error
        if size <= 0 or size > MAX_IMAGE_BYTES:
            raise P04CalibrationError(f"{meaning} must be between 1 byte and 50 MiB")
        try:
            return path.read_bytes()
        except OSError as error:
            raise P04CalibrationError(f"{meaning} is unreadable") from error

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        P04CalibrationService._atomic_write_bytes(
            path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)


def _required_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise P04CalibrationError(f"{key} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise P04CalibrationError(f"{key} must be a non-blank string")
    return value.strip()


def _required_number(payload: dict[str, Any], key: str) -> float:
    return _finite_number(payload.get(key), key)


def _required_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise P04CalibrationError(f"{key} must be an integer")
    return value


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return _finite_number(value, "optional number")


def _optional_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise P04CalibrationError("optional string must be non-blank or null")
    return value.strip()


def _finite_number(value: Any, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise P04CalibrationError(f"{field_name} must be a finite number")
    result = float(value)
    if not (float("-inf") < result < float("inf")):
        raise P04CalibrationError(f"{field_name} must be a finite number")
    return result


def load_calibration_camera_endpoint(
    secret_file: Path, camera_id: str
) -> LocalRtspEndpoint:
    """Load one selected camera endpoint without returning it in diagnostics."""

    if camera_id not in CAMERA_ENDPOINT_KEYS:
        raise P04CalibrationError("camera_id must be one of office-cam-01 through office-cam-04")
    key = CAMERA_ENDPOINT_KEYS[camera_id]
    value = os.environ.get(key, "")
    if secret_file.is_file():
        for line in secret_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            local_key, local_value = stripped.split("=", 1)
            if local_key.strip() == key:
                value = local_value.strip()
    try:
        return LocalRtspEndpoint(camera_id, key, value)
    except Exception as error:
        raise P04CalibrationError(f"{camera_id} RTSP endpoint is missing or malformed") from error


def load_p04_camera3_endpoint(secret_file: Path) -> LocalRtspEndpoint:
    """Backward-compatible Camera 3 endpoint loader."""

    return load_calibration_camera_endpoint(secret_file, "office-cam-03")
