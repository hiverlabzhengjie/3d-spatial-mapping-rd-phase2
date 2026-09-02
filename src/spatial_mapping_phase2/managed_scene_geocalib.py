"""Scene-local orchestration for the pinned GeoCalib worker.

The combined console intentionally stays in the small web runtime.  Model work is delegated to
the already validated native GeoCalib Python environment through an immutable request/result
boundary.  No endpoint values or client media are written to Git.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p04_calibration_domain import FrameReviewStatus
from spatial_mapping_phase2.runtime_environment import repository_source_environment


class ManagedSceneGeoCalibError(ValueError):
    """Raised when scene-specific intrinsic evidence cannot be produced safely."""


EXPECTED_DISTORTED_WEIGHT_SHA256 = (
    "13cc505928e3ff4eb26c00bff73861ab2b11b804a546323456cf5462e1f8f447"
)


@dataclass(frozen=True, slots=True)
class ManagedSceneGeoCalibConfig:
    python_executable: Path
    source_directory: Path
    torch_home: Path
    repository_root: Path
    minimum_frames_per_camera: int = 3

    def validate(self) -> None:
        for path, label in (
            (self.python_executable, "GeoCalib Python executable"),
            (self.source_directory, "GeoCalib source directory"),
            (self.repository_root / "src", "repository source directory"),
        ):
            if not path.exists():
                raise ManagedSceneGeoCalibError(f"{label} is unavailable: {path}")
        if not self.torch_home.is_dir():
            raise ManagedSceneGeoCalibError(
                f"GeoCalib model cache is unavailable: {self.torch_home}"
            )
        weight_path = self.torch_home / "hub" / "geocalib" / "distorted.tar"
        if (
            not weight_path.is_file()
            or _file_sha256(weight_path) != EXPECTED_DISTORTED_WEIGHT_SHA256
        ):
            raise ManagedSceneGeoCalibError(
                "GeoCalib distorted checkpoint is missing or differs from the pinned identity"
            )
        if self.minimum_frames_per_camera < 2:
            raise ManagedSceneGeoCalibError("GeoCalib requires at least two frames per camera")


class ManagedSceneGeoCalibRunner:
    """Build or reuse one hash-bound independent-intrinsic evidence set."""

    def __init__(self, config: ManagedSceneGeoCalibConfig, output_root: Path) -> None:
        self.config = config
        self.output_root = output_root.resolve()

    @property
    def current_evidence_path(self) -> Path:
        return self.output_root / "current-evidence.json"

    def prepare(
        self,
        camera_ids: Sequence[str],
        calibration_services: Mapping[str, Any],
    ) -> Path:
        self.config.validate()
        cameras = [
            self._camera_request(camera_id, calibration_services.get(camera_id))
            for camera_id in camera_ids
        ]
        request_core: dict[str, Any] = {
            "schema_version": "managed-scene-geocalib-request-v1",
            "camera_model": "simple_radial",
            "weights": "distorted",
            "geocalib_source_directory": str(self.config.source_directory.resolve()),
            "torch_home": str(self.config.torch_home.resolve()),
            "cameras": cameras,
        }
        request_sha256 = _payload_sha256(request_core)
        current = _read_optional_json(self.current_evidence_path)
        if (
            current is not None
            and current.get("schema_version") == "xr03-independent-intrinsic-estimates-v1"
            and current.get("request_sha256") == request_sha256
        ):
            return self.current_evidence_path

        run_root = self.output_root / "runs" / f"geocalib-{request_sha256[:16]}"
        for existing in sorted(run_root.glob("attempt-*/evidence.json"), reverse=True):
            result = _required_result(existing, request_sha256, camera_ids)
            self.output_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(self.current_evidence_path, result)
            return self.current_evidence_path
        attempts = tuple(run_root.glob("attempt-*"))
        run_directory = run_root / f"attempt-{len(attempts) + 1:03d}"
        request_path = run_directory / "request.json"
        result_path = run_directory / "evidence.json"
        run_directory.mkdir(parents=True)
        request = {
            **request_core,
            "request_sha256": request_sha256,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "output_path": str(result_path.resolve()),
        }
        _write_json(request_path, request)
        self._run_worker(request_path)
        result = _required_result(result_path, request_sha256, camera_ids)
        self.output_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.current_evidence_path, result)
        return self.current_evidence_path

    def _camera_request(self, camera_id: str, service: Any) -> dict[str, Any]:
        if service is None:
            raise ManagedSceneGeoCalibError(
                f"{camera_id} has no scene-local calibration workspace"
            )
        try:
            state = service.load_state()
        except Exception as error:
            raise ManagedSceneGeoCalibError(
                f"{camera_id} calibration workspace is not initialized"
            ) from error
        retained = [
            frame for frame in state.frames if frame.status is not FrameReviewStatus.REJECTED
        ]
        if state.approved_frame is None:
            raise ManagedSceneGeoCalibError(
                f"{camera_id}: approve one primary calibration frame first"
            )
        if len(retained) < self.config.minimum_frames_per_camera:
            raise ManagedSceneGeoCalibError(
                f"{camera_id}: capture at least {self.config.minimum_frames_per_camera} "
                "usable calibration frames before determining intrinsics"
            )
        retained = retained[-self.config.minimum_frames_per_camera :]
        sizes = {(frame.image_width_pixels, frame.image_height_pixels) for frame in retained}
        profiles = {frame.profile_version for frame in retained}
        if len(sizes) != 1 or len(profiles) != 1:
            raise ManagedSceneGeoCalibError(
                f"{camera_id}: retained GeoCalib frames must share one resolution and profile"
            )
        return {
            "camera_id": camera_id,
            "profile_version": retained[0].profile_version,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "path": str(service.frame_image_path(frame.frame_id).resolve()),
                    "sha256": frame.sha256,
                    "width_pixels": frame.image_width_pixels,
                    "height_pixels": frame.image_height_pixels,
                }
                for frame in retained
            ],
        }

    def _run_worker(self, request_path: Path) -> None:
        environment = repository_source_environment(self.config.repository_root)
        try:
            completed = subprocess.run(
                [
                    str(self.config.python_executable.resolve()),
                    "-m",
                    "spatial_mapping_phase2.managed_scene_geocalib_worker",
                    str(request_path.resolve()),
                ],
                cwd=self.config.repository_root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=15 * 60,
            )
        except subprocess.TimeoutExpired as error:
            raise ManagedSceneGeoCalibError(
                "Scene GeoCalib exceeded the 15-minute bounded runtime"
            ) from error
        except OSError as error:
            raise ManagedSceneGeoCalibError(
                f"Scene GeoCalib worker could not start ({type(error).__name__})"
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            summary = detail[-1] if detail else "worker exited without diagnostics"
            raise ManagedSceneGeoCalibError(f"Scene GeoCalib failed: {summary}")


def _required_result(path: Path, request_sha256: str, camera_ids: Sequence[str]) -> dict[str, Any]:
    value = _read_optional_json(path)
    if value is None:
        raise ManagedSceneGeoCalibError("GeoCalib worker produced no evidence file")
    estimates = value.get("estimates")
    if (
        value.get("schema_version") != "xr03-independent-intrinsic-estimates-v1"
        or value.get("request_sha256") != request_sha256
        or not isinstance(estimates, list)
        or tuple(str(item.get("camera_id")) for item in estimates if isinstance(item, dict))
        != tuple(camera_ids)
    ):
        raise ManagedSceneGeoCalibError("GeoCalib evidence identity or camera roster is malformed")
    return value


def _payload_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManagedSceneGeoCalibError(f"Intrinsic evidence is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ManagedSceneGeoCalibError("Intrinsic evidence must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _write_json(temporary, value)
    temporary.replace(path)
