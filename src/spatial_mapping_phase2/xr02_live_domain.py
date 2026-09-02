"""Typed scene selection and live evidence contracts for XR02 WP4."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.xr02_local_domain import SceneContextKey
from spatial_mapping_phase2.xr02_local_pipeline import build_scene_context

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class XR02LiveContractError(ValueError):
    """Raised when live configuration or evidence violates the WP4 contract."""


class LiveServiceState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    SCENE_UPDATE_AVAILABLE = "scene_update_available"
    STOPPING = "stopping"
    FAILED = "failed"


class CameraLiveState(StrEnum):
    STARTING = "starting"
    CURRENT = "current"
    STALE = "stale"
    RECONNECTING = "reconnecting"
    MISSING = "missing"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or self.bytes < 0 or not _SHA256.fullmatch(self.sha256):
            raise XR02LiveContractError("file identity is invalid")

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class AdoptedSceneSelection:
    """One exact operator-selected scene epoch; never hot-swapped while active."""

    scene: SceneContextKey
    operator_state: FileIdentity
    active_geometry_artifact_id: str
    active_floor_artifact_id: str
    geometry: FileIdentity
    floor: FileIdentity
    static_rerun: FileIdentity
    p06_calibration: FileIdentity
    p07_pose: FileIdentity
    p08_floor_manifest: FileIdentity

    @property
    def selection_signature_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene.as_dict(),
            "scene_context_sha256": self.scene.context_sha256,
            "operator_state": self.operator_state.as_dict(),
            "active_geometry_artifact_id": self.active_geometry_artifact_id,
            "active_floor_artifact_id": self.active_floor_artifact_id,
            "geometry": self.geometry.as_dict(),
            "floor": self.floor.as_dict(),
            "static_rerun": self.static_rerun.as_dict(),
            "p06_calibration": self.p06_calibration.as_dict(),
            "p07_pose": self.p07_pose.as_dict(),
            "p08_floor_manifest": self.p08_floor_manifest.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class CameraHealthEvidence:
    camera_id: str
    state: CameraLiveState
    generation: int
    decoded_frames: int
    reconnects: int
    replaced_frames: int
    stale_snapshots: int
    failure_class: str | None
    frame_age_ms: float | None
    failure_detail: str | None = None
    capture_backend: str = "pyav"
    capture_process_state: str = "stopped"
    tracker_epoch: int = 0
    delivered_frames: int = 0
    supervisor_restarts: int = 0
    restart_reason: str | None = None
    process_heartbeat_age_ms: float | None = None

    def __post_init__(self) -> None:
        counters = (
            self.generation,
            self.decoded_frames,
            self.reconnects,
            self.replaced_frames,
            self.stale_snapshots,
            self.tracker_epoch,
            self.delivered_frames,
            self.supervisor_restarts,
        )
        if any(value < 0 for value in counters):
            raise XR02LiveContractError("camera health counters must be non-negative")
        if self.frame_age_ms is not None and (
            not math.isfinite(self.frame_age_ms) or self.frame_age_ms < 0
        ):
            raise XR02LiveContractError("camera frame age must be finite and non-negative")
        if self.process_heartbeat_age_ms is not None and (
            not math.isfinite(self.process_heartbeat_age_ms) or self.process_heartbeat_age_ms < 0
        ):
            raise XR02LiveContractError("camera heartbeat age must be finite and non-negative")
        if not self.capture_backend or not self.capture_process_state:
            raise XR02LiveContractError("camera capture backend and process state are required")

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "state": self.state.value,
            "generation": self.generation,
            "decoded_frames": self.decoded_frames,
            "reconnects": self.reconnects,
            "replaced_frames": self.replaced_frames,
            "stale_snapshots": self.stale_snapshots,
            "failure_class": self.failure_class,
            "failure_detail": self.failure_detail,
            "frame_age_ms": self.frame_age_ms,
            "capture_backend": self.capture_backend,
            "capture_process_state": self.capture_process_state,
            "tracker_epoch": self.tracker_epoch,
            "delivered_frames": self.delivered_frames,
            "supervisor_restarts": self.supervisor_restarts,
            "restart_reason": self.restart_reason,
            "process_heartbeat_age_ms": self.process_heartbeat_age_ms,
        }


def resolve_adopted_scene(
    operator_state_path: Path,
    p06_path: Path,
    p07_path: Path,
    p08_floor_manifest_path: Path | None,
    camera_policy_sha256: str | None = None,
) -> AdoptedSceneSelection:
    """Resolve and hash the current approved P08 selection without modifying its catalog."""

    operator = _json_object(operator_state_path)
    if operator.get("geometry_approved") is not True or operator.get("floor_approved") is not True:
        raise XR02LiveContractError("current geometry and floor must both be operator-approved")
    geometry_id = _required_string(operator, "active_geometry_artifact_id")
    floor_id = _required_string(operator, "active_floor_artifact_id")
    geometry_path = Path(_required_string(operator, "geometry_source_path"))
    geometry = identify_file(geometry_path, _required_string(operator, "geometry_source_sha256"))
    floor_root = Path(_required_string(operator, "current_floor_output_directory"))
    floor = identify_file(floor_root / "authoritative_floor_plane.npz")
    current_floor_manifest = floor_root / "floor-completion-manifest.json"
    if not current_floor_manifest.is_file():
        if p08_floor_manifest_path is None:
            raise XR02LiveContractError("current floor-completion manifest is unavailable")
        current_floor_manifest = p08_floor_manifest_path
    static_entry = _selected_runtime_entry(operator, floor_id)
    static_rerun = identify_file(
        Path(_required_string(static_entry, "path")),
        _required_string(static_entry, "sha256"),
    )
    p06 = identify_file(p06_path)
    p07 = identify_file(p07_path)
    p08_manifest = identify_file(current_floor_manifest)
    policy_epoch = "" if camera_policy_sha256 is None else f"-{camera_policy_sha256[:8]}"
    scene_epoch = f"office-{geometry.sha256[:8]}-{floor.sha256[:8]}{policy_epoch}"
    scene = build_scene_context(
        scene_id="office",
        scene_epoch_id=scene_epoch,
        geometry_sha256=geometry.sha256,
        floor_sha256=floor.sha256,
        calibration_authority={"p06": p06.sha256, "p07": p07.sha256},
        camera_policy_sha256=camera_policy_sha256,
    )
    return AdoptedSceneSelection(
        scene=scene,
        operator_state=identify_file(operator_state_path),
        active_geometry_artifact_id=geometry_id,
        active_floor_artifact_id=floor_id,
        geometry=geometry,
        floor=floor,
        static_rerun=static_rerun,
        p06_calibration=p06,
        p07_pose=p07,
        p08_floor_manifest=p08_manifest,
    )


def identify_file(path: Path, expected_sha256: str | None = None) -> FileIdentity:
    resolved = path.resolve()
    if not resolved.is_file():
        raise XR02LiveContractError(f"required file is unavailable: {resolved.name}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise XR02LiveContractError(f"required file identity changed: {resolved.name}")
    return FileIdentity(str(resolved), resolved.stat().st_size, digest)


def _selected_runtime_entry(operator: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    records = operator.get("runtime_artifacts")
    if not isinstance(records, list):
        raise XR02LiveContractError("operator runtime artifact list is malformed")
    matches = [
        item
        for item in records
        if isinstance(item, dict) and item.get("artifact_id") == artifact_id
    ]
    if len(matches) != 1 or matches[0].get("selected") is not True:
        raise XR02LiveContractError("active floor artifact is not one selected runtime record")
    return matches[0]


def _json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XR02LiveContractError(f"cannot read scene selection: {path.name}") from error
    if not isinstance(loaded, dict):
        raise XR02LiveContractError("scene selection must be an object")
    return loaded


def _required_string(value: dict[str, Any], key: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str) or not selected:
        raise XR02LiveContractError(f"scene selection {key} is unavailable")
    return selected


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
