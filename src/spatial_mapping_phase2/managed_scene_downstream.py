"""Scene-local adapters for post-review updates and XR02 live operations.

The accepted XR02 worker still uses its frozen four-camera boundary.  Managed scenes therefore
materialize a private, hash-bound compatibility package that aliases their four scene camera IDs
to the worker's canonical camera slots.  Calibration, geometry, floor, endpoints and policy all
come from the managed scene; no Office evidence is copied into a new scene.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from spatial_mapping_phase2.managed_scene_capture import (
    SceneCaptureService,
    SceneEndpointLoader,
)
from spatial_mapping_phase2.p08_scene_updates import SceneUpdateSchedule
from spatial_mapping_phase2.p08_workflow import P08WorkflowError, SceneWorkspaceRepository
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS as XR02_CAMERA_IDS
from spatial_mapping_phase2.xr01_update_pipeline import (
    CapturedJpeg,
    SceneUpdatePipelineAdapter,
)
from spatial_mapping_phase2.xr02_deployment import load_xr02_deployment
from spatial_mapping_phase2.xr03_camera_policy import (
    CameraPolicyRepository,
    SceneCameraPolicy,
)
from spatial_mapping_phase2.xr03_live_operations import SupervisedXR02Worker


class ManagedSceneDownstreamError(RuntimeError):
    """Raised when scene-local accepted inputs cannot satisfy a downstream contract."""


class _LiveWorker(Protocol):
    def status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ManagedFreshFrameProvider:
    """Expose ephemeral managed-scene previews to the scene-update pipeline."""

    service: SceneCaptureService

    def capture_jpeg(self, camera_id: str) -> CapturedJpeg:
        frame = self.service.preview(camera_id)
        return CapturedJpeg(camera_id, frame.content, frame.observed_at_utc)


@dataclass(frozen=True)
class ManagedSceneUpdatePipelineAdapter:
    """Resolve the accepted scene calibration baseline only when an update is run."""

    repository: SceneWorkspaceRepository
    frame_provider: ManagedFreshFrameProvider
    camera_ids: tuple[str, ...]
    input_output_root: Path
    reconstruction: Any
    floor: Any
    floor_preview: Any
    camera_policy_provider: Callable[[], Mapping[str, Any]]

    def run(
        self,
        update_id: str,
        schedule: SceneUpdateSchedule,
        cancel_event: threading.Event,
    ) -> Mapping[str, Any]:
        state = self.repository.read_operator_state()
        if state is None:
            raise P08WorkflowError("scene update requires an approved scene result")
        geometry_id = state.get("active_geometry_artifact_id")
        if not isinstance(geometry_id, str) or not geometry_id:
            raise P08WorkflowError("scene update cannot resolve the active calibration baseline")
        baseline = _resolve_or_record_calibration_baseline(self.repository, geometry_id)
        return SceneUpdatePipelineAdapter(
            frame_provider=self.frame_provider,
            camera_ids=self.camera_ids,
            baseline_input_directory=baseline,
            input_output_root=self.input_output_root,
            reconstruction=self.reconstruction,
            floor=self.floor,
            floor_preview=self.floor_preview,
            camera_policy_provider=self.camera_policy_provider,
        ).run(update_id, schedule, cancel_event)


@dataclass(frozen=True)
class ManagedSceneLiveConfig:
    python_executable: Path
    worker_script: Path | None
    worker_module: str | None
    base_deployment_config: Path
    mediamtx_binary: Path
    ffmpeg_binary: Path
    recording_free_space_reserve_gb: float
    port: int


class ManagedSceneXR02Worker:
    """Lazily launch XR02 against one managed scene's current approved epoch."""

    def __init__(
        self,
        config: ManagedSceneLiveConfig,
        repository: SceneWorkspaceRepository,
        endpoint_keys: Mapping[str, str],
        secret_file: Path,
        output_root: Path,
        worker_factory: Callable[..., _LiveWorker] = SupervisedXR02Worker,
    ) -> None:
        self.config = config
        self.repository = repository
        self.endpoint_keys = dict(endpoint_keys)
        self.secret_file = secret_file.resolve()
        self.output_root = output_root.resolve()
        self.worker_factory = worker_factory
        self._worker: _LiveWorker | None = None
        self._deployment_signature: str | None = None
        self._lock = threading.RLock()

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._worker is not None:
                return dict(self._worker.status())
            self._prepare_deployment()
            return {
                "schema": "xr02.wp4.operator_status.v5",
                "active": False,
                "active_mode": None,
                "operator_state": "ready",
                "pending_run": None,
                "saved_recordings": [],
                "recent_live_runs": [],
                "recording_available": True,
                "output_directory": None,
            }

    def start_live(
        self,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "start_live",
            resumed_from_session_id=resumed_from_session_id,
            scene_update_id=scene_update_id,
        )

    def start_recording(self) -> dict[str, Any]:
        return self._call("start_recording")

    def stop(self, *, reason: str) -> dict[str, Any]:
        return self._call("stop", reason=reason)

    def open_rerun(self) -> dict[str, Any]:
        return self._call("open_rerun")

    def reset_trails(self) -> dict[str, Any]:
        return self._call("reset_trails")

    def export_evidence_snapshot(self) -> dict[str, Any]:
        return self._call("export_evidence_snapshot")

    def view_recording(self, session_id: str) -> dict[str, Any]:
        return self._call("view_recording", session_id)

    def save_recording(self, session_id: str, label: str) -> dict[str, Any]:
        return self._call("save_recording", session_id, label)

    def delete_recording(self, session_id: str, confirmation: str) -> dict[str, Any]:
        return self._call("delete_recording", session_id, confirmation)

    def close(self) -> None:
        with self._lock:
            if self._worker is not None:
                self._worker.close()
                self._worker = None

    def _call(self, method: str, *args: object, **kwargs: object) -> dict[str, Any]:
        with self._lock:
            worker = self._ensure_worker()
            action = getattr(worker, method)
            return dict(action(*args, **kwargs))

    def _ensure_worker(self) -> _LiveWorker:
        deployment_path, signature = self._prepare_deployment()
        if self._worker is not None and signature != self._deployment_signature:
            status = self._worker.status()
            if status.get("active") is True:
                raise ManagedSceneDownstreamError(
                    "stop Live operations before changing the accepted scene epoch"
                )
            self._worker.close()
            self._worker = None
        if self._worker is None:
            arguments = (
                "--deployment-config",
                str(deployment_path),
                "--output-root",
                str(self.output_root),
                "--mediamtx-binary",
                str(self.config.mediamtx_binary),
                "--record-trial-video",
                "--ffmpeg-binary",
                str(self.config.ffmpeg_binary),
                "--recording-free-space-reserve-gb",
                str(self.config.recording_free_space_reserve_gb),
            )
            self._worker = self.worker_factory(
                self.config.python_executable,
                self.config.worker_script,
                arguments,
                worker_module=self.config.worker_module,
                port=self.config.port,
            )
            self._deployment_signature = signature
        return self._worker

    def _prepare_deployment(self) -> tuple[Path, str]:
        scene = self.repository.load()
        camera_ids = tuple(camera.camera_id for camera in scene.cameras if camera.enabled)
        if len(camera_ids) != len(XR02_CAMERA_IDS):
            raise ManagedSceneDownstreamError(
                "XR02 Live operations currently require exactly four enabled cameras"
            )
        if tuple(self.endpoint_keys) != camera_ids:
            raise ManagedSceneDownstreamError("managed live camera bindings changed")
        state = self.repository.read_operator_state()
        if (
            state is None
            or state.get("geometry_approved") is not True
            or state.get("floor_approved") is not True
        ):
            raise ManagedSceneDownstreamError("approve the current final result first")
        geometry_id = state.get("active_geometry_artifact_id")
        floor_directory = state.get("current_floor_output_directory")
        if not isinstance(geometry_id, str) or not isinstance(floor_directory, str):
            raise ManagedSceneDownstreamError("accepted scene selection is incomplete")
        p06_source = (
            _resolve_or_record_calibration_baseline(self.repository, geometry_id)
            / "input-manifest.json"
        )
        floor_root = Path(floor_directory).resolve()
        floor_manifest = floor_root / "floor-completion-manifest.json"
        floor_plane = floor_root / "authoritative_floor_plane.npz"
        policy_repository = CameraPolicyRepository.open(self.repository.camera_policy_path)
        policy = policy_repository.active(require_lens=True, require_overlap=True)
        source_identities = {
            "operator": _sha256(self.repository.operator_state_path),
            "p06": _sha256(p06_source),
            "floor_manifest": _sha256(floor_manifest),
            "floor": _sha256(floor_plane),
            "policy": policy.sha256,
        }
        signature = _canonical_sha256(source_identities)
        package = self.repository.root / "managed-downstream" / "xr02" / signature[:16]
        deployment_path = package / "deployment.json"
        package.mkdir(parents=True, exist_ok=True)
        alias = dict(zip(camera_ids, XR02_CAMERA_IDS, strict=True))
        p06 = _json_object(p06_source)
        p06_cameras = _camera_records(p06)
        translated_p06 = {
            **p06,
            "cameras": [
                {**record, "camera_id": alias[camera_id]} for camera_id, record in p06_cameras
            ],
            "managed_scene_camera_aliases": alias,
        }
        p06_path = package / "p06-calibration.json"
        _write_json(p06_path, translated_p06)
        p07_path = package / "p07-pose.json"
        _write_json(
            p07_path,
            {
                "schema_version": "managed-scene-xr02-pose-compatibility-v1",
                "managed_scene_camera_aliases": alias,
                "cameras": [
                    _p07_camera(alias[camera_id], record) for camera_id, record in p06_cameras
                ],
            },
        )
        policy_path = package / "camera-policy.sqlite3"
        translated_policy = _translate_policy(policy, alias)
        if policy_path.is_file():
            retained_policy = CameraPolicyRepository.open(policy_path).active(
                require_lens=True, require_overlap=True
            )
            if retained_policy.sha256 != translated_policy.sha256:
                raise ManagedSceneDownstreamError(
                    "managed Live policy package differs from its scene epoch"
                )
        else:
            translated_repository = CameraPolicyRepository(
                policy_path,
                translated_policy.project_id,
                translated_policy.scene_id,
                translated_policy.camera_ids,
            )
            translated_repository.apply(
                "managed-scene-policy",
                translated_policy,
                expected_revision=None,
                confirm_impacts=False,
            )
        environment_path = package / "xr02-secrets.env"
        endpoint_loader = SceneEndpointLoader(self.endpoint_keys, self.secret_file)
        resolutions = endpoint_loader.resolve()
        if any(item.endpoint is None for item in resolutions):
            raise ManagedSceneDownstreamError("all four live camera endpoints must be configured")
        environment_path.write_text(
            "".join(
                f"PHASE2_RTSP_CAMERA_{index}={item.endpoint.for_read_only_adapter()}\n"
                for index, item in enumerate(resolutions, start=1)
                if item.endpoint is not None
            ),
            encoding="utf-8",
        )
        base = load_xr02_deployment(self.config.base_deployment_config)
        deployment = {
            "schema_version": "xr02-worker-deployment-v1",
            "operator_state": str(self.repository.operator_state_path),
            "p06_calibration_manifest": str(p06_path),
            "p07_geometry_manifest": str(p07_path),
            "p08_floor_manifest": str(floor_manifest),
            "p08_floor": str(floor_plane),
            "detector_model": str(base.detector_model),
            "reid_model": str(base.reid_model),
            "environment_file": str(environment_path),
            "camera_policy": str(policy_path),
            "ultralytics_config": str(base.ultralytics_config),
            "wp2_overlay": None if base.wp2_overlay is None else str(base.wp2_overlay),
            "hashes": {
                "p06": _sha256(p06_path),
                "p07": _sha256(p07_path),
                "p08_floor_manifest": source_identities["floor_manifest"],
                "p08_floor": source_identities["floor"],
                "detector": base.hashes.detector,
                "reid": base.hashes.reid,
            },
        }
        _write_json(deployment_path, deployment)
        return deployment_path, signature


def _p07_camera(camera_id: str, p06_camera: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = _mapping(p06_camera, "intrinsics")
    matrix = intrinsics.get("K_pinhole")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ManagedSceneDownstreamError("managed camera intrinsics are malformed")
    width = float(intrinsics.get("width_pixels", 0))
    height = float(intrinsics.get("height_pixels", 0))
    if width <= 0 or height <= 0:
        raise ManagedSceneDownstreamError("managed camera dimensions are malformed")
    scaled = [[float(value) for value in row] for row in matrix]
    for column in range(3):
        scaled[0][column] *= 504.0 / width
        scaled[1][column] *= 280.0 / height
    seed = _mapping(p06_camera, "seed_transform")
    transform = seed.get("T_world_from_camera")
    if not isinstance(transform, list) or len(transform) != 4:
        raise ManagedSceneDownstreamError("managed camera transform is malformed")
    return {
        "camera_id": camera_id,
        "processed_intrinsics": scaled,
        "T_world_from_camera": transform,
    }


def _resolve_or_record_calibration_baseline(
    repository: SceneWorkspaceRepository, geometry_id: str
) -> Path:
    selection_path = repository.root / "managed-downstream" / "calibration-baseline.json"
    if selection_path.is_file():
        selection = _json_object(selection_path)
        selected_path = selection.get("path")
        expected_sha256 = selection.get("input_manifest_sha256")
        if not isinstance(selected_path, str) or not isinstance(expected_sha256, str):
            raise ManagedSceneDownstreamError("managed calibration baseline record is malformed")
        baseline = Path(selected_path).resolve()
        if _sha256(baseline / "input-manifest.json") != expected_sha256:
            raise ManagedSceneDownstreamError("managed calibration baseline identity changed")
        return baseline
    baseline = repository.root / "calibrated-reconstruction-inputs" / geometry_id
    manifest_path = baseline / "input-manifest.json"
    if not manifest_path.is_file():
        raise ManagedSceneDownstreamError("managed calibration baseline is unavailable")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        selection_path,
        {
            "schema_version": "managed-scene-calibration-baseline-v1",
            "path": str(baseline.resolve()),
            "input_manifest_sha256": _sha256(manifest_path),
            "authority": "fixed accepted intrinsics and camera poses for downstream operations",
        },
    )
    return baseline


def _translate_policy(policy: SceneCameraPolicy, aliases: Mapping[str, str]) -> SceneCameraPolicy:
    return SceneCameraPolicy.build(
        policy.project_id,
        "office",
        XR02_CAMERA_IDS,
        [
            {
                "group_id": group.group_id,
                "lens_model": group.lens_model,
                "camera_ids": [aliases[camera_id] for camera_id in group.camera_ids],
            }
            for group in policy.intrinsic_groups
        ],
        [
            {
                "camera_id_a": aliases[review.camera_id_a],
                "camera_id_b": aliases[review.camera_id_b],
                "verdict": review.verdict.value,
            }
            for review in policy.overlap_pair_reviews
        ],
    )


def _camera_records(value: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records = value.get("cameras")
    if not isinstance(records, list):
        raise ManagedSceneDownstreamError("managed calibration camera list is malformed")
    output: list[tuple[str, dict[str, Any]]] = []
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("camera_id"), str):
            raise ManagedSceneDownstreamError("managed calibration camera record is malformed")
        output.append((str(item["camera_id"]), item))
    return output


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, dict):
        raise ManagedSceneDownstreamError(f"managed calibration {key} is malformed")
    return selected


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManagedSceneDownstreamError(f"managed input is unreadable: {path.name}") from error
    if not isinstance(value, dict):
        raise ManagedSceneDownstreamError(f"managed input must be an object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ManagedSceneDownstreamError(f"managed input is unavailable: {path.name}") from error


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
