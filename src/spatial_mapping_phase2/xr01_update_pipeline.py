"""Fresh RTSP-frame preparation and full-chain adapter for XR01 scene updates."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p08_scene_updates import SceneUpdateSchedule

Array = NDArray[Any]
UPDATE_PROCESS_RESOLUTION = 448


class SceneUpdatePipelineError(RuntimeError):
    """Raised when a fresh-frame update cannot produce a complete immutable candidate."""


@dataclass(frozen=True)
class CapturedJpeg:
    camera_id: str
    content: bytes
    observed_at_utc: str


class FreshFrameProvider(Protocol):
    def capture_jpeg(self, camera_id: str) -> CapturedJpeg: ...


class ReconstructionRunner(Protocol):
    def run(
        self,
        job_id: str,
        cancel_event: threading.Event,
        *,
        input_run_directory: Path | None = None,
        process_resolution: int | None = None,
        camera_ids: Sequence[str] | None = None,
        camera_policy_sha256: str | None = None,
    ) -> Mapping[str, Any]: ...


class DynamicFloorRunner(Protocol):
    def run_for_source(
        self,
        job_id: str,
        source_path: Path,
        source_sha256: str,
        cancel_event: threading.Event,
    ) -> Mapping[str, Any]: ...


class DynamicFloorPreviewRunner(Protocol):
    def run_for_geometry(
        self,
        run_directory: Path,
        geometry_manifest_path: Path,
        cancel_event: threading.Event,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PreparedUpdateInput:
    directory: Path
    input_manifest: dict[str, Any]


@dataclass(frozen=True)
class SceneUpdatePipelineAdapter:
    """Run capture, DA3, combined geometry, floor and final preview as one transaction."""

    frame_provider: FreshFrameProvider
    camera_ids: tuple[str, ...]
    baseline_input_directory: Path
    input_output_root: Path
    reconstruction: ReconstructionRunner
    floor: DynamicFloorRunner
    floor_preview: DynamicFloorPreviewRunner
    camera_policy_provider: Callable[[], Mapping[str, Any]] | None = None

    def run(
        self,
        update_id: str,
        schedule: SceneUpdateSchedule,
        cancel_event: threading.Event,
    ) -> Mapping[str, Any]:
        prepared = prepare_scene_update_inputs(
            update_id,
            schedule,
            self.frame_provider,
            self.camera_ids,
            self.baseline_input_directory,
            self.input_output_root,
            cancel_event,
        )
        reconstruction_arguments: dict[str, Any] = {
            "input_run_directory": prepared.directory,
            "process_resolution": UPDATE_PROCESS_RESOLUTION,
        }
        if self.camera_policy_provider is not None:
            policy_status = self.camera_policy_provider()
            active = policy_status.get("active_policy")
            if not isinstance(active, Mapping) or active.get("lens_complete") is not True:
                raise SceneUpdatePipelineError(
                    "scene update requires a complete active lens-group policy"
                )
            if tuple(active.get("camera_ids", ())) != self.camera_ids:
                raise SceneUpdatePipelineError(
                    "active camera policy differs from the scene-update camera roster"
                )
            policy_sha256 = active.get("policy_sha256")
            if not isinstance(policy_sha256, str) or len(policy_sha256) != 64:
                raise SceneUpdatePipelineError("active camera policy SHA-256 is malformed")
            reconstruction_arguments.update(
                camera_ids=self.camera_ids,
                camera_policy_sha256=policy_sha256,
            )
        reconstruction = dict(
            self.reconstruction.run(
                update_id,
                cancel_event,
                **reconstruction_arguments,
            )
        )
        geometry = _object(reconstruction, "combined_geometry")
        geometry_manifest = _identity_object(reconstruction, "geometry_manifest")
        floor_job_id = f"{update_id}-floor"
        floor = dict(
            self.floor.run_for_source(
                floor_job_id,
                Path(_string(geometry, "path")),
                _string(geometry, "sha256"),
                cancel_event,
            )
        )
        floor_directory = Path(_string(floor, "output_directory"))
        final = dict(
            self.floor_preview.run_for_geometry(
                floor_directory,
                Path(_string(geometry_manifest, "path")),
                cancel_event,
            )
        )
        rerun = _identity_object(final, "rerun")
        return {
            "schema_version": "xr01-scene-update-result-v1",
            "update_id": update_id,
            "trigger_mode": schedule.mode.value,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "capture": {
                "input_directory": str(prepared.directory),
                "input_manifest": _identity(prepared.directory / "input-manifest.json"),
                "frame_count_per_camera": schedule.capture_frame_count,
                "spacing_seconds": schedule.capture_spacing_seconds,
                "median_composite": schedule.capture_frame_count > 1,
            },
            "process_resolution": UPDATE_PROCESS_RESOLUTION,
            "reconstruction": reconstruction,
            "floor": floor,
            "final_preview": final,
            "geometry_artifact": _identity_object(reconstruction, "rerun"),
            "final_artifact": rerun,
            "geometry_source": geometry,
            "floor_job_id": floor_job_id,
            "floor_output_directory": str(floor_directory),
            "complete_chain": True,
        }


def prepare_scene_update_inputs(
    update_id: str,
    schedule: SceneUpdateSchedule,
    frame_provider: FreshFrameProvider,
    camera_ids: Sequence[str],
    baseline_input_directory: Path,
    output_root: Path,
    cancel_event: threading.Event,
) -> PreparedUpdateInput:
    """Capture concurrent sets and derive hash-bound pinhole inputs without endpoint disclosure."""

    baseline = baseline_input_directory.resolve()
    baseline_manifest_path = baseline / "input-manifest.json"
    baseline_run_path = baseline / "run-manifest.json"
    manifest = _read_json(baseline_manifest_path)
    records = _camera_records(manifest)
    ordered_ids = tuple(camera_ids)
    if tuple(records) != ordered_ids:
        raise SceneUpdatePipelineError("baseline camera order differs from the configured scene")
    output = output_root.resolve() / update_id
    if output.exists():
        raise SceneUpdatePipelineError("scene update input directory already exists")
    raw_directory = output / "raw"
    derivative_directory = output / "inputs"
    raw_directory.mkdir(parents=True)
    derivative_directory.mkdir()

    sets: list[dict[str, CapturedJpeg]] = []
    for index in range(schedule.capture_frame_count):
        if cancel_event.is_set():
            raise SceneUpdatePipelineError("scene update capture was cancelled")
        with ThreadPoolExecutor(max_workers=len(ordered_ids)) as executor:
            futures = {
                camera_id: executor.submit(frame_provider.capture_jpeg, camera_id)
                for camera_id in ordered_ids
            }
            captured = {camera_id: futures[camera_id].result() for camera_id in ordered_ids}
        if any(frame.camera_id != camera_id for camera_id, frame in captured.items()):
            raise SceneUpdatePipelineError("fresh frame provider changed camera identity")
        sets.append(captured)
        if index + 1 < schedule.capture_frame_count and cancel_event.wait(
            schedule.capture_spacing_seconds
        ):
            raise SceneUpdatePipelineError("scene update median capture was cancelled")

    updated_records: list[dict[str, Any]] = []
    for camera_id in ordered_ids:
        baseline_record = records[camera_id]
        frames: list[Array] = []
        source_records: list[dict[str, Any]] = []
        for index, captured_set in enumerate(sets, start=1):
            frame = captured_set[camera_id]
            raw_path = raw_directory / f"{camera_id}-{index:02d}.jpg"
            raw_path.write_bytes(frame.content)
            image = cv2.imdecode(np.frombuffer(frame.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise SceneUpdatePipelineError(f"{camera_id} returned an unreadable JPEG")
            frames.append(image)
            source_records.append(
                {
                    **_identity(raw_path),
                    "observed_at_utc": frame.observed_at_utc,
                    "set_index": index,
                }
            )
        if any(image.shape != frames[0].shape for image in frames):
            raise SceneUpdatePipelineError(f"{camera_id} frame dimensions changed within capture")
        composite = median_rgb_stack(frames) if len(frames) > 1 else frames[0]
        composite_path = raw_directory / f"{camera_id}-selected.png"
        if not cv2.imwrite(str(composite_path), composite):
            raise SceneUpdatePipelineError(f"failed to write {camera_id} selected frame")
        pinhole = _undistort(composite, baseline_record, camera_id)
        derivative_path = derivative_directory / f"{camera_id}-pinhole.png"
        if not cv2.imwrite(str(derivative_path), pinhole):
            raise SceneUpdatePipelineError(f"failed to write {camera_id} pinhole derivative")
        updated = json.loads(json.dumps(baseline_record))
        updated["source"] = {
            **_identity(composite_path),
            "captured_frames": source_records,
            "operation": "per-pixel channel median" if len(frames) > 1 else "single fresh frame",
            "frame_count": len(frames),
            "authority": "XR01 fresh RTSP scene-update input; no endpoint or credential retained",
        }
        updated["pinhole_derivative"] = {
            **_identity(derivative_path),
            "operation": "OpenCV simple-radial undistortion at source dimensions",
            "source_intrinsic_label": _object(baseline_record, "pinhole_derivative").get(
                "source_intrinsic_label"
            ),
            "authority": "derived scene-update model input; intrinsics and pose unchanged",
        }
        updated_records.append(updated)

    derived = json.loads(json.dumps(manifest))
    derived["schema_version"] = "xr01-scene-update-da3-input-v1"
    derived["created_at_utc"] = datetime.now(UTC).isoformat()
    derived["purpose"] = "fresh static-scene update using unchanged accepted camera conditions"
    derived["baseline_input_manifest"] = _identity(baseline_manifest_path)
    derived["capture_policy"] = {
        "mode": schedule.mode.value,
        "frame_count_per_camera": schedule.capture_frame_count,
        "spacing_seconds": schedule.capture_spacing_seconds,
        "median_composite": schedule.capture_frame_count > 1,
        "four_camera_sets_captured_concurrently": True,
    }
    derived["cameras"] = updated_records
    _write_json(output / "input-manifest.json", derived)
    shutil.copy2(baseline_run_path, output / "run-manifest.json")
    _write_json(
        output / "capture-provenance.json",
        {
            "schema_version": "xr01-scene-update-capture-v1",
            "update_id": update_id,
            "input_manifest": _identity(output / "input-manifest.json"),
            "baseline_run_manifest": _identity(output / "run-manifest.json"),
            "camera_ids": list(ordered_ids),
            "contains_credentials": False,
        },
    )
    return PreparedUpdateInput(output, derived)


def median_rgb_stack(images_bgr: Sequence[Array]) -> Array:
    if len(images_bgr) < 3 or len(images_bgr) % 2 == 0:
        raise SceneUpdatePipelineError("median stack requires an odd set of at least three frames")
    first = np.asarray(images_bgr[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise SceneUpdatePipelineError("median stack frames must be uint8 colour images")
    if any(np.asarray(image).shape != first.shape for image in images_bgr):
        raise SceneUpdatePipelineError("median stack frame shapes differ")
    stack = np.stack(images_bgr, axis=0)
    return np.asarray(np.median(stack, axis=0).astype(np.uint8))


def _undistort(image: Array, record: Mapping[str, Any], camera_id: str) -> Array:
    intrinsic = _object(record, "intrinsics")
    matrix = np.asarray(intrinsic.get("K_pinhole"), dtype=np.float64)
    distortion = intrinsic.get("distortion")
    height, width = image.shape[:2]
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise SceneUpdatePipelineError(f"{camera_id} baseline pinhole matrix is malformed")
    if not isinstance(distortion, list) or len(distortion) != 1:
        raise SceneUpdatePipelineError(f"{camera_id} baseline distortion is malformed")
    coefficients = np.asarray([float(distortion[0]), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix,
        coefficients,
        np.eye(3, dtype=np.float64),
        matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return np.asarray(
        cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    )


def _camera_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = manifest.get("cameras")
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise SceneUpdatePipelineError("baseline camera records are malformed")
    records = {str(item.get("camera_id")): dict(item) for item in values}
    if len(records) != len(values):
        raise SceneUpdatePipelineError("baseline camera IDs are duplicated")
    return records


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SceneUpdatePipelineError(f"workflow input is unreadable: {path.name}") from error
    if not isinstance(value, dict):
        raise SceneUpdatePipelineError(f"workflow input must be an object: {path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _object(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise SceneUpdatePipelineError(f"{key} must be an object")
    return dict(result)


def _identity_object(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = _object(value, key)
    _string(result, "path")
    _string(result, "sha256")
    return result


def _string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise SceneUpdatePipelineError(f"{key} must be a non-blank string")
    return result.strip()
