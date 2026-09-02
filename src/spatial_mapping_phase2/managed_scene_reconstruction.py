"""Scene-local source preparation and runtime configuration for generic DA3 reconstruction."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatial_mapping_phase2.managed_scene_capture import (
    SceneBundleStatus,
    SceneCaptureRepository,
    SceneCaptureStatus,
    SceneStorageMode,
)
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy


class ManagedSceneReconstructionError(ValueError):
    """Raised when a managed scene cannot produce exact reconstruction inputs."""


@dataclass(frozen=True, slots=True)
class ManagedSceneReconstructionConfig:
    """Machine-local common paths for a managed scene's isolated DA3 run."""

    python_executable: Path
    repository_root: Path
    source_directory: Path
    checkpoint_directory: Path
    process_resolution: int = 448

    def __post_init__(self) -> None:
        if self.process_resolution not in {392, 420, 448, 476, 504}:
            raise ManagedSceneReconstructionError(
                "managed reconstruction resolution is outside the supported DA3 envelope"
            )


class ManagedSceneReconstructionInputBuilder:
    """Materialize one exact, reviewable source image from the current complete bundle."""

    def __init__(
        self,
        repository: SceneCaptureRepository,
        scene_id: str,
        camera_ids: tuple[str, ...],
    ) -> None:
        self.repository = repository
        self.scene_id = scene_id
        self.camera_ids = camera_ids
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ManagedSceneReconstructionError(
                "managed reconstruction requires a unique camera roster"
            )

    def build(self, policy: SceneCameraPolicy, output_directory: Path) -> Path:
        """Create an immutable Office-independent source manifest for one reconstruction job."""

        if policy.camera_ids != self.camera_ids:
            raise ManagedSceneReconstructionError(
                "active camera policy differs from the managed scene roster"
            )
        current = self.repository.current_bundle()
        if current is None:
            raise ManagedSceneReconstructionError(
                "capture every scene camera and select one complete bundle"
            )
        bundle_path, bundle, selection_source = current
        if bundle.scene_id != self.scene_id or bundle.status is not SceneBundleStatus.COMPLETE:
            raise ManagedSceneReconstructionError(
                "the current capture bundle is incomplete or belongs to another scene"
            )
        selected_ids = tuple(frame.camera_id for frame in bundle.selected_frames)
        if selected_ids != self.camera_ids:
            raise ManagedSceneReconstructionError(
                "the current capture bundle differs from the enabled scene roster"
            )
        session = self.repository.read_session(bundle.source_session_id)
        if session.scene_id != self.scene_id:
            raise ManagedSceneReconstructionError(
                "the current capture session belongs to another scene"
            )
        session_ids = tuple(binding.camera_id for binding in session.camera_roster)
        if session_ids != self.camera_ids:
            raise ManagedSceneReconstructionError(
                "the current capture session differs from the enabled scene roster"
            )

        output = output_directory.resolve()
        created = False
        try:
            output.mkdir(parents=True, exist_ok=False)
            created = True
            inputs = output / "inputs"
            inputs.mkdir()
            selected_by_camera = {frame.camera_id: frame for frame in bundle.selected_frames}
            result_by_camera = {result.camera_id: result for result in session.results}
            camera_records = [
                self._materialize_camera(
                    camera_id,
                    selected_by_camera[camera_id],
                    result_by_camera[camera_id],
                    inputs,
                )
                for camera_id in self.camera_ids
            ]
            session_path = self.repository.session_directory(session.session_id) / "session.json"
            manifest = {
                "schema_version": "managed-scene-da3-source-input-v1",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "scene_id": self.scene_id,
                "camera_policy_sha256": policy.sha256,
                "camera_order": list(self.camera_ids),
                "capture_bundle": _identity(bundle_path),
                "capture_bundle_id": bundle.bundle_id,
                "capture_selection_source": selection_source,
                "capture_session": _identity(session_path),
                "capture_overall_skew_ns": bundle.overall_skew_ns,
                "cameras": camera_records,
                "authority": (
                    "scene-local immutable source-image derivatives from the current complete "
                    "capture bundle; no Office calibration or geometry authority"
                ),
            }
            manifest_path = output / "input-manifest.json"
            _write_json_exclusive(manifest_path, manifest)
            _write_json_exclusive(
                output / "run-manifest.json",
                {
                    "schema_version": "managed-scene-da3-source-input-run-v1",
                    "success": True,
                    "input_manifest": _identity(manifest_path),
                    "camera_policy_sha256": policy.sha256,
                    "source_images_materialized": len(camera_records),
                },
            )
        except Exception as error:
            if created and output.is_dir():
                failure = output / "failure.json"
                if not failure.exists():
                    _write_json_exclusive(
                        failure,
                        {
                            "schema_version": "managed-scene-da3-source-input-failure-v1",
                            "created_at_utc": datetime.now(UTC).isoformat(),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
            raise
        return output

    def _materialize_camera(
        self,
        camera_id: str,
        selected: Any,
        result: Any,
        inputs: Path,
    ) -> dict[str, Any]:
        if result.status is not SceneCaptureStatus.CAPTURED or result.artifact is None:
            raise ManagedSceneReconstructionError(f"{camera_id} has no captured source artifact")
        if selected.profile_version != result.profile.profile_version:
            raise ManagedSceneReconstructionError(f"{camera_id} selected profile is stale")
        source_frame = next(
            (frame for frame in result.frames if frame.frame_id == selected.frame_id), None
        )
        if source_frame is None:
            raise ManagedSceneReconstructionError(f"{camera_id} selected frame is stale")
        artifact_path = (self.repository.root / result.artifact.relative_path).resolve()
        if not artifact_path.is_relative_to(self.repository.root):
            raise ManagedSceneReconstructionError(f"{camera_id} source artifact escaped storage")
        if not artifact_path.is_file() or _sha256(artifact_path) != result.artifact.sha256:
            raise ManagedSceneReconstructionError(
                f"{camera_id} source artifact is missing or changed"
            )

        output_path = inputs / f"{camera_id}-selected.png"
        decoded_pts: int | None
        if result.artifact.storage_mode is SceneStorageMode.DECODED_FRAME_FALLBACK:
            image = cv2.imread(str(artifact_path), cv2.IMREAD_COLOR)
            decoded_pts = source_frame.source_pts
            selection_mode = "exact-single-decoded-fallback"
        else:
            image, decoded_pts = _decode_nearest_video_frame(
                artifact_path, source_frame.source_pts
            )
            selection_mode = (
                "exact-source-pts"
                if decoded_pts == source_frame.source_pts
                else "nearest-decodable-pts"
            )
        if image is None or image.shape[:2] != (
            result.profile.height_pixels,
            result.profile.width_pixels,
        ):
            raise ManagedSceneReconstructionError(
                f"{camera_id} materialized image differs from the captured stream profile"
            )
        if not cv2.imwrite(str(output_path), image):
            raise ManagedSceneReconstructionError(
                f"failed to write the selected reconstruction image for {camera_id}"
            )
        pts_delta = (
            None
            if source_frame.source_pts is None or decoded_pts is None
            else decoded_pts - source_frame.source_pts
        )
        time_base = source_frame.source_time_base
        delta_seconds = (
            None
            if pts_delta is None or time_base is None
            else pts_delta * time_base.numerator / time_base.denominator
        )
        return {
            "camera_id": camera_id,
            "source": _identity(output_path),
            "capture_artifact": {
                **_identity(artifact_path),
                "storage_mode": result.artifact.storage_mode.value,
            },
            "capture_profile": {
                "profile_version": result.profile.profile_version,
                "width_pixels": result.profile.width_pixels,
                "height_pixels": result.profile.height_pixels,
                "codec": result.profile.codec,
                "crop": result.profile.crop,
                "rotation_degrees": result.profile.rotation_degrees,
            },
            "selected_frame": {
                "frame_id": selected.frame_id,
                "source_pts": source_frame.source_pts,
                "source_time_base": (None if time_base is None else time_base.to_dict()),
                "acquisition_monotonic_ns": selected.acquisition_monotonic_ns,
            },
            "materialization": {
                "operation": "decode/copy selected capture evidence to lossless PNG",
                "selection_mode": selection_mode,
                "decoded_source_pts": decoded_pts,
                "source_pts_delta": pts_delta,
                "source_time_delta_seconds": delta_seconds,
                "color_semantics": (
                    "lossless PNG with ordinary image RGB semantics; OpenCV used BGR arrays "
                    "only at the encode boundary"
                ),
            },
            "evaluation_mask": {"rectangles_xyxy_derivative_pixels": []},
        }


def _decode_nearest_video_frame(
    path: Path, target_pts: int | None
) -> tuple[np.ndarray[Any, Any], int | None]:
    if target_pts is None:
        raise ManagedSceneReconstructionError(
            "packet-preserving capture requires a selected source PTS"
        )
    try:
        av = importlib.import_module("av")
        best_image: np.ndarray[Any, Any] | None = None
        best_pts: int | None = None
        best_distance: int | None = None
        with av.open(str(path), mode="r") as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                raise ManagedSceneReconstructionError("capture artifact has no video stream")
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                distance = abs(int(frame.pts) - target_pts)
                if best_distance is None or distance < best_distance:
                    best_image = np.asarray(frame.to_ndarray(format="bgr24"))
                    best_pts = int(frame.pts)
                    best_distance = distance
                if distance == 0:
                    break
        if best_image is None:
            raise ManagedSceneReconstructionError(
                "capture artifact contains no decodable frame with source timing"
            )
        return best_image, best_pts
    except ManagedSceneReconstructionError:
        raise
    except Exception as error:
        raise ManagedSceneReconstructionError(
            "failed to decode the selected capture artifact"
        ) from error


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise ManagedSceneReconstructionError(
            "immutable reconstruction input already exists"
        ) from error
