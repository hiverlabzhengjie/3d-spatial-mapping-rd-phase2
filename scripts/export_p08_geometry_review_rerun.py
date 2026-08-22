"""Build a camera-rich Geometry Review Rerun derivative from a completed reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from export_p07_all4_da3_cloud import _array_sha256, write_geometry_review_rerun

from spatial_mapping_phase2.p07_geometry import (
    GeometryPatch,
    p06_processed_image_to_rgb,
    validate_all4_da3_camera_order,
)
from spatial_mapping_phase2.rerun_camera_visualization import RerunCameraFrustum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_directory = arguments.run_dir.resolve()
    rerun_path = run_directory / "all-four-da3-combined-geometry-review-v2.rrd"
    manifest_path = run_directory / "geometry-review-manifest-v2.json"
    if rerun_path.exists() or manifest_path.exists():
        parser.error("geometry review output already exists; artifacts are immutable")
    result = run(run_directory, rerun_path, manifest_path)
    print(json.dumps(result, indent=2, sort_keys=True))


def run(run_directory: Path, rerun_path: Path, manifest_path: Path) -> dict[str, Any]:
    geometry_manifest_path = run_directory / "geometry-manifest.json"
    geometry_manifest = _read_json(geometry_manifest_path)
    if geometry_manifest.get("success") is not True:
        raise ValueError("reconstruction geometry manifest is not successful")
    camera_order = validate_all4_da3_camera_order(_string_list(geometry_manifest, "camera_order"))
    inputs = _mapping(geometry_manifest, "inputs")
    raw_path = _identity_path(_mapping(inputs, "selected_raw"), "selected DA3 raw output")
    combined_record = _mapping(geometry_manifest, "combined")
    combined_path = _identity_path(combined_record, "combined geometry")
    camera_records = _list(geometry_manifest, "camera_records")
    records_by_id = {str(record.get("camera_id")): record for record in camera_records}
    if set(records_by_id) != set(camera_order):
        raise ValueError("geometry camera records do not match camera order")

    with np.load(combined_path, allow_pickle=False) as archive:
        points = np.asarray(archive["points"])
        combined = GeometryPatch(
            camera_id="all-four-view-pose-conditioned-da3-diagnostic",
            points=points,
            colors_rgb=np.asarray(archive["colors_rgb"]),
            confidence=np.asarray(archive["confidence"]),
            evaluation_mask_keep=np.ones(len(points), dtype=np.bool_),
            pixel_uv=np.zeros((len(points), 2), dtype=np.int32),
            source_pixel_count=np.asarray(archive["source_pixel_count"]),
            frame_id="facility-world-x-plan-left-y-plan-down-z-up",
            units="metres",
        )
    with np.load(raw_path, allow_pickle=False) as archive:
        processed_images = np.asarray(archive["processed_images"])
        processed_intrinsics = np.asarray(archive["intrinsics"])
    if len(processed_images) != len(camera_order) or len(processed_intrinsics) != len(
        camera_order
    ):
        raise ValueError("selected DA3 raw output camera count changed")

    cameras: list[RerunCameraFrustum] = []
    camera_context: list[dict[str, Any]] = []
    for index, camera_id in enumerate(camera_order):
        record = records_by_id[camera_id]
        K_processed = processed_intrinsics[index]
        if _array_sha256(K_processed) != record.get("processed_intrinsics_array_sha256"):
            raise ValueError(f"{camera_id} processed intrinsic identity changed")
        frame_rgb = p06_processed_image_to_rgb(processed_images[index])
        T_world_from_camera = np.asarray(record.get("T_world_from_camera"), dtype=np.float64)
        camera = RerunCameraFrustum(
            camera_id=camera_id,
            T_world_from_camera=T_world_from_camera,
            K_processed=K_processed,
            frame_rgb=frame_rgb,
            image_plane_distance_metres=1.0,
            axis_length_metres=0.75,
        )
        cameras.append(camera)
        camera_context.append(
            {
                "camera_id": camera_id,
                "T_world_from_camera_array_sha256": _array_sha256(camera.T_world_from_camera),
                "processed_intrinsics_array_sha256": _array_sha256(camera.K_processed),
                "processed_frame_rgb_array_sha256": _array_sha256(camera.frame_rgb),
                "processed_frame_resolution_xy": list(camera.resolution_xy),
            }
        )

    entity_paths = write_geometry_review_rerun(rerun_path, combined, cameras)
    manifest = {
        "schema_version": "p08-static-geometry-review-rerun-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "success": True,
        "derivative_type": "visualization-only-camera-rich-geometry-review",
        "inputs": {
            "geometry_manifest": _identity(geometry_manifest_path),
            "combined_geometry": _identity(combined_path),
            "selected_raw_output": _identity(raw_path),
        },
        "camera_order": list(camera_order),
        "cameras": camera_context,
        "rerun": {**_identity(rerun_path), "entity_paths": entity_paths},
        "camera_frustum_count": len(cameras),
        "camera_rgb_image_plane_count": len(cameras),
        "camera_orientation_axis_count": len(cameras),
        "fixed_camera_centre_count": len(cameras),
        "world_space_camera_label_count": len(cameras),
        "combined_geometry_unchanged": True,
        "pose_or_intrinsic_adjustment": False,
        "camera_context_display_only": True,
        "authority": "visualization-only derivative of the exact completed static reconstruction",
    }
    _write_json_exclusive(manifest_path, manifest)
    return {**manifest, "manifest": _identity(manifest_path)}


def _identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _identity_path(record: dict[str, Any], label: str) -> Path:
    path = Path(_required_string(record, "path"))
    if not path.is_file() or _identity(path)["sha256"] != _required_string(record, "sha256"):
        raise ValueError(f"{label} identity mismatch")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"required mapping is malformed: {key}")
    return result


def _list(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    result = value.get(key)
    if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
        raise ValueError(f"required record list is malformed: {key}")
    return result


def _string_list(value: dict[str, Any], key: str) -> list[str]:
    result = value.get(key)
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise ValueError(f"required string list is malformed: {key}")
    return result


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"required string is malformed: {key}")
    return result.strip()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
