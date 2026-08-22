"""Build and record one combined cloud from the all-four-view DA3 diagnostic output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil  # type: ignore[import-untyped]
from numpy.typing import NDArray

from spatial_mapping_phase2.p06_da3_evaluation import resize_boolean_mask
from spatial_mapping_phase2.p07_geometry import (
    GeometryPatch,
    T_world_from_da3_T_camera_from_world,
    back_project_depth,
    concatenate_diagnostic_candidate,
    filter_by_confidence,
    filter_by_evaluation_mask,
    filter_by_range,
    floor_plane_diagnostic,
    p06_processed_image_to_rgb,
    patch_bounds,
    statistical_outlier_filter,
    transform_to_provisional_facility,
    validate_all4_da3_camera_order,
    voxel_downsample,
)
from spatial_mapping_phase2.rerun_camera_visualization import (
    RerunCameraFrustum,
    log_camera_frustum,
)

Array = NDArray[Any]
CONFIDENCE_MINIMUM = 1.0
MINIMUM_DEPTH_METRES = 0.5
MAXIMUM_DEPTH_METRES = 15.0
VOXEL_SIZE_METRES = 0.05
OUTLIER_NEIGHBOUR_COUNT = 20
OUTLIER_STANDARD_DEVIATION_RATIO = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-run-dir", type=Path, required=True)
    args = parser.parse_args()
    output_manifest = args.diagnostic_run_dir / "geometry-manifest.json"
    if output_manifest.exists() or (args.diagnostic_run_dir / "geometry").exists():
        parser.error("diagnostic geometry already exists; artifacts are immutable")
    try:
        result = run(args.diagnostic_run_dir)
    except Exception as error:
        _write_json(
            args.diagnostic_run_dir / "geometry-failure.json",
            {
                "schema_version": "p07-all4-da3-geometry-failure-v1",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "authority": "retained non-operational diagnostic failure only",
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


def run(run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    process = psutil.Process()
    inference_path = run_dir / "inference-manifest.json"
    inference = _read_json(inference_path)
    if (
        inference.get("success") is not True
        or inference.get("schema_version") != "p07-all4-pose-conditioned-da3-diagnostic-v1"
    ):
        raise ValueError("all-four-view DA3 inference manifest is not successful")
    camera_order = validate_all4_da3_camera_order(_string_list(inference, "camera_order"))
    repetitions = _list(inference, "repetitions")
    if len(repetitions) != 2:
        raise ValueError("all-four diagnostic requires exactly two repetitions")
    first_raw = _identity_path(_mapping(repetitions[0], "raw"), "selected raw")
    second_raw = _identity_path(_mapping(repetitions[1], "raw"), "repeat raw")
    if _mapping(inference, "repeatability").get("all_required_fields_exact") is not True:
        raise ValueError("all-four diagnostic required raw fields are not exact across repeats")
    with np.load(first_raw, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name])
            for name in ("depth", "confidence", "extrinsics", "intrinsics", "processed_images")
        }
        is_metric = int(np.asarray(archive["is_metric"]).item())
    with np.load(second_raw, allow_pickle=False) as archive:
        for name, selected in arrays.items():
            if not np.array_equal(selected, np.asarray(archive[name])):
                raise ValueError(f"repeat raw field differs during export: {name}")
    if is_metric != 1:
        raise ValueError("all-four-view DA3 output is not metric-flagged")

    input_cameras = {
        str(item["camera_id"]): item for item in _list(_mapping(inference, "inputs"), "cameras")
    }
    if tuple(input_cameras) != camera_order:
        raise ValueError("inference input camera provenance order changed")
    geometry_dir = run_dir / "geometry"
    geometry_dir.mkdir()
    world_patches: dict[str, GeometryPatch] = {}
    camera_records: list[dict[str, Any]] = []
    camera_visualizations: list[RerunCameraFrustum] = []
    for index, camera_id in enumerate(camera_order):
        camera_started = time.perf_counter()
        input_record = input_cameras[camera_id]
        mask_record = _mapping(input_record, "evaluation_mask")
        mask_path = _identity_path(mask_record, f"{camera_id} evaluation mask")
        source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source_mask is None:
            raise ValueError(f"{camera_id} evaluation mask is unreadable")
        depth = arrays["depth"][index]
        confidence = arrays["confidence"][index]
        K_processed = arrays["intrinsics"][index]
        processed_rgb = p06_processed_image_to_rgb(arrays["processed_images"][index])
        processed_mask = resize_boolean_mask(
            source_mask > 0, (int(depth.shape[0]), int(depth.shape[1]))
        )
        raw_patch = back_project_depth(
            camera_id,
            depth,
            K_processed,
            processed_rgb,
            confidence,
            processed_mask,
        )
        camera_dir = geometry_dir / camera_id
        camera_dir.mkdir()
        chain: list[dict[str, Any]] = []
        chain.append(
            _write_patch(
                camera_dir / "v00-raw-camera-space.npz",
                raw_patch,
                "raw-back-projection",
                {"source": "all-four-view pose-conditioned DA3 public aligned prediction"},
                None,
            )
        )
        confidence_result = filter_by_confidence(raw_patch, CONFIDENCE_MINIMUM)
        chain.append(
            _write_filter(camera_dir / "v01-confidence.npz", confidence_result, chain[-1])
        )
        range_result = filter_by_range(
            confidence_result.patch, MINIMUM_DEPTH_METRES, MAXIMUM_DEPTH_METRES
        )
        chain.append(_write_filter(camera_dir / "v02-range.npz", range_result, chain[-1]))
        mask_result = filter_by_evaluation_mask(range_result.patch)
        chain.append(_write_filter(camera_dir / "v03-mask.npz", mask_result, chain[-1]))
        voxel_result = voxel_downsample(mask_result.patch, VOXEL_SIZE_METRES)
        chain.append(_write_filter(camera_dir / "v04-voxel.npz", voxel_result, chain[-1]))
        outlier_result = statistical_outlier_filter(
            voxel_result.patch,
            OUTLIER_NEIGHBOUR_COUNT,
            OUTLIER_STANDARD_DEVIATION_RATIO,
        )
        chain.append(_write_filter(camera_dir / "v05-outlier.npz", outlier_result, chain[-1]))
        T_world_from_camera = T_world_from_da3_T_camera_from_world(arrays["extrinsics"][index])
        camera_visualizations.append(
            RerunCameraFrustum(
                camera_id=camera_id,
                T_world_from_camera=T_world_from_camera,
                K_processed=K_processed,
                frame_rgb=processed_rgb,
                image_plane_distance_metres=1.0,
                axis_length_metres=0.75,
            )
        )
        world_patch = transform_to_provisional_facility(outlier_result.patch, T_world_from_camera)
        input_T_camera_from_world = np.asarray(
            input_record["T_camera_from_world_for_DA3"], dtype=np.float64
        )
        input_T_world_from_camera = T_world_from_da3_T_camera_from_world(input_T_camera_from_world)
        pose_delta = _pose_delta(input_T_world_from_camera, T_world_from_camera)
        chain.append(
            _write_patch(
                camera_dir / "v06-da3-aligned-world.npz",
                world_patch,
                "DA3-predicted-T_world_from_camera-copy",
                {
                    "input_transform_direction": "T_camera_from_world_for_DA3",
                    "output_transform_direction": (
                        "T_camera_from_world predicted and public-aligned by DA3; inverted to "
                        "T_world_from_camera"
                    ),
                    "T_world_from_camera": T_world_from_camera.tolist(),
                    "pose_delta_from_supplied_seed": pose_delta,
                    "manual_pose_change": False,
                    "ICP": False,
                    "bundle_adjustment": False,
                },
                chain[-1],
            )
        )
        world_patches[camera_id] = world_patch
        camera_records.append(
            {
                "camera_id": camera_id,
                "view_index": index,
                "processed_intrinsics": K_processed.tolist(),
                "processed_intrinsics_array_sha256": _array_sha256(K_processed),
                "processed_frame_rgb_array_sha256": _array_sha256(processed_rgb),
                "processed_frame_resolution_xy": [
                    int(processed_rgb.shape[1]),
                    int(processed_rgb.shape[0]),
                ],
                "predicted_T_camera_from_world_array_sha256": _array_sha256(
                    arrays["extrinsics"][index]
                ),
                "T_world_from_camera": T_world_from_camera.tolist(),
                "pose_delta_from_supplied_seed": pose_delta,
                "processed_mask_array_sha256": _array_sha256(processed_mask),
                "filter_chain": chain,
                "final_point_count": world_patch.point_count,
                "represented_source_pixel_count": int(np.sum(world_patch.source_pixel_count)),
                "floor_diagnostic": floor_plane_diagnostic(world_patch),
                "bounds": patch_bounds(world_patch),
                "elapsed_seconds": time.perf_counter() - camera_started,
            }
        )

    combined = concatenate_diagnostic_candidate(
        "all-four-view-pose-conditioned-da3-diagnostic", world_patches
    )
    combined_path = geometry_dir / "all-four-da3-combined-diagnostic.npz"
    np.savez_compressed(
        combined_path,
        points=combined.points,
        colors_rgb=combined.colors_rgb,
        confidence=combined.confidence,
        source_pixel_count=combined.source_pixel_count,
        source_camera_index=combined.source_camera_index,
        camera_ids=np.asarray(combined.camera_ids),
    )
    rerun_path = run_dir / "all-four-da3-combined-diagnostic.rrd"
    entity_paths = write_geometry_review_rerun(rerun_path, combined, camera_visualizations)
    manifest: dict[str, Any] = {
        "schema_version": "p07-all4-da3-combined-geometry-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "success": True,
        "case_id": combined.case_id,
        "camera_order": list(camera_order),
        "inputs": {
            "inference_manifest": _identity(inference_path),
            "selected_raw": _identity(first_raw),
            "repeat_raw": _identity(second_raw),
            "raw_required_fields_exact": True,
        },
        "frozen_filter_parameters": {
            "confidence_minimum": CONFIDENCE_MINIMUM,
            "minimum_depth_metres": MINIMUM_DEPTH_METRES,
            "maximum_depth_metres": MAXIMUM_DEPTH_METRES,
            "per_view_voxel_size_metres": VOXEL_SIZE_METRES,
            "outlier_neighbour_count": OUTLIER_NEIGHBOUR_COUNT,
            "outlier_standard_deviation_ratio": OUTLIER_STANDARD_DEVIATION_RATIO,
            "retuned": False,
        },
        "camera_records": camera_records,
        "combined": {
            **_identity(combined_path),
            "point_count": combined.point_count,
            "represented_source_pixel_count": combined.represented_source_pixel_count,
            "source_camera_membership_preserved": True,
            "operation": "deterministic concatenation after unchanged per-view P07 filters",
            "cross_camera_voxel_merge": False,
            "surface_completion": False,
        },
        "rerun": {
            **_identity(rerun_path),
            "entity_paths": entity_paths,
            "point_cloud_entity_count": 1,
            "camera_frustum_count": len(camera_visualizations),
            "camera_rgb_image_plane_count": len(camera_visualizations),
            "camera_orientation_axis_count": len(camera_visualizations),
            "fixed_camera_centre_count": len(camera_visualizations),
            "world_space_camera_label_count": len(camera_visualizations),
            "camera_context_display_only": True,
            "camera_frustum_display_plane_distance_metres": 1.0,
            "camera_axis_display_length_metres": 0.75,
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "rss_end_bytes": process.memory_info().rss,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "operational_fusion_gate_status": "rejected-unchanged",
        "accepted_geometry": False,
        "replaces_D041_working_geometry": False,
        "authority": (
            "all-four-view pose-conditioned DA3 diagnostic only; one combined inspection cloud; "
            "no pose, edge, facility-frame, XYZ-accuracy, operational-fusion or "
            "accepted-geometry authority"
        ),
    }
    manifest_path = run_dir / "geometry-manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest": _identity(manifest_path)}


def write_geometry_review_rerun(
    path: Path, combined: Any, cameras: list[RerunCameraFrustum]
) -> list[str]:
    import rerun as rr
    import rerun.blueprint as rrb
    from rerun.error_utils import RerunWarning

    warnings.simplefilter("error", RerunWarning)
    rr.init(
        "p07-all-four-view-pose-conditioned-da3-diagnostic",
        recording_id="all-four-da3-combined-geometry-review",
        spawn=False,
    )
    rr.save(str(path))
    cloud_path = "diagnostic/all_four_da3/combined_actual_rgb"
    centre_path = "diagnostic/all_four_da3/fixed_camera_centres"
    axis_path = "diagnostic/all_four_da3/fixed_optical_axes"
    camera_root = "diagnostic/all_four_da3/cameras"
    label_root = "diagnostic/all_four_da3/camera_labels"
    rr.log(
        cloud_path,
        rr.Points3D(combined.points, colors=combined.colors_rgb, radii=0.018),
        static=True,
    )
    centres = np.asarray([camera.T_world_from_camera[:3, 3] for camera in cameras])
    axes = np.asarray([camera.T_world_from_camera[:3, 2] for camera in cameras])
    rr.log(
        centre_path,
        rr.Points3D(
            centres,
            radii=0.12,
            colors=[[255, 214, 64]] * len(cameras),
        ),
        static=True,
    )
    rr.log(axis_path, rr.Arrows3D(origins=centres, vectors=axes), static=True)
    paths = [f"/{cloud_path}", f"/{centre_path}", f"/{axis_path}"]
    for index, camera in enumerate(cameras):
        label_offset = np.array(
            [(-0.45 if index % 2 == 0 else 0.45), 0.0, 0.72 + 0.08 * index],
            dtype=np.float64,
        )
        paths.extend(
            log_camera_frustum(
                rr,
                camera_root,
                label_root,
                camera,
                label_position_world=camera.T_world_from_camera[:3, 3] + label_offset,
                label_text=f"{camera.camera_id} | calibrated fixed working pose",
            )
        )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/",
                contents="diagnostic/all_four_da3/**",
                name="Static reconstruction | combined geometry review",
            ),
            auto_layout=False,
            auto_views=False,
        ),
        make_active=True,
        make_default=True,
    )
    rr.disconnect()
    return paths


def _write_filter(path: Path, result: Any, parent: dict[str, Any]) -> dict[str, Any]:
    return _write_patch(
        path,
        result.patch,
        result.operation,
        result.statistics(),
        parent,
    )


def _write_patch(
    path: Path,
    patch: GeometryPatch,
    operation: str,
    parameters: dict[str, Any],
    parent: dict[str, Any] | None,
) -> dict[str, Any]:
    np.savez_compressed(
        path,
        points=patch.points,
        colors_rgb=patch.colors_rgb,
        confidence=patch.confidence,
        evaluation_mask_keep=patch.evaluation_mask_keep,
        pixel_uv=patch.pixel_uv,
        source_pixel_count=patch.source_pixel_count,
    )
    return {
        **_identity(path),
        "version": path.stem.split("-", 1)[0],
        "camera_id": patch.camera_id,
        "frame_id": patch.frame_id,
        "units": patch.units,
        "operation": operation,
        "parameters": parameters,
        "parent_sha256": None if parent is None else parent["sha256"],
        "point_count": patch.point_count,
        "represented_source_pixel_count": int(np.sum(patch.source_pixel_count)),
        "array_sha256": {
            "points": _array_sha256(patch.points),
            "colors_rgb": _array_sha256(patch.colors_rgb),
            "confidence": _array_sha256(patch.confidence),
            "evaluation_mask_keep": _array_sha256(patch.evaluation_mask_keep),
            "pixel_uv": _array_sha256(patch.pixel_uv),
            "source_pixel_count": _array_sha256(patch.source_pixel_count),
        },
    }


def _pose_delta(input_T: Array, output_T: Array) -> dict[str, float]:
    if np.array_equal(input_T, output_T):
        return {
            "camera_centre_shift_metres": 0.0,
            "rotation_change_degrees": 0.0,
        }
    translation = np.asarray(output_T[:3, 3] - input_T[:3, 3], dtype=np.float64)
    input_rotation = _nearest_proper_rotation(input_T[:3, :3])
    output_rotation = _nearest_proper_rotation(output_T[:3, :3])
    relative_rotation = output_rotation @ input_rotation.T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "camera_centre_shift_metres": float(np.linalg.norm(translation)),
        "rotation_change_degrees": math.degrees(math.acos(cosine)),
    }


def _nearest_proper_rotation(value: Array) -> Array:
    left, _, right_transpose = np.linalg.svd(np.asarray(value, dtype=np.float64))
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return np.asarray(rotation, dtype=np.float64)


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _identity_path(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or _identity(path)["sha256"] != record.get("sha256"):
        raise ValueError(f"{label} identity mismatch")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


if __name__ == "__main__":
    main()
