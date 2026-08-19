"""Analyze P06 raw DA3 outputs and render explicitly non-authoritative diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p06_da3_evaluation import (
    RAW_FIELDS,
    P06EvidenceError,
    confidence_change_diagnostic,
    depth_change_diagnostic,
    file_sha256,
    masked_distribution,
    projection_depth_diagnostic,
    repeatability_comparison,
    resize_boolean_mask,
    validate_complete_case_matrix,
    validate_prediction_arrays,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("analysis output directory already exists; P06 never overwrites evidence")
    input_manifest = _read_json(args.input_manifest)
    run_manifest = _read_json(args.run_manifest)
    if run_manifest.get("input_manifest_sha256") != file_sha256(args.input_manifest):
        parser.error("run manifest is not bound to the supplied input manifest")
    if run_manifest.get("success") is not True:
        parser.error(
            "run manifest is not successful; retain failure rather than analyze partial data"
        )
    case_values = input_manifest.get("case_matrix")
    if not isinstance(case_values, list) or not all(
        isinstance(item, dict) for item in case_values
    ):
        parser.error("input case matrix is malformed")
    cases = validate_complete_case_matrix(case_values)
    camera_values = input_manifest.get("cameras")
    cameras = _records_by_id(camera_values, "camera_id", "input cameras")
    run_case_values = run_manifest.get("cases")
    run_cases = _records_by_id(run_case_values, "case_id", "run cases")
    if tuple(run_cases) != tuple(case.case_id for case in cases):
        parser.error("run cases differ from the frozen case order")
    args.output_dir.mkdir(parents=True)
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir()

    loaded: dict[str, list[dict[str, NDArray[Any]]]] = {}
    case_analyses: list[dict[str, Any]] = []
    for case in cases:
        run_case = run_cases[case.case_id]
        repetitions = run_case.get("repetitions")
        if run_case.get("success") is not True or not isinstance(repetitions, list):
            parser.error(f"case has no successful repetitions: {case.case_id}")
        if len(repetitions) != 2:
            parser.error(f"case does not contain exactly two repetitions: {case.case_id}")
        raw_values: list[dict[str, NDArray[Any]]] = []
        for repetition in repetitions:
            if not isinstance(repetition, dict):
                parser.error(f"case repetition record is malformed: {case.case_id}")
            raw_path = Path(str(repetition.get("raw_path")))
            if file_sha256(raw_path) != repetition.get("raw_sha256"):
                parser.error(f"raw output identity mismatch: {raw_path}")
            with np.load(raw_path, allow_pickle=False) as archive:
                arrays = {field: np.asarray(archive[field]) for field in RAW_FIELDS}
            validate_prediction_arrays(arrays, len(case.camera_ids))
            raw_values.append(arrays)
        loaded[case.case_id] = raw_values
        first, second = raw_values
        repeatability = {
            field: repeatability_comparison(first[field], second[field]) for field in RAW_FIELDS
        }
        view_summaries: list[dict[str, Any]] = []
        for index, camera_id in enumerate(case.camera_ids):
            mask = _processed_mask(cameras[camera_id], first["depth"][index].shape)
            view_summaries.append(
                {
                    "camera_id": camera_id,
                    "depth_metres": masked_distribution(first["depth"][index], mask),
                    "confidence": masked_distribution(first["confidence"][index], mask),
                    "evaluation_valid_pixel_count": int(np.count_nonzero(mask)),
                    "output_authority": "diagnostic only; no accepted geometry or accuracy",
                }
            )
            preview_path = preview_dir / f"{case.case_id}-{camera_id}.png"
            _write_preview(
                preview_path,
                first["processed_images"][index],
                first["depth"][index],
                first["confidence"][index],
                mask,
                f"{case.case_id} / {camera_id}",
            )
        case_analyses.append(
            {
                **case.to_dict(),
                "repeatability": repeatability,
                "views": view_summaries,
                "performance": {
                    "repetitions": [
                        {
                            "repetition": repetition.get("repetition"),
                            "elapsed_seconds": repetition.get("elapsed_seconds"),
                            "rss_gib_after_case": repetition.get("rss_gib_after_case"),
                            "vram_peak_allocated_gib": repetition.get(
                                "vram_peak_allocated_gib"
                            ),
                            "vram_peak_reserved_gib": repetition.get("vram_peak_reserved_gib"),
                        }
                        for repetition in repetitions
                        if isinstance(repetition, dict)
                    ]
                },
            }
        )

    single_case_id = {
        camera_id: f"single-{camera_id}" for camera_id in sorted(cameras)
    }
    baseline_comparisons: list[dict[str, Any]] = []
    projection_comparisons: list[dict[str, Any]] = []
    for case in cases:
        if not case.uses_input_camera_parameters:
            continue
        arrays = loaded[case.case_id][0]
        for index, camera_id in enumerate(case.camera_ids):
            baseline = loaded[single_case_id[camera_id]][0]
            mask = _processed_mask(cameras[camera_id], arrays["depth"][index].shape)
            baseline_comparisons.append(
                {
                    "case_id": case.case_id,
                    "camera_id": camera_id,
                    **depth_change_diagnostic(baseline["depth"][0], arrays["depth"][index], mask),
                    "confidence_change": confidence_change_diagnostic(
                        baseline["confidence"][0], arrays["confidence"][index], mask
                    ),
                }
            )
        for source_index, source_camera in enumerate(case.camera_ids):
            for target_index, target_camera in enumerate(case.camera_ids):
                if source_index == target_index:
                    continue
                source_mask = _processed_mask(
                    cameras[source_camera], arrays["depth"][source_index].shape
                )
                target_mask = _processed_mask(
                    cameras[target_camera], arrays["depth"][target_index].shape
                )
                projection_comparisons.append(
                    {
                        "case_id": case.case_id,
                        "source_camera_id": source_camera,
                        "target_camera_id": target_camera,
                        **projection_depth_diagnostic(
                            arrays["depth"][source_index],
                            arrays["intrinsics"][source_index],
                            np.asarray(
                                cameras[source_camera]["seed_transform"][
                                    "T_world_from_camera"
                                ],
                                dtype=np.float64,
                            ),
                            arrays["depth"][target_index],
                            arrays["intrinsics"][target_index],
                            np.asarray(
                                cameras[target_camera]["seed_transform"][
                                    "T_world_from_camera"
                                ],
                                dtype=np.float64,
                            ),
                            source_mask,
                            target_mask,
                        ),
                    }
                )

    cameras_123 = loaded["posed-diagnostic-cameras-1-2-3"][0]
    cameras_13 = loaded["posed-diagnostic-cameras-1-3"][0]
    removal_comparisons: list[dict[str, Any]] = []
    for camera_id, index_123, index_13 in (
        ("office-cam-01", 0, 0),
        ("office-cam-03", 2, 1),
    ):
        mask = _processed_mask(cameras[camera_id], cameras_123["depth"][index_123].shape)
        removal_comparisons.append(
            {
                "camera_id": camera_id,
                "baseline_case_id": "posed-diagnostic-cameras-1-3",
                "candidate_case_id": "posed-diagnostic-cameras-1-2-3",
                **depth_change_diagnostic(
                    cameras_13["depth"][index_13],
                    cameras_123["depth"][index_123],
                    mask,
                ),
                "confidence_change": confidence_change_diagnostic(
                    cameras_13["confidence"][index_13],
                    cameras_123["confidence"][index_123],
                    mask,
                ),
                "interpretation_boundary": (
                    "effect of including weak Camera 2 in the DA3 invocation; not proof that "
                    "Camera 2 is connected or that either output is accurate"
                ),
            }
        )

    cameras_23 = loaded["posed-diagnostic-cameras-2-3"][0]
    direct_pair_comparisons: list[dict[str, Any]] = []
    for camera_id, index_123, index_23 in (
        ("office-cam-02", 1, 1),
        ("office-cam-03", 2, 0),
    ):
        mask = _processed_mask(cameras[camera_id], cameras_123["depth"][index_123].shape)
        direct_pair_comparisons.append(
            {
                "camera_id": camera_id,
                "baseline_case_id": "posed-diagnostic-cameras-1-2-3",
                "candidate_case_id": "posed-diagnostic-cameras-2-3",
                **depth_change_diagnostic(
                    cameras_123["depth"][index_123],
                    cameras_23["depth"][index_23],
                    mask,
                ),
                "confidence_change": confidence_change_diagnostic(
                    cameras_123["confidence"][index_123],
                    cameras_23["confidence"][index_23],
                    mask,
                ),
                "interpretation_boundary": (
                    "effect of removing Camera 1 and evaluating the direct Camera 2/3 pair "
                    "with Camera 3 ordered first; not pose, connectivity or accuracy evidence"
                ),
            }
        )

    analysis: dict[str, Any] = {
        "schema_version": "p06-da3-analysis-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_manifest_sha256": file_sha256(args.input_manifest),
        "run_manifest_sha256": file_sha256(args.run_manifest),
        "authority": (
            "diagnostic comparison only; no accepted geometry, pose, connectivity, facility "
            "frame, scale, accuracy or fusion authority"
        ),
        "case_analyses": case_analyses,
        "single_view_comparisons": baseline_comparisons,
        "pose_conditioned_projection_depth_diagnostics": projection_comparisons,
        "camera_2_removal_sensitivity": removal_comparisons,
        "direct_camera_2_3_vs_three_view": direct_pair_comparisons,
        "policy": {
            "office-cam-01": "single-view metric baseline retained; multi-view diagnostic only",
            "office-cam-02": (
                "single-view weak/failure baseline plus D037 direct-pair diagnostic; weak "
                "non-anchor with no registration or connectivity authority"
            ),
            "office-cam-03": (
                "single-view metric baseline plus D037 direct-pair diagnostic as ordered "
                "provisional reference; no registration or connectivity authority"
            ),
            "office-cam-04": "single-view metric only by D036 policy",
            "pose_conditioned_component_selection": (
                "none; no validated component or accepted connectivity edge exists"
            ),
            "cloud_fusion": "prohibited and not performed",
        },
        "preview_manifest": [],
    }
    for preview_path in sorted(preview_dir.glob("*.png")):
        analysis["preview_manifest"].append(
            {
                "path": str(preview_path.resolve()),
                "sha256": file_sha256(preview_path),
                "byte_count": preview_path.stat().st_size,
                "authority": "visual diagnostic only",
            }
        )
    analysis_path = args.output_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "analysis_path": str(analysis_path),
                "analysis_sha256": file_sha256(analysis_path),
                "preview_count": len(analysis["preview_manifest"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _records_by_id(value: Any, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a list of records")
    records = {str(item.get(key)): item for item in value}
    if len(records) != len(value):
        raise ValueError(f"{label} contains duplicate identities")
    return records


def _processed_mask(camera: dict[str, Any], shape: tuple[int, ...]) -> NDArray[Any]:
    if len(shape) != 2:
        raise P06EvidenceError("processed depth shape must be two-dimensional")
    mask_record = camera.get("evaluation_mask")
    if not isinstance(mask_record, dict):
        raise P06EvidenceError("camera evaluation mask record is malformed")
    mask_path = Path(str(mask_record.get("path")))
    if file_sha256(mask_path) != mask_record.get("sha256"):
        raise P06EvidenceError("camera evaluation mask identity mismatch")
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise P06EvidenceError("camera evaluation mask is unreadable")
    return resize_boolean_mask(mask_image > 0, (shape[0], shape[1]))


def _write_preview(
    path: Path,
    image: NDArray[Any],
    depth: NDArray[Any],
    confidence: NDArray[Any],
    mask: NDArray[Any],
    label: str,
) -> None:
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_color = _colorize(depth, mask, cv2.COLORMAP_TURBO)
    confidence_color = _colorize(confidence, mask, cv2.COLORMAP_VIRIDIS)
    for panel in (bgr, depth_color, confidence_color):
        panel[~mask] = (180, 0, 180)
    canvas = np.concatenate([bgr, depth_color, confidence_color], axis=1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        f"{label} | INPUT / DEPTH / CONFIDENCE | DIAGNOSTIC - NO GEOMETRY AUTHORITY",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise P06EvidenceError(f"failed to write diagnostic preview: {path}")


def _colorize(
    values: NDArray[Any], mask: NDArray[Any], color_map: int
) -> NDArray[Any]:
    array = np.asarray(values, dtype=np.float64)
    selected = array[mask & np.isfinite(array)]
    if selected.size == 0:
        raise P06EvidenceError("preview has no finite selected values")
    low, high = np.quantile(selected, [0.05, 0.95])
    if not high > low:
        high = low + 1.0
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.asarray(cv2.applyColorMap((normalized * 255).astype(np.uint8), color_map))


if __name__ == "__main__":
    main()
