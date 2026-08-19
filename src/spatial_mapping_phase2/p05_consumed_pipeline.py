"""Pipeline for explicitly non-validated consumed-eight provisional orientations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from spatial_mapping_phase2.p04_pose_domain import PoseSolveError
from spatial_mapping_phase2.p05_consumed_orientation import (
    ConsumedOrientationConfig,
    solve_consumed_eight_orientation,
)
from spatial_mapping_phase2.p05_fixed_center_orientation import rotation_difference_degrees
from spatial_mapping_phase2.p05_pose_candidates import (
    D034_CAMERA_IDS,
    build_d034_intrinsic_policy_candidates,
)


def evaluate_consumed_eight_camera(
    workspace: Path,
    correspondence_export_path: Path,
    fleet_manifest_path: Path,
    facility_export_path: Path,
    fixed_center_world_metres: list[float],
    *,
    config: ConsumedOrientationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate all eight points with Candidate B preferred and challengers retained."""

    selected_config = config or ConsumedOrientationConfig()
    correspondence = _read_json(correspondence_export_path)
    fleet = _read_json(fleet_manifest_path)
    camera_id = str(correspondence.get("camera_id"))
    if camera_id not in D034_CAMERA_IDS:
        raise ValueError("consumed-eight correspondence has an unsupported camera identity")
    if correspondence.get("status") != "ready-for-pose-input-review":
        raise ValueError("consumed-eight correspondence is not ready for pose input review")
    approved = correspondence.get("approved_frame")
    if not isinstance(approved, dict) or approved.get("camera_id") != camera_id:
        raise ValueError("consumed-eight approved-frame camera binding is invalid")
    if approved.get("image_width_pixels") != 1920 or approved.get("image_height_pixels") != 1080:
        raise ValueError("consumed-eight method requires the 1920x1080 stream profile")
    if _sha256(workspace / str(approved["relative_path"])) != approved.get("sha256"):
        raise ValueError("consumed-eight approved frame differs from its immutable artifact")
    facility_reference = correspondence.get("facility_reference")
    if not isinstance(facility_reference, dict) or facility_reference.get(
        "export_sha256"
    ) != _sha256(facility_export_path):
        raise ValueError("consumed-eight facility reference differs from selected P02")
    landmarks = correspondence.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != 8:
        raise ValueError("consumed-eight method requires exactly eight exported landmarks")
    if any(item.get("frame_id") != approved.get("frame_id") for item in landmarks):
        raise ValueError("consumed-eight landmarks must bind to the approved frame")
    center = np.asarray(fixed_center_world_metres, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("consumed-eight centre must contain three finite values")
    ids = [str(item["landmark_id"]) for item in landmarks]
    world, image = _landmark_arrays(landmarks)
    evaluations: list[dict[str, Any]] = []
    solved_records: list[dict[str, Any]] = []
    for candidate in build_d034_intrinsic_policy_candidates(fleet, camera_id):
        record = candidate.to_dict()
        try:
            solution = solve_consumed_eight_orientation(
                candidate.intrinsics,
                center.tolist(),
                ids,
                world.tolist(),
                image.tolist(),
                config=selected_config,
            )
        except PoseSolveError as error:
            record.update(
                {"status": "rejected-consumed-evidence", "rejection_reason": str(error)}
            )
            evaluations.append(record)
            continue
        orientation = solution.to_dict()
        record.update(
            {
                "status": "passed-consumed-evidence",
                "orientation": orientation,
            }
        )
        evaluations.append(record)
        solved_records.append(record)
    selected_record: dict[str, Any] | None = None
    if solved_records:
        d033 = next(
            (record for record in solved_records if record["label"] == "d033-candidate-b"),
            None,
        )
        if d033 is not None:
            selected_record = d033
        else:
            solved_records.sort(
                key=lambda record: (
                    -int(record["orientation"]["consensus_count"]),
                    float(record["orientation"]["inlier_reprojection_rmse_pixels"]),
                    float(record["orientation"]["maximum_perturbation_rotation_degrees"]),
                    str(record["label"]),
                )
            )
            selected_record = solved_records[0]
        selected_record["status"] = "selected-provisional-consumed-evidence"
    selected_solution = (
        None if selected_record is None else selected_record["orientation"]
    )
    selected_rotation = (
        None
        if selected_solution is None
        else np.asarray(selected_solution["T_world_from_camera"], dtype=np.float64)[:3, :3]
    )
    challenger_angles: dict[str, float] = {}
    if selected_rotation is not None:
        for record in evaluations:
            if record is selected_record or "orientation" not in record:
                continue
            orientation = record["orientation"]
            if not isinstance(orientation, dict):
                continue
            transform = np.asarray(orientation["T_world_from_camera"], dtype=np.float64)
            difference = rotation_difference_degrees(selected_rotation, transform[:3, :3])
            record["rotation_difference_from_d033_degrees"] = difference
            challenger_angles[str(record["label"])] = difference
    status = "rejected" if selected_solution is None else "provisional-consumed-evidence"
    strength = None if selected_solution is None else selected_solution["evidence_strength"]
    return {
        "schema_version": "p05-consumed-eight-provisional-camera-v1",
        "camera_id": camera_id,
        "status": status,
        "evidence_strength": strength,
        "validation_status": "unavailable; all eight observations consumed by estimation",
        "fixed_center": {
            "C_world_camera_metres": center.tolist(),
            "authority": "owner-directed P02 revision-3 mounting prior",
            "optimization_influence": "exact and immutable; translation absent from optimization",
        },
        "inputs": {
            "correspondence_export_path": str(correspondence_export_path.resolve()),
            "correspondence_export_sha256": _sha256(correspondence_export_path),
            "approved_frame_id": approved["frame_id"],
            "approved_frame_sha256": approved["sha256"],
            "fleet_manifest_sha256": _sha256(fleet_manifest_path),
            "facility_export_sha256": _sha256(facility_export_path),
            "original_roles": {
                "solve": [
                    str(item["landmark_id"])
                    for item in landmarks
                    if item.get("role") == "solve"
                ],
                "held-out-now-consumed": [
                    str(item["landmark_id"])
                    for item in landmarks
                    if item.get("role") == "held-out"
                ],
            },
        },
        "algorithm": {
            "method": "all-three-of-eight Wahba consensus plus rotation-only Huber refinement",
            "config": selected_config.to_dict(),
            "selected_intrinsic_policy": (
                "D033 Candidate B preferred; immutable challenger rollback if Candidate B fails"
            ),
            "strict_d034_influence": "none; selected D034 run remains immutable",
        },
        "intrinsic_candidates": evaluations,
        "selected_intrinsic_label": (
            None if selected_record is None else selected_record["label"]
        ),
        "selected_intrinsic_reason": (
            None
            if selected_record is None
            else "D033 Candidate B preferred because it passed"
            if selected_record["label"] == "d033-candidate-b"
            else "D033 Candidate B failed; selected best passing immutable rollback challenger"
        ),
        "selected_orientation": selected_solution,
        "challenger_rotation_differences_degrees": challenger_angles,
        "authority_note": (
            "Best-available provisional hypothesis from consumed evidence. It is not strict D034 "
            "validation, acceptance, connectivity authority or permission to start P06."
        ),
    }


def _landmark_arrays(items: list[dict[str, Any]]) -> tuple[Any, Any]:
    world = np.asarray(
        [
            [
                item["world_point"]["x_metres"],
                item["world_point"]["y_metres"],
                item["world_point"]["z_metres"],
            ]
            for item in items
        ],
        dtype=np.float64,
    )
    image = np.asarray(
        [[item["image_point"]["u"], item["image_point"]["v"]] for item in items],
        dtype=np.float64,
    )
    return world, image


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required immutable artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
