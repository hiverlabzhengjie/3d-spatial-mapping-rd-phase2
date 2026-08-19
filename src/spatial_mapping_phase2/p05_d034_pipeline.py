"""D034 solve-only camera comparison pipeline with no validation-point access."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from spatial_mapping_phase2.p04_pose_domain import PoseSolveError
from spatial_mapping_phase2.p05_fixed_center_orientation import (
    D034SolverConfig,
    select_d034_solve_set,
    solve_d034_orientation,
)
from spatial_mapping_phase2.p05_pose_candidates import (
    D034_CAMERA_IDS,
    build_d034_intrinsic_policy_candidates,
)


def evaluate_d034_camera(
    workspace: Path,
    correspondence_export_path: Path,
    fleet_manifest_path: Path,
    facility_export_path: Path,
    fixed_center_world_metres: list[float],
    fixed_center_authority: str,
    *,
    config: D034SolverConfig | None = None,
) -> dict[str, Any]:
    """Select four solve points and compare intrinsics without loading held-out coordinates."""

    selected_config = config or D034SolverConfig()
    correspondence = _read_json(correspondence_export_path)
    fleet = _read_json(fleet_manifest_path)
    facility = _read_json(facility_export_path)
    camera_id = str(correspondence.get("camera_id"))
    if camera_id not in D034_CAMERA_IDS:
        raise ValueError("D034 correspondence has an unsupported camera identity")
    if correspondence.get("status") != "ready-for-pose-input-review":
        raise ValueError("D034 correspondence is not ready for pose input review")
    approved = correspondence.get("approved_frame")
    if not isinstance(approved, dict) or approved.get("camera_id") != camera_id:
        raise ValueError("D034 approved frame camera binding is invalid")
    if approved.get("image_width_pixels") != 1920 or approved.get("image_height_pixels") != 1080:
        raise ValueError("D034 requires the compatible 1920x1080 stream profile")
    if _sha256(workspace / str(approved["relative_path"])) != approved.get("sha256"):
        raise ValueError("D034 approved frame identity differs from its immutable artifact")
    facility_reference = correspondence.get("facility_reference")
    if not isinstance(facility_reference, dict) or facility_reference.get(
        "export_sha256"
    ) != _sha256(facility_export_path):
        raise ValueError("D034 facility reference differs from the selected P02 export")
    landmarks = correspondence.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError("D034 correspondence landmarks are missing")
    solve_items = [item for item in landmarks if item.get("role") == "solve"]
    diagnostic_items = [item for item in landmarks if item.get("role") == "held-out"]
    if len(solve_items) < 4:
        raise ValueError(
            "D034 needs at least four eligible solve points before freezing exactly four"
        )
    if any(item.get("frame_id") != approved.get("frame_id") for item in landmarks):
        raise ValueError("D034 landmarks must bind to the approved immutable frame")
    center = np.asarray(fixed_center_world_metres, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("D034 fixed centre must contain three finite world-metre values")
    world, image = _landmark_arrays(solve_items)
    solve_ids = [str(item["landmark_id"]) for item in solve_items]
    selection = select_d034_solve_set(
        solve_ids,
        world.tolist(),
        image.tolist(),
        center.tolist(),
        int(approved["image_width_pixels"]),
        int(approved["image_height_pixels"]),
        minimum_information_ratio=selected_config.minimum_information_ratio,
    )
    indices = list(selection.selected_indices)
    selected_world = world[indices]
    selected_image = image[indices]
    candidates = build_d034_intrinsic_policy_candidates(fleet, camera_id)
    evaluations: list[dict[str, Any]] = []
    solved: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    for candidate in candidates:
        record = candidate.to_dict()
        try:
            solution = solve_d034_orientation(
                candidate.intrinsics,
                center.tolist(),
                selection.selected_landmark_ids,
                selected_world.tolist(),
                selected_image.tolist(),
                config=selected_config,
            )
        except PoseSolveError as error:
            record.update({"status": "rejected-solve-only", "rejection_reason": str(error)})
            evaluations.append(record)
            continue
        errors = sorted(solution.solve_reprojection_errors_pixels)[:3]
        trimmed_rmse = math.sqrt(sum(value * value for value in errors) / 3)
        score = (
            -float(len(solution.inlier_indices)),
            trimmed_rmse,
            solution.maximum_perturbation_rotation_degrees,
            solution.subset_spread_degrees,
        )
        record.update(
            {
                "status": "passed-solve-only-gates",
                "solve_only_rank_metrics": {
                    "consensus_count": len(solution.inlier_indices),
                    "trimmed_three_rmse_pixels": trimmed_rmse,
                    "maximum_perturbation_rotation_degrees": (
                        solution.maximum_perturbation_rotation_degrees
                    ),
                    "subset_spread_degrees": solution.subset_spread_degrees,
                },
                "orientation": solution.to_dict(),
            }
        )
        evaluations.append(record)
        solved.append((score, record))
    selected_label: str | None = None
    selection_reason: str | None = None
    operational_status = "rejected"
    if solved:
        solved.sort(key=lambda value: value[0])
        best_score, best_record = solved[0]
        tied = [
            record
            for score, record in solved
            if score[0] == best_score[0]
            and score[1] <= best_score[1] + selected_config.ambiguity_trimmed_rmse_pixels
        ]
        d033 = next(
            (record for record in tied if record.get("label") == "d033-candidate-b"), None
        )
        selected_record = d033 if d033 is not None else best_record
        selected_label = str(selected_record["label"])
        selected_record["status"] = "selected-provisional-solve-only"
        selection_reason = (
            "D033 retained within the frozen solve-only material-tie window"
            if d033 is not None
            else "lowest frozen solve-only rank among candidates passing every D034 gate"
        )
        operational_status = "provisional"
    return {
        "schema_version": "p05-d034-fixed-centre-camera-v1",
        "camera_id": camera_id,
        "operational_status": operational_status,
        "strict_validation_status": "unavailable-two-genuinely-unconsumed-points",
        "fixed_center": {
            "C_world_camera_metres": center.tolist(),
            "authority": fixed_center_authority,
            "optimization_influence": "exact and immutable; translation is never optimized",
        },
        "inputs": {
            "correspondence_export_path": str(correspondence_export_path.resolve()),
            "correspondence_export_sha256": _sha256(correspondence_export_path),
            "approved_frame_id": approved["frame_id"],
            "approved_frame_sha256": approved["sha256"],
            "fleet_manifest_sha256": _sha256(fleet_manifest_path),
            "facility_export_sha256": _sha256(facility_export_path),
        },
        "solve_selection": selection.to_dict(),
        "excluded_previously_inspected_diagnostic_point_ids": [
            str(item["landmark_id"]) for item in diagnostic_items
        ],
        "algorithm": {
            "decision_authority": "D034",
            "config": selected_config.to_dict(),
            "held_out_data_loaded": False,
            "legacy_6dof_influence": "none; separate diagnostic evidence only",
        },
        "intrinsic_candidates": evaluations,
        "selected_intrinsic_label": selected_label,
        "selected_intrinsic_reason": selection_reason,
        "authority_note": (
            "Solve-only provisional D034 evidence. Strict status cannot be accepted until exactly "
            "two genuinely unconsumed points are evaluated once against the frozen manifest."
        ),
        "facility_record_loaded_for_identity_only": bool(facility),
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
