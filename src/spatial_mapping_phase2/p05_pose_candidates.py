"""P05 per-camera intrinsic-policy and independent pose candidate evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolveError,
    angular_difference_degrees,
    solve_camera_pose,
)

D033_INTRINSICS = CameraIntrinsics(
    "simple_radial",
    1920,
    1080,
    1401.71728515625,
    1401.7171630859375,
    960.0000610351562,
    540.0,
    (-0.27859073877334595,),
)
RANSAC_INITIALIZATION_THRESHOLD_PIXELS = 30.0
SUPPORTED_CAMERA_IDS = ("office-cam-01", "office-cam-02", "office-cam-04")
D034_CAMERA_IDS = ("office-cam-01", "office-cam-02", "office-cam-03", "office-cam-04")


@dataclass(frozen=True, slots=True)
class IntrinsicPolicyCandidate:
    label: str
    authority: str
    intrinsics: CameraIntrinsics
    source_manifest_sha256: str | None
    policy_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "authority": self.authority,
            "intrinsics": self.intrinsics.to_dict(),
            "source_manifest_sha256": self.source_manifest_sha256,
            "policy_role": self.policy_role,
        }


@dataclass(frozen=True, slots=True)
class RoleSwapAssignment:
    demoted_validation_ids: tuple[str, str]
    promoted_solve_ids: tuple[str, str]
    solve_items: tuple[dict[str, Any], ...]
    validation_items: tuple[dict[str, Any], ...]


def build_camera2_role_swap_assignments(
    landmarks: list[dict[str, Any]],
) -> tuple[RoleSwapAssignment, ...]:
    """Promote both held-out points and demote every pair of the six original solve points."""

    solve = tuple(item for item in landmarks if item.get("role") == "solve")
    held = tuple(item for item in landmarks if item.get("role") == "held-out")
    if len(solve) != 6 or len(held) != 2:
        raise ValueError("Camera 2 role swaps require six solve and two held-out landmarks")
    promoted_ids = (str(held[0]["landmark_id"]), str(held[1]["landmark_id"]))
    output: list[RoleSwapAssignment] = []
    for first, second in combinations(range(len(solve)), 2):
        demoted = (solve[first], solve[second])
        retained = tuple(
            item for index, item in enumerate(solve) if index not in {first, second}
        )
        output.append(
            RoleSwapAssignment(
                (str(demoted[0]["landmark_id"]), str(demoted[1]["landmark_id"])),
                promoted_ids,
                (*retained, *held),
                demoted,
            )
        )
    return tuple(output)


def evaluate_camera2_role_swaps(
    correspondence: dict[str, Any],
    candidates: tuple[IntrinsicPolicyCandidate, ...],
) -> dict[str, Any]:
    """Evaluate all requested 6/2 role swaps without changing the frozen source export."""

    if correspondence.get("camera_id") != "office-cam-02":
        raise ValueError("role-swap diagnostic requires the Camera 2 correspondence export")
    landmarks = correspondence.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError("Camera 2 correspondence landmarks are missing")
    assignments = build_camera2_role_swap_assignments(landmarks)
    cases: list[dict[str, Any]] = []
    for assignment in assignments:
        solve_world, solve_image = _landmark_arrays(list(assignment.solve_items))
        validation_world, validation_image = _landmark_arrays(
            list(assignment.validation_items)
        )
        for candidate in candidates:
            record: dict[str, Any] = {
                "candidate_label": candidate.label,
                "demoted_validation_ids": list(assignment.demoted_validation_ids),
                "promoted_solve_ids": list(assignment.promoted_solve_ids),
                "solve_landmark_ids": [item["landmark_id"] for item in assignment.solve_items],
                "validation_landmark_ids": [
                    item["landmark_id"] for item in assignment.validation_items
                ],
            }
            try:
                solution = solve_camera_pose(
                    candidate.intrinsics,
                    solve_world.tolist(),
                    solve_image.tolist(),
                    validation_world.tolist(),
                    validation_image.tolist(),
                    ransac_threshold_pixels=RANSAC_INITIALIZATION_THRESHOLD_PIXELS,
                )
                record.update(
                    {
                        "status": "diagnostic-pose-candidate",
                        "pose": solution.to_dict(),
                    }
                )
            except PoseSolveError as error:
                record.update({"status": "rejected-role-swap", "reason": str(error)})
            cases.append(record)
    return {
        "schema_version": "p05-camera2-role-swap-diagnostic-v1",
        "camera_id": "office-cam-02",
        "original_solve_ids": [
            item["landmark_id"] for item in landmarks if item.get("role") == "solve"
        ],
        "original_held_out_ids": [
            item["landmark_id"] for item in landmarks if item.get("role") == "held-out"
        ],
        "assignment_count": len(assignments),
        "candidate_count_per_assignment": len(candidates),
        "cases": cases,
        "authority_note": (
            "Role assignments are diagnostic and selected using former held-out evidence. "
            "They cannot establish independent validation or an accepted Camera 2 pose."
        ),
    }


def build_intrinsic_policy_candidates(
    fleet_manifest: dict[str, Any], camera_id: str
) -> tuple[IntrinsicPolicyCandidate, ...]:
    """Build the frozen D033/per-camera/fleet comparison set for one P05 camera."""

    if camera_id not in SUPPORTED_CAMERA_IDS:
        raise ValueError("P05 pose candidates support Cameras 1, 2 and 4")
    return _build_intrinsic_policy_candidates(fleet_manifest, camera_id)


def build_d034_intrinsic_policy_candidates(
    fleet_manifest: dict[str, Any], camera_id: str
) -> tuple[IntrinsicPolicyCandidate, ...]:
    """Build the same immutable challenger set for any D034 camera."""

    if camera_id not in D034_CAMERA_IDS:
        raise ValueError("D034 intrinsic candidates support Cameras 1, 2, 3 and 4")
    return _build_intrinsic_policy_candidates(fleet_manifest, camera_id)


def _build_intrinsic_policy_candidates(
    fleet_manifest: dict[str, Any], camera_id: str
) -> tuple[IntrinsicPolicyCandidate, ...]:
    if fleet_manifest.get("decision_authority") != "D027":
        raise ValueError("fleet manifest must retain D027 authority")
    models = fleet_manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("fleet manifest models are missing")
    radial = next(
        (item for item in models if item.get("camera_model") == "simple_radial"), None
    )
    if not isinstance(radial, dict):
        raise ValueError("simple-radial fleet evidence is missing")
    estimates = radial.get("per_camera_estimates")
    if not isinstance(estimates, list):
        raise ValueError("per-camera intrinsic estimates are missing")
    independent = next((item for item in estimates if item.get("camera_id") == camera_id), None)
    if not isinstance(independent, dict):
        raise ValueError(f"frozen independent estimate is missing for {camera_id}")
    evaluations = radial.get("camera3_pose_evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("retained fleet profile evidence is missing")

    def fleet_profile(method: str) -> CameraIntrinsics:
        label = f"leave-camera3-out:{method}"
        item = next((value for value in evaluations if value.get("label") == label), None)
        if not isinstance(item, dict) or not isinstance(item.get("intrinsics"), dict):
            raise ValueError(f"retained fleet profile is missing: {method}")
        return _intrinsics_from_dict(item["intrinsics"])

    candidates = (
        IntrinsicPolicyCandidate(
            "d033-candidate-b",
            "D033 provisional shared board-free default and nominal prior",
            D033_INTRINSICS,
            None,
            "starting-default-and-challenger",
        ),
        IntrinsicPolicyCandidate(
            "frozen-independent-geocalib",
            "D027 frozen camera-specific multi-frame GeoCalib estimate",
            CameraIntrinsics(
                "simple_radial",
                1920,
                1080,
                float(independent["fx_pixels"]),
                float(independent["fy_pixels"]),
                float(independent["cx_pixels"]),
                float(independent["cy_pixels"]),
                (float(independent["distortion"][0]),),
            ),
            str(independent["manifest_sha256"]),
            "mandatory-per-camera-challenger-and-rollback",
        ),
        IntrinsicPolicyCandidate(
            "d027-fleet-arithmetic",
            "D027 leave-Camera-3-out equal-camera arithmetic fleet profile",
            fleet_profile("equal-camera-arithmetic-mean"),
            None,
            "retained-fleet-challenger",
        ),
        IntrinsicPolicyCandidate(
            "d029-huber" if camera_id == "office-cam-01" else "d027-fleet-huber",
            (
                "D029 former Camera 1 robust-fleet default retained as a frozen challenger"
                if camera_id == "office-cam-01"
                else "D027 leave-Camera-3-out equal-camera Huber fleet profile"
            ),
            fleet_profile("equal-camera-componentwise-huber"),
            None,
            "additional-frozen-challenger",
        ),
    )
    return candidates


def evaluate_camera_pose_candidates(
    workspace: Path,
    correspondence_export_path: Path,
    fleet_manifest_path: Path,
    facility_export_path: Path,
) -> dict[str, Any]:
    """Evaluate frozen intrinsic candidates with solve-only PnP and held-out diagnostics."""

    correspondence = _read_json(correspondence_export_path)
    fleet = _read_json(fleet_manifest_path)
    facility = _read_json(facility_export_path)
    camera_id = str(correspondence.get("camera_id"))
    if camera_id not in SUPPORTED_CAMERA_IDS:
        raise ValueError("correspondence export has an unsupported P05 camera identity")
    if correspondence.get("status") != "ready-for-pose-input-review":
        raise ValueError("correspondence export is not ready for pose input review")
    approved = correspondence.get("approved_frame")
    if not isinstance(approved, dict) or approved.get("camera_id") != camera_id:
        raise ValueError("approved frame camera binding is invalid")
    if approved.get("image_width_pixels") != 1920 or approved.get("image_height_pixels") != 1080:
        raise ValueError("D033 comparison requires a compatible 1920x1080 frame")
    frame_path = workspace / str(approved["relative_path"])
    if _sha256(frame_path) != approved.get("sha256"):
        raise ValueError("approved frame identity differs from the immutable artifact")
    if correspondence["facility_reference"]["export_sha256"] != _sha256(
        facility_export_path
    ):
        raise ValueError("facility-reference identity differs from the selected P02 export")
    landmarks = correspondence.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError("correspondence landmarks are missing")
    solve_items = [item for item in landmarks if item.get("role") == "solve"]
    held_items = [item for item in landmarks if item.get("role") == "held-out"]
    if len(solve_items) != 6 or len(held_items) != 2:
        raise ValueError("P05 pose comparison requires six solve and two held-out landmarks")
    if any(item.get("frame_id") != approved.get("frame_id") for item in landmarks):
        raise ValueError("all landmarks must bind to the approved immutable frame")
    solve_world, solve_image = _landmark_arrays(solve_items)
    held_world, held_image = _landmark_arrays(held_items)
    prior = _mounting_prior(facility, camera_id)
    rough_pan = _rough_pan(facility, camera_id)
    candidates = build_intrinsic_policy_candidates(fleet, camera_id)
    gravity, gravity_source = _independent_gravity(fleet, camera_id)

    evaluations: list[dict[str, Any]] = []
    for candidate in candidates:
        record = candidate.to_dict()
        try:
            solution = solve_camera_pose(
                candidate.intrinsics,
                solve_world.tolist(),
                solve_image.tolist(),
                held_world.tolist(),
                held_image.tolist(),
                ransac_threshold_pixels=RANSAC_INITIALIZATION_THRESHOLD_PIXELS,
            )
        except PoseSolveError as error:
            record["status"] = "rejected-pose-candidate"
            record["rejection_reason"] = str(error)
            evaluations.append(record)
            continue
        record["status"] = "provisional-pose-candidate"
        record["pose"] = solution.to_dict()
        record["diagnostics"] = _physical_diagnostics(
            solution.T_world_from_camera,
            solution.camera_position_world_metres,
            solution.optical_axis_world,
            prior,
            rough_pan,
            gravity,
        )
        record["leave_one_out"] = _leave_one_out(
            candidate.intrinsics, solve_items, solve_world, solve_image, solution
        )
        evaluations.append(record)

    return {
        "schema_version": "p05-per-camera-pose-candidate-comparison-v1",
        "camera_id": camera_id,
        "profile_version": approved["profile_version"],
        "inputs": {
            "correspondence_export_path": str(correspondence_export_path.resolve()),
            "correspondence_export_sha256": _sha256(correspondence_export_path),
            "approved_frame_id": approved["frame_id"],
            "approved_frame_sha256": approved["sha256"],
            "solve_landmark_ids": [item["landmark_id"] for item in solve_items],
            "held_out_landmark_ids": [item["landmark_id"] for item in held_items],
            "fleet_manifest_sha256": _sha256(fleet_manifest_path),
            "facility_export_sha256": _sha256(facility_export_path),
            "independent_gravity_manifest_sha256": gravity_source,
        },
        "method": {
            "intrinsic_policy": "D033",
            "initialization": "OpenCV solvePnPRansac/EPNP on undistorted normalized pixels",
            "ransac_threshold_pixels": RANSAC_INITIALIZATION_THRESHOLD_PIXELS,
            "ransac_threshold_authority": "initialization only; not an acceptance tolerance",
            "refinement": "bounded SciPy least_squares with Huber loss on RANSAC inliers",
            "held_out_influence": "none",
            "mounting_prior_influence": "none; diagnostic after solving",
        },
        "candidates": evaluations,
        "authority_note": (
            "D033 Candidate B is the starting default, not forced calibration. Results remain "
            "provisional pending camera-specific review and the post-P05 D030 checkpoint."
        ),
    }


def _intrinsics_from_dict(value: dict[str, Any]) -> CameraIntrinsics:
    distortion = value.get("distortion")
    if not isinstance(distortion, list) or len(distortion) != 1:
        raise ValueError("retained simple-radial profile has invalid distortion")
    return CameraIntrinsics(
        "simple_radial",
        int(value["width_pixels"]),
        int(value["height_pixels"]),
        float(value["fx_pixels"]),
        float(value["fy_pixels"]),
        float(value["cx_pixels"]),
        float(value["cy_pixels"]),
        (float(distortion[0]),),
    )


def _landmark_arrays(
    items: list[dict[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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


def _mounting_prior(
    facility: dict[str, Any], camera_id: str
) -> NDArray[np.float64]:
    item = _camera_prior(facility, camera_id)
    point = item["C_world_mount_prior"]
    return np.asarray(
        [point["x_metres"], point["y_metres"], point["z_metres"]], dtype=np.float64
    )


def _rough_pan(facility: dict[str, Any], camera_id: str) -> NDArray[np.float64]:
    item = _camera_prior(facility, camera_id)
    return np.asarray(item["rough_pan_vector_world_xy"], dtype=np.float64)


def _camera_prior(facility: dict[str, Any], camera_id: str) -> dict[str, Any]:
    priors = facility.get("camera_mounting_priors")
    if not isinstance(priors, list):
        raise ValueError("facility camera priors are missing")
    item = next((value for value in priors if value.get("camera_id") == camera_id), None)
    if not isinstance(item, dict):
        raise ValueError(f"mounting prior is missing for {camera_id}")
    return item


def _independent_gravity(
    fleet: dict[str, Any], camera_id: str
) -> tuple[NDArray[np.float64], str]:
    radial = next(item for item in fleet["models"] if item["camera_model"] == "simple_radial")
    estimate = next(
        item for item in radial["per_camera_estimates"] if item["camera_id"] == camera_id
    )
    path = Path(str(estimate["manifest_path"]))
    if _sha256(path) != estimate["manifest_sha256"]:
        raise ValueError("independent GeoCalib manifest identity changed")
    manifest = _read_json(path)
    rows = np.asarray(manifest["candidate"]["individual_gravity_camera"], dtype=np.float64)
    gravity = np.mean(rows, axis=0)
    gravity /= np.linalg.norm(gravity)
    return gravity, str(estimate["manifest_sha256"])


def _physical_diagnostics(
    transform_rows: tuple[tuple[float, ...], ...],
    camera_position: tuple[float, float, float],
    optical_axis: tuple[float, float, float],
    mounting_prior: NDArray[np.float64],
    rough_pan_xy: NDArray[np.float64],
    gravity_camera: NDArray[np.float64],
) -> dict[str, Any]:
    transform = np.asarray(transform_rows)
    vertical_world = transform[:3, :3] @ gravity_camera
    position = np.asarray(camera_position)
    optical = np.asarray(optical_axis)
    delta = position - mounting_prior
    optical_xy = optical[:2]
    pan_error: float | None = None
    if np.linalg.norm(optical_xy) > 1e-9 and np.linalg.norm(rough_pan_xy) > 1e-9:
        denominator = float(np.linalg.norm(optical_xy) * np.linalg.norm(rough_pan_xy))
        cosine = float(np.dot(optical_xy, rough_pan_xy) / denominator)
        pan_error = math.degrees(
            math.acos(max(-1.0, min(1.0, cosine)))
        )
    ground_intersection: list[float] | None = None
    if optical[2] < -1e-9 and position[2] > 0:
        intersection = position + (-position[2] / optical[2]) * optical
        ground_intersection = [float(intersection[0]), float(intersection[1]), 0.0]
    return {
        "geocalib_vertical_up_camera_mean": gravity_camera.tolist(),
        "geocalib_vertical_up_transformed_to_world": vertical_world.tolist(),
        "vertical_up_alignment_error_degrees": angular_difference_degrees(
            tuple(float(value) for value in vertical_world), (0.0, 0.0, 1.0)
        ),
        "camera_height_world_metres": float(position[2]),
        "viewing_downward": bool(optical[2] < 0),
        "optical_axis_ground_intersection_world_metres": ground_intersection,
        "rough_pan_xy_alignment_error_degrees": pan_error,
        "d021_mounting_prior_world_metres": mounting_prior.tolist(),
        "d021_solved_minus_mounting_prior_metres": delta.tolist(),
        "d021_horizontal_discrepancy_metres": float(np.linalg.norm(delta[:2])),
        "d021_position_discrepancy_3d_metres": float(np.linalg.norm(delta)),
        "d021_height_discrepancy_metres": float(delta[2]),
    }


def _leave_one_out(
    intrinsics: CameraIntrinsics,
    landmarks: list[dict[str, Any]],
    world: NDArray[np.float64],
    image: NDArray[np.float64],
    full_solution: Any,
) -> dict[str, Any]:
    full_position = np.asarray(full_solution.camera_position_world_metres)
    full_rotation = np.asarray(full_solution.T_world_from_camera)[:3, :3]
    cases: list[dict[str, Any]] = []
    for index, landmark in enumerate(landmarks):
        keep = [value for value in range(len(landmarks)) if value != index]
        case: dict[str, Any] = {"omitted_landmark_id": landmark["landmark_id"]}
        try:
            solution = solve_camera_pose(
                intrinsics,
                world[keep].tolist(),
                image[keep].tolist(),
                ransac_threshold_pixels=RANSAC_INITIALIZATION_THRESHOLD_PIXELS,
            )
            position_delta = np.asarray(solution.camera_position_world_metres) - full_position
            rotation = np.asarray(solution.T_world_from_camera)[:3, :3]
            relative = full_rotation.T @ rotation
            relative_rvec, _ = cv2.Rodrigues(relative)
            case.update(
                {
                    "status": "solved",
                    "camera_position_delta_norm_metres": float(
                        np.linalg.norm(position_delta)
                    ),
                    "orientation_delta_degrees": math.degrees(
                        float(np.linalg.norm(relative_rvec))
                    ),
                }
            )
        except PoseSolveError as error:
            case.update({"status": "failed", "reason": str(error)})
        cases.append(case)
    solved = [case for case in cases if case["status"] == "solved"]
    return {
        "cases": cases,
        "solved_case_count": len(solved),
        "maximum_position_delta_metres": (
            max(float(case["camera_position_delta_norm_metres"]) for case in solved)
            if solved
            else None
        ),
        "maximum_orientation_delta_degrees": (
            max(float(case["orientation_delta_degrees"]) for case in solved)
            if solved
            else None
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required immutable artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
