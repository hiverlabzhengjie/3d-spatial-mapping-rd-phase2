"""Compare P04 intrinsic candidates with independent PnP and held-out evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolveError,
    angular_difference_degrees,
    solve_camera_pose,
)

DISTORTION_COUNTS = {"pinhole": 0, "simple_radial": 1, "simple_divisional": 1, "radial": 2}
MOUNTING_PRIOR_WORLD_METRES = (8.959630, 5.516815, 3.10)
RANSAC_INITIALIZATION_THRESHOLD_PIXELS = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--intrinsic-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.workspace, arguments.intrinsic_manifest, arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def run(workspace: Path, intrinsic_manifests: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError("output directory already exists")
    state_path = workspace / "state.json"
    state = _read_json(state_path)
    if state.get("camera_id") != "office-cam-03" or state.get("revision") != 20:
        raise RuntimeError("pose run requires reviewed Camera 3 workspace revision 20")
    approved_frames = [frame for frame in state["frames"] if frame["status"] == "approved"]
    if len(approved_frames) != 1:
        raise RuntimeError("pose run requires exactly one approved frame")
    approved = approved_frames[0]
    frame_path = workspace / approved["relative_path"]
    if _sha256(frame_path) != approved["sha256"]:
        raise RuntimeError("approved frame hash does not match its immutable identity")
    solve_landmarks = [item for item in state["landmarks"] if item["role"] == "solve"]
    held_landmarks = [item for item in state["landmarks"] if item["role"] == "held-out"]
    if len(solve_landmarks) != 6 or len(held_landmarks) != 2:
        raise RuntimeError("pose pilot requires the reviewed six-solve/two-held-out set")
    if any(
        item.get("z_source") != "Owner-reported laser-pointer measurement"
        for item in state["landmarks"]
    ):
        raise RuntimeError("all reviewed landmarks must retain the owner-reported Z source")
    solve_world, solve_image = _landmark_arrays(solve_landmarks)
    held_world, held_image = _landmark_arrays(held_landmarks)

    candidates: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for manifest_path in intrinsic_manifests:
        manifest = _read_json(manifest_path)
        model = manifest["candidate"]["camera_model"]
        if model in seen_models or model not in DISTORTION_COUNTS:
            raise RuntimeError("provide one valid manifest for each distinct camera model")
        seen_models.add(model)
        _verify_intrinsic_binding(manifest, approved)
        camera_row = manifest["candidate"]["shared_camera_parameters"][0]
        distortion_count = DISTORTION_COUNTS[model]
        intrinsics = CameraIntrinsics(
            model,
            round(camera_row[0]),
            round(camera_row[1]),
            camera_row[2],
            camera_row[3],
            camera_row[4],
            camera_row[5],
            tuple(camera_row[6 : 6 + distortion_count]),
        )
        record: dict[str, Any] = {
            "camera_model": model,
            "intrinsics": intrinsics.to_dict(),
            "intrinsic_manifest_path": str(manifest_path.resolve()),
            "intrinsic_manifest_sha256": _sha256(manifest_path),
            "status": "provisional-pose-candidate",
        }
        try:
            solution = solve_camera_pose(
                intrinsics,
                solve_world,
                solve_image,
                held_world,
                held_image,
                ransac_threshold_pixels=RANSAC_INITIALIZATION_THRESHOLD_PIXELS,
            )
            record["pose"] = solution.to_dict()
            record["diagnostics"] = _diagnostics(
                solution.T_world_from_camera,
                solution.camera_position_world_metres,
                solution.optical_axis_world,
                manifest["candidate"]["individual_gravity_camera"],
            )
            record["leave_one_out"] = _leave_one_out(
                intrinsics, solve_landmarks, solve_world, solve_image, solution
            )
        except PoseSolveError as error:
            record["status"] = "rejected-pose-candidate"
            record["rejection_reason"] = str(error)
        candidates.append(record)
    if seen_models != set(DISTORTION_COUNTS):
        raise RuntimeError("pose comparison requires all four retained intrinsic models")

    output: dict[str, Any] = {
        "schema_version": "p04-pose-candidate-comparison-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "camera_id": state["camera_id"],
        "profile_version": approved["profile_version"],
        "coordinate_convention": {
            "world": "P02 provisional facility frame in metres: +X plan-left, +Y plan-down, +Z up",
            "camera": "OpenCV optical frame: +X right, +Y down, +Z forward",
            "reported_transform": "T_world_from_camera",
        },
        "inputs": {
            "workspace_revision": state["revision"],
            "workspace_state_sha256": _sha256(state_path),
            "approved_frame_id": approved["frame_id"],
            "approved_frame_sha256": approved["sha256"],
            "solve_landmark_ids": [item["landmark_id"] for item in solve_landmarks],
            "held_out_landmark_ids": [item["landmark_id"] for item in held_landmarks],
            "p02_facility_export_sha256": state["facility_reference"]["export_sha256"],
            "z_source": "Owner-reported laser-pointer measurement",
        },
        "method": {
            "initialization": (
                "OpenCV solvePnPRansac/EPNP on GeoCalib-undistorted normalized pixels"
            ),
            "ransac_threshold_pixels": RANSAC_INITIALIZATION_THRESHOLD_PIXELS,
            "ransac_threshold_authority": (
                "initialization-only setting selected after the retained 8-pixel attempt formed "
                "no four-point consensus; it is not an acceptance tolerance"
            ),
            "refinement": "SciPy bounded least_squares, Huber loss, inliers only",
            "refinement_rotation_bound_radians": 0.5,
            "refinement_translation_bound_metres": 5.0,
            "held_out_influence": "none",
            "mounting_prior_influence": "none; compared only after solving",
            "geocalib_vertical_convention": (
                "GeoCalib Gravity.vec3d is the image/world vertical-up direction "
                "(level camera [0,-1,0]); compared with world +Z"
            ),
        },
        "candidates": candidates,
        "authority_note": (
            "All results are provisional P04 evidence. P02 revision 3 has unknown horizontal "
            "uncertainty and cannot independently support a full XYZ-accuracy claim."
        ),
    }
    output_dir.mkdir(parents=True)
    for candidate in candidates:
        if candidate["status"] == "provisional-pose-candidate":
            candidate["overlay_artifact"] = _render_overlay(
                frame_path,
                candidate,
                solve_landmarks,
                held_landmarks,
                output_dir,
            )
    output_path = output_dir / "manifest.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _verify_intrinsic_binding(manifest: dict[str, Any], approved: dict[str, Any]) -> None:
    if manifest.get("status") != "provisional-candidate":
        raise RuntimeError("intrinsic manifest is not a retained provisional candidate")
    if (
        manifest.get("camera_id") != "office-cam-03"
        or manifest.get("profile_version") != approved["profile_version"]
    ):
        raise RuntimeError("intrinsic candidate camera/profile binding differs")
    matching = [frame for frame in manifest["frames"] if frame["sha256"] == approved["sha256"]]
    if (
        len(matching) != 1
        or matching[0]["camera_id"] != approved["camera_id"]
        or matching[0]["profile_version"] != approved["profile_version"]
    ):
        raise RuntimeError("intrinsic candidate does not contain the approved immutable P04 frame")


def _diagnostics(
    transform_rows: tuple[tuple[float, ...], ...],
    camera_position: tuple[float, float, float],
    optical_axis: tuple[float, float, float],
    gravity_rows: list[list[float]],
) -> dict[str, Any]:
    transform = np.asarray(transform_rows)
    rotation_world_from_camera = transform[:3, :3]
    gravity_camera = np.mean(np.asarray(gravity_rows, dtype=np.float64), axis=0)
    gravity_camera /= np.linalg.norm(gravity_camera)
    vertical_up_world = rotation_world_from_camera @ gravity_camera
    vertical_error = angular_difference_degrees(vertical_up_world, (0.0, 0.0, 1.0))
    position = np.asarray(camera_position)
    prior = np.asarray(MOUNTING_PRIOR_WORLD_METRES)
    delta = position - prior
    optical = np.asarray(optical_axis)
    ground_intersection: list[float] | None = None
    if optical[2] < -1e-9 and position[2] > 0:
        distance = -position[2] / optical[2]
        intersection = position + distance * optical
        ground_intersection = [float(intersection[0]), float(intersection[1]), 0.0]
    return {
        "geocalib_vertical_up_camera_mean": gravity_camera.tolist(),
        "geocalib_vertical_up_transformed_to_world": vertical_up_world.tolist(),
        "vertical_up_alignment_error_degrees": vertical_error,
        "camera_height_world_metres": float(position[2]),
        "optical_axis_ground_intersection_world_metres": ground_intersection,
        "viewing_downward": bool(optical[2] < 0),
        "d021_mounting_prior_world_metres": list(MOUNTING_PRIOR_WORLD_METRES),
        "d021_solved_minus_mounting_prior_metres": delta.tolist(),
        "d021_position_discrepancy_3d_metres": float(np.linalg.norm(delta)),
        "d021_horizontal_discrepancy_metres": float(np.linalg.norm(delta[:2])),
        "d021_height_discrepancy_metres": float(delta[2]),
    }


def _leave_one_out(
    intrinsics: CameraIntrinsics,
    landmarks: list[dict[str, Any]],
    world: Any,
    image: Any,
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
                world[keep],
                image[keep],
                ransac_threshold_pixels=RANSAC_INITIALIZATION_THRESHOLD_PIXELS,
            )
            position_delta = np.asarray(solution.camera_position_world_metres) - full_position
            rotation = np.asarray(solution.T_world_from_camera)[:3, :3]
            relative = full_rotation.T @ rotation
            relative_rvec, _ = cv2.Rodrigues(relative)
            case.update(
                {
                    "status": "solved",
                    "camera_position_delta_metres": position_delta.tolist(),
                    "camera_position_delta_norm_metres": float(np.linalg.norm(position_delta)),
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
            max(case["camera_position_delta_norm_metres"] for case in solved) if solved else None
        ),
        "maximum_orientation_delta_degrees": (
            max(case["orientation_delta_degrees"] for case in solved) if solved else None
        ),
    }


def _landmark_arrays(landmarks: list[dict[str, Any]]) -> tuple[Any, Any]:
    world = np.asarray(
        [
            [
                item["world_point"]["x_metres"],
                item["world_point"]["y_metres"],
                item["world_point"]["z_metres"],
            ]
            for item in landmarks
        ],
        dtype=np.float64,
    )
    image = np.asarray(
        [[item["image_point"]["u"], item["image_point"]["v"]] for item in landmarks],
        dtype=np.float64,
    )
    return world, image


def _render_overlay(
    frame_path: Path,
    candidate: dict[str, Any],
    solve_landmarks: list[dict[str, Any]],
    held_landmarks: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("cannot decode approved frame for pose overlay")
    pose = candidate["pose"]
    inliers = set(pose["ransac_inlier_indices"])
    for index, (landmark, projected, error) in enumerate(
        zip(
            solve_landmarks,
            pose["solve_projected_pixels"],
            pose["solve_reprojection_errors_pixels"],
            strict=True,
        )
    ):
        observed = _pixel_tuple(landmark["image_point"])
        estimate = (round(projected[0]), round(projected[1]))
        colour = (0, 210, 0) if index in inliers else (0, 140, 255)
        cv2.line(frame, observed, estimate, colour, 2, cv2.LINE_AA)
        cv2.circle(frame, observed, 7, colour, 2, cv2.LINE_AA)
        cv2.drawMarker(frame, estimate, (0, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{landmark['landmark_id']} {error:.1f}px",
            (observed[0] + 9, observed[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
    for landmark, projected, error in zip(
        held_landmarks,
        pose["held_out_projected_pixels"],
        pose["held_out_reprojection_errors_pixels"],
        strict=True,
    ):
        observed = _pixel_tuple(landmark["image_point"])
        estimate = (round(projected[0]), round(projected[1]))
        cv2.line(frame, observed, estimate, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, observed, 8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, estimate, (255, 0, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"HELD {landmark['landmark_id']} {error:.1f}px",
            (observed[0] + 9, observed[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(frame, (8, 8), (780, 48), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "circle=observed, red/magenta cross=projected, line=residual",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path = output_dir / f"overlay-{candidate['camera_model']}.jpg"
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError("cannot write pose overlay")
    return {"relative_path": path.name, "sha256": _sha256(path), "byte_count": path.stat().st_size}


def _pixel_tuple(point: dict[str, Any]) -> tuple[int, int]:
    return round(point["u"]), round(point["v"])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required input does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON object is malformed: {path.name}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
