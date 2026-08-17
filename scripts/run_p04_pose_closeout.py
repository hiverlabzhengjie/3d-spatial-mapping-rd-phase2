"""Run the bounded P04 Camera 3 desktop pose closeout and measured envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from spatial_mapping_phase2.p04_pose_closeout import (
    PoseEnvelope,
    application_projection_envelope,
    build_pose_envelope,
    derive_provisional_application_tolerances,
)
from spatial_mapping_phase2.p04_pose_domain import CameraIntrinsics

MOUNTING_PRIOR_WORLD_METRES = np.asarray((8.959630, 5.516815, 3.10))
P01_TAPE_HEIGHT_METRES = 3.15
EXPECTED_LABELS = {
    "camera3-frozen-historical",
    "leave-camera3-out:equal-camera-arithmetic-mean",
    "leave-camera3-out:equal-camera-componentwise-median",
    "leave-camera3-out:equal-camera-componentwise-huber",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--fleet-study-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = run(arguments.workspace, arguments.fleet_study_manifest, arguments.output_dir)
    print(json.dumps(output, indent=2, sort_keys=True))


def run(workspace: Path, fleet_study_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError("output directory already exists")
    state_path = workspace / "state.json"
    state = _read_json(state_path)
    fleet = _read_json(fleet_study_path)
    if state.get("revision") != 20 or fleet.get("decision_authority") != "D027":
        raise RuntimeError("closeout requires frozen revision 20 and the D027 fleet study")
    solve_items = [item for item in state["landmarks"] if item["role"] == "solve"]
    held_items = [item for item in state["landmarks"] if item["role"] == "held-out"]
    solve_world, solve_image = _landmark_arrays(solve_items)
    held_world, held_image = _landmark_arrays(held_items)
    radial = next(model for model in fleet["models"] if model["camera_model"] == "simple_radial")
    inputs = [
        item for item in radial["camera3_pose_evaluations"] if item["label"] in EXPECTED_LABELS
    ]
    if {item["label"] for item in inputs} != EXPECTED_LABELS:
        raise RuntimeError("closeout profile set is incomplete")

    results: list[dict[str, Any]] = []
    envelopes: list[PoseEnvelope] = []
    for item in sorted(inputs, key=lambda value: value["label"]):
        intrinsics = _intrinsics(item["intrinsics"])
        envelope = build_pose_envelope(
            intrinsics, solve_world, solve_image, held_world, held_image
        )
        envelopes.append(envelope)
        summary = envelope.summary()
        results.append(
            {
                "label": item["label"],
                "intrinsics": intrinsics.to_dict(),
                "source_manifests": item["source_manifests"],
                "envelope_summary": summary,
                "physical_summary": _physical_summary(envelope),
                "reference_case": _reference_case(envelope),
                "cases": [case.to_dict() for case in envelope.cases],
            }
        )
    combined = _combined_summary(envelopes)
    projection = application_projection_envelope(
        envelopes,
        minimum_focal_pixels=min(result["intrinsics"]["fx_pixels"] for result in results),
    )
    tolerances = derive_provisional_application_tolerances(combined, projection)
    output: dict[str, Any] = {
        "schema_version": "p04-pose-closeout-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "camera_id": "office-cam-03",
        "profile_version": "stream-profile-v1",
        "coordinate_convention": {
            "world": "P02 provisional facility frame: metres, +X plan-left, +Y plan-down, +Z up",
            "camera": "OpenCV optical frame: +X right, +Y down, +Z forward",
            "reported_transform": "T_world_from_camera",
        },
        "frozen_inputs": {
            "workspace_revision": state["revision"],
            "workspace_state_sha256": _sha256(state_path),
            "fleet_study_manifest_sha256": _sha256(fleet_study_path),
            "solve_landmark_ids": [item["landmark_id"] for item in solve_items],
            "held_out_landmark_ids": [item["landmark_id"] for item in held_items],
            "held_out_tuning_influence": "none",
        },
        "method": {
            "solve_points_in_each_refinement": 6,
            "initializers": ["ransac-epnp", "all-epnp", "all-sqpnp", "all-iterative"],
            "robust_losses": ["huber", "soft_l1", "cauchy"],
            "robust_scales_pixels": [3.0, 6.0, 12.0],
            "selection_rule": (
                "report complete stable envelope; do not choose minimum held-out residual"
            ),
            "reference_case": "ransac-epnp plus Huber loss at 6 pixels, fixed before execution",
        },
        "profiles": results,
        "combined_profile_and_solver_envelope": combined,
        "application_projection_envelope": projection,
        "provisional_application_tolerances": tolerances,
        "authority_note": (
            "Measured closeout evidence only. P02 horizontal uncertainty remains unknown; the "
            "control tower, not P04, decides stage and tolerance acceptance."
        ),
    }
    output_dir.mkdir(parents=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _physical_summary(envelope: PoseEnvelope) -> dict[str, Any]:
    positions = np.asarray(
        [case.bounded_solution.camera_position_world_metres for case in envelope.cases]
    )
    deltas = positions - MOUNTING_PRIOR_WORLD_METRES
    horizontal = np.linalg.norm(deltas[:, :2], axis=1)
    distance = np.linalg.norm(deltas, axis=1)
    return {
        "mounting_prior_world_metres": MOUNTING_PRIOR_WORLD_METRES.tolist(),
        "horizontal_mount_discrepancy_min_metres": float(min(horizontal)),
        "horizontal_mount_discrepancy_median_metres": float(median(tuple(horizontal))),
        "horizontal_mount_discrepancy_max_metres": float(max(horizontal)),
        "three_dimensional_mount_discrepancy_min_metres": float(min(distance)),
        "three_dimensional_mount_discrepancy_median_metres": float(median(tuple(distance))),
        "three_dimensional_mount_discrepancy_max_metres": float(max(distance)),
        "height_minus_mount_prior_min_metres": float(min(deltas[:, 2])),
        "height_minus_mount_prior_max_metres": float(max(deltas[:, 2])),
        "height_minus_p01_tape_min_metres": float(min(positions[:, 2] - P01_TAPE_HEIGHT_METRES)),
        "height_minus_p01_tape_max_metres": float(max(positions[:, 2] - P01_TAPE_HEIGHT_METRES)),
    }


def _reference_case(envelope: PoseEnvelope) -> dict[str, object]:
    match = next(
        case
        for case in envelope.cases
        if case.initializer == "ransac-epnp"
        and case.robust_loss == "huber"
        and case.robust_scale_pixels == 6.0
    )
    return match.to_dict()


def _combined_summary(envelopes: list[PoseEnvelope]) -> dict[str, Any]:
    cases = [case for envelope in envelopes for case in envelope.cases]
    positions = np.asarray([case.bounded_solution.camera_position_world_metres for case in cases])
    centre = np.asarray([median(tuple(column)) for column in positions.T])
    axes = np.asarray([case.bounded_solution.optical_axis_world for case in cases])
    height = positions[:, 2]
    return {
        "case_count": len(cases),
        "median_camera_position_world_metres": centre.tolist(),
        "camera_position_min_world_metres": np.min(positions, axis=0).tolist(),
        "camera_position_max_world_metres": np.max(positions, axis=0).tolist(),
        "maximum_position_distance_from_combined_median_metres": float(
            max(np.linalg.norm(positions - centre, axis=1))
        ),
        "maximum_pairwise_optical_axis_angle_degrees": _maximum_pairwise_angle(axes),
        "camera_height_min_metres": float(min(height)),
        "camera_height_median_metres": float(median(tuple(height))),
        "camera_height_max_metres": float(max(height)),
        "maximum_horizontal_mount_discrepancy_metres": float(
            max(np.linalg.norm((positions - MOUNTING_PRIOR_WORLD_METRES)[:, :2], axis=1))
        ),
        "maximum_three_dimensional_mount_discrepancy_metres": float(
            max(np.linalg.norm(positions - MOUNTING_PRIOR_WORLD_METRES, axis=1))
        ),
    }


def _maximum_pairwise_angle(axes: Any) -> float:
    maximum = 0.0
    for index, left in enumerate(axes):
        for right in axes[index + 1 :]:
            dot = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            maximum = max(maximum, math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0)))))
    return maximum


def _intrinsics(value: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        value["model"],
        value["width_pixels"],
        value["height_pixels"],
        value["fx_pixels"],
        value["fy_pixels"],
        value["cx_pixels"],
        value["cy_pixels"],
        tuple(value["distortion"]),
    )


def _landmark_arrays(items: list[dict[str, Any]]) -> tuple[Any, Any]:
    world = np.asarray(
        [
            [
                item["world_point"]["x_metres"],
                item["world_point"]["y_metres"],
                item["world_point"]["z_metres"],
            ]
            for item in items
        ]
    )
    image = np.asarray(
        [[item["image_point"]["u"], item["image_point"]["v"]] for item in items]
    )
    return world, image


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
