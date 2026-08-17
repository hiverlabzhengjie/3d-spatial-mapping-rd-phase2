"""Run the D027 leave-Camera-3-out intrinsic-fleet experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from spatial_mapping_phase2.p04_intrinsic_fleet import (
    CameraIntrinsicEstimate,
    build_fleet_profiles,
    summarize_between_camera_variation,
)
from spatial_mapping_phase2.p04_pose_domain import (
    CameraIntrinsics,
    PoseSolveError,
    solve_camera_pose,
)

EXPECTED_CAMERAS = {f"office-cam-{index:02d}" for index in range(1, 5)}
DISTORTION_COUNTS = {"simple_radial": 1, "simple_divisional": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--fleet-manifest", type=Path, action="append", required=True)
    parser.add_argument("--camera3-baseline-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.workspace,
        arguments.fleet_manifest,
        arguments.camera3_baseline_manifest,
        arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def run(
    workspace: Path,
    fleet_manifest_paths: list[Path],
    baseline_manifest_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError("output directory already exists")
    state_path = workspace / "state.json"
    state = _read_json(state_path)
    if state.get("camera_id") != "office-cam-03" or state.get("revision") != 20:
        raise RuntimeError("fleet validation requires frozen Camera 3 workspace revision 20")
    solve_items = [item for item in state["landmarks"] if item["role"] == "solve"]
    held_items = [item for item in state["landmarks"] if item["role"] == "held-out"]
    if len(solve_items) != 6 or len(held_items) != 2:
        raise RuntimeError("frozen validation requires six solve and two held-out points")
    solve_world, solve_image = _landmark_arrays(solve_items)
    held_world, held_image = _landmark_arrays(held_items)

    fleet_manifests = [(_read_json(path), path) for path in fleet_manifest_paths]
    baseline_manifests = {
        manifest["candidate"]["camera_model"]: (manifest, path)
        for path in baseline_manifest_paths
        for manifest in [_read_json(path)]
    }
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for manifest, path in fleet_manifests:
        model = manifest["candidate"]["camera_model"]
        if model not in DISTORTION_COUNTS:
            raise RuntimeError("fleet input uses an unauthorized camera model")
        grouped.setdefault(model, []).append((manifest, path))
    if set(grouped) != set(DISTORTION_COUNTS) or set(baseline_manifests) != set(DISTORTION_COUNTS):
        raise RuntimeError("fleet study requires both authorized models and both baselines")

    model_results: list[dict[str, Any]] = []
    for model in sorted(grouped):
        entries = grouped[model]
        if {manifest["camera_id"] for manifest, _ in entries} != EXPECTED_CAMERAS:
            raise RuntimeError(f"{model} requires exactly one manifest for every camera")
        estimates = tuple(_estimate(manifest) for manifest, _ in entries)
        _verify_manifest_set(entries)
        profiles = build_fleet_profiles(estimates, exclude_camera_id="office-cam-03")
        camera3_same_frame = next(
            estimate for estimate in estimates if estimate.camera_id == "office-cam-03"
        )
        baseline_manifest, baseline_path = baseline_manifests[model]
        baseline = _estimate(baseline_manifest)
        if baseline.camera_id != "office-cam-03":
            raise RuntimeError("historical baseline must belong to Camera 3")
        candidates: list[dict[str, Any]] = []
        candidate_inputs = [
            ("camera3-frozen-historical", baseline, [str(baseline_path.resolve())]),
            (
                "camera3-same-frame-independent",
                camera3_same_frame,
                [
                    str(path.resolve())
                    for manifest, path in entries
                    if manifest["camera_id"] == "office-cam-03"
                ],
            ),
        ]
        for label, estimate, sources in candidate_inputs:
            candidates.append(
                _evaluate_pose(
                    label,
                    _intrinsics_from_estimate(estimate),
                    sources,
                    solve_world,
                    solve_image,
                    held_world,
                    held_image,
                )
            )
        source_paths = [
            str(path.resolve())
            for manifest, path in entries
            if manifest["camera_id"] != "office-cam-03"
        ]
        for profile in profiles:
            intrinsics = CameraIntrinsics(
                profile.model,
                profile.width_pixels,
                profile.height_pixels,
                profile.fx_pixels,
                profile.fy_pixels,
                profile.cx_pixels,
                profile.cy_pixels,
                profile.distortion,
            )
            candidate = _evaluate_pose(
                f"leave-camera3-out:{profile.method}",
                intrinsics,
                source_paths,
                solve_world,
                solve_image,
                held_world,
                held_image,
            )
            candidate["fleet_profile"] = profile.to_dict()
            candidates.append(candidate)
        model_results.append(
            {
                "camera_model": model,
                "per_camera_estimates": [
                    _estimate_dict(estimate, path)
                    for estimate, (_, path) in zip(estimates, entries, strict=True)
                ],
                "between_all_four_cameras": summarize_between_camera_variation(estimates),
                "between_training_cameras_1_2_4": summarize_between_camera_variation(
                    estimates, exclude_camera_id="office-cam-03"
                ),
                "parameter_leave_one_camera_out": _parameter_leave_one_out(estimates),
                "camera3_pose_evaluations": candidates,
            }
        )

    output: dict[str, Any] = {
        "schema_version": "p04-intrinsic-fleet-study-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "decision_authority": "D027",
        "profile_version": "stream-profile-v1",
        "image_size_pixels": [1920, 1080],
        "training_camera_ids": ["office-cam-01", "office-cam-02", "office-cam-04"],
        "excluded_validation_camera_id": "office-cam-03",
        "frozen_validation": {
            "workspace_revision": state["revision"],
            "workspace_state_sha256": _sha256(state_path),
            "solve_landmark_ids": [item["landmark_id"] for item in solve_items],
            "held_out_landmark_ids": [item["landmark_id"] for item in held_items],
            "held_out_tuning_influence": "none",
        },
        "pooling_contract": {
            "unit_of_weight": "one independent shared-intrinsic estimate per camera",
            "methods_fixed_before_validation": [
                "equal-camera-arithmetic-mean",
                "equal-camera-componentwise-median",
                "equal-camera-componentwise-huber",
            ],
            "normalized_before_pooling": ["fx/W", "fy/H", "cx/W", "cy/H", "distortion"],
            "geocalib_cross_camera_shared_intrinsics": False,
        },
        "models": model_results,
        "authority_note": (
            "Fleet profiles are D027 study evidence. Camera 1, 2 and 4 poses remain unvalidated, "
            "and P04 does not accept its own intrinsic or pose recommendation."
        ),
    }
    output_dir.mkdir(parents=True)
    output_path = output_dir / "manifest.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _estimate(manifest: dict[str, Any]) -> CameraIntrinsicEstimate:
    if manifest.get("status") != "provisional-candidate":
        raise RuntimeError("fleet input must be a retained provisional GeoCalib candidate")
    model = manifest["candidate"]["camera_model"]
    count = DISTORTION_COUNTS.get(model)
    if count is None:
        raise RuntimeError("unsupported fleet model")
    row = manifest["candidate"]["shared_camera_parameters"][0]
    return CameraIntrinsicEstimate(
        manifest["camera_id"],
        manifest["profile_version"],
        model,
        round(row[0]),
        round(row[1]),
        row[2],
        row[3],
        row[4],
        row[5],
        tuple(row[6 : 6 + count]),
        manifest["candidate"]["stability"]["focal_cv"],
    )


def _verify_manifest_set(entries: list[tuple[dict[str, Any], Path]]) -> None:
    identities = set()
    for manifest, _ in entries:
        if manifest["profile_version"] != "stream-profile-v1":
            raise RuntimeError("fleet manifest profile is incompatible")
        identities.add(
            (
                manifest["identities"]["geocalib_source_commit"],
                manifest["identities"]["weight_sha256"],
            )
        )
        if len(manifest["frames"]) != 3:
            raise RuntimeError("each fleet camera must contribute exactly three frames")
        for frame in manifest["frames"]:
            path = Path(frame["source_path"])
            if frame["camera_id"] != manifest["camera_id"] or _sha256(path) != frame["sha256"]:
                raise RuntimeError("fleet frame identity or hash differs from its manifest")
    if len(identities) != 1:
        raise RuntimeError("fleet manifests must use one GeoCalib source and checkpoint")


def _evaluate_pose(
    label: str,
    intrinsics: CameraIntrinsics,
    source_manifests: list[str],
    solve_world: Any,
    solve_image: Any,
    held_world: Any,
    held_image: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label,
        "intrinsics": intrinsics.to_dict(),
        "source_manifests": source_manifests,
        "status": "provisional-pose-evaluation",
    }
    try:
        pose = solve_camera_pose(
            intrinsics,
            solve_world,
            solve_image,
            held_world,
            held_image,
            ransac_threshold_pixels=30.0,
        )
        result["pose"] = pose.to_dict()
    except PoseSolveError as error:
        result["status"] = "rejected-pose-evaluation"
        result["rejection_reason"] = str(error)
    return result


def _intrinsics_from_estimate(estimate: CameraIntrinsicEstimate) -> CameraIntrinsics:
    return CameraIntrinsics(
        estimate.model,
        estimate.width_pixels,
        estimate.height_pixels,
        estimate.fx_pixels,
        estimate.fy_pixels,
        estimate.cx_pixels,
        estimate.cy_pixels,
        estimate.distortion,
    )


def _estimate_dict(estimate: CameraIntrinsicEstimate, path: Path) -> dict[str, Any]:
    return {
        "camera_id": estimate.camera_id,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "focal_pixels": estimate.focal_pixels,
        "fx_pixels": estimate.fx_pixels,
        "fy_pixels": estimate.fy_pixels,
        "cx_pixels": estimate.cx_pixels,
        "cy_pixels": estimate.cy_pixels,
        "distortion": list(estimate.distortion),
        "within_camera_focal_cv": estimate.within_camera_focal_cv,
    }


def _parameter_leave_one_out(
    estimates: tuple[CameraIntrinsicEstimate, ...],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for excluded in sorted(estimates, key=lambda item: item.camera_id):
        profile_results: list[dict[str, Any]] = []
        for profile in build_fleet_profiles(estimates, exclude_camera_id=excluded.camera_id):
            predicted_focal = (profile.fx_pixels + profile.fy_pixels) / 2.0
            profile_results.append(
                {
                    "method": profile.method,
                    "predicted_focal_pixels": predicted_focal,
                    "observed_focal_pixels": excluded.focal_pixels,
                    "signed_focal_error_pixels": predicted_focal - excluded.focal_pixels,
                    "signed_focal_error_fraction": (
                        predicted_focal - excluded.focal_pixels
                    )
                    / excluded.focal_pixels,
                    "predicted_distortion": list(profile.distortion),
                    "observed_distortion": list(excluded.distortion),
                    "signed_distortion_error": [
                        predicted - observed
                        for predicted, observed in zip(
                            profile.distortion, excluded.distortion, strict=True
                        )
                    ],
                }
            )
        results.append(
            {"excluded_camera_id": excluded.camera_id, "profile_predictions": profile_results}
        )
    return results


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
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
