"""Open one sealed D034 validation pair exactly once against a frozen camera result."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p04_pose_domain import CameraIntrinsics
from spatial_mapping_phase2.p05_fixed_center_orientation import (
    evaluate_d034_frozen_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-camera-result", type=Path, required=True)
    parser.add_argument("--validation-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("one-time validation output already exists")
    camera = _read_json(args.frozen_camera_result)
    seal = _read_json(args.validation_seal)
    if camera.get("schema_version") != "p05-d034-fixed-centre-camera-v1":
        parser.error("frozen camera result has the wrong schema")
    if seal.get("schema_version") != "p05-d034-validation-seal-v1":
        parser.error("validation input is not a D034 seal")
    if seal.get("status") != "sealed-unconsumed":
        parser.error("D034 validation seal is not marked unconsumed")
    if seal.get("camera_id") != camera.get("camera_id"):
        parser.error("D034 validation camera differs from the frozen result")
    selected = next(
        (
            value
            for value in camera.get("intrinsic_candidates", [])
            if value.get("status") == "selected-provisional-solve-only"
        ),
        None,
    )
    if not isinstance(selected, dict):
        parser.error("camera has no frozen D034 orientation eligible for validation")
    orientation = selected.get("orientation")
    intrinsics_value = selected.get("intrinsics")
    if not isinstance(orientation, dict) or not isinstance(intrinsics_value, dict):
        parser.error("selected D034 candidate is malformed")
    items = seal.get("validation_landmarks")
    if not isinstance(items, list) or len(items) != 2:
        parser.error("D034 seal must contain exactly two validation landmarks")
    if any(item.get("role") != "d034-validation" for item in items):
        parser.error("D034 seal contains a non-validation role")
    if any(item.get("frame_id") != seal.get("approved_frame_id") for item in items):
        parser.error("D034 validation does not bind to the sealed approved frame")
    result = evaluate_d034_frozen_validation(
        _intrinsics(intrinsics_value),
        orientation["T_world_from_camera"],
        orientation["solve_landmark_ids"],
        [str(item["landmark_id"]) for item in items],
        [
            [
                item["world_point"]["x_metres"],
                item["world_point"]["y_metres"],
                item["world_point"]["z_metres"],
            ]
            for item in items
        ],
        [[item["image_point"]["u"], item["image_point"]["v"]] for item in items],
        threshold_pixels=float(camera["algorithm"]["config"]["solve_inlier_threshold_pixels"]),
    )
    output = {
        "schema_version": "p05-d034-one-time-validation-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "camera_id": camera["camera_id"],
        "frozen_camera_result_path": str(args.frozen_camera_result.resolve()),
        "frozen_camera_result_sha256": _sha256(args.frozen_camera_result),
        "validation_seal_path": str(args.validation_seal.resolve()),
        "validation_seal_sha256": _sha256(args.validation_seal),
        "validation": result.to_dict(),
        "post_opening_rule": (
            "result is final; do not rerun, tune, change intrinsic or replace the solve set"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


def _intrinsics(value: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        str(value["model"]),
        int(value["width_pixels"]),
        int(value["height_pixels"]),
        float(value["fx_pixels"]),
        float(value["fy_pixels"]),
        float(value["cx_pixels"]),
        float(value["cy_pixels"]),
        tuple(float(item) for item in value["distortion"]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
