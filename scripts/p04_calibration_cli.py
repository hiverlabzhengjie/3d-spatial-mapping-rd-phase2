"""Manage the local P04 Camera 3 calibration correspondence workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p04_calibration_domain import FrameReviewStatus
from spatial_mapping_phase2.p04_calibration_service import P04CalibrationService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="Bind a P02 export and rendered plan")
    initialize.add_argument("--facility-export", type=Path, required=True)
    initialize.add_argument("--plan-image", type=Path, required=True)
    initialize.add_argument(
        "--camera-id",
        choices=tuple(f"office-cam-0{index}" for index in range(1, 5)),
        default="office-cam-03",
    )

    add_frame = subparsers.add_parser("add-frame", help="Import an immutable frame candidate")
    add_frame.add_argument("--source", type=Path, required=True)
    add_frame.add_argument("--frame-id", required=True)
    add_frame.add_argument("--profile-version", default="stream-profile-v1")
    add_frame.add_argument("--expected-sha256")

    review = subparsers.add_parser("review-frame", help="Approve or reject a candidate frame")
    review.add_argument("--frame-id", required=True)
    review.add_argument("--status", choices=("approved", "rejected"), required=True)
    review.add_argument("--note")

    landmark = subparsers.add_parser("add-landmark", help="Save one linked correspondence")
    landmark.add_argument("--landmark-id", required=True)
    landmark.add_argument("--name", required=True)
    landmark.add_argument("--physical-meaning", required=True)
    landmark.add_argument("--frame-id", required=True)
    landmark.add_argument("--image-u", type=float, required=True)
    landmark.add_argument("--image-v", type=float, required=True)
    landmark.add_argument("--plan-u", type=float, required=True)
    landmark.add_argument("--plan-v", type=float, required=True)
    landmark.add_argument("--z-metres", type=float, required=True)
    landmark.add_argument("--z-source")
    landmark.add_argument("--z-uncertainty-metres", type=float)
    landmark.add_argument("--role", choices=("solve", "held-out"), required=True)

    remove = subparsers.add_parser("remove-landmark", help="Remove from the current revision")
    remove.add_argument("--landmark-id", required=True)
    z_sources = subparsers.add_parser(
        "set-missing-z-sources",
        help="Bind one operator-supplied measurement source to blank landmark Z sources",
    )
    z_sources.add_argument("--source", required=True)
    subparsers.add_parser("status", help="Print the credential-free workspace state")
    subparsers.add_parser("export", help="Write a correspondence snapshot")

    arguments = parser.parse_args()
    service = P04CalibrationService(arguments.workspace)
    result: dict[str, Any]
    if arguments.command == "init":
        result = service.initialize(
            arguments.facility_export, arguments.plan_image, arguments.camera_id
        ).to_dict()
    elif arguments.command == "add-frame":
        result = service.add_frame(
            arguments.source,
            arguments.frame_id,
            arguments.profile_version,
            arguments.expected_sha256,
        ).to_dict()
    elif arguments.command == "review-frame":
        result = service.review_frame(
            arguments.frame_id,
            FrameReviewStatus(arguments.status),
            arguments.note,
        ).to_dict()
    elif arguments.command == "add-landmark":
        result = service.add_landmark(
            {
                "landmark_id": arguments.landmark_id,
                "name": arguments.name,
                "physical_meaning": arguments.physical_meaning,
                "frame_id": arguments.frame_id,
                "image_point": {"u": arguments.image_u, "v": arguments.image_v},
                "plan_point": {"u": arguments.plan_u, "v": arguments.plan_v},
                "z_metres": arguments.z_metres,
                "z_source": arguments.z_source,
                "z_uncertainty_metres": arguments.z_uncertainty_metres,
                "role": arguments.role,
            }
        ).to_dict()
    elif arguments.command == "remove-landmark":
        result = service.remove_landmark(arguments.landmark_id).to_dict()
    elif arguments.command == "set-missing-z-sources":
        result = service.set_missing_z_sources(arguments.source).to_dict()
    elif arguments.command == "export":
        path, payload = service.export_snapshot()
        result = {"path": str(path), "export": payload}
    else:
        result = service.state_response()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
