"""Run D033-bound intrinsic and pose comparisons for P05 Cameras 1, 2 and 4."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from spatial_mapping_phase2.p05_pose_candidates import evaluate_camera_pose_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-manifest", type=Path, required=True)
    parser.add_argument("--facility-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-input",
        action="append",
        nargs=2,
        metavar=("WORKSPACE", "CORRESPONDENCE_EXPORT"),
        required=True,
    )
    arguments = parser.parse_args()
    if arguments.output_dir.exists():
        parser.error("output directory already exists")
    results = [
        evaluate_camera_pose_candidates(
            Path(workspace),
            Path(export),
            arguments.fleet_manifest,
            arguments.facility_export,
        )
        for workspace, export in arguments.camera_input
    ]
    camera_ids = [result["camera_id"] for result in results]
    if sorted(camera_ids) != ["office-cam-01", "office-cam-02", "office-cam-04"]:
        parser.error("provide exactly one input for Cameras 1, 2 and 4")
    arguments.output_dir.mkdir(parents=True)
    for result in results:
        camera_path = arguments.output_dir / f"{result['camera_id']}.json"
        camera_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": "p05-pose-candidate-batch-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "intrinsic_policy": "D033",
        "camera_ids": camera_ids,
        "camera_results": [f"{camera_id}.json" for camera_id in camera_ids],
        "authority_note": (
            "Candidate B starts every camera comparison but is not forced. No candidate is "
            "accepted by this batch."
        ),
    }
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
