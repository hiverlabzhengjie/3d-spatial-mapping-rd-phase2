"""Run the solve-only D034 fixed-centre comparison for all four cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p05_d034_pipeline import evaluate_d034_camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-manifest", type=Path, required=True)
    parser.add_argument("--facility-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-6dof-dir", type=Path, required=True)
    parser.add_argument("--camera3-diagnostic-6dof", type=Path, required=True)
    parser.add_argument(
        "--camera-input",
        action="append",
        nargs=2,
        metavar=("WORKSPACE", "CORRESPONDENCE_EXPORT"),
        required=True,
    )
    parser.add_argument(
        "--camera3-surveyed-center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        required=True,
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    facility = json.loads(args.facility_export.read_text(encoding="utf-8"))
    priors = facility.get("camera_mounting_priors")
    if not isinstance(priors, list):
        parser.error("selected facility export has no camera mounting priors")
    centers: dict[str, tuple[list[float], str]] = {}
    for prior in priors:
        if not isinstance(prior, dict):
            continue
        camera_id = str(prior.get("camera_id"))
        point = prior.get("C_world_mount_prior")
        if not isinstance(point, dict):
            continue
        centers[camera_id] = (
            [float(point["x_metres"]), float(point["y_metres"]), float(point["z_metres"])],
            "D034 owner-authorized P02 revision-3 mounting XYZ as exact optical centre",
        )
    centers["office-cam-03"] = (
        [float(value) for value in args.camera3_surveyed_center],
        "owner physical survey supplied 2026-08-18; D034 exact optical centre",
    )
    results: list[dict[str, Any]] = []
    for workspace_value, export_value in args.camera_input:
        export_path = Path(export_value)
        source = json.loads(export_path.read_text(encoding="utf-8"))
        camera_id = str(source.get("camera_id"))
        if camera_id not in centers:
            results.append(
                {
                    "schema_version": "p05-d034-fixed-centre-camera-v1",
                    "camera_id": camera_id,
                    "operational_status": "unregistered",
                    "reason": "no owner-reviewed D034 fixed-centre XYZ",
                }
            )
            continue
        center, authority = centers[camera_id]
        results.append(
            evaluate_d034_camera(
                Path(workspace_value),
                export_path,
                args.fleet_manifest,
                args.facility_export,
                center,
                authority,
            )
        )
    camera_ids = sorted(str(result["camera_id"]) for result in results)
    required = [f"office-cam-0{index}" for index in range(1, 5)]
    if camera_ids != required:
        parser.error("provide exactly one immutable correspondence input for every camera")
    args.output_dir.mkdir(parents=True)
    for result in results:
        (args.output_dir / f"{result['camera_id']}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    registry_cameras = []
    for result in sorted(results, key=lambda value: str(value["camera_id"])):
        selected = next(
            (
                value
                for value in result.get("intrinsic_candidates", [])
                if value.get("status") == "selected-provisional-solve-only"
            ),
            None,
        )
        diagnostic_path = (
            args.camera3_diagnostic_6dof
            if result["camera_id"] == "office-cam-03"
            else args.diagnostic_6dof_dir / f"{result['camera_id']}.json"
        )
        diagnostic: dict[str, Any] = (
            {
                "path": str(diagnostic_path.resolve()),
                "sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
                "authority": "historical diagnostic only; zero D034 influence",
            }
            if diagnostic_path.is_file()
            else {
                "path": None,
                "sha256": None,
                "authority": "no separate 6-DoF camera result supplied",
            }
        )
        registry_cameras.append(
            {
                "camera_id": result["camera_id"],
                "operational_status": result["operational_status"],
                "fixed_center": result.get("fixed_center"),
                "selected_intrinsic_label": result.get("selected_intrinsic_label"),
                "operational_orientation": None if selected is None else selected["orientation"],
                "rejected_solve_outlier": (
                    None
                    if selected is None
                    else selected["orientation"]["rejected_solve_landmark_id"]
                ),
                "strict_validation_status": result.get("strict_validation_status"),
                "diagnostic_6dof": diagnostic,
            }
        )
    registry = {
        "schema_version": "p05-d034-camera-registry-v1",
        "decision_authority": "D034",
        "camera_count": 4,
        "cameras": registry_cameras,
        "authority_note": (
            "Provisional solve-only registry. No camera can be accepted before its two-point "
            "one-time D034 validation and physical overlay review."
        ),
    }
    connectivity = {
        "schema_version": "p05-d034-connectivity-v1",
        "decision_authority": "D034",
        "nodes": [
            {"camera_id": value["camera_id"], "status": value["operational_status"]}
            for value in registry_cameras
        ],
        "operational_edges": [],
        "status": "no-authorized-operational-edges",
        "reason": (
            "D034 strict validation is unavailable for every camera; historical overlap remains "
            "diagnostic and cannot establish operational connectivity."
        ),
    }
    (args.output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "connectivity.json").write_text(
        json.dumps(connectivity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "p05-d034-fixed-centre-batch-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "decision_authority": "D034",
        "camera_ids": camera_ids,
        "camera_results": [f"{camera_id}.json" for camera_id in camera_ids],
        "camera_registry": "registry.json",
        "connectivity_graph": "connectivity.json",
        "held_out_data_loaded": False,
        "strict_validation_status": "not-run; needs two new unconsumed points per camera",
        "legacy_6dof_status": "retained separately as diagnostic only",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
