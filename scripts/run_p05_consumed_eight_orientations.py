"""Run best-available P02-centred rotations from all eight consumed observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p05_consumed_pipeline import evaluate_consumed_eight_camera


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
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    facility = _read_json(args.facility_export)
    priors = facility.get("camera_mounting_priors")
    if not isinstance(priors, list):
        parser.error("selected P02 export has no camera mounting priors")
    centers: dict[str, list[float]] = {}
    for prior in priors:
        if not isinstance(prior, dict):
            continue
        point = prior.get("C_world_mount_prior")
        if not isinstance(point, dict):
            continue
        centers[str(prior.get("camera_id"))] = [
            float(point["x_metres"]),
            float(point["y_metres"]),
            float(point["z_metres"]),
        ]
    results: list[dict[str, Any]] = []
    for workspace_value, export_value in args.camera_input:
        export_path = Path(export_value)
        source = _read_json(export_path)
        camera_id = str(source.get("camera_id"))
        if camera_id not in centers:
            parser.error(f"selected P02 export has no centre for {camera_id}")
        results.append(
            evaluate_consumed_eight_camera(
                Path(workspace_value),
                export_path,
                args.fleet_manifest,
                args.facility_export,
                centers[camera_id],
            )
        )
    camera_ids = sorted(str(result["camera_id"]) for result in results)
    if camera_ids != [f"office-cam-0{index}" for index in range(1, 5)]:
        parser.error("provide exactly one immutable input for each of the four cameras")
    args.output_dir.mkdir(parents=True)
    for result in results:
        (args.output_dir / f"{result['camera_id']}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    registry = {
        "schema_version": "p05-consumed-eight-provisional-registry-v1",
        "authority": "non-operational best-available provisional evidence",
        "cameras": [
            {
                "camera_id": result["camera_id"],
                "status": result["status"],
                "evidence_strength": result["evidence_strength"],
                "fixed_center": result["fixed_center"],
                "selected_intrinsic_label": result["selected_intrinsic_label"],
                "provisional_orientation": result["selected_orientation"],
                "validation_status": result["validation_status"],
            }
            for result in sorted(results, key=lambda value: str(value["camera_id"]))
        ],
        "connectivity_authority": "none",
        "strict_d034_registry_influence": "none",
    }
    (args.output_dir / "provisional-registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "p05-consumed-eight-provisional-batch-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method_record": "docs/stages/P05/CONSUMED_EIGHT_PROVISIONAL_METHOD.md",
        "camera_ids": camera_ids,
        "camera_results": [f"{camera_id}.json" for camera_id in camera_ids],
        "provisional_registry": "provisional-registry.json",
        "facility_export_sha256": _sha256(args.facility_export),
        "fleet_manifest_sha256": _sha256(args.fleet_manifest),
        "all_observations_consumed": True,
        "strict_validation": "unavailable",
        "operational_or_connectivity_authority": "none",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
