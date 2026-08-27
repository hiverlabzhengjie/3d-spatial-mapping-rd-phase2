"""Verify a private P09 live-run manifest and Rerun recording structurally."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS, TrackingState

REQUIRED_RERUN_ENTITIES = {
    "/p09/status",
    "/p09/telemetry/processing_latency_ms",
    "/p09/world/axes",
    "/p09/world/current_anonymous_xy",
    "/p09/world/p07_v2_cloud_actual_rgb",
    "/p09/world/p08_floor_z0",
}


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest = _json_object(manifest_path)
    schema_version = manifest.get("schema_version")
    if (
        schema_version
        not in {
            "p09-live-demo-v1",
            "p09-live-demo-v2-yolo11-cuda",
            "p09-live-demo-v3-yolo11-live-rerun",
        }
        or manifest.get("anonymous_single_person_only") is not True
        or manifest.get("dynamic_DA3_invoked") is not False
        or manifest.get("credentials_persisted") is not False
        or manifest.get("client_frames_persisted_separately") is not False
    ):
        raise ValueError("P09 live manifest escaped its authority or privacy contract")
    if schema_version in {
        "p09-live-demo-v2-yolo11-cuda",
        "p09-live-demo-v3-yolo11-live-rerun",
    }:
        detector = _mapping(manifest, "detector")
        cuda = _mapping(detector, "cuda")
        if (
            detector.get("family") != "YOLO11n"
            or detector.get("backend") != "ultralytics-pytorch-cuda"
            or detector.get("cpu_fallback_allowed") is not False
            or cuda.get("device") != "cuda:0"
        ):
            raise ValueError("P09 YOLO11 manifest lacks required CUDA evidence")
    rerun = _mapping(manifest, "rerun")
    rerun_path = Path(str(rerun.get("path"))).resolve()
    if (
        not rerun_path.is_file()
        or rerun_path.stat().st_size != rerun.get("byte_count")
        or _sha256(rerun_path) != rerun.get("sha256")
    ):
        raise ValueError("P09 Rerun identity changed")
    runtime = _mapping(manifest, "runtime")
    ticks = runtime.get("ticks")
    if not isinstance(ticks, list) or not ticks:
        raise ValueError("P09 run has no completed tick evidence")
    if schema_version == "p09-live-demo-v3-yolo11-live-rerun":
        rerun_stream = _mapping(runtime, "rerun_stream")
        archive_tick_count = rerun_stream.get("archive_tick_count")
        live_tick_count = rerun_stream.get("live_tick_count")
        live_connection_count = rerun_stream.get("live_connection_count")
        if (
            rerun_stream.get("sink_mode") != "independent-archive-and-live-tcp"
            or archive_tick_count != len(ticks)
            or not isinstance(live_tick_count, int)
            or not 0 <= live_tick_count <= len(ticks)
            or not isinstance(live_connection_count, int)
            or live_connection_count < 0
            or (
                live_connection_count > 0
                and (
                    live_tick_count == 0
                    or rerun_stream.get("live_connected_before_close") is not True
                )
            )
        ):
            raise ValueError("P09 live/archive Rerun sink evidence is malformed")
    allowed_states = {state.value for state in TrackingState}
    for tick in ticks:
        if not isinstance(tick, dict) or tick.get("state") not in allowed_states:
            raise ValueError("P09 tick state is malformed")
        cameras = tick.get("cameras")
        if not isinstance(cameras, list):
            raise ValueError("P09 tick lacks per-camera evidence")
        camera_ids = tuple(camera.get("camera_id") for camera in cameras)
        if len(camera_ids) != len(set(camera_ids)) or not set(camera_ids) <= set(CAMERA_IDS):
            raise ValueError("P09 tick camera evidence is duplicated or unknown")
        if tick.get("current_xy_metres") is not None and tick.get("state") not in {
            TrackingState.TRACKED_FUSED.value,
            TrackingState.TRACKED_SINGLE_CAMERA.value,
        }:
            raise ValueError("non-tracked tick contains a current position")
    entities = _rerun_entity_paths(rerun_path)
    expected = set(REQUIRED_RERUN_ENTITIES)
    for camera_id in CAMERA_IDS:
        expected.update(
            {
                f"/p09/live/{camera_id}",
                f"/p09/live/{camera_id}/detections",
                f"/p09/live/{camera_id}/footpoints",
                f"/p09/world/cameras/{camera_id}",
                f"/p09/world/camera_labels/{camera_id}",
                f"/p09/world/candidates/{camera_id}",
            }
        )
    missing = expected - entities
    if missing:
        raise ValueError(f"P09 Rerun is missing required entities: {sorted(missing)}")
    serialized = manifest_path.read_text(encoding="utf-8").lower()
    if "rtsp://" in serialized or "rtsps://" in serialized or "phase2_rtsp_camera" in serialized:
        raise ValueError("P09 manifest contains endpoint material")
    status_counts: dict[str, int] = {}
    for tick in ticks:
        state = str(tick["state"])
        status_counts[state] = status_counts.get(state, 0) + 1
    return {
        "schema_version": "p09-live-demo-verification-v1",
        "status": "passed",
        "manifest": _identity(manifest_path),
        "rerun": _identity(rerun_path),
        "tick_count": len(ticks),
        "state_counts": status_counts,
        "entity_count": len(entities),
        "required_entities_present": True,
        "credentials_absent": True,
    }


def _rerun_entity_paths(path: Path) -> set[str]:
    import rerun.dataframe as rdf

    recording = rdf.load_recording(path)
    return {
        str(component).removeprefix("Component(").split(":", 1)[0]
        for component in recording.schema().component_columns()
    }


def _json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("P09 manifest must be a JSON object")
    return loaded


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, dict):
        raise ValueError(f"P09 {key} record is malformed")
    return selected


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serialized = json.dumps(verify(args.manifest.resolve()), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.resolve().write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
