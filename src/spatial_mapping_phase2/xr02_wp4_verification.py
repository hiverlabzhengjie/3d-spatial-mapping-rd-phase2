"""Independent structural verification for retained XR02 WP4 live evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS
from spatial_mapping_phase2.xr02_global_journal import verify_global_journal
from spatial_mapping_phase2.xr02_journal import verify_journal
from spatial_mapping_phase2.xr02_mediamtx import (
    MEDIAMTX_VERSION,
    MEDIAMTX_WINDOWS_EXE_SHA256,
)

_REQUIRED_ENTITIES = {
    "/xr02/status/live",
    "/xr02/status/scene_context",
    "/xr02/status/service",
    "/xr02/telemetry/processing_latency_ms",
    "/xr02/world/static/adopted_cloud",
    "/xr02/world/static/axes",
    "/xr02/world/static/floor_z0",
}


def verify_wp4_live(manifest_path: Path) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    manifest = _object(manifest_path)
    if (
        manifest.get("schema")
        not in {
            "xr02.wp4.live_manifest.v1",
            "xr02.wp4.live_manifest.v2",
            "xr02.wp4.live_manifest.v3",
            "xr02.wp4.live_manifest.v4",
        }
        or manifest.get("credentials_persisted") is not False
        or (
            manifest.get("client_frames_persisted_separately") is not False
            and manifest.get("schema") != "xr02.wp4.live_manifest.v4"
        )
        or manifest.get("dynamic_da3_invoked") is not False
        or manifest.get("trackstudio_used") is not False
    ):
        raise ValueError("WP4 manifest escaped its bounded live contract")
    claims = _mapping(manifest, "claims")
    if any(value is not False for value in claims.values()):
        raise ValueError("WP4 manifest contains an unsupported completion or production claim")
    ingress = manifest.get("media_ingress")
    manifest_schema = manifest.get("schema")
    if manifest_schema in {
        "xr02.wp4.live_manifest.v2",
        "xr02.wp4.live_manifest.v3",
        "xr02.wp4.live_manifest.v4",
    }:
        if not isinstance(ingress, dict):
            raise ValueError("WP4 v2 manifest lacks media-ingress evidence")
        _verify_media_ingress(ingress)
    service = _mapping(manifest, "service")
    service_config = _mapping(service, "config")
    status = _mapping(service, "status")
    worker = _mapping(status, "worker")
    ticks = service.get("tick_summaries")
    health = service.get("camera_health_samples")
    if not isinstance(ticks, list) or not ticks or not isinstance(health, list) or not health:
        raise ValueError("WP4 run lacks tick or camera-health evidence")
    if (
        worker.get("failed_ticks") != 0
        or worker.get("busy") is not False
        or worker.get("pending") not in {None, False}
    ):
        raise ValueError("WP4 worker did not finish cleanly")
    publication_worker: dict[str, Any] | None = None
    if manifest_schema in {"xr02.wp4.live_manifest.v3", "xr02.wp4.live_manifest.v4"}:
        publication_worker = _mapping(status, "publication_worker")
        if (
            publication_worker.get("failed_items") != 0
            or publication_worker.get("busy") is not False
            or status.get("publisher_failure_class") is not None
        ):
            raise ValueError("WP4 publication worker did not finish cleanly")
    if service.get("clock_ticks") != int(worker.get("completed_ticks", -1)) + int(
        worker.get("busy_dropped_ticks", -1)
    ):
        raise ValueError("WP4 bounded admission counters disagree")
    if [item.get("tick_index") for item in ticks] != list(range(len(ticks))):
        raise ValueError("WP4 association ticks are not contiguous")
    scene = _mapping(manifest, "scene")
    scene_context = scene.get("scene_context_sha256")
    if any(item.get("decision_signature_sha256") is None for item in ticks):
        raise ValueError("WP4 ticks lack deterministic association signatures")
    for sample in health:
        cameras = sample.get("cameras") if isinstance(sample, dict) else None
        if not isinstance(cameras, list):
            raise ValueError("WP4 camera-health sample is malformed")
        ids = tuple(item.get("camera_id") for item in cameras if isinstance(item, dict))
        if ids != CAMERA_IDS:
            raise ValueError("WP4 camera-health sample changed the office roster")

    artifacts = _mapping(manifest, "artifacts")
    rerun_path = _verified_identity(_mapping(artifacts, "rerun"))
    global_path = _verified_identity(_mapping(artifacts, "global_journal"))
    local_identity = _mapping(artifacts, "local_journal")
    if local_identity.get("present") is False:
        local_path = Path(str(local_identity.get("path"))).resolve()
        if local_path.exists():
            raise ValueError("WP4 absent local journal unexpectedly exists")
        local_records = 0
    else:
        local_path = _verified_identity(local_identity)
        local_records = verify_journal(local_path).records
    global_result = verify_global_journal(global_path)
    association_ticks = ticks
    if manifest_schema in {"xr02.wp4.live_manifest.v3", "xr02.wp4.live_manifest.v4"}:
        association_ticks = [item for item in ticks if item.get("association_updated") is True]
        if [item.get("association_tick_index") for item in association_ticks] != list(
            range(len(association_ticks))
        ):
            raise ValueError("WP4 global association ticks are not contiguous")
    if global_result.records != len(association_ticks):
        raise ValueError("WP4 global journal count disagrees with association cadence")
    rerun_evidence = _mapping(service, "rerun")
    expected_publications = len(ticks)
    if manifest_schema in {"xr02.wp4.live_manifest.v3", "xr02.wp4.live_manifest.v4"}:
        if publication_worker is None:
            raise ValueError("WP4 publication telemetry disappeared")
        expected_publications = int(publication_worker.get("completed_items", -1)) + int(
            service.get("forced_final_publications", -1)
        )
    if rerun_evidence.get("archive_tick_count") != expected_publications:
        raise ValueError("WP4 Rerun archive count disagrees with publication cadence")
    pipeline_profile = _mapping(service, "pipeline_profile")
    if pipeline_profile.get("cpu_fallback_allowed") is not False:
        raise ValueError("WP4 did not preserve the CUDA-only model gate")
    entities = _rerun_entity_paths(rerun_path)
    expected = set(_REQUIRED_ENTITIES)
    for camera_id in CAMERA_IDS:
        expected.update(
            {
                f"/xr02/live/cameras/{camera_id}",
                f"/xr02/world/static/cameras/{camera_id}",
                f"/xr02/world/static/camera_labels/{camera_id}",
            }
        )
    missing = expected - entities
    if missing:
        raise ValueError(f"WP4 Rerun is missing required entities: {sorted(missing)}")
    rerun_profile = rerun_evidence.get("presentation_profile")
    if rerun_profile is not None:
        if rerun_profile not in {
            "xr02-wp4-diagnostic-v2",
            "xr02-wp4-operator-people-summary-v3",
            "xr02-wp4-operator-people-summary-v4-stale-hold",
        }:
            raise ValueError("WP4 Rerun presentation profile is unknown")
        presentation_entities = {"/xr02/diagnostics/active_tracks"}
        if rerun_profile == "xr02-wp4-diagnostic-v2":
            presentation_entities.update(
                {
                    "/xr02/diagnostics/events",
                    "/xr02/diagnostics/legend",
                    "/xr02/telemetry/assignment_evidence/ambiguous",
                    "/xr02/telemetry/assignment_evidence/duplicate",
                    "/xr02/telemetry/track_states/confirmed",
                }
            )
        elif rerun_evidence.get("default_visible_panels") != [
            "facility_3d",
            "four_live_cameras",
            "people_in_scene",
        ]:
            raise ValueError("WP4 operator presentation panel inventory changed")
        presentation_missing = presentation_entities - entities
        if presentation_missing:
            raise ValueError(
                "WP4 Rerun is missing diagnostic presentation entities: "
                f"{sorted(presentation_missing)}"
            )
    admission = status.get("inference_admission")
    if admission is not None:
        if (
            not isinstance(admission, dict)
            or admission.get("profile") != "one-running-one-overwriteable-latest-pending-v1"
            or admission.get("fifo") is not False
            or admission.get("maximum_pending_items") != 1
            or admission.get("maximum_pending_age_ms")
            != service_config.get("inference_pending_max_age_ms")
        ):
            raise ValueError("WP4 inference microbuffer contract changed")
    trial_recording_verified = False
    if manifest_schema == "xr02.wp4.live_manifest.v4":
        trial_recording_verified = _verify_trial_recording(
            manifest_path,
            manifest,
            artifacts,
        )
    if local_records and not any(
        path.startswith("/xr02/world/global_tracks/g_") for path in entities
    ):
        raise ValueError("WP4 Rerun lacks a safe scene-global track entity")
    if (
        b"rtsp://" in manifest_path.read_bytes().lower()
        or b"rtsps://" in manifest_path.read_bytes().lower()
    ):
        raise ValueError("WP4 manifest contains an endpoint value")
    credential_paths = [rerun_path, global_path]
    if local_records:
        credential_paths.append(local_path)
    _assert_credentials_absent(*credential_paths)
    return {
        "schema": (
            "xr02.wp4.live_verification.v4"
            if manifest_schema == "xr02.wp4.live_manifest.v4"
            else "xr02.wp4.live_verification.v3"
            if manifest_schema == "xr02.wp4.live_manifest.v3"
            else "xr02.wp4.live_verification.v2"
        ),
        "status": "passed",
        "manifest": _identity(manifest_path),
        "scene_context_sha256": scene_context,
        "tick_count": len(ticks),
        "local_observation_records": local_records,
        "global_association_records": global_result.records,
        "local_tracking_ticks": len(ticks),
        "publication_ticks": expected_publications,
        "busy_dropped_ticks": worker.get("busy_dropped_ticks"),
        "worker_failures": worker.get("failed_ticks"),
        "entity_count": len(entities),
        "required_entities_present": True,
        "credentials_absent": True,
        "media_ingress_verified": manifest_schema
        in {
            "xr02.wp4.live_manifest.v2",
            "xr02.wp4.live_manifest.v3",
            "xr02.wp4.live_manifest.v4",
        },
        "trial_recording_verified": trial_recording_verified,
    }


def _verify_trial_recording(
    manifest_path: Path,
    manifest: dict[str, Any],
    artifacts: dict[str, Any],
) -> bool:
    recording = _mapping(manifest, "trial_recording")
    retained_value = manifest.get("client_frames_persisted_separately")
    if retained_value not in {True, False}:
        raise ValueError("WP4 v4 trial recording retention flag is malformed")
    retained = retained_value is True
    video_artifacts = artifacts.get("trial_video")
    if not isinstance(video_artifacts, list):
        raise ValueError("WP4 v4 trial-video artifact inventory is malformed")
    if not retained:
        if recording.get("enabled") is not False or video_artifacts:
            raise ValueError("WP4 disabled trial recording contains artifacts")
        return True
    if (
        recording.get("schema") != "xr02.wp4.trial_recording.v1"
        or recording.get("enabled") is not True
        or recording.get("credentials_persisted") is not False
        or recording.get("video_mode") != "encoded-stream-copy"
        or recording.get("reconnect_policy") != "supervised per-camera segmented restart"
    ):
        raise ValueError("WP4 trial recording escaped its credential-safe contract")
    binary = Path(str(recording.get("ffmpeg_binary_path"))).resolve()
    if (
        not binary.is_file()
        or _sha256(binary) != recording.get("ffmpeg_binary_sha256")
        or not str(recording.get("ffmpeg_version", "")).lower().startswith("ffmpeg version")
    ):
        raise ValueError("WP4 FFmpeg runtime identity changed")
    captures = recording.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("WP4 enabled trial recording has no capture attempts")
    expected_root = (manifest_path.parent / "trial-video").resolve()
    seen_camera_ids: set[str] = set()
    recorded_artifacts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for value in captures:
        if not isinstance(value, dict):
            raise ValueError("WP4 trial recording capture is malformed")
        camera_id = value.get("camera_id")
        generation = value.get("generation")
        artifact = value.get("artifact")
        if (
            camera_id not in CAMERA_IDS
            or not isinstance(generation, int)
            or generation <= 0
            or not isinstance(artifact, dict)
        ):
            raise ValueError("WP4 trial recording segment identity is malformed")
        path = Path(str(artifact.get("path"))).resolve()
        if path.parent != expected_root or path in seen_paths:
            raise ValueError("WP4 trial recording segment escaped its run directory")
        seen_paths.add(path)
        seen_camera_ids.add(str(camera_id))
        if artifact.get("present") is True:
            _verified_identity(artifact)
        elif artifact.get("present") is False:
            if path.exists():
                raise ValueError("WP4 absent trial-video segment unexpectedly exists")
        else:
            raise ValueError("WP4 trial-video artifact presence is malformed")
        recorded_artifacts.append(artifact)
    if seen_camera_ids != set(CAMERA_IDS) or video_artifacts != recorded_artifacts:
        raise ValueError("WP4 trial recording inventory changed")
    return True


def _verify_media_ingress(value: dict[str, Any]) -> None:
    if value.get("schema") == "xr02.mediamtx.direct_diagnostic.v1":
        if value.get("credentials_persisted") is not False:
            raise ValueError("direct diagnostic ingress persisted credentials")
        return
    if (
        value.get("schema") != "xr02.mediamtx.gateway_evidence.v1"
        or value.get("version") != MEDIAMTX_VERSION
        or value.get("binary_sha256") != MEDIAMTX_WINDOWS_EXE_SHA256
        or value.get("credentials_persisted") is not False
        or value.get("source_on_demand") is not False
        or value.get("always_available") is not False
        or value.get("rtsp_transport") != "tcp"
    ):
        raise ValueError("WP4 MediaMTX ingress escaped D069")
    binary = Path(str(value.get("binary_path"))).resolve()
    config = Path(str(value.get("config_path"))).resolve()
    if (
        not binary.is_file()
        or _sha256(binary) != value.get("binary_sha256")
        or not config.is_file()
        or _sha256(config) != value.get("config_sha256")
    ):
        raise ValueError("WP4 MediaMTX runtime identity changed")
    config_bytes = config.read_bytes().lower()
    if b"rtsp://" in config_bytes or b"rtsps://" in config_bytes:
        raise ValueError("WP4 MediaMTX config persisted an endpoint")


def _verified_identity(value: dict[str, Any]) -> Path:
    path = Path(str(value.get("path"))).resolve()
    if (
        value.get("present") is not True
        or not path.is_file()
        or path.stat().st_size != value.get("bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(f"WP4 artifact identity changed: {path.name}")
    return path


def _rerun_entity_paths(path: Path) -> set[str]:
    import rerun.dataframe as rdf

    recording = rdf.load_recording(path)
    return {
        str(component).removeprefix("Component(").split(":", 1)[0]
        for component in recording.schema().component_columns()
    }


def _assert_credentials_absent(*paths: Path) -> None:
    forbidden = (b"rtsp://", b"rtsps://", b"phase2_rtsp_camera")
    for path in paths:
        lowered = path.read_bytes().lower()
        if any(marker in lowered for marker in forbidden):
            raise ValueError(f"WP4 artifact contains endpoint material: {path.name}")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("WP4 manifest must be an object")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, dict):
        raise ValueError(f"WP4 {key} record is malformed")
    return selected


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
