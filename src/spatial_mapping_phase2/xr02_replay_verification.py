"""Structural verification for XR02 WP2 replay evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.xr02_journal import verify_journal

EXPECTED_PROFILES = {"botsort-fixed-v1", "deepocsort-fixed-v1"}
REQUIRED_RERUN_ENTITIES = {
    "/xr02/status/static_scene",
    "/xr02/world/p08_floor_z0",
    *(f"/xr02/cameras/office-cam-0{index}" for index in range(1, 5)),
}


class XR02ReplayVerificationError(ValueError):
    """Raised when retained WP2 evidence no longer satisfies its bounded contract."""


def verify_wp2_replay(manifest_path: Path) -> dict[str, object]:
    manifest = _json_object(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scope") != "XR02 WP2 camera-local tracking only; no global association"
    ):
        raise XR02ReplayVerificationError("WP2 manifest escaped its authorized scope")
    scene = _mapping(manifest, "scene_context")
    if not _is_sha256(scene.get("geometry_sha256")) or not _is_sha256(
        scene.get("floor_sha256")
    ):
        raise XR02ReplayVerificationError("WP2 scene authority identities are malformed")
    supervision = _mapping(manifest, "supervision_gate")
    if supervision.get("version") != "0.30.0" or not isinstance(
        supervision.get("frames_converted"), int
    ):
        raise XR02ReplayVerificationError("Supervision adapter gate is malformed")
    profiles = manifest.get("profiles")
    if (
        not isinstance(profiles, list)
        or {profile.get("profile_id") for profile in profiles if isinstance(profile, dict)}
        != EXPECTED_PROFILES
    ):
        raise XR02ReplayVerificationError("WP2 tracker profile set changed")
    verified_journals: dict[str, object] = {}
    for profile in profiles:
        if not isinstance(profile, dict):
            raise XR02ReplayVerificationError("WP2 tracker profile is malformed")
        profile_id = str(profile["profile_id"])
        kwargs = _mapping(profile, "tracker_kwargs")
        if profile_id.startswith("botsort") and kwargs.get("use_cmc") is not False:
            raise XR02ReplayVerificationError("BoT-SORT CMC was re-enabled")
        if profile_id.startswith("deepocsort") and kwargs.get("cmc_off") is not True:
            raise XR02ReplayVerificationError("Deep-OC-SORT CMC was re-enabled")
        if profile.get("deterministic_second_pass") is not True:
            raise XR02ReplayVerificationError("tracker replay was not deterministic")
        journal = _mapping(profile, "journal")
        journal_path = Path(str(journal.get("path")))
        result = verify_journal(journal_path)
        if result.records != journal.get("records") or result.final_sha256 != journal.get(
            "final_sha256"
        ):
            raise XR02ReplayVerificationError("journal verification identity changed")
        verified_journals[profile_id] = {
            "records": result.records,
            "final_sha256": result.final_sha256,
        }
    rerun = _mapping(manifest, "rerun")
    rerun_path = Path(str(rerun.get("path")))
    if (
        not rerun_path.is_file()
        or rerun_path.stat().st_size != rerun.get("bytes")
        or _sha256(rerun_path) != rerun.get("sha256")
    ):
        raise XR02ReplayVerificationError("WP2 Rerun identity changed")
    entities = _rerun_entity_paths(rerun_path)
    missing = REQUIRED_RERUN_ENTITIES - entities
    if missing:
        raise XR02ReplayVerificationError(f"WP2 Rerun entities are missing: {sorted(missing)}")
    serialized = manifest_path.read_text(encoding="utf-8").lower()
    if "rtsp://" in serialized or "rtsps://" in serialized or "phase2_rtsp_camera" in serialized:
        raise XR02ReplayVerificationError("WP2 evidence contains endpoint material")
    return {
        "schema_version": "xr02-wp2-replay-verification-v1",
        "status": "passed",
        "manifest": _identity(manifest_path),
        "rerun": _identity(rerun_path),
        "entity_count": len(entities),
        "required_entities_present": True,
        "journals": verified_journals,
        "credentials_absent": True,
        "global_association_present": False,
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
        raise XR02ReplayVerificationError("WP2 manifest must contain an object")
    return loaded


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, dict):
        raise XR02ReplayVerificationError(f"WP2 {key} record is malformed")
    return selected


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )
