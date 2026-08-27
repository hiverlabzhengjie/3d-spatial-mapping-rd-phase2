"""Structural verification for retained XR02 WP3 replay evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.xr02_global_domain import SignalProfile
from spatial_mapping_phase2.xr02_global_journal import verify_global_journal

_EXPECTED_WP2_CHAINS = {
    "botsort-fixed-v1": "fe8dfe23e9b929de28a01e1b309b1c819970f90107afac44fcfa4e15b80003e1",
    "deepocsort-fixed-v1": "d6c15fe6fb4b61562904081622d2f1911d6368452a31adc82dc7e019fecb72c2",
}


class XR02WP3VerificationError(ValueError):
    """Raised when WP3 retained evidence violates its bounded contract."""


def verify_wp3_replay(manifest_path: Path) -> dict[str, object]:
    manifest = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    if manifest.get("schema") != "xr02.wp3.replay_manifest.v1":
        raise XR02WP3VerificationError("WP3 manifest schema changed")
    scope = manifest.get("scope")
    if not isinstance(scope, str) or "no live RTSP or operator service" not in scope:
        raise XR02WP3VerificationError("WP3 manifest escaped its authorized scope")
    profiles = manifest.get("profiles")
    if profiles != [profile.value for profile in SignalProfile]:
        raise XR02WP3VerificationError("WP3 signal profile set changed")
    fixtures = _list_of_objects(manifest.get("fixture_partitions"), "fixture partitions")
    if [item.get("partition") for item in fixtures] != ["tuning", "heldout"]:
        raise XR02WP3VerificationError("tuning and heldout fixture roles changed")
    for fixture in fixtures:
        _verify_identity(Path(str(fixture.get("path"))), fixture)

    authority = _list_of_objects(manifest.get("input_authority"), "input authority")
    if {str(item.get("tracker_profile")) for item in authority} != set(_EXPECTED_WP2_CHAINS):
        raise XR02WP3VerificationError("WP2 tracker authority set changed")
    for item in authority:
        profile = str(item.get("tracker_profile"))
        if item.get("journal_final_sha256") != _EXPECTED_WP2_CHAINS[profile]:
            raise XR02WP3VerificationError("WP2 journal authority chain changed")
        if item.get("cross_camera_timing_authority") != (
            "not_claimed; WP2 camera clips were replayed sequentially and are schema inputs only"
        ):
            raise XR02WP3VerificationError("WP2 timing limitation disappeared")

    determinism = _list_of_objects(manifest.get("determinism"), "determinism")
    if len(determinism) != 6 or any(
        item.get("matched") is not True
        or item.get("first_signature_sha256") != item.get("second_signature_sha256")
        for item in determinism
    ):
        raise XR02WP3VerificationError("WP3 replay is not deterministic")

    journals = _list_of_objects(manifest.get("journals"), "journals")
    if len(journals) != 6:
        raise XR02WP3VerificationError("WP3 evidence journal set is incomplete")
    verified_journals: dict[str, object] = {}
    for journal in journals:
        path = Path(str(journal.get("path")))
        _verify_identity(path, journal)
        result = verify_global_journal(path)
        if (
            result.records != journal.get("records")
            or result.final_sha256 != journal.get("final_sha256")
            or result.signature_sha256 != journal.get("decision_signature_sha256")
        ):
            raise XR02WP3VerificationError("WP3 global journal verification changed")
        key = f"{journal.get('partition')}:{journal.get('profile')}"
        verified_journals[key] = {"records": result.records, "final_sha256": result.final_sha256}

    gate = _object(manifest.get("gate"), "gate")
    if (
        gate.get("passed") is not True
        or float(gate.get("combined", 0.0)) <= float(gate.get("strongest_simple_baseline", 1.0))
        or int(gate.get("combined_false_merge_pairs", 1)) != 0
        or int(gate.get("combined_ambiguous_observations", 0)) <= 0
    ):
        raise XR02WP3VerificationError("WP3 heldout ablation gate failed")
    scorecard = _object(manifest.get("scorecard"), "scorecard")
    _verify_identity(Path(str(scorecard.get("path"))), scorecard)
    rerun = _object(manifest.get("rerun"), "rerun")
    rerun_path = Path(str(rerun.get("path")))
    _verify_identity(rerun_path, rerun)
    entities = _rerun_entities(rerun_path)
    required = {"/xr02/wp3/status", "/xr02/wp3/world/floor_z0"}
    if not required.issubset(entities) or not any("/trajectories/" in item for item in entities):
        raise XR02WP3VerificationError("WP3 Rerun floor trajectories are incomplete")
    claims = _object(manifest.get("claims"), "claims")
    if any(claims.get(key) is not False for key in claims):
        raise XR02WP3VerificationError("WP3 bounded-claim flags changed")
    serialized = manifest_path.read_text(encoding="utf-8").lower()
    if "rtsp://" in serialized or "rtsps://" in serialized or "phase2_rtsp_camera" in serialized:
        raise XR02WP3VerificationError("WP3 evidence contains endpoint or credential material")
    return {
        "schema": "xr02.wp3.replay_verification.v1",
        "status": "passed",
        "manifest": _identity(manifest_path),
        "journal_count": len(verified_journals),
        "journals": verified_journals,
        "rerun_entity_count": len(entities),
        "required_rerun_entities_present": True,
        "heldout_ablation_gate_passed": True,
        "ambiguous_state_present": True,
        "credentials_absent": True,
        "live_rtsp_used": False,
    }


def _rerun_entities(path: Path) -> set[str]:
    import rerun.dataframe as rdf

    recording = rdf.load_recording(path)
    return {
        str(component).removeprefix("Component(").split(":", 1)[0]
        for component in recording.schema().component_columns()
    }


def _verify_identity(path: Path, expected: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != expected.get("bytes")
        or _sha256(path) != expected.get("sha256")
    ):
        raise XR02WP3VerificationError(f"artifact identity changed: {path}")


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise XR02WP3VerificationError(f"{label} must be an object")
    return value


def _list_of_objects(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise XR02WP3VerificationError(f"{label} must be a list of objects")
    return value
