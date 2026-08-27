"""Replay retained XR02 WP4 local evidence through the current association policy.

This is an association-policy comparison, not a detector/tracker rerun. The retained
content-addressed appearance evidence is reused with the live two-second cache bound;
the source journals and recordings remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spatial_mapping_phase2.xr02_association import (  # noqa: E402
    AssociationConfig,
    SceneGlobalAssociator,
    office_topology,
)
from spatial_mapping_phase2.xr02_global_domain import (  # noqa: E402
    AssociationObservation,
    AssociationTickResult,
    GlobalTrackState,
    SignalProfile,
)
from spatial_mapping_phase2.xr02_global_journal import (  # noqa: E402
    GlobalAssociationJournal,
    verify_global_journal,
)
from spatial_mapping_phase2.xr02_journal import verify_journal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay retained WP4 observations through the current association policy"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario-tick",
        type=int,
        action="append",
        default=[],
        help="Optional tick to copy into the human-inspection section; repeat as needed",
    )
    args = parser.parse_args()
    report = replay_association(
        args.source.resolve(),
        args.output.resolve(),
        scenario_ticks=tuple(args.scenario_tick),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def replay_association(
    source: Path,
    output: Path,
    *,
    scenario_ticks: tuple[int, ...] = (),
) -> dict[str, object]:
    local_path = source / "wp4-local-observations.jsonl"
    global_path = source / "wp4-global-association.jsonl"
    manifest_path = source / "wp4-live-manifest.json"
    for path in (local_path, global_path, manifest_path):
        if not path.is_file():
            raise RuntimeError(f"retained WP4 input is missing: {path}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("replay output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    local_verification = verify_journal(local_path)
    global_verification = verify_global_journal(global_path)
    local_payloads = _load_local_payloads(local_path)
    original_payloads = _load_global_payloads(global_path)
    if len(original_payloads) != global_verification.records:
        raise RuntimeError("verified global-journal record count changed during replay")

    first_payload = original_payloads[0]
    scene_context = _required_string(first_payload, "scene_context_sha256")
    associator = SceneGlobalAssociator(
        scene_context,
        office_topology(),
        SignalProfile.COMBINED,
        AssociationConfig.continuity_live(),
    )
    replay_journal_path = output / "wp4-reachable-radius-association.jsonl"
    replay_journal = GlobalAssociationJournal(replay_journal_path)
    embedding_cache: dict[str, tuple[int, str, tuple[float, ...]]] = {}
    replay_results: list[AssociationTickResult] = []
    for original in original_payloads:
        tick_index = _required_int(original, "tick_index")
        evaluated_ns = _required_int(original, "evaluated_monotonic_ns")
        assignments = _required_list(original, "assignments")
        observations: list[AssociationObservation] = []
        for raw_assignment in assignments:
            assignment = _required_mapping(raw_assignment, "assignment")
            observation_id = _required_string(assignment, "observation_id")
            payload = local_payloads.get(observation_id)
            if payload is None:
                raise RuntimeError(f"local evidence missing for {observation_id}")
            observation = _to_association_observation(
                source, payload, embedding_cache, scene_context
            )
            if observation is not None:
                observations.append(observation)
        result = associator.process_tick(tick_index, evaluated_ns, tuple(observations))
        replay_journal.append(result)
        replay_results.append(result)

    replay_verification = verify_global_journal(replay_journal_path)
    original_metrics = _metrics_from_payloads(
        original_payloads, local_payloads, scenario_ticks=scenario_ticks
    )
    replay_metrics = _metrics_from_results(
        replay_results, local_payloads, scenario_ticks=scenario_ticks
    )
    report: dict[str, object] = {
        "schema": "xr02.wp4.association_replay.v3",
        "scope": (
            "Association-only replay of immutable retained WP4 local evidence; persisted "
            "appearance cache, no detector/BoT-SORT rerun and no live RTSP"
        ),
        "source": {
            "directory": str(source),
            "manifest_sha256": _sha256_file(manifest_path),
            "local_journal_sha256": _sha256_file(local_path),
            "local_journal_records": local_verification.records,
            "global_journal_sha256": _sha256_file(global_path),
            "global_journal_records": global_verification.records,
            "global_journal_final_sha256": global_verification.final_sha256,
        },
        "profile_id": associator.profile_id,
        "original": original_metrics,
        "replay": replay_metrics,
        "replay_journal": {
            "path": str(replay_journal_path),
            "sha256": _sha256_file(replay_journal_path),
            "records": replay_verification.records,
            "final_sha256": replay_verification.final_sha256,
            "signature_sha256": replay_verification.signature_sha256,
        },
        "limitations": [
            (
                "The retained local journal persists appearance at 2 Hz, while the live path "
                "also used transient fresh embeddings."
            ),
            (
                "The 8 Hz BoT-SORT frame-rate correction requires a model/video rerun or the "
                "next live trial; association-only replay does not prove it."
            ),
            (
                "The owner trial description is scenario context, not a frame-by-frame "
                "ground-truth annotation."
            ),
        ],
    }
    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["comparison"] = {
        "path": str(report_path),
        "sha256": _sha256_file(report_path),
    }
    return report


def _load_local_payloads(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = _required_mapping(json.loads(line), "local record")
            payload = _required_mapping(record.get("payload"), "local payload")
            frame = _required_mapping(payload.get("frame"), "frame")
            observation_id = (
                f"{_required_string(frame, 'frame_id')}.d"
                f"{_required_int(payload, 'detection_index')}"
            )
            result[observation_id] = payload
    return result


def _load_global_payloads(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = _required_mapping(json.loads(line), "global record")
            result.append(_required_mapping(record.get("payload"), "global payload"))
    return result


def _to_association_observation(
    source: Path,
    payload: dict[str, Any],
    embedding_cache: dict[str, tuple[int, str, tuple[float, ...]]],
    scene_context: str,
) -> AssociationObservation | None:
    if payload.get("projection_status") != "valid":
        return None
    frame = _required_mapping(payload.get("frame"), "frame")
    track = _required_mapping(payload.get("track"), "track")
    world_xy = _required_number_list(payload, "world_xy_metres", 2)
    bbox = _required_number_list(payload, "bbox_xyxy", 4)
    crop = _required_mapping(payload.get("crop_quality"), "crop quality")
    local_id = _required_string(track, "stable_id")
    observed_ns = _required_int(frame, "acquisition_monotonic_ns")
    embedding_sha: str | None = None
    embedding_vector: tuple[float, ...] | None = None
    raw_embedding = payload.get("embedding")
    if raw_embedding is not None:
        reference = _required_mapping(raw_embedding, "embedding reference")
        embedding_sha = _required_string(reference, "sha256")
        relative_path = _required_string(reference, "relative_path")
        embedding_path = source / Path(relative_path)
        if _sha256_file(embedding_path) != embedding_sha:
            raise RuntimeError("retained embedding identity changed")
        vector = np.asarray(np.load(embedding_path, allow_pickle=False), dtype=np.float32)
        embedding_vector = tuple(float(value) for value in vector.reshape(-1))
        embedding_cache[local_id] = (observed_ns, embedding_sha, embedding_vector)
    else:
        cached = embedding_cache.get(local_id)
        if cached is not None:
            cached_ns, cached_sha, cached_vector = cached
            age_seconds = (observed_ns - cached_ns) / 1_000_000_000.0
            if 0.0 <= age_seconds <= 2.0:
                embedding_sha = cached_sha
                embedding_vector = cached_vector

    confidence = _required_float(payload, "confidence")
    quality = max(
        1e-6,
        min(1.0, confidence * _required_float(crop, "visible_fraction")),
    )
    return AssociationObservation(
        scene_context_sha256=scene_context,
        observation_id=(
            f"{_required_string(frame, 'frame_id')}.d{_required_int(payload, 'detection_index')}"
        ),
        local_track_stable_id=local_id,
        camera_id=_required_string(frame, "camera_id"),
        tracker_profile=_required_string(track, "tracker_profile"),
        observed_monotonic_ns=observed_ns,
        world_xy_metres=(world_xy[0], world_xy[1]),
        confidence=confidence,
        quality_weight=quality,
        embedding_reference_sha256=embedding_sha,
        embedding=embedding_vector,
        bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )


def _metrics_from_payloads(
    payloads: list[dict[str, Any]],
    local_payloads: dict[str, dict[str, Any]],
    *,
    scenario_ticks: tuple[int, ...],
) -> dict[str, object]:
    assignments = [
        _required_mapping(item, "assignment")
        for payload in payloads
        for item in _required_list(payload, "assignments")
    ]
    return _summarize(
        payloads,
        assignments,
        local_payloads,
        scenario_ticks=scenario_ticks,
    )


def _metrics_from_results(
    results: list[AssociationTickResult],
    local_payloads: dict[str, dict[str, Any]],
    *,
    scenario_ticks: tuple[int, ...],
) -> dict[str, object]:
    payloads = [result.as_dict() for result in results]
    assignments = [assignment.as_dict() for result in results for assignment in result.assignments]
    return _summarize(
        payloads,
        assignments,
        local_payloads,
        scenario_ticks=scenario_ticks,
    )


def _summarize(
    tick_payloads: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    local_payloads: dict[str, dict[str, Any]],
    *,
    scenario_ticks: tuple[int, ...],
) -> dict[str, object]:
    state_counts = Counter(_required_string(item, "state") for item in assignments)
    reason_counts = Counter(_required_string(item, "reason") for item in assignments)
    global_ids = {
        value for item in assignments if (value := item.get("global_track_id")) is not None
    }
    local_ids = {_required_string(item, "local_track_stable_id") for item in assignments}
    globals_by_local: dict[str, set[str]] = defaultdict(set)
    low_confidence_counted = 0
    for assignment in assignments:
        global_id = assignment.get("global_track_id")
        if global_id is None or assignment.get("state") == GlobalTrackState.DUPLICATE.value:
            continue
        if isinstance(global_id, str):
            globals_by_local[_required_string(assignment, "local_track_stable_id")].add(global_id)
        local = local_payloads.get(_required_string(assignment, "observation_id"))
        if local is not None and _required_float(local, "confidence") < 0.65:
            low_confidence_counted += 1

    scenarios: dict[str, object] = {}
    observed_positions: dict[str, tuple[int, tuple[float, float], tuple[str, ...]]] = {}
    jumps_over_one = 0
    jumps_over_two = 0
    raw_speed_over_limit = 0
    trail_breaks = 0
    observed_people_counts: Counter[int] = Counter()
    same_camera_global_collisions = 0
    births: list[dict[str, object]] = []
    for payload in tick_payloads:
        tick = _required_int(payload, "tick_index")
        current_assignments = [
            _required_mapping(item, "assignment")
            for item in _required_list(payload, "assignments")
        ]
        current_tracks = [
            _required_mapping(item, "track") for item in _required_list(payload, "tracks")
        ]
        observed_ids = {
            value
            for item in current_assignments
            if (value := item.get("global_track_id")) is not None
            and item.get("state") != GlobalTrackState.DUPLICATE.value
        }
        observed_people_counts[len(observed_ids)] += 1
        camera_global_counts = Counter(
            (
                _required_string(item, "camera_id"),
                _required_string_value(item.get("global_track_id"), "global track id"),
            )
            for item in current_assignments
            if item.get("global_track_id") is not None
            and item.get("state") != GlobalTrackState.DUPLICATE.value
        )
        same_camera_global_collisions += sum(
            count - 1 for count in camera_global_counts.values() if count > 1
        )
        for item in current_assignments:
            if item.get("reason") != "confirmed_new_scene_global_track":
                continue
            local = local_payloads.get(_required_string(item, "observation_id"))
            births.append(
                {
                    "tick_index": tick,
                    "camera_id": _required_string(item, "camera_id"),
                    "global_track_id": item.get("global_track_id"),
                    "local_track_stable_id": _required_string(item, "local_track_stable_id"),
                    "confidence": (
                        None if local is None else _required_float(local, "confidence")
                    ),
                }
            )
        if tick in scenario_ticks:
            scenarios[str(tick)] = {
                "observed_people": len(observed_ids),
                "assignments": current_assignments,
            }
        track_by_id = {_required_string(item, "global_track_id"): item for item in current_tracks}
        for global_id in observed_ids:
            if not isinstance(global_id, str):
                continue
            track = track_by_id.get(global_id)
            if track is None:
                continue
            xy_values = _required_number_list(track, "last_world_xy_metres", 2)
            xy = (xy_values[0], xy_values[1])
            observed_ns = _required_int(track, "last_observed_monotonic_ns")
            cameras = tuple(
                _required_string_value(item, "camera id")
                for item in _required_list(track, "camera_ids")
            )
            previous = observed_positions.get(global_id)
            if previous is not None:
                previous_ns, previous_xy, previous_cameras = previous
                elapsed = (observed_ns - previous_ns) / 1_000_000_000.0
                distance = math.dist(previous_xy, xy)
                if distance > 1.0:
                    jumps_over_one += 1
                if distance > 2.0:
                    jumps_over_two += 1
                if elapsed > 0 and distance / elapsed > 3.5:
                    raw_speed_over_limit += 1
                if (
                    elapsed > 1.0
                    or distance > 2.0
                    or (cameras != previous_cameras and distance > 1.0)
                ):
                    trail_breaks += 1
            observed_positions[global_id] = (observed_ns, xy, cameras)

    return {
        "ticks": len(tick_payloads),
        "assignments": len(assignments),
        "global_ids_assigned": len(global_ids),
        "local_track_ids": len(local_ids),
        "assignment_states": dict(sorted(state_counts.items())),
        "assignment_reasons": dict(sorted(reason_counts.items())),
        "low_confidence_counted_assignments": low_confidence_counted,
        "observed_people_tick_counts": {
            str(count): ticks for count, ticks in sorted(observed_people_counts.items())
        },
        "maximum_observed_people": max(observed_people_counts, default=0),
        "same_camera_global_collisions": same_camera_global_collisions,
        "local_tracks_mapped_to_multiple_globals": {
            local_id: sorted(ids)
            for local_id, ids in sorted(globals_by_local.items())
            if len(ids) > 1
        },
        "confirmed_identity_births": births,
        "jumps_over_1m": jumps_over_one,
        "jumps_over_2m": jumps_over_two,
        "raw_speed_over_3_5mps": raw_speed_over_limit,
        "rerun_trail_breaks_required": trail_breaks,
        "scenario_ticks": scenarios,
    }


def _required_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _required_list(value: dict[str, Any], key: str) -> list[object]:
    result = value.get(key)
    if not isinstance(result, list):
        raise RuntimeError(f"{key} must be an array")
    return result


def _required_string(value: dict[str, Any], key: str) -> str:
    return _required_string_value(value.get(key), key)


def _required_string_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def _required_int(value: dict[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise RuntimeError(f"{key} must be an integer")
    return result


def _required_float(value: dict[str, Any], key: str) -> float:
    result = value.get(key)
    if not isinstance(result, int | float) or isinstance(result, bool):
        raise RuntimeError(f"{key} must be numeric")
    number = float(result)
    if not math.isfinite(number):
        raise RuntimeError(f"{key} must be finite")
    return number


def _required_number_list(value: dict[str, Any], key: str, expected_length: int) -> list[float]:
    result = _required_list(value, key)
    if len(result) != expected_length:
        raise RuntimeError(f"{key} has the wrong length")
    return [_required_float({"value": item}, "value") for item in result]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
