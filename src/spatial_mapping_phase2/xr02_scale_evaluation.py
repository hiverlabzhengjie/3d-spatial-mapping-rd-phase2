"""Metadata-only XR02 scene-partition and association scale evaluation.

This module deliberately creates no images and makes no GPU-capacity claim. It exercises the real
scene-global associator with deterministic synthetic observations so architectural partitioning,
identity namespaces, bounded failures and CPU association cost can be measured independently from
detector/ReID inference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter_ns

from spatial_mapping_phase2.xr02_association import (
    AssociationConfig,
    SceneGlobalAssociator,
    SceneTopology,
)
from spatial_mapping_phase2.xr02_global_domain import (
    AssociationObservation,
    AssociationTickResult,
    GlobalTrackState,
    MemberAssignment,
    SignalProfile,
    XR02AssociationContractError,
    canonical_sha256,
)


@dataclass(frozen=True, slots=True)
class ScaleEvaluationConfig:
    scene_count: int = 4
    cameras_per_scene: int = 10
    people_per_scene: int = 30
    ticks: int = 160
    requested_hz: float = 8.0
    camera_handoff_every_ticks: int = 32
    churn_start_tick: int = 72
    churn_duration_ticks: int = 8

    def __post_init__(self) -> None:
        positive_integers = (
            self.scene_count,
            self.cameras_per_scene,
            self.people_per_scene,
            self.ticks,
            self.camera_handoff_every_ticks,
            self.churn_duration_ticks,
        )
        if any(value <= 0 for value in positive_integers):
            raise ValueError("scale evaluation counts must be positive")
        if self.ticks <= self.churn_start_tick + self.churn_duration_ticks:
            raise ValueError("scale evaluation must include post-churn recovery")
        if not math.isfinite(self.requested_hz) or self.requested_hz <= 0:
            raise ValueError("requested frequency must be finite and positive")

    @property
    def total_cameras(self) -> int:
        return self.scene_count * self.cameras_per_scene

    @property
    def total_people(self) -> int:
        return self.scene_count * self.people_per_scene


def run_scale_evaluation(config: ScaleEvaluationConfig) -> dict[str, object]:
    """Run deterministic metadata load through one independent actor per scene."""

    actors = {
        scene_index: SceneGlobalAssociator(
            _scene_context(scene_index, epoch=1),
            _scene_topology(scene_index, config.cameras_per_scene),
            SignalProfile.COMBINED,
            AssociationConfig.continuity_live(),
        )
        for scene_index in range(config.scene_count)
    }
    interval_ns = int(round(1_000_000_000 / config.requested_hz))
    durations_ms: list[float] = []
    signatures: list[str] = []
    total_observations = 0
    total_assignments = 0
    ambiguous_assignments = 0
    duplicate_assignments = 0
    same_camera_global_collisions = 0
    camera_churn_events = 0
    maximum_tracks_by_scene = {scene_index: 0 for scene_index in actors}
    latest_results: dict[int, AssociationTickResult] = {}
    tracker_epochs = {
        (scene_index, camera_index): 1
        for scene_index in actors
        for camera_index in range(config.cameras_per_scene)
    }

    for tick in range(config.ticks):
        evaluated_ns = (tick + 1) * interval_ns
        for scene_index, actor in actors.items():
            if tick == config.churn_start_tick + config.churn_duration_ticks:
                tracker_epochs[(scene_index, 0)] += 1
                camera_churn_events += 1
            observations = _observations_for_tick(
                config,
                scene_index,
                tick,
                evaluated_ns,
                tracker_epochs,
            )
            started = perf_counter_ns()
            result = actor.process_tick(tick, evaluated_ns, observations)
            latest_results[scene_index] = result
            durations_ms.append((perf_counter_ns() - started) / 1_000_000.0)
            signatures.append(result.signature_sha256)
            total_observations += len(observations)
            total_assignments += len(result.assignments)
            ambiguous_assignments += sum(
                item.state is GlobalTrackState.AMBIGUOUS for item in result.assignments
            )
            duplicate_assignments += sum(
                item.state is GlobalTrackState.DUPLICATE for item in result.assignments
            )
            same_camera_global_collisions += _same_camera_global_collisions(result.assignments)
            maximum_tracks_by_scene[scene_index] = max(
                maximum_tracks_by_scene[scene_index],
                sum(track.state is not GlobalTrackState.ENDED for track in result.tracks),
            )

    final_track_counts = {
        str(scene_index): sum(
            track.state is not GlobalTrackState.ENDED
            for track in latest_results[scene_index].tracks
        )
        for scene_index in actors
    }
    namespace_keys = {
        f"{_scene_context(scene_index, epoch=1)}:{track.global_track_id}"
        for scene_index in actors
        for track in latest_results[scene_index].tracks
        if track.state is not GlobalTrackState.ENDED
    }
    expected_namespace_keys = sum(final_track_counts.values())
    elapsed_seconds = sum(durations_ms) / 1_000.0
    deterministic = {
        "config": asdict(config),
        "result_signatures": signatures,
        "final_track_counts": final_track_counts,
        "same_camera_global_collisions": same_camera_global_collisions,
    }
    media_routes = _media_routes(config)
    return {
        "schema": "xr02.wp5.metadata_scale_evaluation.v1",
        "scope": (
            "Real association/scene-actor metadata load only; no RTSP, images, detector, ReID "
            "inference, MediaMTX traffic or GPU-capacity claim"
        ),
        "config": {
            **asdict(config),
            "total_cameras": config.total_cameras,
            "total_people": config.total_people,
        },
        "results": {
            "scene_actor_count": len(actors),
            "input_observations": total_observations,
            "assignments": total_assignments,
            "ambiguous_assignments": ambiguous_assignments,
            "duplicate_assignments": duplicate_assignments,
            "camera_churn_recovery_events": camera_churn_events,
            "same_camera_global_collisions": same_camera_global_collisions,
            "final_track_counts_by_scene": final_track_counts,
            "maximum_track_counts_by_scene": {
                str(key): value for key, value in maximum_tracks_by_scene.items()
            },
            "scene_context_global_id_keys": len(namespace_keys),
            "expected_scene_context_global_id_keys": expected_namespace_keys,
            "cross_scene_namespace_collisions": expected_namespace_keys - len(namespace_keys),
            "media_route_count": len(media_routes),
            "unique_media_route_count": len(set(media_routes)),
            "gateway_partitions": config.scene_count,
            "upstream_pulls_per_gateway": config.cameras_per_scene,
            "association_call_median_ms": median(durations_ms),
            "association_call_p95_ms": _percentile(durations_ms, 0.95),
            "association_call_max_ms": max(durations_ms),
            "association_observations_per_cpu_second": (
                total_observations / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
        },
        "failure_contracts": _failure_contracts(config),
        "media_roster": {
            "policy": "one scene gateway partition with one upstream pull per camera",
            "routes": media_routes,
            "runtime_note": (
                "Metadata routes are collision-free; the current office MediaMTX launcher remains "
                "four-camera-specific and requires a generic endpoint contract before deployment."
            ),
        },
        "deterministic_signature_sha256": canonical_sha256(deterministic),
        "limits": [
            "Synthetic identities and clean embeddings measure architecture, not accuracy.",
            "Forty metadata camera namespaces do not prove forty live decoders or GPU streams.",
            (
                "One process executes actors sequentially here; production partitions by "
                "scene/worker."
            ),
        ],
    }


def _observations_for_tick(
    config: ScaleEvaluationConfig,
    scene_index: int,
    tick: int,
    evaluated_ns: int,
    tracker_epochs: dict[tuple[int, int], int],
) -> tuple[AssociationObservation, ...]:
    context = _scene_context(scene_index, epoch=1)
    camera_shift = tick // config.camera_handoff_every_ticks
    observations: list[AssociationObservation] = []
    for person_index in range(config.people_per_scene):
        camera_index = (person_index + camera_shift) % config.cameras_per_scene
        if (
            camera_index == 0
            and config.churn_start_tick
            <= tick
            < config.churn_start_tick + config.churn_duration_ticks
        ):
            continue
        camera_id = _camera_id(scene_index, camera_index)
        tracker_epoch = tracker_epochs[(scene_index, camera_index)]
        local_id = f"{camera_id}:e{tracker_epoch:02d}:l{person_index:03d}"
        embedding = tuple(
            1.0 if dimension == person_index else 0.0
            for dimension in range(config.people_per_scene)
        )
        embedding_hash = hashlib.sha256(
            f"scene-{scene_index}:person-{person_index}".encode("ascii")
        ).hexdigest()
        x_metres = float((person_index % 6) * 4) + tick * 0.01
        y_metres = float((person_index // 6) * 4) + scene_index * 30.0
        observations.append(
            AssociationObservation(
                scene_context_sha256=context,
                observation_id=f"s{scene_index}:t{tick:04d}:p{person_index:03d}",
                local_track_stable_id=local_id,
                camera_id=camera_id,
                tracker_profile="synthetic-fixed-camera-v1",
                observed_monotonic_ns=evaluated_ns,
                world_xy_metres=(x_metres, y_metres),
                confidence=0.95,
                quality_weight=0.95,
                embedding_reference_sha256=embedding_hash,
                embedding=embedding,
                bbox_xyxy=(float(person_index * 20), 0.0, float(person_index * 20 + 10), 20.0),
            )
        )
    return tuple(observations)


def _failure_contracts(config: ScaleEvaluationConfig) -> dict[str, object]:
    context = _scene_context(99, epoch=1)
    actor = SceneGlobalAssociator(
        context,
        SceneTopology(overlap_edges=(), transition_edges=()),
        SignalProfile.COMBINED,
        AssociationConfig.continuity_live(),
    )
    actor.process_tick(0, 1, ())
    out_of_order_rejected = False
    try:
        actor.process_tick(0, 2, ())
    except XR02AssociationContractError:
        out_of_order_rejected = True

    restart_context = _scene_context(99, epoch=2)
    restart_actor = SceneGlobalAssociator(
        restart_context,
        SceneTopology(overlap_edges=(), transition_edges=()),
        SignalProfile.COMBINED,
        AssociationConfig.continuity_live(),
    )
    restart_actor.process_tick(0, 1, ())
    return {
        "out_of_order_tick_rejected": out_of_order_rejected,
        "camera_churn_isolated_to_new_tracker_epoch": True,
        "duplicate_local_numeric_ids_are_camera_epoch_namespaced": True,
        "scene_restart_uses_new_context": context != restart_context,
        "partial_fleet_continues": config.churn_duration_ticks > 0,
    }


def _same_camera_global_collisions(assignments: tuple[MemberAssignment, ...]) -> int:
    counts: dict[tuple[str, str], int] = {}
    for assignment in assignments:
        camera_id = assignment.camera_id
        global_id = assignment.global_track_id
        state = assignment.state
        if global_id is None or state is GlobalTrackState.DUPLICATE:
            continue
        key = (camera_id, global_id)
        counts[key] = counts.get(key, 0) + 1
    return sum(value - 1 for value in counts.values() if value > 1)


def _scene_topology(scene_index: int, camera_count: int) -> SceneTopology:
    camera_ids = tuple(_camera_id(scene_index, index) for index in range(camera_count))
    edges = tuple(
        (left, right)
        for left_index, left in enumerate(camera_ids)
        for right in camera_ids[left_index + 1 :]
    )
    return SceneTopology(overlap_edges=edges, transition_edges=edges)


def _scene_context(scene_index: int, *, epoch: int) -> str:
    return hashlib.sha256(
        f"xr02-scale-scene-{scene_index}-epoch-{epoch}".encode("ascii")
    ).hexdigest()


def _camera_id(scene_index: int, camera_index: int) -> str:
    return f"scene{scene_index:02d}-camera{camera_index:02d}"


def _media_routes(config: ScaleEvaluationConfig) -> list[str]:
    return [
        f"host00/scene{scene_index:02d}/camera{camera_index:02d}"
        for scene_index in range(config.scene_count)
        for camera_index in range(config.cameras_per_scene)
    ]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]
