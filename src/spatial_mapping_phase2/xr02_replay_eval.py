"""Bounded anonymous replay fixtures and labelled WP3 association metrics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.xr02_association import (
    AssociationConfig,
    SceneGlobalAssociator,
    SceneTopology,
)
from spatial_mapping_phase2.xr02_global_domain import (
    AssociationObservation,
    AssociationTickResult,
    GlobalTrackState,
    SignalProfile,
    XR02AssociationContractError,
    canonical_sha256,
)


@dataclass(frozen=True, slots=True)
class LabelledObservation:
    observation: AssociationObservation
    anonymous_ground_truth_id: str


@dataclass(frozen=True, slots=True)
class ReplayTick:
    tick_index: int
    evaluated_monotonic_ns: int
    labelled_observations: tuple[LabelledObservation, ...]


@dataclass(frozen=True, slots=True)
class ReplayScenario:
    scenario_id: str
    description: str
    scene_context_sha256: str
    ticks: tuple[ReplayTick, ...]


@dataclass(frozen=True, slots=True)
class ReplayPartition:
    partition: str
    source_path: str
    source_sha256: str
    scenarios: tuple[ReplayScenario, ...]


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    scenario: ReplayScenario
    profile: SignalProfile
    profile_id: str
    results: tuple[AssociationTickResult, ...]
    metrics: dict[str, int | float | str]
    decision_signature_sha256: str


def load_replay_partition(path: Path) -> ReplayPartition:
    raw_bytes = path.read_bytes()
    loaded = json.loads(raw_bytes)
    root = _mapping(loaded, "fixture root")
    if root.get("schema") != "xr02.wp3.labelled_replay.v1":
        raise XR02AssociationContractError("WP3 replay fixture schema changed")
    partition = _string(root, "partition")
    if partition not in {"tuning", "heldout"}:
        raise XR02AssociationContractError("replay partition must be tuning or heldout")
    scene_context = _string(root, "scene_context_sha256")
    raw_scenarios = root.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise XR02AssociationContractError("replay partition requires scenarios")
    scenarios: list[ReplayScenario] = []
    for raw_scenario in raw_scenarios:
        scenario = _mapping(raw_scenario, "scenario")
        scenario_id = _string(scenario, "scenario_id")
        raw_ticks = scenario.get("ticks")
        if not isinstance(raw_ticks, list) or not raw_ticks:
            raise XR02AssociationContractError("scenario requires ticks")
        ticks: list[ReplayTick] = []
        for tick_index, raw_tick in enumerate(raw_ticks):
            tick = _mapping(raw_tick, "tick")
            time_ns = _integer(tick, "time_ns")
            rows = tick.get("observations")
            if not isinstance(rows, list):
                raise XR02AssociationContractError("tick observations must be a list")
            labelled: list[LabelledObservation] = []
            for row_value in rows:
                row = _mapping(row_value, "labelled observation")
                embedding_raw = row.get("embedding")
                embedding = None
                embedding_sha = None
                if embedding_raw is not None:
                    if not isinstance(embedding_raw, list) or not embedding_raw:
                        raise XR02AssociationContractError("fixture embedding must be a vector")
                    embedding = tuple(float(value) for value in embedding_raw)
                    embedding_sha = canonical_sha256(embedding)
                world_xy = row.get("world_xy_metres")
                if not isinstance(world_xy, list) or len(world_xy) != 2:
                    raise XR02AssociationContractError("fixture world XY must contain two values")
                observation = AssociationObservation(
                    scene_context_sha256=scene_context,
                    observation_id=_string(row, "observation_id"),
                    local_track_stable_id=_string(row, "local_track_stable_id"),
                    camera_id=_string(row, "camera_id"),
                    tracker_profile=_string(row, "tracker_profile"),
                    observed_monotonic_ns=time_ns + _integer(row, "time_offset_ns", default=0),
                    world_xy_metres=(float(world_xy[0]), float(world_xy[1])),
                    confidence=_number(row, "confidence", default=0.90),
                    quality_weight=_number(row, "quality_weight", default=0.90),
                    embedding_reference_sha256=embedding_sha,
                    embedding=embedding,
                )
                labelled.append(
                    LabelledObservation(
                        observation=observation,
                        anonymous_ground_truth_id=_string(row, "anonymous_ground_truth_id"),
                    )
                )
            ticks.append(ReplayTick(tick_index, time_ns, tuple(labelled)))
        scenarios.append(
            ReplayScenario(
                scenario_id=scenario_id,
                description=_string(scenario, "description"),
                scene_context_sha256=scene_context,
                ticks=tuple(ticks),
            )
        )
    return ReplayPartition(
        partition=partition,
        source_path=str(path.resolve()),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        scenarios=tuple(scenarios),
    )


def run_scenario(
    scenario: ReplayScenario,
    profile: SignalProfile,
    topology: SceneTopology,
    config: AssociationConfig | None = None,
) -> ScenarioRun:
    """Run without exposing anonymous labels to the association engine."""

    associator = SceneGlobalAssociator(
        scenario.scene_context_sha256,
        topology,
        profile,
        config,
    )
    results = tuple(
        associator.process_tick(
            tick.tick_index,
            tick.evaluated_monotonic_ns,
            tuple(labelled.observation for labelled in tick.labelled_observations),
        )
        for tick in scenario.ticks
    )
    metrics = evaluate_scenario(scenario, results)
    signature = canonical_sha256([result.signature_sha256 for result in results])
    return ScenarioRun(scenario, profile, associator.profile_id, results, metrics, signature)


def evaluate_scenario(
    scenario: ReplayScenario, results: tuple[AssociationTickResult, ...]
) -> dict[str, int | float | str]:
    if len(scenario.ticks) != len(results):
        raise XR02AssociationContractError("scenario and result tick counts disagree")
    ground_truth: dict[str, str] = {}
    predictions: dict[str, str | None] = {}
    states: Counter[str] = Counter()
    by_identity_tick: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    by_identity_tick_cameras: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    dedup_true_positive = 0
    dedup_false_negative = 0
    dedup_false_positive = 0
    dedup_true_negative = 0

    for tick, result in zip(scenario.ticks, results, strict=True):
        result_assignments = {item.observation_id: item for item in result.assignments}
        if len(result_assignments) != len(tick.labelled_observations):
            raise XR02AssociationContractError("result omitted replay observations")
        labelled_by_id = {
            item.observation.observation_id: item for item in tick.labelled_observations
        }
        for observation_id, labelled in labelled_by_id.items():
            assignment = result_assignments[observation_id]
            ground_truth[observation_id] = labelled.anonymous_ground_truth_id
            predictions[observation_id] = assignment.global_track_id
            states[assignment.state.value] += 1
            if assignment.global_track_id is not None:
                by_identity_tick[labelled.anonymous_ground_truth_id][tick.tick_index].append(
                    assignment.global_track_id
                )
            by_identity_tick_cameras[labelled.anonymous_ground_truth_id][tick.tick_index].add(
                labelled.observation.camera_id
            )
        for left, right in combinations(tick.labelled_observations, 2):
            if left.observation.camera_id == right.observation.camera_id:
                continue
            same_truth = left.anonymous_ground_truth_id == right.anonymous_ground_truth_id
            left_prediction = predictions[left.observation.observation_id]
            right_prediction = predictions[right.observation.observation_id]
            same_prediction = left_prediction is not None and left_prediction == right_prediction
            if same_truth and same_prediction:
                dedup_true_positive += 1
            elif same_truth:
                dedup_false_negative += 1
            elif same_prediction:
                dedup_false_positive += 1
            else:
                dedup_true_negative += 1

    identity_switches = 0
    handoff_attempts = 0
    handoff_successes = 0
    continuity_correct = 0
    continuity_assigned = 0
    for identity, per_tick in by_identity_tick.items():
        chronological = sorted(per_tick)
        previous_id: str | None = None
        previous_tick: int | None = None
        all_ids = [item for tick_ids in per_tick.values() for item in tick_ids]
        if all_ids:
            mode_count = Counter(all_ids).most_common(1)[0][1]
            continuity_correct += mode_count
            continuity_assigned += len(all_ids)
        for tick_index in chronological:
            tick_ids = set(per_tick[tick_index])
            selected = sorted(tick_ids)[0] if len(tick_ids) == 1 else None
            if selected is None:
                previous_id = None
                previous_tick = tick_index
                continue
            if previous_id is not None and selected != previous_id:
                identity_switches += 1
            if previous_tick is not None:
                previous_cameras = by_identity_tick_cameras[identity][previous_tick]
                current_cameras = by_identity_tick_cameras[identity][tick_index]
                if previous_cameras != current_cameras:
                    handoff_attempts += 1
                    if selected == previous_id:
                        handoff_successes += 1
            previous_id = selected
            previous_tick = tick_index

    false_merge_pairs = 0
    true_identity_pairs = 0
    recovered_identity_pairs = 0
    for left_id, right_id in combinations(sorted(ground_truth), 2):
        same_truth = ground_truth[left_id] == ground_truth[right_id]
        left_prediction = predictions[left_id]
        same_prediction = left_prediction is not None and left_prediction == predictions[right_id]
        if same_truth:
            true_identity_pairs += 1
            recovered_identity_pairs += int(same_prediction)
        elif same_prediction:
            false_merge_pairs += 1

    dedup_precision = _ratio(dedup_true_positive, dedup_true_positive + dedup_false_positive)
    dedup_recall = _ratio(dedup_true_positive, dedup_true_positive + dedup_false_negative)
    pairwise_precision = _ratio(
        recovered_identity_pairs,
        recovered_identity_pairs + false_merge_pairs,
    )
    pairwise_recall = _ratio(recovered_identity_pairs, true_identity_pairs)
    return {
        "metric_scope": "bounded anonymous fixture; not MOTChallenge IDF1/HOTA",
        "observation_count": len(ground_truth),
        "identity_switches": identity_switches,
        "false_merge_pairs": false_merge_pairs,
        "ambiguous_observations": states[GlobalTrackState.AMBIGUOUS.value],
        "duplicate_observations": states[GlobalTrackState.DUPLICATE.value],
        "dedup_precision": dedup_precision,
        "dedup_recall": dedup_recall,
        "pairwise_identity_precision": pairwise_precision,
        "pairwise_identity_recall": pairwise_recall,
        "pairwise_identity_f1": _f1(pairwise_precision, pairwise_recall),
        "continuity_modal_fraction": _ratio(continuity_correct, continuity_assigned),
        "handoff_attempts": handoff_attempts,
        "handoff_success_rate": _ratio(handoff_successes, handoff_attempts),
        "state_counts": json.dumps(dict(sorted(states.items())), sort_keys=True),
    }


def aggregate_runs(runs: tuple[ScenarioRun, ...]) -> dict[str, int | float | str]:
    if not runs:
        raise XR02AssociationContractError("cannot aggregate an empty replay")
    weighted_fields = (
        "dedup_precision",
        "dedup_recall",
        "pairwise_identity_precision",
        "pairwise_identity_recall",
        "pairwise_identity_f1",
        "continuity_modal_fraction",
        "handoff_success_rate",
    )
    totals: dict[str, int | float | str] = {
        "profile_id": runs[0].profile_id,
        "scenario_count": len(runs),
        "observation_count": sum(int(run.metrics["observation_count"]) for run in runs),
        "identity_switches": sum(int(run.metrics["identity_switches"]) for run in runs),
        "false_merge_pairs": sum(int(run.metrics["false_merge_pairs"]) for run in runs),
        "ambiguous_observations": sum(int(run.metrics["ambiguous_observations"]) for run in runs),
        "metric_scope": "macro-average over bounded anonymous fixture scenarios",
    }
    for field in weighted_fields:
        totals[field] = sum(float(run.metrics[field]) for run in runs) / len(runs)
    totals["decision_signature_sha256"] = canonical_sha256(
        [run.decision_signature_sha256 for run in runs]
    )
    return totals


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise XR02AssociationContractError(f"{label} must be an object")
    return value


def _string(value: dict[str, Any], key: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str) or not selected:
        raise XR02AssociationContractError(f"{key} must be a non-empty string")
    return selected


def _integer(value: dict[str, Any], key: str, *, default: int | None = None) -> int:
    selected = value.get(key, default)
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise XR02AssociationContractError(f"{key} must be an integer")
    return selected


def _number(value: dict[str, Any], key: str, *, default: float) -> float:
    selected = value.get(key, default)
    if not isinstance(selected, int | float) or isinstance(selected, bool):
        raise XR02AssociationContractError(f"{key} must be numeric")
    return float(selected)
