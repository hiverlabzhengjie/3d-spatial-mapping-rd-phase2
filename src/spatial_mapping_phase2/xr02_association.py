"""Deterministic complete-link deduplication and Hungarian global association."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from spatial_mapping_phase2.xr02_global_domain import (
    AssociationObservation,
    AssociationTickResult,
    CandidateEvidence,
    CandidateKind,
    CandidateOutcome,
    CostComponents,
    GlobalTrackSnapshot,
    GlobalTrackState,
    MemberAssignment,
    SignalProfile,
    XR02AssociationContractError,
    canonical_sha256,
    cosine_distance,
)
from spatial_mapping_phase2.xr02_topology import SceneTopology as SceneTopology

_INVALID_COST = 1_000_000.0


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    config_id: str = "wp3-association-v1"
    overlap_max_time_seconds: float = 0.30
    overlap_max_spatial_metres: float = 1.50
    overlap_max_appearance_distance: float = 0.35
    assignment_max_spatial_metres: float = 2.50
    assignment_max_appearance_distance: float = 0.45
    minimum_quality_weight: float = 0.10
    dedup_ambiguity_margin: float = 0.08
    assignment_ambiguity_margin: float = 0.08
    occluded_after_seconds: float = 1.10
    reacquisition_max_seconds: float = 4.00
    ended_after_seconds: float = 4.01
    lost_gate_multiplier: float = 0.75
    isolated_camera_gate_multiplier: float = 0.65
    minimum_confirmation_hits: int = 2
    gallery_size: int = 8
    continuity_policy_enabled: bool = False
    measurement_uncertainty_metres: float = 1.0
    maximum_effective_speed_mps: float = 3.5
    velocity_smoothing: float = 0.65
    dormant_after_seconds: float | None = None
    dormant_gate_multiplier: float = 0.55
    interaction_hold_seconds: float = 1.0
    interaction_world_distance_metres: float = 1.5
    interaction_min_bbox_intersection_ratio: float = 0.01
    binding_switch_confirmation_hits: int = 3
    binding_switch_margin: float = 0.15
    new_global_confirmation_hits: int = 2
    appearance_pending_hold_seconds: float = 3.0
    new_global_minimum_confidence: float = 0.65
    binding_release_confirmation_hits: int = 8
    binding_challenge_max_gap_seconds: float = 0.50
    binding_observation_grace_seconds: float = 1.0

    @classmethod
    def continuity_live(cls) -> AssociationConfig:
        """Return the owner-authorized post-Trial-3 live policy.

        The default constructor deliberately remains the accepted WP3 replay profile.
        """

        return cls(
            config_id="wp4-reachable-radius-continuity-v5",
            reacquisition_max_seconds=60.0,
            ended_after_seconds=120.0,
            continuity_policy_enabled=True,
            dormant_after_seconds=8.0,
            new_global_confirmation_hits=4,
        )

    def __post_init__(self) -> None:
        positive = (
            self.overlap_max_time_seconds,
            self.overlap_max_spatial_metres,
            self.overlap_max_appearance_distance,
            self.assignment_max_spatial_metres,
            self.assignment_max_appearance_distance,
            self.occluded_after_seconds,
            self.reacquisition_max_seconds,
            self.ended_after_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise XR02AssociationContractError(
                "association thresholds must be finite and positive"
            )
        if self.ended_after_seconds <= self.reacquisition_max_seconds:
            raise XR02AssociationContractError("ended threshold must exceed reacquisition window")
        if not 0 < self.minimum_quality_weight <= 1:
            raise XR02AssociationContractError("minimum quality must be within (0, 1]")
        if not 0 < self.lost_gate_multiplier <= 1:
            raise XR02AssociationContractError("lost gate multiplier must be within (0, 1]")
        if not 0 < self.isolated_camera_gate_multiplier <= 1:
            raise XR02AssociationContractError("isolated gate multiplier must be within (0, 1]")
        if self.minimum_confirmation_hits <= 0 or self.gallery_size <= 0:
            raise XR02AssociationContractError("confirmation and gallery sizes must be positive")
        if self.continuity_policy_enabled:
            continuity_positive = (
                self.measurement_uncertainty_metres,
                self.maximum_effective_speed_mps,
                self.interaction_hold_seconds,
                self.interaction_world_distance_metres,
                self.interaction_min_bbox_intersection_ratio,
                self.binding_switch_margin,
                self.appearance_pending_hold_seconds,
                self.binding_challenge_max_gap_seconds,
                self.binding_observation_grace_seconds,
            )
            if any(not math.isfinite(value) or value <= 0 for value in continuity_positive):
                raise XR02AssociationContractError(
                    "continuity thresholds must be finite and positive"
                )
            if not 0 < self.velocity_smoothing <= 1:
                raise XR02AssociationContractError("velocity smoothing must be within (0, 1]")
            if not 0 < self.dormant_gate_multiplier <= 1:
                raise XR02AssociationContractError("dormant gate multiplier must be within (0, 1]")
            if self.dormant_after_seconds is None or not (
                self.occluded_after_seconds < self.dormant_after_seconds < self.ended_after_seconds
            ):
                raise XR02AssociationContractError(
                    "dormant threshold must fall between occlusion and ending"
                )
            if (
                self.binding_switch_confirmation_hits <= 1
                or self.new_global_confirmation_hits <= 1
                or self.binding_release_confirmation_hits <= 1
            ):
                raise XR02AssociationContractError(
                    "continuity confirmation counts must exceed one"
                )
            if not 0 < self.new_global_minimum_confidence <= 1:
                raise XR02AssociationContractError("new-global confidence must be within (0, 1]")


@dataclass(slots=True)
class _Cluster:
    members: list[AssociationObservation]
    ambiguous: bool = False

    @property
    def cluster_id(self) -> str:
        digest = canonical_sha256(sorted(item.observation_id for item in self.members))[:16]
        return f"cluster:{digest}"

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted({item.camera_id for item in self.members}))

    @property
    def world_xy(self) -> tuple[float, float]:
        weights = [item.quality_weight for item in self.members]
        total = sum(weights)
        return (
            sum(
                item.world_xy_metres[0] * weight
                for item, weight in zip(self.members, weights, strict=True)
            )
            / total,
            sum(
                item.world_xy_metres[1] * weight
                for item, weight in zip(self.members, weights, strict=True)
            )
            / total,
        )

    @property
    def quality_weight(self) -> float:
        return sum(item.quality_weight for item in self.members) / len(self.members)

    @property
    def embedding(self) -> tuple[float, ...] | None:
        vectors = [item.embedding for item in self.members if item.embedding is not None]
        if not vectors:
            return None
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise XR02AssociationContractError("cluster embedding dimensions disagree")
        mean = tuple(
            sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)
        )
        norm = math.sqrt(sum(value * value for value in mean))
        return None if norm <= 0 else tuple(value / norm for value in mean)

    @property
    def primary(self) -> AssociationObservation:
        return sorted(
            self.members,
            key=lambda item: (-item.quality_weight, -item.confidence, item.observation_id),
        )[0]

    @property
    def observed_monotonic_ns(self) -> int:
        return max(item.observed_monotonic_ns for item in self.members)

    @property
    def local_track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.local_track_stable_id for item in self.members))


@dataclass(slots=True)
class _Track:
    global_track_id: str
    profile_id: str
    state: GlobalTrackState
    last_world_xy_metres: tuple[float, float]
    last_observed_monotonic_ns: int
    camera_ids: tuple[str, ...]
    hit_count: int
    last_local_track_ids: tuple[str, ...]
    gallery: list[tuple[float, ...]] = field(default_factory=list)
    velocity_xy_metres_per_second: tuple[float, float] = (0.0, 0.0)
    position_uncertainty_metres: float | None = None

    @property
    def embedding(self) -> tuple[float, ...] | None:
        if not self.gallery:
            return None
        dimension = len(self.gallery[0])
        mean = tuple(
            sum(vector[index] for vector in self.gallery) / len(self.gallery)
            for index in range(dimension)
        )
        norm = math.sqrt(sum(value * value for value in mean))
        return None if norm <= 0 else tuple(value / norm for value in mean)

    def snapshot(self) -> GlobalTrackSnapshot:
        return GlobalTrackSnapshot(
            global_track_id=self.global_track_id,
            state=self.state,
            last_world_xy_metres=self.last_world_xy_metres,
            last_observed_monotonic_ns=self.last_observed_monotonic_ns,
            camera_ids=self.camera_ids,
            hit_count=self.hit_count,
            profile_id=self.profile_id,
            velocity_xy_metres_per_second=(
                self.velocity_xy_metres_per_second
                if self.position_uncertainty_metres is not None
                else None
            ),
            position_uncertainty_metres=self.position_uncertainty_metres,
        )


@dataclass(slots=True)
class _LocalBinding:
    global_track_id: str
    last_reliable_observed_ns: int
    challenge_count: int = 0
    last_challenge_observed_ns: int | None = None


@dataclass(slots=True)
class _SwapChallenge:
    left_global_id: str
    right_global_id: str
    count: int
    last_tick: int


@dataclass(slots=True)
class _NewTrackConfirmation:
    count: int
    last_tick: int


class SceneGlobalAssociator:
    """One deterministic scene actor; observations and outputs are immutable evidence."""

    def __init__(
        self,
        scene_context_sha256: str,
        topology: SceneTopology,
        profile: SignalProfile,
        config: AssociationConfig | None = None,
    ) -> None:
        self.scene_context_sha256 = scene_context_sha256
        self.topology = topology
        self.profile = profile
        self.config = config or AssociationConfig()
        self.profile_id = f"{profile.value}:{self.config.config_id}"
        self._tracks: dict[str, _Track] = {}
        self._next_global_id = 1
        self._last_tick = -1
        self._last_time_ns = -1
        self._local_bindings: dict[str, _LocalBinding] = {}
        self._interaction_until_by_local: dict[str, int] = {}
        self._swap_challenges: dict[tuple[str, str], _SwapChallenge] = {}
        self._new_track_confirmations: dict[str, _NewTrackConfirmation] = {}

    def process_tick(
        self,
        tick_index: int,
        evaluated_monotonic_ns: int,
        observations: tuple[AssociationObservation, ...],
    ) -> AssociationTickResult:
        if tick_index != self._last_tick + 1:
            raise XR02AssociationContractError("association ticks must be contiguous")
        if evaluated_monotonic_ns <= self._last_time_ns:
            raise XR02AssociationContractError("association time must increase strictly")
        if any(item.scene_context_sha256 != self.scene_context_sha256 for item in observations):
            raise XR02AssociationContractError("observation scene context changed")
        if len({item.observation_id for item in observations}) != len(observations):
            raise XR02AssociationContractError("observation IDs must be unique within a tick")

        self._advance_lifecycle(evaluated_monotonic_ns)
        if self.config.continuity_policy_enabled:
            self._prune_continuity_state(tick_index)
            self._update_interaction_windows(evaluated_monotonic_ns, observations)
        clusters, candidates = self._deduplicate(tick_index, observations)
        assignments, assignment_candidates, updated_ids = self._assign(
            tick_index, evaluated_monotonic_ns, clusters
        )
        candidates.extend(assignment_candidates)
        self._advance_unmatched(evaluated_monotonic_ns, updated_ids)
        self._last_tick = tick_index
        self._last_time_ns = evaluated_monotonic_ns
        return AssociationTickResult(
            scene_context_sha256=self.scene_context_sha256,
            tick_index=tick_index,
            evaluated_monotonic_ns=evaluated_monotonic_ns,
            profile_id=self.profile_id,
            assignments=tuple(sorted(assignments, key=lambda item: item.observation_id)),
            tracks=tuple(self._tracks[key].snapshot() for key in sorted(self._tracks)),
            candidates=tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        )

    def _deduplicate(
        self, tick_index: int, observations: tuple[AssociationObservation, ...]
    ) -> tuple[list[_Cluster], list[CandidateEvidence]]:
        ordered = sorted(observations, key=lambda item: item.observation_id)
        compatibility: dict[frozenset[str], bool] = {}
        pair_costs: dict[frozenset[str], CostComponents] = {}
        candidates: list[CandidateEvidence] = []
        compatible_by_observation: dict[str, list[tuple[float, str]]] = {
            item.observation_id: [] for item in ordered
        }
        for left, right in combinations(ordered, 2):
            costs, accepted, reason = self._overlap_cost(left, right)
            key = frozenset((left.observation_id, right.observation_id))
            compatibility[key] = accepted
            pair_costs[key] = costs
            outcome = CandidateOutcome.ACCEPTED if accepted else CandidateOutcome.REJECTED
            candidate_id = _candidate_id(
                "pair", tick_index, left.observation_id, right.observation_id
            )
            candidates.append(
                CandidateEvidence(
                    candidate_id,
                    CandidateKind.OVERLAP_PAIR,
                    left.observation_id,
                    right.observation_id,
                    self.profile_id,
                    outcome,
                    reason,
                    costs,
                )
            )
            if accepted and costs.normalized_total is not None:
                compatible_by_observation[left.observation_id].append(
                    (costs.normalized_total, right.observation_id)
                )
                compatible_by_observation[right.observation_id].append(
                    (costs.normalized_total, left.observation_id)
                )

        ambiguous_ids: set[str] = set()
        for observation_id, matches in compatible_by_observation.items():
            ranked = sorted(matches)
            if len(ranked) < 2 or ranked[1][0] - ranked[0][0] > self.config.dedup_ambiguity_margin:
                continue
            peer_key = frozenset((ranked[0][1], ranked[1][1]))
            if not compatibility.get(peer_key, False):
                ambiguous_ids.add(observation_id)

        clusters = [_Cluster([item], item.observation_id in ambiguous_ids) for item in ordered]
        cluster_by_observation = {
            item.observation_id: cluster for item, cluster in zip(ordered, clusters, strict=True)
        }
        accepted_pairs = sorted(
            (
                (pair_costs[key].normalized_total or 0.0, tuple(sorted(key)))
                for key, accepted in compatibility.items()
                if accepted and not (set(key) & ambiguous_ids)
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _, pair_ids in accepted_pairs:
            left_cluster = cluster_by_observation[pair_ids[0]]
            right_cluster = cluster_by_observation[pair_ids[1]]
            if left_cluster is right_cluster:
                continue
            if not self._clusters_pairwise_compatible(left_cluster, right_cluster, compatibility):
                merge_cost = pair_costs[frozenset(pair_ids)]
                candidates.append(
                    CandidateEvidence(
                        _candidate_id(
                            "merge", tick_index, left_cluster.cluster_id, right_cluster.cluster_id
                        ),
                        CandidateKind.CLUSTER_MERGE,
                        left_cluster.cluster_id,
                        right_cluster.cluster_id,
                        self.profile_id,
                        CandidateOutcome.REJECTED,
                        "incomplete_pairwise_compatibility",
                        merge_cost,
                    )
                )
                continue
            left_cluster.members.extend(right_cluster.members)
            left_cluster.members.sort(key=lambda item: item.observation_id)
            clusters.remove(right_cluster)
            for member in right_cluster.members:
                cluster_by_observation[member.observation_id] = left_cluster
        return sorted(clusters, key=lambda item: item.cluster_id), candidates

    def _clusters_pairwise_compatible(
        self,
        left: _Cluster,
        right: _Cluster,
        compatibility: dict[frozenset[str], bool],
    ) -> bool:
        cameras = left.cameras + right.cameras
        if len(cameras) != len(set(cameras)):
            return False
        return all(
            compatibility.get(frozenset((a.observation_id, b.observation_id)), False)
            for a in left.members
            for b in right.members
        )

    def _overlap_cost(
        self, left: AssociationObservation, right: AssociationObservation
    ) -> tuple[CostComponents, bool, str]:
        time_gap = abs(left.observed_monotonic_ns - right.observed_monotonic_ns) / 1_000_000_000.0
        spatial = _distance(left.world_xy_metres, right.world_xy_metres)
        appearance = _optional_cosine(left.embedding, right.embedding)
        quality_penalty = 1.0 - min(left.quality_weight, right.quality_weight)
        topology_allowed = self.topology.overlaps(left.camera_id, right.camera_id)
        total = self._normalized_cost(spatial, appearance, time_gap, quality_penalty, overlap=True)
        costs = CostComponents(
            spatial, appearance, time_gap, quality_penalty, topology_allowed, total
        )
        if left.camera_id == right.camera_id:
            return costs, False, "same_camera_cannot_link"
        if not topology_allowed:
            return costs, False, "camera_pair_not_in_overlap_topology"
        if time_gap > self.config.overlap_max_time_seconds:
            return costs, False, "overlap_time_gate"
        if min(left.quality_weight, right.quality_weight) < self.config.minimum_quality_weight:
            return costs, False, "quality_gate"
        return self._profile_gate(spatial, appearance, costs, overlap=True)

    def _assign(
        self,
        tick_index: int,
        now_ns: int,
        clusters: list[_Cluster],
    ) -> tuple[list[MemberAssignment], list[CandidateEvidence], set[str]]:
        if not self.config.continuity_policy_enabled:
            return self._assign_legacy(tick_index, now_ns, clusters)
        return self._assign_continuity(tick_index, now_ns, clusters)

    def _assign_legacy(
        self,
        tick_index: int,
        now_ns: int,
        clusters: list[_Cluster],
    ) -> tuple[list[MemberAssignment], list[CandidateEvidence], set[str]]:
        assignments: list[MemberAssignment] = []
        candidates: list[CandidateEvidence] = []
        updated_ids: set[str] = set()
        assignable = [cluster for cluster in clusters if not cluster.ambiguous]
        for cluster in clusters:
            if cluster.ambiguous:
                assignments.extend(self._ambiguous_assignments(cluster, "overlap_candidate_tie"))
        active_tracks = [
            self._tracks[key]
            for key in sorted(self._tracks)
            if self._tracks[key].state is not GlobalTrackState.ENDED
        ]
        accepted_costs: dict[tuple[str, str], float] = {}
        accepted_by_cluster: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for cluster in assignable:
            for track in active_tracks:
                costs, accepted, reason = self._assignment_cost(cluster, track, now_ns)
                outcome = CandidateOutcome.ACCEPTED if accepted else CandidateOutcome.REJECTED
                candidate = CandidateEvidence(
                    _candidate_id("assign", tick_index, cluster.cluster_id, track.global_track_id),
                    CandidateKind.GLOBAL_ASSIGNMENT,
                    cluster.cluster_id,
                    track.global_track_id,
                    self.profile_id,
                    outcome,
                    reason,
                    costs,
                )
                candidates.append(candidate)
                if accepted and costs.normalized_total is not None:
                    accepted_costs[(cluster.cluster_id, track.global_track_id)] = (
                        costs.normalized_total
                    )
                    accepted_by_cluster[cluster.cluster_id].append(
                        (costs.normalized_total, track.global_track_id)
                    )

        ambiguous_cluster_ids: set[str] = set()
        for cluster_id, options in accepted_by_cluster.items():
            ranked = sorted(options)
            if (
                len(ranked) >= 2
                and ranked[1][0] - ranked[0][0] <= self.config.assignment_ambiguity_margin
            ):
                ambiguous_cluster_ids.add(cluster_id)
                for candidate_index, candidate in enumerate(candidates):
                    if (
                        candidate.left_id == cluster_id
                        and candidate.outcome is CandidateOutcome.ACCEPTED
                    ):
                        candidates[candidate_index] = CandidateEvidence(
                            candidate.candidate_id,
                            candidate.kind,
                            candidate.left_id,
                            candidate.right_id,
                            candidate.profile_id,
                            CandidateOutcome.AMBIGUOUS,
                            "assignment_candidate_tie",
                            candidate.costs,
                        )

        matrix_clusters = [
            cluster for cluster in assignable if cluster.cluster_id not in ambiguous_cluster_ids
        ]
        matched_clusters: set[str] = set()
        if matrix_clusters and active_tracks:
            matrix = np.full(
                (len(matrix_clusters), len(active_tracks)), _INVALID_COST, dtype=np.float64
            )
            for row, cluster in enumerate(matrix_clusters):
                for column, track in enumerate(active_tracks):
                    value = accepted_costs.get((cluster.cluster_id, track.global_track_id))
                    if value is not None:
                        matrix[row, column] = value
            rows, columns = linear_sum_assignment(matrix)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                if matrix[row, column] >= _INVALID_COST:
                    continue
                cluster = matrix_clusters[row]
                track = active_tracks[column]
                self._update_track(track, cluster, now_ns)
                updated_ids.add(track.global_track_id)
                matched_clusters.add(cluster.cluster_id)
                assignments.extend(
                    self._cluster_assignments(cluster, track, "gated_hungarian_match")
                )

        for cluster in assignable:
            if cluster.cluster_id in ambiguous_cluster_ids:
                assignments.extend(
                    self._ambiguous_assignments(cluster, "assignment_candidate_tie")
                )
            elif cluster.cluster_id not in matched_clusters:
                if accepted_by_cluster.get(cluster.cluster_id):
                    assignments.extend(
                        self._ambiguous_assignments(cluster, "one_to_one_assignment_conflict")
                    )
                    for candidate_index, candidate in enumerate(candidates):
                        if (
                            candidate.left_id == cluster.cluster_id
                            and candidate.outcome is CandidateOutcome.ACCEPTED
                        ):
                            candidates[candidate_index] = CandidateEvidence(
                                candidate.candidate_id,
                                candidate.kind,
                                candidate.left_id,
                                candidate.right_id,
                                candidate.profile_id,
                                CandidateOutcome.AMBIGUOUS,
                                "one_to_one_assignment_conflict",
                                candidate.costs,
                            )
                else:
                    track = self._new_track(cluster, now_ns)
                    updated_ids.add(track.global_track_id)
                    assignments.extend(
                        self._cluster_assignments(cluster, track, "new_scene_global_track")
                    )
        return assignments, candidates, updated_ids

    def _assign_continuity(
        self,
        tick_index: int,
        now_ns: int,
        clusters: list[_Cluster],
    ) -> tuple[list[MemberAssignment], list[CandidateEvidence], set[str]]:
        """Apply the live identity ladder: retain, recover, interact, then create."""

        assignments: list[MemberAssignment] = []
        candidates: list[CandidateEvidence] = []
        updated_ids: set[str] = set()
        assignable: list[_Cluster] = []
        for cluster in clusters:
            if cluster.ambiguous:
                assignments.extend(self._ambiguous_assignments(cluster, "overlap_candidate_tie"))
            else:
                assignable.append(cluster)

        active_tracks = [
            self._tracks[key]
            for key in sorted(self._tracks)
            if self._tracks[key].state is not GlobalTrackState.ENDED
        ]
        evaluated: dict[tuple[str, str], tuple[CostComponents, bool, str]] = {}
        accepted_by_cluster: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for cluster in assignable:
            for track in active_tracks:
                costs, accepted, reason = self._assignment_cost(
                    cluster, track, now_ns, apply_binding_bonus=False
                )
                evaluated[(cluster.cluster_id, track.global_track_id)] = (
                    costs,
                    accepted,
                    reason,
                )
                candidates.append(
                    CandidateEvidence(
                        _candidate_id(
                            "assign", tick_index, cluster.cluster_id, track.global_track_id
                        ),
                        CandidateKind.GLOBAL_ASSIGNMENT,
                        cluster.cluster_id,
                        track.global_track_id,
                        self.profile_id,
                        CandidateOutcome.ACCEPTED if accepted else CandidateOutcome.REJECTED,
                        reason,
                        costs,
                    )
                )
                if accepted and costs.normalized_total is not None:
                    accepted_by_cluster[cluster.cluster_id].append(
                        (costs.normalized_total, track.global_track_id)
                    )

        released_binding_reason: dict[str, str] = {}
        excluded_tracks_by_cluster: dict[str, set[str]] = defaultdict(set)
        for cluster in assignable:
            release_reason, released_global_ids = self._observe_binding_evidence(now_ns, cluster)
            if release_reason is not None:
                released_binding_reason[cluster.cluster_id] = release_reason
                excluded_tracks_by_cluster[cluster.cluster_id].update(released_global_ids)

        swapped_local_ids, challenged_local_ids = self._evaluate_interaction_swaps(
            tick_index, now_ns, assignable
        )
        bound_by_track: dict[str, list[_Cluster]] = defaultdict(list)
        unbound: list[_Cluster] = []
        blocked_cluster_ids: set[str] = set()
        for cluster in assignable:
            bound_ids = self._bound_global_ids(cluster)
            if len(bound_ids) > 1:
                assignments.extend(
                    self._ambiguous_assignments(cluster, "conflicting_persistent_local_bindings")
                )
                blocked_cluster_ids.add(cluster.cluster_id)
            elif bound_ids:
                bound_by_track[next(iter(bound_ids))].append(cluster)
            else:
                unbound.append(cluster)

        reserved_tracks: set[str] = set()
        for global_track_id in sorted(bound_by_track):
            bound_track = self._tracks.get(global_track_id)
            binding_candidates = sorted(
                bound_by_track[global_track_id],
                key=lambda cluster: self._binding_claim_rank(cluster, global_track_id, evaluated),
            )
            if bound_track is None or bound_track.state is GlobalTrackState.ENDED:
                unbound.extend(binding_candidates)
                continue
            winner = binding_candidates[0]
            for conflict in binding_candidates[1:]:
                if (
                    conflict.primary.confidence >= self.config.new_global_minimum_confidence
                    and self._same_camera_separated(winner, conflict, now_ns)
                ):
                    self._release_cluster_binding(conflict, global_track_id)
                    released_binding_reason[conflict.cluster_id] = (
                        "released_same_camera_distinct_binding"
                    )
                    excluded_tracks_by_cluster[conflict.cluster_id].add(global_track_id)
                    unbound.append(conflict)
                    continue
                assignments.extend(
                    self._ambiguous_assignments(conflict, "persistent_binding_one_to_one_conflict")
                )
                blocked_cluster_ids.add(conflict.cluster_id)
            costs, accepted, gate_reason = evaluated[(winner.cluster_id, global_track_id)]
            if not self._binding_observation_trusted(winner, global_track_id, now_ns, evaluated):
                assignments.extend(
                    self._ambiguous_assignments(
                        winner, "persistent_binding_unverified_observation"
                    )
                )
                blocked_cluster_ids.add(winner.cluster_id)
                continue
            hard_conflict = gate_reason in {
                "impossible_speed_gate",
                "transition_topology_gate",
                "reacquisition_time_gate",
            }
            if hard_conflict:
                assignments.extend(
                    self._ambiguous_assignments(winner, "persistent_binding_motion_conflict")
                )
                blocked_cluster_ids.add(winner.cluster_id)
                continue
            interacting = self._cluster_interaction_active(winner, now_ns)
            challenged = bool(set(winner.local_track_ids) & challenged_local_ids)
            switched = bool(set(winner.local_track_ids) & swapped_local_ids)
            if challenged:
                reason = "persistent_binding_interaction_challenge"
            elif switched:
                reason = "interaction_reciprocal_switch_confirmed"
            elif interacting:
                reason = "persistent_binding_interaction_hold"
            else:
                reason = "persistent_local_binding"
            if not challenged:
                self._update_track(
                    bound_track,
                    winner,
                    now_ns,
                    allow_gallery=accepted and not interacting,
                )
                updated_ids.add(global_track_id)
            self._bind_cluster(
                winner,
                global_track_id,
                now_ns,
                refresh_reliable_evidence=(
                    winner.primary.confidence >= self.config.new_global_minimum_confidence
                    and (accepted or gate_reason == "appearance_unavailable")
                ),
            )
            reserved_tracks.add(global_track_id)
            assignments.extend(self._cluster_assignments(winner, bound_track, reason))

        unbound = [cluster for cluster in unbound if cluster.cluster_id not in blocked_cluster_ids]
        assigned_cluster_by_track: dict[str, _Cluster] = {
            global_track_id: sorted(
                clusters_for_track,
                key=lambda cluster: self._binding_claim_rank(cluster, global_track_id, evaluated),
            )[0]
            for global_track_id, clusters_for_track in bound_by_track.items()
            if global_track_id in reserved_tracks
        }
        available_tracks = [
            track for track in active_tracks if track.global_track_id not in reserved_tracks
        ]
        ambiguous_cluster_ids: set[str] = set()
        for cluster in unbound:
            assignment_options = sorted(
                option
                for option in accepted_by_cluster.get(cluster.cluster_id, [])
                if option[1] not in reserved_tracks
            )
            if (
                len(assignment_options) >= 2
                and assignment_options[1][0] - assignment_options[0][0]
                <= self.config.assignment_ambiguity_margin
            ):
                ambiguous_cluster_ids.add(cluster.cluster_id)
                self._mark_candidates_ambiguous(
                    candidates, cluster.cluster_id, "assignment_candidate_tie"
                )

        matrix_clusters = [
            cluster for cluster in unbound if cluster.cluster_id not in ambiguous_cluster_ids
        ]
        matched_clusters: set[str] = set()
        if matrix_clusters and available_tracks:
            matrix = np.full(
                (len(matrix_clusters), len(available_tracks)), _INVALID_COST, dtype=np.float64
            )
            for row, cluster in enumerate(matrix_clusters):
                for column, track in enumerate(available_tracks):
                    item = evaluated.get((cluster.cluster_id, track.global_track_id))
                    if (
                        cluster.primary.confidence >= self.config.new_global_minimum_confidence
                        and track.global_track_id
                        not in excluded_tracks_by_cluster[cluster.cluster_id]
                        and item is not None
                        and item[1]
                        and item[0].normalized_total is not None
                    ):
                        matrix[row, column] = item[0].normalized_total
            rows, columns = linear_sum_assignment(matrix)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                if matrix[row, column] >= _INVALID_COST:
                    continue
                cluster = matrix_clusters[row]
                track = available_tracks[column]
                interacting = self._cluster_interaction_active(cluster, now_ns)
                self._update_track(track, cluster, now_ns, allow_gallery=not interacting)
                self._bind_cluster(
                    cluster,
                    track.global_track_id,
                    now_ns,
                    refresh_reliable_evidence=True,
                )
                self._clear_new_confirmation(cluster)
                updated_ids.add(track.global_track_id)
                reserved_tracks.add(track.global_track_id)
                assigned_cluster_by_track[track.global_track_id] = cluster
                matched_clusters.add(cluster.cluster_id)
                assignments.extend(
                    self._cluster_assignments(cluster, track, "gated_reachable_hungarian_match")
                )

        for cluster in unbound:
            if cluster.cluster_id in ambiguous_cluster_ids:
                assignments.extend(
                    self._ambiguous_assignments(cluster, "assignment_candidate_tie")
                )
                continue
            if cluster.cluster_id in matched_clusters:
                continue
            if cluster.primary.confidence < self.config.new_global_minimum_confidence:
                self._clear_new_confirmation(cluster)
                assignments.extend(
                    self._ambiguous_assignments(
                        cluster,
                        released_binding_reason.get(
                            cluster.cluster_id, "new_global_birth_below_confidence"
                        ),
                    )
                )
                continue
            simultaneous_distinct_tracks = {
                global_track_id
                for global_track_id, assigned_cluster in assigned_cluster_by_track.items()
                if cluster.primary.confidence >= self.config.new_global_minimum_confidence
                and self._same_camera_separated(cluster, assigned_cluster, now_ns)
            }
            excluded_tracks = (
                excluded_tracks_by_cluster[cluster.cluster_id] | simultaneous_distinct_tracks
            )
            for global_track_id in simultaneous_distinct_tracks:
                self._set_candidate_outcome(
                    candidates,
                    cluster.cluster_id,
                    global_track_id,
                    CandidateOutcome.REJECTED,
                    "simultaneous_same_camera_distinct",
                )
            # A valid target that the one-to-one solve already reserved is a conflict,
            # not evidence for a new person. Consult every accepted edge here rather
            # than only the still-available subset.
            conflicting_options = [
                option
                for option in accepted_by_cluster.get(cluster.cluster_id, [])
                if option[1] not in excluded_tracks
            ]
            if conflicting_options:
                assignments.extend(
                    self._ambiguous_assignments(cluster, "one_to_one_assignment_conflict")
                )
                self._mark_candidates_ambiguous(
                    candidates, cluster.cluster_id, "one_to_one_assignment_conflict"
                )
                continue
            pending_options = self._appearance_pending_options(cluster, active_tracks, evaluated)
            pending_options = [
                option
                for option in pending_options
                if option[0].global_track_id not in excluded_tracks
            ]
            if len(pending_options) > 1:
                assignments.extend(
                    self._ambiguous_assignments(
                        cluster, "multiple_reachable_predecessors_appearance_unresolved"
                    )
                )
                for track, _, _ in pending_options:
                    self._set_candidate_outcome(
                        candidates,
                        cluster.cluster_id,
                        track.global_track_id,
                        CandidateOutcome.AMBIGUOUS,
                        "multiple_reachable_predecessors_appearance_unresolved",
                    )
                self._clear_new_confirmation(cluster)
                continue
            if pending_options:
                track, _, gate_reason = pending_options[0]
                if track.global_track_id in reserved_tracks:
                    assignments.extend(
                        self._ambiguous_assignments(
                            cluster, "reachable_predecessor_one_to_one_conflict"
                        )
                    )
                    self._set_candidate_outcome(
                        candidates,
                        cluster.cluster_id,
                        track.global_track_id,
                        CandidateOutcome.AMBIGUOUS,
                        "reachable_predecessor_one_to_one_conflict",
                    )
                    self._clear_new_confirmation(cluster)
                    continue
                if gate_reason == "appearance_gate":
                    recovery_reason = "unique_physical_recovery_overrides_appearance_mismatch"
                    self._update_track(track, cluster, now_ns, allow_gallery=False)
                    self._bind_cluster(
                        cluster,
                        track.global_track_id,
                        now_ns,
                        refresh_reliable_evidence=True,
                    )
                    self._clear_new_confirmation(cluster)
                    updated_ids.add(track.global_track_id)
                    reserved_tracks.add(track.global_track_id)
                    matched_clusters.add(cluster.cluster_id)
                    assignments.extend(self._cluster_assignments(cluster, track, recovery_reason))
                    self._set_candidate_outcome(
                        candidates,
                        cluster.cluster_id,
                        track.global_track_id,
                        CandidateOutcome.ACCEPTED,
                        recovery_reason,
                    )
                    continue
                pending_reason = "unique_reachable_predecessor_appearance_pending"
                assignments.extend(self._ambiguous_assignments(cluster, pending_reason))
                self._clear_new_confirmation(cluster)
                self._set_candidate_outcome(
                    candidates,
                    cluster.cluster_id,
                    track.global_track_id,
                    CandidateOutcome.AMBIGUOUS,
                    pending_reason,
                )
                continue
            birth_block_reason = self._new_track_birth_block_reason(cluster)
            if birth_block_reason is not None:
                self._clear_new_confirmation(cluster)
                assignments.extend(
                    self._ambiguous_assignments(
                        cluster,
                        released_binding_reason.get(cluster.cluster_id, birth_block_reason),
                    )
                )
                continue
            if not self._new_track_confirmed(cluster, tick_index):
                assignments.extend(
                    self._ambiguous_assignments(cluster, "provisional_new_global_track")
                )
                continue
            track = self._new_track(cluster, now_ns)
            self._bind_cluster(
                cluster,
                track.global_track_id,
                now_ns,
                refresh_reliable_evidence=True,
            )
            updated_ids.add(track.global_track_id)
            assignments.extend(
                self._cluster_assignments(cluster, track, "confirmed_new_scene_global_track")
            )

        return assignments, candidates, updated_ids

    def _appearance_pending_options(
        self,
        cluster: _Cluster,
        active_tracks: list[_Track],
        evaluated: dict[tuple[str, str], tuple[CostComponents, bool, str]],
    ) -> list[tuple[_Track, CostComponents, str]]:
        """Return short-gap predecessors that remain physically plausible.

        The normal combined solve recovers an identity when appearance and reachability
        agree. Missing or conflicting appearance does not force a match and does not
        authorize a replacement identity during this short hold.
        """

        options: list[tuple[_Track, CostComponents, str]] = []
        for track in active_tracks:
            item = evaluated.get((cluster.cluster_id, track.global_track_id))
            if item is None:
                continue
            costs, accepted, reason = item
            if accepted or reason not in {"appearance_unavailable", "appearance_gate"}:
                continue
            if costs.time_gap_seconds > self.config.appearance_pending_hold_seconds:
                continue
            options.append((track, costs, reason))
        return sorted(options, key=lambda item: item[0].global_track_id)

    def _new_track_birth_block_reason(self, cluster: _Cluster) -> str | None:
        """Require positive evidence before creating a new anonymous identity."""

        if cluster.embedding is None:
            return "new_global_birth_requires_appearance"
        if cluster.primary.confidence < self.config.new_global_minimum_confidence:
            return "new_global_birth_below_confidence"
        return None

    @staticmethod
    def _candidate_rank(
        value: tuple[CostComponents, bool, str] | None,
    ) -> float:
        if value is None or value[0].normalized_total is None:
            return _INVALID_COST
        return value[0].normalized_total

    @staticmethod
    def _mark_candidates_ambiguous(
        candidates: list[CandidateEvidence], cluster_id: str, reason: str
    ) -> None:
        for index, candidate in enumerate(candidates):
            if (
                candidate.left_id == cluster_id
                and candidate.kind is CandidateKind.GLOBAL_ASSIGNMENT
                and candidate.outcome is CandidateOutcome.ACCEPTED
            ):
                candidates[index] = CandidateEvidence(
                    candidate.candidate_id,
                    candidate.kind,
                    candidate.left_id,
                    candidate.right_id,
                    candidate.profile_id,
                    CandidateOutcome.AMBIGUOUS,
                    reason,
                    candidate.costs,
                )

    @staticmethod
    def _set_candidate_outcome(
        candidates: list[CandidateEvidence],
        cluster_id: str,
        track_id: str,
        outcome: CandidateOutcome,
        reason: str,
    ) -> None:
        for index, candidate in enumerate(candidates):
            if (
                candidate.left_id == cluster_id
                and candidate.right_id == track_id
                and candidate.kind is CandidateKind.GLOBAL_ASSIGNMENT
            ):
                candidates[index] = CandidateEvidence(
                    candidate.candidate_id,
                    candidate.kind,
                    candidate.left_id,
                    candidate.right_id,
                    candidate.profile_id,
                    outcome,
                    reason,
                    candidate.costs,
                )
                return

    def _update_interaction_windows(
        self,
        now_ns: int,
        observations: tuple[AssociationObservation, ...],
    ) -> None:
        interaction_ids = {
            item.local_track_stable_id
            for item in observations
            if item.interaction_evidence and item.bbox_xyxy is None
        }
        for left, right in combinations(observations, 2):
            if left.camera_id != right.camera_id:
                continue
            boxes_intersect = (
                left.bbox_xyxy is not None
                and right.bbox_xyxy is not None
                and _bbox_intersection_ratio(left.bbox_xyxy, right.bbox_xyxy)
                >= self.config.interaction_min_bbox_intersection_ratio
            )
            spatially_converged = (
                (left.interaction_evidence or right.interaction_evidence)
                and (left.bbox_xyxy is None or right.bbox_xyxy is None)
                and _distance(left.world_xy_metres, right.world_xy_metres)
                <= self.config.interaction_world_distance_metres
            )
            if boxes_intersect or spatially_converged:
                interaction_ids.update((left.local_track_stable_id, right.local_track_stable_id))
        hold_ns = int(round(self.config.interaction_hold_seconds * 1_000_000_000))
        for local_id in interaction_ids:
            self._interaction_until_by_local[local_id] = max(
                self._interaction_until_by_local.get(local_id, -1), now_ns + hold_ns
            )
        self._interaction_until_by_local = {
            key: value
            for key, value in self._interaction_until_by_local.items()
            if value >= now_ns
        }

    def _prune_continuity_state(self, tick_index: int) -> None:
        active_global_ids = {
            key for key, track in self._tracks.items() if track.state is not GlobalTrackState.ENDED
        }
        self._local_bindings = {
            key: value
            for key, value in self._local_bindings.items()
            if value.global_track_id in active_global_ids
        }
        self._new_track_confirmations = {
            key: value
            for key, value in self._new_track_confirmations.items()
            if value.last_tick >= tick_index - 1
        }

    def _bound_global_ids(self, cluster: _Cluster) -> set[str]:
        result: set[str] = set()
        for local_id in cluster.local_track_ids:
            binding = self._local_bindings.get(local_id)
            if binding is None:
                continue
            track = self._tracks.get(binding.global_track_id)
            if track is None or track.state is GlobalTrackState.ENDED:
                self._local_bindings.pop(local_id, None)
                continue
            result.add(binding.global_track_id)
        return result

    def _cluster_interaction_active(self, cluster: _Cluster, now_ns: int) -> bool:
        return any(
            self._interaction_until_by_local.get(local_id, -1) >= now_ns
            for local_id in cluster.local_track_ids
        )

    def _observe_binding_evidence(
        self,
        now_ns: int,
        cluster: _Cluster,
    ) -> tuple[str | None, set[str]]:
        """Release only a repeatedly contradicted local binding.

        Camera absence never challenges a binding because this method sees only current
        observations. A real box interaction resets the challenge. This keeps the anonymous
        global identity stubborn through blackout and occlusion while preventing a weak box
        that has latched onto furniture from refreshing the person indefinitely.
        """

        # A credible observation on the same persistent local track is identity
        # continuity evidence even when its approximate floor projection or ReID crop
        # is temporarily poor. Geometry may reject the current XY update below, but it
        # must not erode the binding itself. Explicit simultaneous same-camera people
        # are resolved separately by _same_camera_separated.
        if self._cluster_interaction_active(cluster, now_ns):
            for local_id in cluster.local_track_ids:
                binding = self._local_bindings.get(local_id)
                if binding is not None:
                    binding.challenge_count = 0
                    binding.last_challenge_observed_ns = None
            return None, set()

        released_reason: str | None = None
        released_global_ids: set[str] = set()
        maximum_gap_ns = int(round(self.config.binding_challenge_max_gap_seconds * 1_000_000_000))
        for local_id in cluster.local_track_ids:
            binding = self._local_bindings.get(local_id)
            if binding is None:
                continue
            if cluster.primary.confidence >= self.config.new_global_minimum_confidence:
                binding.last_reliable_observed_ns = cluster.observed_monotonic_ns
                binding.challenge_count = 0
                binding.last_challenge_observed_ns = None
                continue

            previous_ns = binding.last_challenge_observed_ns
            binding.challenge_count = (
                binding.challenge_count + 1
                if previous_ns is not None
                and 0 <= cluster.observed_monotonic_ns - previous_ns <= maximum_gap_ns
                else 1
            )
            binding.last_challenge_observed_ns = cluster.observed_monotonic_ns
            if binding.challenge_count < self.config.binding_release_confirmation_hits:
                continue
            released_global_ids.add(binding.global_track_id)
            self._local_bindings.pop(local_id, None)
            released_reason = "released_low_confidence_persistent_binding"
        return released_reason, released_global_ids

    def _binding_claim_rank(
        self,
        cluster: _Cluster,
        global_track_id: str,
        evaluated: dict[tuple[str, str], tuple[CostComponents, bool, str]],
    ) -> tuple[int, int, int, float, float, str]:
        item = evaluated.get((cluster.cluster_id, global_track_id))
        accepted = item is not None and item[1]
        current_reliable = (
            cluster.primary.confidence >= self.config.new_global_minimum_confidence
            and item is not None
            and (item[1] or item[2] == "appearance_unavailable")
        )
        last_reliable_ns = max(
            (
                binding.last_reliable_observed_ns
                for local_id in cluster.local_track_ids
                if (binding := self._local_bindings.get(local_id)) is not None
                and binding.global_track_id == global_track_id
            ),
            default=-1,
        )
        return (
            0 if current_reliable else 1,
            0 if accepted else 1,
            -last_reliable_ns,
            self._candidate_rank(item),
            -cluster.primary.confidence,
            cluster.cluster_id,
        )

    def _binding_observation_trusted(
        self,
        cluster: _Cluster,
        global_track_id: str,
        now_ns: int,
        evaluated: dict[tuple[str, str], tuple[CostComponents, bool, str]],
    ) -> bool:
        del evaluated
        if cluster.primary.confidence >= self.config.new_global_minimum_confidence:
            return True
        grace_ns = int(round(self.config.binding_observation_grace_seconds * 1_000_000_000))
        return any(
            0 <= now_ns - binding.last_reliable_observed_ns <= grace_ns
            for local_id in cluster.local_track_ids
            if (binding := self._local_bindings.get(local_id)) is not None
            and binding.global_track_id == global_track_id
        )

    def _same_camera_separated(
        self,
        left: _Cluster,
        right: _Cluster,
        now_ns: int,
    ) -> bool:
        if self._cluster_interaction_active(left, now_ns) or self._cluster_interaction_active(
            right, now_ns
        ):
            return False
        for left_member in left.members:
            for right_member in right.members:
                if left_member.camera_id != right_member.camera_id:
                    continue
                if (
                    abs(left_member.observed_monotonic_ns - right_member.observed_monotonic_ns)
                    / 1_000_000_000.0
                    > self.config.overlap_max_time_seconds
                ):
                    continue
                if left_member.bbox_xyxy is None or right_member.bbox_xyxy is None:
                    continue
                if (
                    _bbox_intersection_ratio(left_member.bbox_xyxy, right_member.bbox_xyxy)
                    < self.config.interaction_min_bbox_intersection_ratio
                ):
                    return True
        return False

    def _release_cluster_binding(self, cluster: _Cluster, global_track_id: str) -> None:
        for local_id in cluster.local_track_ids:
            binding = self._local_bindings.get(local_id)
            if binding is not None and binding.global_track_id == global_track_id:
                self._local_bindings.pop(local_id, None)

    def _bind_cluster(
        self,
        cluster: _Cluster,
        global_track_id: str,
        now_ns: int,
        *,
        refresh_reliable_evidence: bool,
    ) -> None:
        for local_id in cluster.local_track_ids:
            binding = self._local_bindings.get(local_id)
            interaction_until = self._interaction_until_by_local.get(local_id, -1)
            if binding is not None and binding.global_track_id != global_track_id:
                if interaction_until < now_ns:
                    raise XR02AssociationContractError(
                        "persistent local identity changed outside an interaction window"
                    )
            if binding is None or binding.global_track_id != global_track_id:
                self._local_bindings[local_id] = _LocalBinding(
                    global_track_id=global_track_id,
                    last_reliable_observed_ns=(
                        cluster.observed_monotonic_ns if refresh_reliable_evidence else now_ns
                    ),
                )
            elif refresh_reliable_evidence:
                binding.last_reliable_observed_ns = cluster.observed_monotonic_ns
                binding.challenge_count = 0
                binding.last_challenge_observed_ns = None

    def _evaluate_interaction_swaps(
        self,
        tick_index: int,
        now_ns: int,
        clusters: list[_Cluster],
    ) -> tuple[set[str], set[str]]:
        """Confirm only reciprocal, sustained swaps after a real interaction window."""

        eligible_by_camera: dict[
            str, list[tuple[_Cluster, AssociationObservation, _LocalBinding]]
        ] = defaultdict(list)
        for cluster in clusters:
            if len(cluster.members) != 1:
                continue
            observation = cluster.members[0]
            binding = self._local_bindings.get(observation.local_track_stable_id)
            if (
                binding is None
                or self._interaction_until_by_local.get(observation.local_track_stable_id, -1)
                < now_ns
            ):
                continue
            eligible_by_camera[observation.camera_id].append((cluster, observation, binding))

        qualified_keys: set[tuple[str, str]] = set()
        challenged_ids: set[str] = set()
        swapped_ids: set[str] = set()
        used_ids: set[str] = set()
        proposals: list[
            tuple[
                float,
                str,
                str,
                _LocalBinding,
                _LocalBinding,
            ]
        ] = []
        for camera_id in sorted(eligible_by_camera):
            rows = sorted(
                eligible_by_camera[camera_id],
                key=lambda item: item[1].local_track_stable_id,
            )
            for left, right in combinations(rows, 2):
                left_cluster, left_observation, left_binding = left
                right_cluster, right_observation, right_binding = right
                if left_binding.global_track_id == right_binding.global_track_id:
                    continue
                left_track = self._tracks.get(left_binding.global_track_id)
                right_track = self._tracks.get(right_binding.global_track_id)
                if (
                    left_track is None
                    or right_track is None
                    or left_track.state is GlobalTrackState.ENDED
                    or right_track.state is GlobalTrackState.ENDED
                ):
                    continue
                left_direct = self._assignment_cost(
                    left_cluster, left_track, now_ns, apply_binding_bonus=False
                )
                right_direct = self._assignment_cost(
                    right_cluster, right_track, now_ns, apply_binding_bonus=False
                )
                left_swapped = self._assignment_cost(
                    left_cluster, right_track, now_ns, apply_binding_bonus=False
                )
                right_swapped = self._assignment_cost(
                    right_cluster, left_track, now_ns, apply_binding_bonus=False
                )
                if not left_swapped[1] or not right_swapped[1]:
                    continue
                direct_total = sum(
                    item[0].normalized_total
                    if item[1] and item[0].normalized_total is not None
                    else _INVALID_COST
                    for item in (left_direct, right_direct)
                )
                swapped_total = sum(
                    item[0].normalized_total or 0.0 for item in (left_swapped, right_swapped)
                )
                improvement = direct_total - swapped_total
                if improvement < 2.0 * self.config.binding_switch_margin:
                    continue
                proposals.append(
                    (
                        -improvement,
                        left_observation.local_track_stable_id,
                        right_observation.local_track_stable_id,
                        left_binding,
                        right_binding,
                    )
                )

        for _, left_id, right_id, left_binding, right_binding in sorted(proposals):
            if left_id in used_ids or right_id in used_ids:
                continue
            key = (left_id, right_id) if left_id < right_id else (right_id, left_id)
            left_global = (
                left_binding.global_track_id
                if key[0] == left_id
                else right_binding.global_track_id
            )
            right_global = (
                right_binding.global_track_id
                if key[1] == right_id
                else left_binding.global_track_id
            )
            previous = self._swap_challenges.get(key)
            count = (
                previous.count + 1
                if previous is not None
                and previous.last_tick == tick_index - 1
                and previous.left_global_id == left_global
                and previous.right_global_id == right_global
                else 1
            )
            self._swap_challenges[key] = _SwapChallenge(
                left_global, right_global, count, tick_index
            )
            qualified_keys.add(key)
            challenged_ids.update(key)
            used_ids.update(key)
            if count < self.config.binding_switch_confirmation_hits:
                continue
            first = self._local_bindings[key[0]]
            second = self._local_bindings[key[1]]
            first.global_track_id, second.global_track_id = (
                second.global_track_id,
                first.global_track_id,
            )
            for binding in (first, second):
                binding.last_reliable_observed_ns = now_ns
                binding.challenge_count = 0
                binding.last_challenge_observed_ns = None
            swapped_ids.update(key)
            challenged_ids.difference_update(key)
            self._swap_challenges.pop(key, None)

        self._swap_challenges = {
            key: value for key, value in self._swap_challenges.items() if key in qualified_keys
        }
        return swapped_ids, challenged_ids

    def _new_track_confirmed(self, cluster: _Cluster, tick_index: int) -> bool:
        key = canonical_sha256(cluster.local_track_ids)
        previous = self._new_track_confirmations.get(key)
        count = (
            previous.count + 1
            if previous is not None and previous.last_tick == tick_index - 1
            else 1
        )
        self._new_track_confirmations[key] = _NewTrackConfirmation(count, tick_index)
        return count >= self.config.new_global_confirmation_hits

    def _clear_new_confirmation(self, cluster: _Cluster) -> None:
        self._new_track_confirmations.pop(canonical_sha256(cluster.local_track_ids), None)

    def _assignment_cost(
        self,
        cluster: _Cluster,
        track: _Track,
        now_ns: int,
        *,
        apply_binding_bonus: bool = True,
    ) -> tuple[CostComponents, bool, str]:
        if self.config.continuity_policy_enabled:
            return self._reachable_assignment_cost(cluster, track, now_ns)
        time_gap = (now_ns - track.last_observed_monotonic_ns) / 1_000_000_000.0
        spatial = _distance(cluster.world_xy, track.last_world_xy_metres)
        appearance = _optional_cosine(cluster.embedding, track.embedding)
        quality_penalty = 1.0 - cluster.quality_weight
        topology_allowed = any(
            self.topology.can_transition(previous, current)
            for previous in track.camera_ids
            for current in cluster.cameras
        )
        total = self._normalized_cost(
            spatial, appearance, time_gap, quality_penalty, overlap=False
        )
        if apply_binding_bonus and any(
            member.local_track_stable_id in track.last_local_track_ids
            for member in cluster.members
        ):
            total *= 0.25
        costs = CostComponents(
            spatial, appearance, time_gap, quality_penalty, topology_allowed, total
        )
        if not topology_allowed:
            return costs, False, "transition_topology_gate"
        if time_gap > self.config.reacquisition_max_seconds:
            return costs, False, "reacquisition_time_gate"
        if cluster.quality_weight < self.config.minimum_quality_weight:
            return costs, False, "quality_gate"

        multiplier = 1.0
        if track.state is GlobalTrackState.LOST:
            multiplier *= self.config.lost_gate_multiplier
        if "office-cam-04" in cluster.cameras or "office-cam-04" in track.camera_ids:
            multiplier *= self.config.isolated_camera_gate_multiplier
        return self._profile_gate(
            spatial,
            appearance,
            costs,
            overlap=False,
            gate_multiplier=multiplier,
        )

    def _reachable_assignment_cost(
        self, cluster: _Cluster, track: _Track, now_ns: int
    ) -> tuple[CostComponents, bool, str]:
        observation_ns = cluster.observed_monotonic_ns
        time_gap = max(
            0.0,
            (observation_ns - track.last_observed_monotonic_ns) / 1_000_000_000.0,
        )
        spatial = _distance(cluster.world_xy, track.last_world_xy_metres)
        uncertainty = (
            track.position_uncertainty_metres or self.config.measurement_uncertainty_metres
        )
        same_camera_continuation = any(
            previous == current for previous in track.camera_ids for current in cluster.cameras
        )
        uncertainty_budget = (
            self.config.measurement_uncertainty_metres
            if same_camera_continuation
            else uncertainty + self.config.measurement_uncertainty_metres
        )
        effective_distance = max(
            0.0,
            spatial - uncertainty_budget,
        )
        effective_speed = effective_distance / max(time_gap, 0.05)
        reachable_radius = uncertainty_budget + self.config.maximum_effective_speed_mps * time_gap
        appearance = _gallery_cosine_distance(cluster.embedding, track.gallery)
        quality_penalty = 1.0 - cluster.quality_weight
        topology_allowed = any(
            self.topology.can_transition(previous, current)
            for previous in track.camera_ids
            for current in cluster.cameras
        )
        total = self._normalized_cost(
            spatial,
            appearance,
            time_gap,
            quality_penalty,
            overlap=False,
            spatial_limit_override=reachable_radius,
        )
        costs = CostComponents(
            spatial,
            appearance,
            time_gap,
            quality_penalty,
            topology_allowed,
            total,
            effective_speed_metres_per_second=effective_speed,
            reachable_radius_metres=reachable_radius,
        )
        if not topology_allowed:
            return costs, False, "transition_topology_gate"
        if time_gap > self.config.reacquisition_max_seconds:
            return costs, False, "reacquisition_time_gate"
        if cluster.quality_weight < self.config.minimum_quality_weight:
            return costs, False, "quality_gate"
        if effective_speed > self.config.maximum_effective_speed_mps:
            return costs, False, "impossible_speed_gate"
        return self._profile_gate(
            spatial,
            appearance,
            costs,
            overlap=False,
            spatial_limit_override=reachable_radius,
        )

    def _profile_gate(
        self,
        spatial: float,
        appearance: float | None,
        costs: CostComponents,
        *,
        overlap: bool,
        gate_multiplier: float = 1.0,
        spatial_limit_override: float | None = None,
    ) -> tuple[CostComponents, bool, str]:
        spatial_limit = (
            spatial_limit_override
            if spatial_limit_override is not None
            else (
                self.config.overlap_max_spatial_metres
                if overlap
                else self.config.assignment_max_spatial_metres
            )
        ) * gate_multiplier
        appearance_limit = (
            self.config.overlap_max_appearance_distance
            if overlap
            else self.config.assignment_max_appearance_distance
        ) * gate_multiplier
        if self.profile is SignalProfile.SPATIAL_ONLY:
            return (
                costs,
                spatial <= spatial_limit,
                "within_spatial_gate" if spatial <= spatial_limit else "spatial_gate",
            )
        if self.profile is SignalProfile.APPEARANCE_ONLY:
            if appearance is None:
                return costs, False, "appearance_unavailable"
            return (
                costs,
                appearance <= appearance_limit,
                "within_appearance_gate" if appearance <= appearance_limit else "appearance_gate",
            )
        if spatial > spatial_limit:
            return costs, False, "spatial_gate"
        if appearance is None:
            return costs, False, "appearance_unavailable"
        if appearance > appearance_limit:
            return costs, False, "appearance_gate"
        return costs, True, "within_combined_gates"

    def _normalized_cost(
        self,
        spatial: float,
        appearance: float | None,
        time_gap: float,
        quality_penalty: float,
        *,
        overlap: bool,
        spatial_limit_override: float | None = None,
    ) -> float:
        spatial_limit = (
            spatial_limit_override
            if spatial_limit_override is not None
            else (
                self.config.overlap_max_spatial_metres
                if overlap
                else self.config.assignment_max_spatial_metres
            )
        )
        appearance_limit = (
            self.config.overlap_max_appearance_distance
            if overlap
            else self.config.assignment_max_appearance_distance
        )
        time_limit = (
            self.config.overlap_max_time_seconds
            if overlap
            else self.config.reacquisition_max_seconds
        )
        spatial_normalized = spatial / spatial_limit
        appearance_normalized = 2.0 if appearance is None else appearance / appearance_limit
        time_normalized = time_gap / time_limit
        if self.profile is SignalProfile.SPATIAL_ONLY:
            return spatial_normalized + 0.05 * time_normalized + 0.05 * quality_penalty
        if self.profile is SignalProfile.APPEARANCE_ONLY:
            return appearance_normalized + 0.05 * time_normalized + 0.05 * quality_penalty
        return (
            0.50 * spatial_normalized
            + 0.40 * appearance_normalized
            + 0.05 * time_normalized
            + 0.05 * quality_penalty
        )

    def _new_track(self, cluster: _Cluster, now_ns: int) -> _Track:
        global_id = f"g:{self._next_global_id:06d}"
        self._next_global_id += 1
        gallery = [] if cluster.embedding is None else [cluster.embedding]
        track = _Track(
            global_track_id=global_id,
            profile_id=self.profile_id,
            state=(
                GlobalTrackState.CONFIRMED
                if self.config.minimum_confirmation_hits <= 1
                else GlobalTrackState.TENTATIVE
            ),
            last_world_xy_metres=cluster.world_xy,
            last_observed_monotonic_ns=(
                cluster.observed_monotonic_ns if self.config.continuity_policy_enabled else now_ns
            ),
            camera_ids=cluster.cameras,
            hit_count=1,
            last_local_track_ids=tuple(
                sorted(item.local_track_stable_id for item in cluster.members)
            ),
            gallery=gallery,
            position_uncertainty_metres=(
                self.config.measurement_uncertainty_metres
                if self.config.continuity_policy_enabled
                else None
            ),
        )
        self._tracks[global_id] = track
        return track

    def _update_track(
        self,
        track: _Track,
        cluster: _Cluster,
        now_ns: int,
        *,
        allow_gallery: bool = True,
    ) -> None:
        if self.config.continuity_policy_enabled:
            observation_ns = max(cluster.observed_monotonic_ns, track.last_observed_monotonic_ns)
            elapsed = (observation_ns - track.last_observed_monotonic_ns) / 1_000_000_000.0
            if elapsed > 0:
                dx = cluster.world_xy[0] - track.last_world_xy_metres[0]
                dy = cluster.world_xy[1] - track.last_world_xy_metres[1]
                distance = math.hypot(dx, dy)
                effective_distance = max(
                    0.0, distance - self.config.measurement_uncertainty_metres
                )
                scale = 0.0 if distance <= 0 else effective_distance / distance
                instantaneous = (dx * scale / elapsed, dy * scale / elapsed)
                speed = math.hypot(*instantaneous)
                if speed > self.config.maximum_effective_speed_mps:
                    cap = self.config.maximum_effective_speed_mps / speed
                    instantaneous = (instantaneous[0] * cap, instantaneous[1] * cap)
                alpha = self.config.velocity_smoothing
                track.velocity_xy_metres_per_second = (
                    alpha * instantaneous[0]
                    + (1.0 - alpha) * track.velocity_xy_metres_per_second[0],
                    alpha * instantaneous[1]
                    + (1.0 - alpha) * track.velocity_xy_metres_per_second[1],
                )
            track.last_observed_monotonic_ns = observation_ns
            track.position_uncertainty_metres = self.config.measurement_uncertainty_metres
        else:
            track.last_observed_monotonic_ns = now_ns
        track.last_world_xy_metres = cluster.world_xy
        track.camera_ids = cluster.cameras
        track.hit_count += 1
        track.last_local_track_ids = tuple(
            sorted(item.local_track_stable_id for item in cluster.members)
        )
        if allow_gallery and cluster.embedding is not None:
            track.gallery.append(cluster.embedding)
            del track.gallery[: -self.config.gallery_size]
        track.state = (
            GlobalTrackState.CONFIRMED
            if track.hit_count >= self.config.minimum_confirmation_hits
            else GlobalTrackState.TENTATIVE
        )

    def _advance_lifecycle(self, now_ns: int) -> None:
        for track in self._tracks.values():
            if track.state is GlobalTrackState.ENDED:
                continue
            elapsed = (now_ns - track.last_observed_monotonic_ns) / 1_000_000_000.0
            if elapsed >= self.config.ended_after_seconds:
                track.state = GlobalTrackState.ENDED
            elif (
                self.config.continuity_policy_enabled
                and self.config.dormant_after_seconds is not None
                and elapsed > self.config.dormant_after_seconds
            ):
                track.state = GlobalTrackState.DORMANT
            elif elapsed > self.config.occluded_after_seconds:
                track.state = GlobalTrackState.LOST
            elif elapsed > 0:
                track.state = GlobalTrackState.OCCLUDED

    def _advance_unmatched(self, now_ns: int, updated_ids: set[str]) -> None:
        for track in self._tracks.values():
            if track.global_track_id in updated_ids or track.state is GlobalTrackState.ENDED:
                continue
            elapsed = (now_ns - track.last_observed_monotonic_ns) / 1_000_000_000.0
            if elapsed >= self.config.ended_after_seconds:
                track.state = GlobalTrackState.ENDED
            elif (
                self.config.continuity_policy_enabled
                and self.config.dormant_after_seconds is not None
                and elapsed > self.config.dormant_after_seconds
            ):
                track.state = GlobalTrackState.DORMANT
            elif elapsed > self.config.occluded_after_seconds:
                track.state = GlobalTrackState.LOST
            else:
                track.state = GlobalTrackState.OCCLUDED

    def _cluster_assignments(
        self, cluster: _Cluster, track: _Track, reason: str
    ) -> list[MemberAssignment]:
        primary_id = cluster.primary.observation_id
        return [
            MemberAssignment(
                observation_id=member.observation_id,
                local_track_stable_id=member.local_track_stable_id,
                camera_id=member.camera_id,
                global_track_id=track.global_track_id,
                state=(
                    track.state
                    if member.observation_id == primary_id
                    else GlobalTrackState.DUPLICATE
                ),
                reason=(reason if member.observation_id == primary_id else "overlap_duplicate"),
            )
            for member in cluster.members
        ]

    def _ambiguous_assignments(self, cluster: _Cluster, reason: str) -> list[MemberAssignment]:
        return [
            MemberAssignment(
                observation_id=member.observation_id,
                local_track_stable_id=member.local_track_stable_id,
                camera_id=member.camera_id,
                global_track_id=None,
                state=GlobalTrackState.AMBIGUOUS,
                reason=reason,
            )
            for member in cluster.members
        ]


def office_topology() -> SceneTopology:
    return SceneTopology(
        overlap_edges=(
            ("office-cam-01", "office-cam-02"),
            ("office-cam-01", "office-cam-03"),
            ("office-cam-02", "office-cam-03"),
        ),
        transition_edges=(
            ("office-cam-01", "office-cam-02"),
            ("office-cam-01", "office-cam-03"),
            ("office-cam-02", "office-cam-03"),
            ("office-cam-01", "office-cam-04"),
            ("office-cam-02", "office-cam-04"),
            ("office-cam-03", "office-cam-04"),
        ),
    )


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _optional_cosine(
    left: tuple[float, ...] | None, right: tuple[float, ...] | None
) -> float | None:
    if left is None or right is None:
        return None
    return cosine_distance(left, right)


def _gallery_cosine_distance(
    embedding: tuple[float, ...] | None,
    gallery: list[tuple[float, ...]],
) -> float | None:
    if embedding is None or not gallery:
        return None
    distances = sorted(cosine_distance(embedding, item) for item in gallery)
    selected = distances[: min(3, len(distances))]
    return sum(selected) / len(selected)


def _bbox_intersection_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / min(left_area, right_area)


def _candidate_id(prefix: str, tick: int, left: str, right: str) -> str:
    digest = canonical_sha256([prefix, tick, left, right])[:16]
    return f"{prefix}:{tick}:{digest}"
