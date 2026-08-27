"""Typed scene-global anonymous association contracts for XR02 WP3."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class XR02AssociationContractError(ValueError):
    """Raised when global-association evidence violates its public contract."""


class SignalProfile(StrEnum):
    SPATIAL_ONLY = "spatial_only"
    APPEARANCE_ONLY = "appearance_only"
    COMBINED = "combined"


class GlobalTrackState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    OCCLUDED = "occluded"
    LOST = "lost"
    DORMANT = "dormant"
    ENDED = "ended"
    AMBIGUOUS = "ambiguous"
    DUPLICATE = "duplicate"


class CandidateKind(StrEnum):
    OVERLAP_PAIR = "overlap_pair"
    CLUSTER_MERGE = "cluster_merge"
    GLOBAL_ASSIGNMENT = "global_assignment"


class CandidateOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class AssociationObservation:
    """One valid-floor WP2 local observation enriched with its referenced embedding."""

    scene_context_sha256: str
    observation_id: str
    local_track_stable_id: str
    camera_id: str
    tracker_profile: str
    observed_monotonic_ns: int
    world_xy_metres: tuple[float, float]
    confidence: float
    quality_weight: float
    embedding_reference_sha256: str | None
    embedding: tuple[float, ...] | None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    interaction_evidence: bool = False

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.scene_context_sha256):
            raise XR02AssociationContractError("scene context must be a lowercase SHA-256")
        for value, label in (
            (self.observation_id, "observation_id"),
            (self.local_track_stable_id, "local_track_stable_id"),
            (self.camera_id, "camera_id"),
            (self.tracker_profile, "tracker_profile"),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise XR02AssociationContractError(f"{label} is invalid")
        if self.observed_monotonic_ns < 0:
            raise XR02AssociationContractError("observation time must be non-negative")
        if not all(math.isfinite(value) for value in self.world_xy_metres):
            raise XR02AssociationContractError("world XY must be finite")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise XR02AssociationContractError("confidence must be within [0, 1]")
        if not math.isfinite(self.quality_weight) or not 0 < self.quality_weight <= 1:
            raise XR02AssociationContractError("quality weight must be within (0, 1]")
        if (self.embedding_reference_sha256 is None) != (self.embedding is None):
            raise XR02AssociationContractError("embedding reference and vector must coexist")
        if self.embedding_reference_sha256 is not None and not _SHA256.fullmatch(
            self.embedding_reference_sha256
        ):
            raise XR02AssociationContractError("embedding reference must be a lowercase SHA-256")
        if self.embedding is not None:
            if not self.embedding or not all(math.isfinite(value) for value in self.embedding):
                raise XR02AssociationContractError("embedding must contain finite values")
            norm = math.sqrt(sum(value * value for value in self.embedding))
            if norm <= 0:
                raise XR02AssociationContractError("embedding norm must be positive")
            normalized = tuple(value / norm for value in self.embedding)
            object.__setattr__(self, "embedding", normalized)
        if self.bbox_xyxy is not None:
            if not all(math.isfinite(value) for value in self.bbox_xyxy):
                raise XR02AssociationContractError("bounding box must be finite")
            x1, y1, x2, y2 = self.bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                raise XR02AssociationContractError("bounding box must have positive area")

    def as_dict(self, *, include_embedding: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "scene_context_sha256": self.scene_context_sha256,
            "observation_id": self.observation_id,
            "local_track_stable_id": self.local_track_stable_id,
            "camera_id": self.camera_id,
            "tracker_profile": self.tracker_profile,
            "observed_monotonic_ns": self.observed_monotonic_ns,
            "world_xy_metres": list(self.world_xy_metres),
            "confidence": self.confidence,
            "quality_weight": self.quality_weight,
            "embedding_reference_sha256": self.embedding_reference_sha256,
        }
        if include_embedding:
            result["embedding"] = None if self.embedding is None else list(self.embedding)
        if self.bbox_xyxy is not None:
            result["bbox_xyxy"] = list(self.bbox_xyxy)
            result["interaction_evidence"] = self.interaction_evidence
        return result


@dataclass(frozen=True, slots=True)
class CostComponents:
    spatial_distance_metres: float | None
    appearance_cosine_distance: float | None
    time_gap_seconds: float
    quality_penalty: float
    topology_allowed: bool
    normalized_total: float | None
    predicted_spatial_distance_metres: float | None = None
    effective_speed_metres_per_second: float | None = None
    motion_gate_metres: float | None = None
    reachable_radius_metres: float | None = None

    def __post_init__(self) -> None:
        optional = (
            self.spatial_distance_metres,
            self.appearance_cosine_distance,
            self.predicted_spatial_distance_metres,
            self.effective_speed_metres_per_second,
            self.motion_gate_metres,
            self.reachable_radius_metres,
        )
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise XR02AssociationContractError("optional cost components must be finite")
        if any(value is not None and value < 0 for value in optional):
            raise XR02AssociationContractError("optional cost components must be non-negative")
        if not math.isfinite(self.time_gap_seconds) or self.time_gap_seconds < 0:
            raise XR02AssociationContractError("time gap must be finite and non-negative")
        if not math.isfinite(self.quality_penalty) or not 0 <= self.quality_penalty <= 1:
            raise XR02AssociationContractError("quality penalty must be within [0, 1]")
        if self.normalized_total is not None and (
            not math.isfinite(self.normalized_total) or self.normalized_total < 0
        ):
            raise XR02AssociationContractError("normalized cost must be finite and non-negative")

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "spatial_distance_metres": self.spatial_distance_metres,
            "appearance_cosine_distance": self.appearance_cosine_distance,
            "time_gap_seconds": self.time_gap_seconds,
            "quality_penalty": self.quality_penalty,
            "topology_allowed": self.topology_allowed,
            "normalized_total": self.normalized_total,
        }
        if self.predicted_spatial_distance_metres is not None:
            result["predicted_spatial_distance_metres"] = self.predicted_spatial_distance_metres
        if self.effective_speed_metres_per_second is not None:
            result["effective_speed_metres_per_second"] = self.effective_speed_metres_per_second
        if self.motion_gate_metres is not None:
            result["motion_gate_metres"] = self.motion_gate_metres
        if self.reachable_radius_metres is not None:
            result["reachable_radius_metres"] = self.reachable_radius_metres
        return result


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    kind: CandidateKind
    left_id: str
    right_id: str
    profile_id: str
    outcome: CandidateOutcome
    reason: str
    costs: CostComponents

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "candidate_id"),
            (self.left_id, "left_id"),
            (self.right_id, "right_id"),
            (self.profile_id, "profile_id"),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise XR02AssociationContractError(f"{label} is invalid")
        if not self.reason:
            raise XR02AssociationContractError("candidate reason is required")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "profile_id": self.profile_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "costs": self.costs.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class MemberAssignment:
    observation_id: str
    local_track_stable_id: str
    camera_id: str
    global_track_id: str | None
    state: GlobalTrackState
    reason: str

    def __post_init__(self) -> None:
        if self.global_track_id is not None and not _IDENTIFIER.fullmatch(self.global_track_id):
            raise XR02AssociationContractError("global track ID is invalid")
        if self.state is GlobalTrackState.AMBIGUOUS and self.global_track_id is not None:
            raise XR02AssociationContractError("ambiguous evidence cannot contain a global ID")
        if self.state is not GlobalTrackState.AMBIGUOUS and self.global_track_id is None:
            raise XR02AssociationContractError("non-ambiguous assignment requires a global ID")
        if not self.reason:
            raise XR02AssociationContractError("assignment reason is required")

    def as_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "local_track_stable_id": self.local_track_stable_id,
            "camera_id": self.camera_id,
            "global_track_id": self.global_track_id,
            "state": self.state.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GlobalTrackSnapshot:
    global_track_id: str
    state: GlobalTrackState
    last_world_xy_metres: tuple[float, float]
    last_observed_monotonic_ns: int
    camera_ids: tuple[str, ...]
    hit_count: int
    profile_id: str
    velocity_xy_metres_per_second: tuple[float, float] | None = None
    position_uncertainty_metres: float | None = None

    def __post_init__(self) -> None:
        if self.state in {GlobalTrackState.AMBIGUOUS, GlobalTrackState.DUPLICATE}:
            raise XR02AssociationContractError(
                "ambiguous/duplicate are evidence, not track states"
            )
        if self.hit_count <= 0 or self.last_observed_monotonic_ns < 0:
            raise XR02AssociationContractError("track history is invalid")
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise XR02AssociationContractError("track cameras must be non-empty and unique")
        if self.velocity_xy_metres_per_second is not None and not all(
            math.isfinite(value) for value in self.velocity_xy_metres_per_second
        ):
            raise XR02AssociationContractError("track velocity must be finite")
        if self.position_uncertainty_metres is not None and (
            not math.isfinite(self.position_uncertainty_metres)
            or self.position_uncertainty_metres <= 0
        ):
            raise XR02AssociationContractError("position uncertainty must be positive")

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "global_track_id": self.global_track_id,
            "state": self.state.value,
            "last_world_xy_metres": list(self.last_world_xy_metres),
            "last_observed_monotonic_ns": self.last_observed_monotonic_ns,
            "camera_ids": list(self.camera_ids),
            "hit_count": self.hit_count,
            "profile_id": self.profile_id,
        }
        if self.velocity_xy_metres_per_second is not None:
            result["velocity_xy_metres_per_second"] = list(self.velocity_xy_metres_per_second)
            result["position_uncertainty_metres"] = self.position_uncertainty_metres
        return result


@dataclass(frozen=True, slots=True)
class AssociationTickResult:
    scene_context_sha256: str
    tick_index: int
    evaluated_monotonic_ns: int
    profile_id: str
    assignments: tuple[MemberAssignment, ...]
    tracks: tuple[GlobalTrackSnapshot, ...]
    candidates: tuple[CandidateEvidence, ...]

    def __post_init__(self) -> None:
        if self.tick_index < 0 or self.evaluated_monotonic_ns < 0:
            raise XR02AssociationContractError("tick identity must be non-negative")
        observation_ids = [item.observation_id for item in self.assignments]
        if len(observation_ids) != len(set(observation_ids)):
            raise XR02AssociationContractError("tick assignments must be unique by observation")

    @property
    def signature_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_candidates=True))

    def as_dict(self, *, include_candidates: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "xr02.global_association_tick.v1",
            "scene_context_sha256": self.scene_context_sha256,
            "tick_index": self.tick_index,
            "evaluated_monotonic_ns": self.evaluated_monotonic_ns,
            "profile_id": self.profile_id,
            "assignments": [item.as_dict() for item in self.assignments],
            "tracks": [item.as_dict() for item in self.tracks],
        }
        if include_candidates:
            result["candidates"] = [item.as_dict() for item in self.candidates]
        return result


def cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise XR02AssociationContractError("embedding dimensions disagree")
    similarity = sum(a * b for a, b in zip(left, right, strict=True))
    return min(2.0, max(0.0, 1.0 - similarity))


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return canonical_sha256(value)
