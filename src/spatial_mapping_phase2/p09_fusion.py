"""Anonymous time/world-XY clustering, robust fusion and honest current-state logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatial_mapping_phase2.p09_tracking_domain import (
    CAMERA_IDS,
    TrackingEstimate,
    TrackingState,
    WorldFloorObservation,
)


class P09FusionError(ValueError):
    """Raised when fusion configuration or temporal evidence is malformed."""


@dataclass(frozen=True, slots=True)
class FusionConfig:
    maximum_observation_age_ms: float
    maximum_cross_camera_skew_ms: float
    spatial_gate_metres: float
    maximum_speed_metres_per_second: float
    motion_gate_slack_metres: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.maximum_observation_age_ms,
                self.maximum_cross_camera_skew_ms,
                self.spatial_gate_metres,
                self.maximum_speed_metres_per_second,
                self.motion_gate_slack_metres,
            )
        ):
            raise P09FusionError("all fusion gates must be finite and positive")


class AnonymousWorldTracker:
    """Stateful one-person tracker; no per-camera IDs or hidden interpolation are used."""

    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self._last_xy: tuple[float, float] | None = None
        self._last_time_ns: int | None = None

    def evaluate(
        self,
        observations: tuple[WorldFloorObservation, ...],
        current_person_counts: dict[str, int],
        evaluated_monotonic_ns: int,
    ) -> TrackingEstimate:
        if evaluated_monotonic_ns < 0:
            raise P09FusionError("evaluation time must be non-negative")
        if set(current_person_counts) != set(CAMERA_IDS):
            raise P09FusionError("person counts must cover the exact P09 camera roster")
        if any(count < 0 for count in current_person_counts.values()):
            raise P09FusionError("person counts must be non-negative")
        if any(count >= 2 for count in current_person_counts.values()):
            return self._non_position(
                TrackingState.MULTI_PERSON_UNSUPPORTED,
                evaluated_monotonic_ns,
                "at least one current camera credibly detects multiple people",
            )
        if len({observation.camera_id for observation in observations}) != len(observations):
            return self._non_position(
                TrackingState.MULTI_PERSON_UNSUPPORTED,
                evaluated_monotonic_ns,
                "multiple current observations from one camera are unsupported",
            )

        fresh = tuple(
            observation
            for observation in observations
            if 0
            <= evaluated_monotonic_ns - observation.acquisition_monotonic_ns
            <= self.config.maximum_observation_age_ms * 1_000_000.0
        )
        if not fresh:
            return self._non_position(
                TrackingState.UNKNOWN,
                evaluated_monotonic_ns,
                "no current fresh projected observation",
            )
        newest = max(observation.acquisition_monotonic_ns for observation in fresh)
        time_gated = tuple(
            observation
            for observation in fresh
            if newest - observation.acquisition_monotonic_ns
            <= self.config.maximum_cross_camera_skew_ms * 1_000_000.0
        )
        if not time_gated:
            return self._non_position(
                TrackingState.UNKNOWN,
                evaluated_monotonic_ns,
                "no observations pass the cross-camera time gate",
            )
        if len(time_gated) == 1:
            observation = time_gated[0]
            if not self._passes_motion_gate(observation.xy_metres, evaluated_monotonic_ns):
                return self._non_position(
                    TrackingState.UNKNOWN,
                    evaluated_monotonic_ns,
                    "single-camera candidate violates bounded motion continuity",
                )
            return self._tracked(
                TrackingState.TRACKED_SINGLE_CAMERA,
                observation.xy_metres,
                (observation.camera_id,),
                (),
                evaluated_monotonic_ns,
                "one fresh valid floor observation",
            )

        selected_indices = _robust_consistent_cluster(time_gated, self.config.spatial_gate_metres)
        if selected_indices is None:
            return self._non_position(
                TrackingState.AMBIGUOUS,
                evaluated_monotonic_ns,
                "current camera observations form incompatible world-XY clusters",
            )
        selected = tuple(time_gated[index] for index in selected_indices)
        rejected = tuple(
            observation.camera_id
            for index, observation in enumerate(time_gated)
            if index not in selected_indices
        )
        xy = _weighted_mean(selected)
        if not self._passes_motion_gate(xy, evaluated_monotonic_ns):
            return self._non_position(
                TrackingState.AMBIGUOUS,
                evaluated_monotonic_ns,
                "fused candidate violates bounded motion continuity",
            )
        return self._tracked(
            TrackingState.TRACKED_FUSED,
            xy,
            tuple(sorted(observation.camera_id for observation in selected)),
            tuple(sorted(rejected)),
            evaluated_monotonic_ns,
            "one robust time/space-consistent anonymous observation cluster",
        )

    def _passes_motion_gate(
        self, xy_metres: tuple[float, float], evaluated_monotonic_ns: int
    ) -> bool:
        if self._last_xy is None or self._last_time_ns is None:
            return True
        elapsed_seconds = max(0.0, (evaluated_monotonic_ns - self._last_time_ns) / 1e9)
        maximum_distance = (
            self.config.maximum_speed_metres_per_second * elapsed_seconds
            + self.config.motion_gate_slack_metres
        )
        return math.dist(self._last_xy, xy_metres) <= maximum_distance

    def _tracked(
        self,
        state: TrackingState,
        xy_metres: tuple[float, float],
        contributing: tuple[str, ...],
        rejected: tuple[str, ...],
        evaluated_monotonic_ns: int,
        reason: str,
    ) -> TrackingEstimate:
        self._last_xy = xy_metres
        self._last_time_ns = evaluated_monotonic_ns
        return TrackingEstimate(
            state,
            evaluated_monotonic_ns,
            xy_metres,
            contributing,
            rejected,
            reason,
        )

    def _non_position(
        self, state: TrackingState, evaluated_monotonic_ns: int, reason: str
    ) -> TrackingEstimate:
        last_age = (
            None
            if self._last_time_ns is None
            else (evaluated_monotonic_ns - self._last_time_ns) / 1_000_000.0
        )
        return TrackingEstimate(
            state,
            evaluated_monotonic_ns,
            None,
            (),
            (),
            reason,
            self._last_xy,
            last_age,
        )


def _robust_consistent_cluster(
    observations: tuple[WorldFloorObservation, ...], spatial_gate_metres: float
) -> tuple[int, ...] | None:
    count = len(observations)
    if count == 2:
        distance = math.dist(observations[0].xy_metres, observations[1].xy_metres)
        return (0, 1) if distance <= spatial_gate_metres else None
    neighbour_sets: list[tuple[int, ...]] = []
    for observation in observations:
        neighbours = tuple(
            other_index
            for other_index, other in enumerate(observations)
            if math.dist(observation.xy_metres, other.xy_metres) <= spatial_gate_metres
        )
        neighbour_sets.append(neighbours)
    maximum = max(len(indices) for indices in neighbour_sets)
    candidates = sorted({indices for indices in neighbour_sets if len(indices) == maximum})
    required = count - 1
    if maximum < required or maximum < 2 or len(candidates) != 1:
        return None
    return candidates[0]


def _weighted_mean(
    observations: tuple[WorldFloorObservation, ...],
) -> tuple[float, float]:
    weights = np.asarray([observation.quality_weight for observation in observations])
    points = np.asarray([observation.xy_metres for observation in observations])
    mean = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
    return float(mean[0]), float(mean[1])
