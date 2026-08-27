from __future__ import annotations

from spatial_mapping_phase2.xr02_association import (
    AssociationConfig,
    SceneGlobalAssociator,
    office_topology,
)
from spatial_mapping_phase2.xr02_global_domain import (
    AssociationObservation,
    GlobalTrackState,
    SignalProfile,
    canonical_sha256,
)

_SCENE = "f" * 64


def _observation(
    observation_id: str,
    local_id: str,
    xy: tuple[float, float],
    embedding: tuple[float, ...] | None,
    time_ns: int,
    *,
    camera_id: str = "office-cam-01",
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 20.0),
    confidence: float = 0.9,
) -> AssociationObservation:
    return AssociationObservation(
        scene_context_sha256=_SCENE,
        observation_id=observation_id,
        local_track_stable_id=local_id,
        camera_id=camera_id,
        tracker_profile="fixture-live-v1",
        observed_monotonic_ns=time_ns,
        world_xy_metres=xy,
        confidence=confidence,
        quality_weight=0.9,
        embedding_reference_sha256=(None if embedding is None else canonical_sha256(embedding)),
        embedding=embedding,
        bbox_xyxy=bbox,
    )


def _continuity_associator() -> SceneGlobalAssociator:
    return SceneGlobalAssociator(
        _SCENE,
        office_topology(),
        SignalProfile.COMBINED,
        AssociationConfig.continuity_live(),
    )


def _seed_one(
    associator: SceneGlobalAssociator,
    *,
    local_id: str = "fixture:cam01:a",
    xy: tuple[float, float] = (1.0, 1.0),
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> str:
    latest = None
    for tick_index in range(4):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (_observation(f"a{tick_index}", local_id, xy, embedding, time_ns),),
        )
        if tick_index < 3:
            assert not latest.tracks
            assert latest.assignments[0].reason == "provisional_new_global_track"
    assert latest is not None
    assert len(latest.tracks) == 1
    global_id = latest.assignments[0].global_track_id
    assert global_id is not None
    return global_id


def _seed_pair(associator: SceneGlobalAssociator) -> dict[str, str]:
    local_a = "fixture:cam01:a"
    local_b = "fixture:cam01:b"
    for tick_index in range(4):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        result = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"a{tick_index}",
                    local_a,
                    (0.0, 0.0),
                    (1.0, 0.0),
                    time_ns,
                    bbox=(0.0, 0.0, 10.0, 20.0),
                ),
                _observation(
                    f"b{tick_index}",
                    local_b,
                    (1.0, 0.0),
                    (0.0, 1.0),
                    time_ns,
                    bbox=(30.0, 0.0, 40.0, 20.0),
                ),
            ),
        )
    mapping = {
        assignment.local_track_stable_id: assignment.global_track_id
        for assignment in result.assignments
    }
    assert all(value is not None for value in mapping.values())
    return {key: value for key, value in mapping.items() if value is not None}


def test_new_global_identity_requires_consecutive_evidence() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    assert global_id == "g:000001"
    assert associator.profile_id == "combined:wp4-reachable-radius-continuity-v5"


def test_unique_short_gap_missing_appearance_waits_without_replacement() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        1_500_000_000,
        (
            _observation(
                "recovery-no-appearance",
                "fixture:cam02:new",
                (1.2, 1.0),
                None,
                1_500_000_000,
                camera_id="office-cam-02",
            ),
        ),
    )
    assert len(result.tracks) == 1
    assert result.tracks[0].global_track_id == global_id
    assert result.assignments[0].global_track_id is None
    assert result.assignments[0].state is GlobalTrackState.AMBIGUOUS
    assert result.assignments[0].reason == "unique_reachable_predecessor_appearance_pending"


def test_repeated_missing_appearance_never_accumulates_replacement_birth() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    latest = None
    for tick_index in range(4, 12):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"appearance-pending-{tick_index}",
                    "fixture:cam02:new",
                    (1.2, 1.0),
                    None,
                    time_ns,
                    camera_id="office-cam-02",
                ),
            ),
        )
        assert latest.assignments[0].global_track_id is None
        assert latest.assignments[0].reason == "unique_reachable_predecessor_appearance_pending"

    assert latest is not None
    assert len(latest.tracks) == 1
    assert latest.tracks[0].global_track_id == global_id


def test_unique_short_gap_physical_recovery_can_outvote_one_appearance_mismatch() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        1_500_000_000,
        (
            _observation(
                "recovery-mismatch",
                "fixture:cam02:new",
                (1.2, 1.0),
                (0.0, 1.0),
                1_500_000_000,
                camera_id="office-cam-02",
            ),
        ),
    )
    assert len(result.tracks) == 1
    assert result.assignments[0].global_track_id == global_id
    assert result.assignments[0].reason == "unique_physical_recovery_overrides_appearance_mismatch"


def test_multiple_physical_recovery_candidates_fail_ambiguous_without_new_id() -> None:
    associator = _continuity_associator()
    _seed_pair(associator)
    result = associator.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "uncertain-recovery",
                "fixture:cam01:new",
                (0.5, 0.0),
                None,
                1_400_000_000,
            ),
        ),
    )
    assert len(result.tracks) == 2
    assert result.assignments[0].global_track_id is None
    assert result.assignments[0].state is GlobalTrackState.AMBIGUOUS
    assert result.assignments[0].reason == "multiple_reachable_predecessors_appearance_unresolved"


def test_missing_appearance_cannot_birth_identity_when_physics_explains_none() -> None:
    associator = _continuity_associator()
    _seed_one(associator)
    result = associator.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "far-no-appearance",
                "fixture:cam01:new",
                (20.0, 20.0),
                None,
                1_400_000_000,
            ),
        ),
    )
    assert len(result.tracks) == 1
    assert result.assignments[0].global_track_id is None
    assert result.assignments[0].reason == "new_global_birth_requires_appearance"


def test_genuinely_new_high_confidence_person_binds_after_four_ticks() -> None:
    associator = _continuity_associator()
    _seed_one(associator)
    latest = None
    for tick_index in range(4, 8):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"new-person-{tick_index}",
                    "fixture:cam01:new-person",
                    (20.0, 20.0),
                    (0.0, 1.0),
                    time_ns,
                ),
            ),
        )
        if tick_index < 7:
            assert latest.assignments[0].reason == "provisional_new_global_track"
            assert latest.assignments[0].global_track_id is None
    assert latest is not None
    assert len(latest.tracks) == 2
    assert latest.assignments[0].global_track_id == "g:000002"
    assert latest.assignments[0].reason == "confirmed_new_scene_global_track"


def test_low_confidence_observation_never_births_global_identity() -> None:
    associator = _continuity_associator()
    _seed_one(associator)
    latest = None
    for tick_index in range(4, 9):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"low-confidence-{tick_index}",
                    "fixture:cam01:low-confidence",
                    (20.0, 20.0),
                    (0.0, 1.0),
                    time_ns,
                    confidence=0.64,
                ),
            ),
        )
        assert latest.assignments[0].reason == "new_global_birth_below_confidence"
        assert latest.assignments[0].global_track_id is None
    assert latest is not None
    assert len(latest.tracks) == 1


def test_low_confidence_unbound_box_cannot_rebind_existing_identity() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        1_500_000_000,
        (
            _observation(
                "weak-lookalike",
                "fixture:cam01:unbound-weak",
                (1.1, 1.0),
                (1.0, 0.0),
                1_500_000_000,
                confidence=0.25,
            ),
        ),
    )
    assert len(result.tracks) == 1
    assert result.tracks[0].global_track_id == global_id
    assert result.assignments[0].global_track_id is None
    assert result.assignments[0].reason == "new_global_birth_below_confidence"


def test_repeated_low_confidence_box_releases_binding_without_deleting_identity() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    latest = None
    for tick_index in range(4, 12):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"weak-{tick_index}",
                    "fixture:cam01:a",
                    (1.0, 1.0),
                    (1.0, 0.0),
                    time_ns,
                    confidence=0.25,
                ),
            ),
        )
        if tick_index < 11:
            assert latest.assignments[0].global_track_id == global_id

    assert latest is not None
    assert len(latest.tracks) == 1
    assert latest.assignments[0].global_track_id is None
    assert latest.assignments[0].reason == "released_low_confidence_persistent_binding"

    recovered = associator.process_tick(
        12,
        2_200_000_000,
        (
            _observation(
                "strong-again",
                "fixture:cam01:a",
                (1.1, 1.0),
                (1.0, 0.0),
                2_200_000_000,
            ),
        ),
    )
    assert recovered.assignments[0].global_track_id == global_id
    assert len(recovered.tracks) == 1


def test_stale_weak_observation_is_not_counted_but_identity_remains_recoverable() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    uncertain = associator.process_tick(
        4,
        2_500_000_000,
        (
            _observation(
                "stale-weak-box",
                "fixture:cam01:a",
                (1.0, 1.0),
                (1.0, 0.0),
                2_500_000_000,
                confidence=0.25,
            ),
        ),
    )
    assert len(uncertain.tracks) == 1
    assert uncertain.tracks[0].global_track_id == global_id
    assert uncertain.assignments[0].global_track_id is None
    assert uncertain.assignments[0].reason == "persistent_binding_unverified_observation"

    recovered = associator.process_tick(
        5,
        2_600_000_000,
        (
            _observation(
                "trusted-person-again",
                "fixture:cam01:a",
                (1.1, 1.0),
                (1.0, 0.0),
                2_600_000_000,
            ),
        ),
    )
    assert recovered.assignments[0].global_track_id == global_id
    assert len(recovered.tracks) == 1


def test_separated_same_camera_people_become_distinct_after_confirmation() -> None:
    associator = _continuity_associator()
    first_global = _seed_one(associator)
    latest = None
    for tick_index in range(4, 8):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"first-{tick_index}",
                    "fixture:cam01:a",
                    (1.0, 1.0),
                    (1.0, 0.0),
                    time_ns,
                    bbox=(0.0, 0.0, 10.0, 20.0),
                ),
                _observation(
                    f"second-{tick_index}",
                    "fixture:cam01:second",
                    (1.2, 1.0),
                    (1.0, 0.0),
                    time_ns,
                    bbox=(30.0, 0.0, 40.0, 20.0),
                ),
            ),
        )

    assert latest is not None
    assigned = {item.local_track_stable_id: item.global_track_id for item in latest.assignments}
    assert assigned["fixture:cam01:a"] == first_global
    assert assigned["fixture:cam01:second"] == "g:000002"
    assert len(latest.tracks) == 2


def test_persistent_local_binding_ignores_one_frame_appearance_flip() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "a4",
                "fixture:cam01:a",
                (1.1, 1.0),
                (0.0, 1.0),
                1_400_000_000,
            ),
        ),
    )
    assert result.assignments[0].global_track_id == global_id
    assert result.assignments[0].reason == "persistent_local_binding"


def test_high_confidence_person_remains_observed_during_appearance_challenge() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        2_500_000_000,
        (
            _observation(
                "high-confidence-mismatch",
                "fixture:cam01:a",
                (1.1, 1.0),
                (0.0, 1.0),
                2_500_000_000,
            ),
        ),
    )
    assert result.assignments[0].global_track_id == global_id
    assert result.assignments[0].reason == "persistent_local_binding"
    assert len(result.tracks) == 1


def test_appearance_disagreement_alone_never_detaches_visible_person() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    latest = None
    for tick_index in range(4, 16):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"appearance-disagreement-{tick_index}",
                    "fixture:cam01:a",
                    (1.1, 1.0),
                    (0.0, 1.0),
                    time_ns,
                ),
            ),
        )
        assert latest.assignments[0].global_track_id == global_id
        assert latest.assignments[0].reason == "persistent_local_binding"
    assert latest is not None
    assert len(latest.tracks) == 1


def test_impossible_teleport_fails_ambiguous_without_spawning_identity() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "a4",
                "fixture:cam01:a",
                (10.0, 10.0),
                (1.0, 0.0),
                1_400_000_000,
            ),
        ),
    )
    assert len(result.tracks) == 1
    assert result.tracks[0].global_track_id == global_id
    assert result.assignments[0].global_track_id is None
    assert result.assignments[0].state is GlobalTrackState.AMBIGUOUS
    assert result.assignments[0].reason == "persistent_binding_motion_conflict"


def test_repeated_xy_conflict_never_releases_high_confidence_identity_binding() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    for tick_index in range(4, 16):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        conflicted = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"noisy-projection-{tick_index}",
                    "fixture:cam01:a",
                    (10.0, 10.0),
                    None,
                    time_ns,
                ),
            ),
        )
        assert conflicted.assignments[0].global_track_id is None
        assert conflicted.assignments[0].reason == "persistent_binding_motion_conflict"
        assert len(conflicted.tracks) == 1
        assert conflicted.tracks[0].global_track_id == global_id

    recovered = associator.process_tick(
        16,
        2_600_000_000,
        (
            _observation(
                "projection-stable-again",
                "fixture:cam01:a",
                (1.1, 1.0),
                (1.0, 0.0),
                2_600_000_000,
            ),
        ),
    )
    assert recovered.assignments[0].global_track_id == global_id
    assert recovered.assignments[0].reason == "persistent_local_binding"
    assert len(recovered.tracks) == 1


def test_same_camera_jump_uses_tighter_uncertainty_than_camera_handoff() -> None:
    same_camera = _continuity_associator()
    same_global = _seed_one(same_camera)
    rejected = same_camera.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "same-camera-jump",
                "fixture:cam01:a",
                (2.6, 1.0),
                (1.0, 0.0),
                1_400_000_000,
            ),
        ),
    )
    assert rejected.tracks[0].global_track_id == same_global
    assert rejected.assignments[0].global_track_id is None
    assert rejected.assignments[0].reason == "persistent_binding_motion_conflict"

    handoff = _continuity_associator()
    handoff_global = _seed_one(handoff)
    accepted = handoff.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "cross-camera-offset",
                "fixture:cam02:new",
                (2.6, 1.0),
                (1.0, 0.0),
                1_400_000_000,
                camera_id="office-cam-02",
            ),
        ),
    )
    assert accepted.assignments[0].global_track_id == handoff_global


def test_plausible_new_local_handoff_reuses_existing_global_identity() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        1_600_000_000,
        (
            _observation(
                "handoff",
                "fixture:cam02:new",
                (1.4, 1.0),
                (1.0, 0.0),
                1_600_000_000,
                camera_id="office-cam-02",
            ),
        ),
    )
    assert result.assignments[0].global_track_id == global_id
    assert result.assignments[0].reason == "gated_reachable_hungarian_match"


def test_strong_appearance_recovers_after_three_seconds_without_direction_gate() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        7_000_000_000,
        (
            _observation(
                "long-gap-turning-recovery",
                "fixture:cam02:reappeared",
                (5.0, 1.0),
                (1.0, 0.0),
                7_000_000_000,
                camera_id="office-cam-02",
            ),
        ),
    )
    assert result.assignments[0].global_track_id == global_id
    assert result.assignments[0].reason == "gated_reachable_hungarian_match"
    accepted = [
        candidate for candidate in result.candidates if candidate.outcome.value == "accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0].costs.predicted_spatial_distance_metres is None
    assert accepted[0].costs.reachable_radius_metres is not None
    assert accepted[0].costs.reachable_radius_metres > 4.0


def test_unbound_observation_outside_reachable_radius_cannot_steal_identity() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    result = associator.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "impossible-unbound-jump",
                "fixture:cam01:new",
                (10.0, 10.0),
                (1.0, 0.0),
                1_400_000_000,
            ),
        ),
    )
    assert len(result.tracks) == 1
    assert result.tracks[0].global_track_id == global_id
    assert result.assignments[0].global_track_id is None
    candidate = next(item for item in result.candidates if item.right_id == global_id)
    assert candidate.reason == "impossible_speed_gate"
    spatial = candidate.costs.spatial_distance_metres
    reachable_radius = candidate.costs.reachable_radius_metres
    assert spatial is not None
    assert reachable_radius is not None
    assert spatial > reachable_radius


def test_overlapping_one_to_one_conflict_does_not_spawn_second_identity() -> None:
    associator = _continuity_associator()
    _seed_one(associator)
    result = associator.process_tick(
        4,
        1_600_000_000,
        (
            _observation(
                "handoff-a",
                "fixture:cam02:new-a",
                (1.3, 1.0),
                (1.0, 0.0),
                1_600_000_000,
                camera_id="office-cam-02",
                bbox=(0.0, 0.0, 10.0, 20.0),
            ),
            _observation(
                "handoff-b",
                "fixture:cam02:new-b",
                (1.4, 1.0),
                (1.0, 0.0),
                1_600_000_000,
                camera_id="office-cam-02",
                bbox=(8.0, 0.0, 18.0, 20.0),
            ),
        ),
    )
    assert len(result.tracks) == 1
    assert sum(item.global_track_id is not None for item in result.assignments) == 1
    ambiguous = [item for item in result.assignments if item.state is GlobalTrackState.AMBIGUOUS]
    assert len(ambiguous) == 1
    assert ambiguous[0].reason == "one_to_one_assignment_conflict"


def test_dormant_identity_is_retained_and_reactivated_inside_bound() -> None:
    associator = _continuity_associator()
    global_id = _seed_one(associator)
    dormant = associator.process_tick(4, 10_000_000_000, ())
    assert dormant.tracks[0].state is GlobalTrackState.DORMANT
    reacquired = associator.process_tick(
        5,
        10_100_000_000,
        (
            _observation(
                "a3",
                "fixture:cam01:a",
                (1.1, 1.0),
                (1.0, 0.0),
                10_100_000_000,
            ),
        ),
    )
    assert reacquired.assignments[0].global_track_id == global_id
    assert reacquired.tracks[0].state is GlobalTrackState.CONFIRMED


def test_identity_cannot_switch_without_same_camera_box_interaction() -> None:
    associator = _continuity_associator()
    seeded = _seed_pair(associator)
    result = associator.process_tick(
        4,
        1_400_000_000,
        (
            _observation(
                "a4",
                "fixture:cam01:a",
                (1.0, 0.0),
                (0.0, 1.0),
                1_400_000_000,
                bbox=(0.0, 0.0, 10.0, 20.0),
            ),
            _observation(
                "b4",
                "fixture:cam01:b",
                (0.0, 0.0),
                (1.0, 0.0),
                1_400_000_000,
                bbox=(30.0, 0.0, 40.0, 20.0),
            ),
        ),
    )
    assigned = {item.local_track_stable_id: item.global_track_id for item in result.assignments}
    assert assigned == seeded
    assert {item.reason for item in result.assignments} == {"persistent_local_binding"}


def test_box_interaction_requires_sustained_reciprocal_evidence_before_swap() -> None:
    associator = _continuity_associator()
    seeded = _seed_pair(associator)
    latest = None
    for tick_index in (4, 5, 6):
        time_ns = 1_000_000_000 + tick_index * 100_000_000
        latest = associator.process_tick(
            tick_index,
            time_ns,
            (
                _observation(
                    f"a{tick_index}",
                    "fixture:cam01:a",
                    (1.0, 0.0),
                    (0.0, 1.0),
                    time_ns,
                    bbox=(8.0, 0.0, 24.0, 20.0),
                ),
                _observation(
                    f"b{tick_index}",
                    "fixture:cam01:b",
                    (0.0, 0.0),
                    (1.0, 0.0),
                    time_ns,
                    bbox=(20.0, 0.0, 36.0, 20.0),
                ),
            ),
        )
        assigned = {
            item.local_track_stable_id: item.global_track_id for item in latest.assignments
        }
        if tick_index < 6:
            assert assigned == seeded
            assert {item.reason for item in latest.assignments} == {
                "persistent_binding_interaction_challenge"
            }

    assert latest is not None
    assigned = {item.local_track_stable_id: item.global_track_id for item in latest.assignments}
    assert assigned["fixture:cam01:a"] == seeded["fixture:cam01:b"]
    assert assigned["fixture:cam01:b"] == seeded["fixture:cam01:a"]
    assert {item.reason for item in latest.assignments} == {
        "interaction_reciprocal_switch_confirmed"
    }
