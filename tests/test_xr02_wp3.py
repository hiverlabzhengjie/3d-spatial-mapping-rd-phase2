from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_mapping_phase2.xr02_association import (
    AssociationConfig,
    SceneGlobalAssociator,
    office_topology,
)
from spatial_mapping_phase2.xr02_global_domain import (
    AssociationObservation,
    CandidateKind,
    GlobalTrackState,
    SignalProfile,
    XR02AssociationContractError,
    canonical_sha256,
)
from spatial_mapping_phase2.xr02_global_journal import (
    GlobalAssociationJournal,
    GlobalAssociationJournalError,
    verify_global_journal,
)
from spatial_mapping_phase2.xr02_journal import ObservationJournal
from spatial_mapping_phase2.xr02_local_domain import (
    CropQuality,
    EmbeddingStatus,
    FootpointSource,
    FrameKey,
    LocalTrackKey,
    LocalTrackObservation,
    SceneContextKey,
    WorldProjectionStatus,
)
from spatial_mapping_phase2.xr02_replay_eval import (
    aggregate_runs,
    load_replay_partition,
    run_scenario,
)
from spatial_mapping_phase2.xr02_wp2_input import load_wp2_journal_read_only

_SCENE = "e" * 64


def _observation(
    observation_id: str,
    camera_id: str,
    xy: tuple[float, float],
    embedding: tuple[float, ...] = (1.0, 0.0),
    *,
    time_ns: int = 1_000_000_000,
    local_id: str | None = None,
) -> AssociationObservation:
    return AssociationObservation(
        scene_context_sha256=_SCENE,
        observation_id=observation_id,
        local_track_stable_id=local_id or f"fixture:{camera_id}:{observation_id}",
        camera_id=camera_id,
        tracker_profile="fixture-v1",
        observed_monotonic_ns=time_ns,
        world_xy_metres=xy,
        confidence=0.9,
        quality_weight=0.9,
        embedding_reference_sha256=canonical_sha256(embedding),
        embedding=embedding,
    )


def test_complete_link_and_same_camera_cannot_link_are_preserved() -> None:
    associator = SceneGlobalAssociator(_SCENE, office_topology(), SignalProfile.COMBINED)
    result = associator.process_tick(
        0,
        1_000_000_000,
        (
            _observation("a1", "office-cam-01", (0.0, 0.0)),
            _observation("a2", "office-cam-02", (0.9, 0.0)),
            _observation("b3", "office-cam-03", (1.8, 0.0)),
            _observation("same", "office-cam-01", (0.1, 0.0), local_id="fixture:cam01:b"),
        ),
    )
    same_camera = [item for item in result.candidates if item.reason == "same_camera_cannot_link"]
    assert same_camera
    assert any(item.state is GlobalTrackState.AMBIGUOUS for item in result.assignments)
    non_ambiguous_ids = {
        item.global_track_id
        for item in result.assignments
        if item.state is not GlobalTrackState.AMBIGUOUS
    }
    assert len(non_ambiguous_ids) >= 2


def test_hungarian_conflict_fails_ambiguous_instead_of_spawning_identity() -> None:
    associator = SceneGlobalAssociator(
        _SCENE,
        office_topology(),
        SignalProfile.SPATIAL_ONLY,
        AssociationConfig(minimum_confirmation_hits=1),
    )
    first = associator.process_tick(
        0,
        1_000_000_000,
        (_observation("seed", "office-cam-01", (1.0, 1.0)),),
    )
    assert len(first.tracks) == 1
    second = associator.process_tick(
        1,
        2_000_000_000,
        (
            _observation("candidate-a", "office-cam-02", (1.1, 1.0), time_ns=2_000_000_000),
            _observation("candidate-b", "office-cam-02", (1.2, 1.0), time_ns=2_000_000_000),
        ),
    )
    assert len(second.tracks) == 1
    assert sum(item.state is GlobalTrackState.AMBIGUOUS for item in second.assignments) == 1
    assert any(
        item.kind is CandidateKind.GLOBAL_ASSIGNMENT
        and item.reason == "one_to_one_assignment_conflict"
        for item in second.candidates
    )


def test_lifecycle_reacquires_within_bound_then_ends() -> None:
    associator = SceneGlobalAssociator(_SCENE, office_topology(), SignalProfile.COMBINED)
    first = associator.process_tick(
        0, 1_000_000_000, (_observation("life-a0", "office-cam-01", (1.0, 1.0)),)
    )
    global_id = first.assignments[0].global_track_id
    lost = associator.process_tick(1, 2_200_000_000, ())
    assert lost.tracks[0].state is GlobalTrackState.LOST
    reacquired = associator.process_tick(
        2,
        4_000_000_000,
        (
            _observation(
                "life-a1",
                "office-cam-03",
                (1.2, 1.0),
                time_ns=4_000_000_000,
            ),
        ),
    )
    assert reacquired.assignments[0].global_track_id == global_id
    ended = associator.process_tick(3, 9_000_000_000, ())
    assert ended.tracks[0].state is GlobalTrackState.ENDED


def test_heldout_ablation_is_deterministic_and_combined_wins_gate() -> None:
    fixture = Path("docs/workstreams/XR02/wp3/fixtures/heldout.json")
    partition = load_replay_partition(fixture)
    aggregates: dict[SignalProfile, dict[str, int | float | str]] = {}
    signatures: dict[SignalProfile, str] = {}
    for profile in SignalProfile:
        first = tuple(
            run_scenario(item, profile, office_topology()) for item in partition.scenarios
        )
        second = tuple(
            run_scenario(item, profile, office_topology()) for item in partition.scenarios
        )
        aggregates[profile] = aggregate_runs(first)
        signatures[profile] = canonical_sha256([item.decision_signature_sha256 for item in first])
        assert signatures[profile] == canonical_sha256(
            [item.decision_signature_sha256 for item in second]
        )
    combined = aggregates[SignalProfile.COMBINED]
    baseline_f1 = max(
        float(aggregates[SignalProfile.SPATIAL_ONLY]["pairwise_identity_f1"]),
        float(aggregates[SignalProfile.APPEARANCE_ONLY]["pairwise_identity_f1"]),
    )
    assert float(combined["pairwise_identity_f1"]) > baseline_f1
    assert combined["false_merge_pairs"] == 0
    assert int(combined["ambiguous_observations"]) > 0


def test_global_journal_detects_tampering(tmp_path: Path) -> None:
    associator = SceneGlobalAssociator(_SCENE, office_topology(), SignalProfile.COMBINED)
    result = associator.process_tick(
        0, 1_000_000_000, (_observation("journal-a", "office-cam-01", (1.0, 1.0)),)
    )
    path = tmp_path / "global.jsonl"
    GlobalAssociationJournal(path).append(result)
    assert verify_global_journal(path).records == 1
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["assignments"][0]["reason"] = "changed"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(GlobalAssociationJournalError, match="content changed"):
        verify_global_journal(path)


def test_global_journal_batch_preserves_every_decision_and_hash_link(tmp_path: Path) -> None:
    associator = SceneGlobalAssociator(_SCENE, office_topology(), SignalProfile.COMBINED)
    first = associator.process_tick(
        0, 1_000_000_000, (_observation("batch-a", "office-cam-01", (1.0, 1.0)),)
    )
    second = associator.process_tick(
        1, 1_100_000_000, (_observation("batch-b", "office-cam-01", (1.1, 1.0)),)
    )
    path = tmp_path / "global-batch.jsonl"
    digests = GlobalAssociationJournal(path).append_batch((first, second))
    assert len(digests) == 2
    assert verify_global_journal(path).records == 2


def test_wp2_adapter_reads_without_rewriting_journal(tmp_path: Path) -> None:
    scene = SceneContextKey("office", "fixture-epoch", "a" * 64, "b" * 64, "c" * 64)
    frame = FrameKey(
        scene,
        "office-cam-01",
        "frame-0",
        0,
        1_000_000_000,
        "2026-08-24T00:00:00Z",
        504,
        280,
    )
    observation = LocalTrackObservation(
        frame=frame,
        track=LocalTrackKey(scene.context_sha256, "office-cam-01", "botsort-fixed-v1", 1),
        detection_index=0,
        confidence=0.8,
        bbox_xyxy=(10.0, 10.0, 50.0, 100.0),
        footpoint_uv=(30.0, 100.0),
        footpoint_source=FootpointSource.BBOX_BOTTOM_CENTER,
        crop_quality=CropQuality(1.0, 3600.0, 4 / 9),
        embedding_status=EmbeddingStatus.NOT_DUE,
        embedding=None,
        projection_status=WorldProjectionStatus.VALID,
        world_xy_metres=(2.0, 3.0),
        projection_reason="fixture",
    )
    path = tmp_path / "botsort-fixed-v1.jsonl"
    ObservationJournal(path).append(observation)
    before = path.read_bytes()
    audit = load_wp2_journal_read_only(path, "botsort-fixed-v1")
    assert len(audit.valid_observations) == 1
    assert audit.missing_embedding_count == 1
    assert path.read_bytes() == before


def test_contract_rejects_mismatched_embedding_reference() -> None:
    with pytest.raises(XR02AssociationContractError, match="must coexist"):
        AssociationObservation(
            _SCENE,
            "bad",
            "fixture:bad",
            "office-cam-01",
            "fixture-v1",
            1,
            (0.0, 0.0),
            0.9,
            0.9,
            "a" * 64,
            None,
        )
