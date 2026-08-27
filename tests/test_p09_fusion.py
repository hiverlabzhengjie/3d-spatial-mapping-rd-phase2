from __future__ import annotations

import pytest

from spatial_mapping_phase2.p09_fusion import AnonymousWorldTracker, FusionConfig
from spatial_mapping_phase2.p09_tracking_domain import (
    CAMERA_IDS,
    FootpointKind,
    LiveFrameIdentity,
    PersonDetection,
    TrackingState,
    WorldFloorObservation,
)


def _config() -> FusionConfig:
    return FusionConfig(250.0, 100.0, 1.0, 3.0, 0.5)


def _observation(camera_id: str, time_ns: int, xy: tuple[float, float]) -> WorldFloorObservation:
    frame = LiveFrameIdentity(
        camera_id,
        f"{camera_id}-frame",
        time_ns,
        "2026-08-20T00:00:00Z",
        None,
        None,
        504,
        280,
    )
    detection = PersonDetection(
        frame,
        0,
        0.9,
        (100.0, 20.0, 200.0, 200.0),
        (150.0, 200.0),
        FootpointKind.BBOX_BOTTOM_CENTER,
    )
    return WorldFloorObservation(detection, xy, 4.0, 0.8, 20.0)


def _counts(**values: int) -> dict[str, int]:
    return {camera_id: values.get(camera_id, 0) for camera_id in CAMERA_IDS}


def test_single_camera_then_compatible_overlap_and_handoff() -> None:
    tracker = AnonymousWorldTracker(_config())
    first = tracker.evaluate(
        (_observation(CAMERA_IDS[0], 1_000_000_000, (1.0, 1.0)),),
        _counts(),
        1_020_000_000,
    )
    assert first.state is TrackingState.TRACKED_SINGLE_CAMERA
    overlap = tracker.evaluate(
        (
            _observation(CAMERA_IDS[0], 1_100_000_000, (1.2, 1.0)),
            _observation(CAMERA_IDS[1], 1_110_000_000, (1.3, 1.1)),
        ),
        _counts(),
        1_120_000_000,
    )
    assert overlap.state is TrackingState.TRACKED_FUSED
    assert overlap.current_xy_metres == pytest.approx((1.25, 1.05), abs=0.02)
    handoff = tracker.evaluate(
        (_observation(CAMERA_IDS[1], 1_300_000_000, (1.6, 1.1)),),
        _counts(),
        1_320_000_000,
    )
    assert handoff.state is TrackingState.TRACKED_SINGLE_CAMERA
    assert handoff.contributing_camera_ids == (CAMERA_IDS[1],)


def test_two_disagreeing_cameras_and_split_clusters_are_ambiguous() -> None:
    tracker = AnonymousWorldTracker(_config())
    two = tracker.evaluate(
        (
            _observation(CAMERA_IDS[0], 1_000_000_000, (1.0, 1.0)),
            _observation(CAMERA_IDS[1], 1_000_000_000, (5.0, 5.0)),
        ),
        _counts(),
        1_010_000_000,
    )
    assert two.state is TrackingState.AMBIGUOUS
    split = tracker.evaluate(
        tuple(
            _observation(camera_id, 2_000_000_000, xy)
            for camera_id, xy in zip(
                CAMERA_IDS,
                ((1.0, 1.0), (1.1, 1.0), (5.0, 5.0), (5.1, 5.0)),
                strict=True,
            )
        ),
        _counts(),
        2_010_000_000,
    )
    assert split.state is TrackingState.AMBIGUOUS


def test_three_cameras_may_reject_exactly_one_spatial_outlier() -> None:
    tracker = AnonymousWorldTracker(_config())
    result = tracker.evaluate(
        (
            _observation(CAMERA_IDS[0], 1_000_000_000, (1.0, 1.0)),
            _observation(CAMERA_IDS[1], 1_010_000_000, (1.2, 1.1)),
            _observation(CAMERA_IDS[2], 1_020_000_000, (8.0, 8.0)),
        ),
        _counts(),
        1_030_000_000,
    )
    assert result.state is TrackingState.TRACKED_FUSED
    assert result.contributing_camera_ids == CAMERA_IDS[:2]
    assert result.rejected_camera_ids == (CAMERA_IDS[2],)


def test_multi_person_and_incompatible_duplicate_camera_fail_closed() -> None:
    tracker = AnonymousWorldTracker(_config())
    multi = tracker.evaluate((), _counts(**{CAMERA_IDS[1]: 2}), 1_000_000_000)
    assert multi.state is TrackingState.MULTI_PERSON_UNSUPPORTED
    duplicate = tracker.evaluate(
        (
            _observation(CAMERA_IDS[0], 1_000_000_000, (1.0, 1.0)),
            _observation(CAMERA_IDS[0], 1_000_000_000, (1.1, 1.0)),
        ),
        _counts(),
        1_010_000_000,
    )
    assert duplicate.state is TrackingState.MULTI_PERSON_UNSUPPORTED


def test_stale_and_motion_rejection_show_last_known_separately() -> None:
    tracker = AnonymousWorldTracker(_config())
    tracked = tracker.evaluate(
        (_observation(CAMERA_IDS[0], 1_000_000_000, (1.0, 1.0)),),
        _counts(),
        1_010_000_000,
    )
    assert tracked.current_xy_metres is not None
    unknown = tracker.evaluate((), _counts(), 1_110_000_000)
    assert unknown.state is TrackingState.UNKNOWN
    assert unknown.current_xy_metres is None
    assert unknown.last_known_xy_metres == (1.0, 1.0)
    assert unknown.last_known_age_ms == pytest.approx(100.0)
    jump = tracker.evaluate(
        (_observation(CAMERA_IDS[1], 1_200_000_000, (10.0, 10.0)),),
        _counts(),
        1_210_000_000,
    )
    assert jump.state is TrackingState.UNKNOWN
    assert jump.current_xy_metres is None


def test_time_gate_can_reduce_to_single_camera_without_using_stale_peer() -> None:
    tracker = AnonymousWorldTracker(_config())
    result = tracker.evaluate(
        (
            _observation(CAMERA_IDS[0], 1_000_000_000, (1.0, 1.0)),
            _observation(CAMERA_IDS[1], 1_150_000_000, (1.1, 1.0)),
        ),
        _counts(),
        1_160_000_000,
    )
    assert result.state is TrackingState.TRACKED_SINGLE_CAMERA
    assert result.contributing_camera_ids == (CAMERA_IDS[1],)
