from __future__ import annotations

from dataclasses import replace

import pytest

from spatial_mapping_phase2.p01_observability import CAMERA_ENDPOINT_KEYS, CAMERA_IDS
from spatial_mapping_phase2.p03_capture_domain import (
    BUNDLE_SCHEMA,
    PROFILE_SCHEMA,
    SESSION_SCHEMA,
    BundleStatus,
    CameraCaptureResult,
    CaptureArtifact,
    CaptureSessionManifest,
    CaptureStatus,
    P03ContractError,
    RationalTimeBase,
    SourceFrame,
    StorageMode,
    StreamProfileIdentity,
    select_closest_bundle,
)

NOW = "2026-08-14T08:00:00Z"
HASH = "a" * 64


def profile(camera_id: str) -> StreamProfileIdentity:
    return StreamProfileIdentity(
        PROFILE_SCHEMA,
        camera_id,
        "stream-profile-v1",
        CAMERA_ENDPOINT_KEYS[camera_id],
        f"binding-{camera_id}",
        1920,
        1080,
        "h264",
        RationalTimeBase(1, 90000),
        None,
        None,
        NOW,
    )


def result(camera_id: str, times: tuple[int, ...]) -> CameraCaptureResult:
    frames = tuple(
        SourceFrame(
            f"{camera_id}-f{index}",
            camera_id,
            "stream-profile-v1",
            index * 3000,
            RationalTimeBase(1, 90000),
            time,
            NOW,
        )
        for index, time in enumerate(times)
    )
    return CameraCaptureResult(
        camera_id,
        profile(camera_id),
        CaptureStatus.CAPTURED,
        frames,
        CaptureArtifact(
            f"captures/p03/s/{camera_id}.mp4",
            HASH,
            10,
            StorageMode.PACKET_PRESERVING_MP4,
            False,
            None,
        ),
        (),
        None,
        None,
    )


def session(results: tuple[CameraCaptureResult, ...]) -> CaptureSessionManifest:
    return CaptureSessionManifest(SESSION_SCHEMA, "session-1", NOW, 1, 1000, "test", HASH, results)


def test_time_base_keeps_original_rational_conversion() -> None:
    assert RationalTimeBase(1, 90000).seconds_for_pts(45000) == 0.5
    with pytest.raises(P03ContractError, match="pts"):
        RationalTimeBase(1, 90000).seconds_for_pts(1.2)  # type: ignore[arg-type]


def test_profile_change_changes_compatibility_identity() -> None:
    original = profile(CAMERA_IDS[0])
    assert replace(original, width_pixels=1280).compatibility_key != original.compatibility_key
    assert (
        replace(original, endpoint_binding_id="binding-new").compatibility_key
        != original.compatibility_key
    )


def test_decoded_fallback_requires_explicit_provenance() -> None:
    with pytest.raises(P03ContractError, match="manifested reason"):
        CaptureArtifact(
            "captures/p03/a.jpg", HASH, 1, StorageMode.DECODED_FRAME_FALLBACK, False, None
        )


def test_selection_is_closest_reports_pairwise_and_overall_skew() -> None:
    results = tuple(result(camera_id, (100, 200, 300)) for camera_id in CAMERA_IDS)
    adjusted = tuple(
        replace(
            item,
            frames=tuple(
                replace(
                    frame,
                    acquisition_monotonic_ns=frame.acquisition_monotonic_ns + i * 10,
                )
                for frame in item.frames
            ),
        )
        for i, item in enumerate(results)
    )
    bundle = select_closest_bundle(session(adjusted), "bundle-1", 220)
    assert bundle.schema_version == BUNDLE_SCHEMA
    assert bundle.status is BundleStatus.COMPLETE
    assert bundle.overall_skew_ns == 30
    assert len(bundle.pairwise_skew) == 6


def test_tie_uses_lexicographically_first_frame_id() -> None:
    first = result(CAMERA_IDS[0], (100, 300))
    bundle = select_closest_bundle(session((first,)), "bundle-1", 200)
    assert bundle.selected_frames[0].frame_id.endswith("f0")
    assert bundle.status is BundleStatus.PARTIAL
    assert bundle.missing_camera_ids == CAMERA_IDS[1:]


def test_missing_and_empty_sessions_are_explicit() -> None:
    empty = select_closest_bundle(session(()), "bundle-empty")
    assert empty.status is BundleStatus.REJECTED
    assert empty.missing_camera_ids == CAMERA_IDS


def test_stale_frame_profile_is_rejected_as_incompatible() -> None:
    item = result(CAMERA_IDS[0], (100,))
    stale_frame = replace(item.frames[0], profile_version="stream-profile-v2")
    bundle = select_closest_bundle(session((replace(item, frames=(stale_frame,)),)), "bundle-x")
    assert bundle.status is BundleStatus.INCOMPATIBLE_PROFILES


def test_session_rejects_duplicate_camera_results() -> None:
    item = result(CAMERA_IDS[0], (100,))
    with pytest.raises(P03ContractError, match="unique"):
        session((item, item))
