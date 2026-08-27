from __future__ import annotations

import threading
import time
from time import sleep

import numpy as np
import pytest

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.p09_live_runtime import (
    BoundedInferenceWorker,
    CapturedFrame,
    DecoderPolicy,
    LatestFrameCoordinator,
    LatestFrameSlot,
    P09LiveRuntimeError,
    PersistentPyAvDecoder,
    paired_source_timing,
)
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS, LiveFrameIdentity


def _frame(camera_id: str, acquisition_ns: int, sequence: int = 0) -> CapturedFrame:
    identity = LiveFrameIdentity(
        camera_id,
        f"{camera_id}-frame-{sequence}",
        acquisition_ns,
        "2026-08-20T00:00:00Z",
        None,
        None,
        4,
        3,
    )
    return CapturedFrame(identity, np.full((3, 4, 3), sequence, dtype=np.uint8))


def test_latest_frame_slot_replaces_without_queue_and_rejects_time_reversal() -> None:
    slot = LatestFrameSlot("office-cam-01")
    slot.publish(_frame("office-cam-01", 100, 1))
    slot.publish(_frame("office-cam-01", 200, 2))
    snapshot = slot.snapshot(250, 1.0)
    assert snapshot is not None
    assert snapshot.identity.frame_id.endswith("-2")
    assert slot.telemetry().published_frames == 2
    assert slot.telemetry().replaced_frames == 1
    with pytest.raises(P09LiveRuntimeError, match="backwards"):
        slot.publish(_frame("office-cam-01", 150, 3))


def test_coordinator_reports_fresh_stale_missing_and_cross_camera_skew() -> None:
    slots = {camera_id: LatestFrameSlot(camera_id) for camera_id in CAMERA_IDS}
    slots[CAMERA_IDS[0]].publish(_frame(CAMERA_IDS[0], 1_000_000_000))
    slots[CAMERA_IDS[1]].publish(_frame(CAMERA_IDS[1], 950_000_000))
    slots[CAMERA_IDS[2]].publish(_frame(CAMERA_IDS[2], 100_000_000))
    snapshot = LatestFrameCoordinator(slots).snapshot(1_050_000_000, 200.0)
    assert tuple(frame.identity.camera_id for frame in snapshot.frames) == CAMERA_IDS[:2]
    assert snapshot.stale_camera_ids == (CAMERA_IDS[2],)
    assert snapshot.missing_camera_ids == (CAMERA_IDS[3],)
    assert snapshot.cross_camera_skew_ms == pytest.approx(50.0)


def test_bounded_worker_drops_busy_tick_and_closes_cleanly() -> None:
    started = threading.Event()
    release = threading.Event()

    def operation(snapshot: object) -> None:
        del snapshot
        started.set()
        release.wait(timeout=2.0)

    empty = LatestFrameCoordinator(
        {camera_id: LatestFrameSlot(camera_id) for camera_id in CAMERA_IDS}
    ).snapshot(1, 1.0)
    worker = BoundedInferenceWorker(operation)
    assert worker.try_submit(empty)
    assert started.wait(timeout=1.0)
    assert not worker.try_submit(empty)
    release.set()
    for _ in range(100):
        if not worker.telemetry().busy:
            break
        sleep(0.005)
    telemetry = worker.telemetry()
    assert telemetry.submitted_ticks == 1
    assert telemetry.busy_dropped_ticks == 1
    assert telemetry.completed_ticks == 1
    worker.close()
    with pytest.raises(P09LiveRuntimeError, match="closed"):
        worker.try_submit(empty)


def test_decoder_reconnect_failure_is_credential_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = LocalRtspEndpoint(
        "office-cam-01",
        "PHASE2_RTSP_CAMERA_1",
        "rtsp://user:secret@example.invalid/live",
    )
    slot = LatestFrameSlot("office-cam-01")

    class FailingAv:
        @staticmethod
        def open(*args: object, **kwargs: object) -> object:
            raise TimeoutError("rtsp://user:secret@example.invalid/live")

    monkeypatch.setattr("spatial_mapping_phase2.p09_live_runtime._av", lambda: FailingAv())
    decoder = PersistentPyAvDecoder(
        endpoint,
        slot,
        DecoderPolicy(0.02, 0.02, 0.01),
    )
    decoder.start()
    deadline = time.monotonic() + 0.25
    while decoder.telemetry().reconnects < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    telemetry = decoder.telemetry()
    decoder.close()
    assert telemetry.reconnects >= 2
    assert telemetry.failure_class == "TimeoutError"
    assert "secret" not in repr(telemetry)


def test_source_pts_and_time_base_are_kept_or_omitted_as_a_pair() -> None:
    assert paired_source_timing(None, "1/90000") == (None, None)
    assert paired_source_timing(123, None) == (None, None)
    assert paired_source_timing(123, "1/90000") == (123, "1/90000")
