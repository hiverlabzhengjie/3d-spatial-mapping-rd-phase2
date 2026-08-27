from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import pytest

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.xr02_rtsp_capture import (
    _SHARED_HEADER_BYTES,
    CaptureBackend,
    CaptureEventKind,
    GenerationAwareLatestFrameSlot,
    SupervisedDecoderPolicy,
    SupervisedRtspDecoder,
    XR02CaptureError,
    _read_shared_frame,
    _SharedFrameWriter,
)


def test_generation_slot_invalidation_prevents_old_frame_reuse() -> None:
    from spatial_mapping_phase2.p09_live_runtime import CapturedFrame
    from spatial_mapping_phase2.p09_tracking_domain import LiveFrameIdentity

    acquired = time.monotonic_ns()
    slot = GenerationAwareLatestFrameSlot("office-cam-01")
    slot.publish(
        CapturedFrame(
            LiveFrameIdentity(
                "office-cam-01",
                "office-cam-01-g000001-000000000000",
                acquired,
                datetime.now(UTC).isoformat(),
                None,
                None,
                2,
                2,
            ),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )
    assert slot.snapshot(acquired + 1, 1_000.0) is not None
    slot.invalidate()
    assert slot.snapshot(acquired + 2, 1_000.0) is None
    assert slot.invalidations == 1


def test_policy_rejects_watchdog_that_cannot_outlive_backend_timeout() -> None:
    with pytest.raises(XR02CaptureError, match="frame-stall"):
        SupervisedDecoderPolicy(read_timeout_seconds=2.0, frame_stall_timeout_seconds=1.0)


def test_shared_frame_transport_round_trips_latest_native_frame_without_pickle() -> None:
    capacity = 4 * 3 * 3
    memory = shared_memory.SharedMemory(create=True, size=_SHARED_HEADER_BYTES + capacity)
    ready = threading.Event()
    writer = _SharedFrameWriter.from_payload(
        {
            "shared_frame_name": memory.name,
            "shared_frame_capacity_bytes": capacity,
            "shared_frame_ready_event": ready,
        }
    )
    assert writer is not None
    frame = np.arange(capacity, dtype=np.uint8).reshape(3, 4, 3)
    try:
        writer.publish(
            {
                "generation": 2,
                "sequence": 7,
                "acquisition_monotonic_ns": 123_000,
                "observed_at_utc_ns": 1_700_000_000_123_456_000,
                "decoded_frames": 8,
                "source_pts": 9,
                "source_time_base": "1/25",
                "frame_bgr": frame,
            }
        )
        assert ready.is_set()
        message = _read_shared_frame(memory)
        assert message is not None
        assert message["generation"] == 2
        assert message["sequence"] == 7
        assert message["source_time_base"] == "1/25"
        received = message["frame_bgr"]
        assert isinstance(received, np.ndarray)
        np.testing.assert_array_equal(received, frame)
    finally:
        writer.close()
        memory.close()
        memory.unlink()


def test_gstreamer_requires_explicit_isolated_overlay() -> None:
    endpoint = LocalRtspEndpoint("office-cam-01", "PHASE2_RTSP_CAMERA_1", "rtsp://127.0.0.1/test")
    slot = GenerationAwareLatestFrameSlot("office-cam-01")
    with pytest.raises(XR02CaptureError, match="overlay"):
        SupervisedRtspDecoder(
            endpoint,
            slot,
            SupervisedDecoderPolicy(),
            CaptureBackend.GSTREAMER,
        )


def test_supervisor_restarts_stalled_media_and_changes_generation() -> None:
    endpoint = LocalRtspEndpoint(
        "office-cam-01",
        "PHASE2_RTSP_CAMERA_1",
        "rtsp://credential-must-not-appear@example.test/stream",
    )
    slot = GenerationAwareLatestFrameSlot("office-cam-01")
    epochs: list[tuple[str, int, int, str]] = []
    policy = SupervisedDecoderPolicy(
        connect_timeout_seconds=0.05,
        read_timeout_seconds=0.05,
        heartbeat_interval_seconds=0.02,
        heartbeat_timeout_seconds=0.80,
        first_frame_timeout_seconds=0.80,
        frame_stall_timeout_seconds=0.16,
        graceful_stop_seconds=0.05,
        restart_delay_seconds=0.02,
        maximum_restart_delay_seconds=0.04,
    )
    decoder = SupervisedRtspDecoder(
        endpoint,
        slot,
        policy,
        epoch_callback=lambda *value: epochs.append(value),
        worker_target=_one_frame_then_media_stall,
    )
    decoder.start()
    deadline = time.monotonic() + 3.0
    try:
        while time.monotonic() < deadline:
            telemetry = decoder.telemetry()
            if telemetry.generation >= 2 and telemetry.delivered_frames >= 2:
                break
            time.sleep(0.02)
        telemetry = decoder.telemetry()
        assert telemetry.generation >= 2
        assert telemetry.tracker_epoch == telemetry.generation
        assert telemetry.watchdog_restarts >= 1
        assert telemetry.restart_reason == "watchdog_frame_stall_timeout"
        assert len(epochs) >= 2
        assert epochs[0][1:3] == (1, 1)
        assert epochs[1][1:3] == (2, 2)
        fresh = slot.snapshot(time.monotonic_ns(), 1_000.0)
        assert fresh is not None
        assert "-g000002-" in fresh.identity.frame_id
        events = decoder.events()
        assert any(item.kind is CaptureEventKind.WATCHDOG_RESTART for item in events)
        assert any(item.kind is CaptureEventKind.RECOVERED for item in events)
        assert "credential-must-not-appear" not in repr(events)
    finally:
        decoder.close()


def _one_frame_then_media_stall(
    camera_id: str,
    endpoint_url: str,
    generation: int,
    payload: dict[str, object],
    stop_event: Any,
    frame_queue: Any,
    event_queue: Any,
) -> None:
    del camera_id, endpoint_url
    frame_queue.put(
        {
            "generation": generation,
            "sequence": 0,
            "acquisition_monotonic_ns": time.monotonic_ns(),
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "source_pts": None,
            "source_time_base": None,
            "decoded_frames": generation,
            "frame_bgr": np.full((2, 2, 3), generation, dtype=np.uint8),
        }
    )
    interval_value = payload["heartbeat_interval_seconds"]
    assert isinstance(interval_value, int | float)
    interval = float(interval_value)
    while not stop_event.wait(interval):
        try:
            event_queue.put_nowait({"kind": "heartbeat", "decoded_frames": generation})
        except Exception:
            return
