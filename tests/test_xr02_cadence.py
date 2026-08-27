from __future__ import annotations

import threading
from time import monotonic_ns

import pytest

from spatial_mapping_phase2.xr02_cadence import (
    DeterministicMultiRateCadence,
    HigherCadenceProfile,
    LatestPendingWorker,
    XR02CadenceError,
    ZeroQueueWorker,
)


def test_higher_cadence_schedule_is_deterministic_and_bounded() -> None:
    profile = HigherCadenceProfile()
    cadence = DeterministicMultiRateCadence(profile)
    decisions = [cadence.decision(index) for index in range(150)]
    assert sum(item.association_due for item in decisions) == 150
    assert sum(item.publication_due for item in decisions) == 38
    assert sum(item.appearance_persistence_due for item in decisions) == 38
    assert profile.appearance_every_n_local_frames == 4
    assert profile.effective_appearance_hz == 2.0
    assert profile.local_track_buffer_frames == 12
    assert profile.as_dict()["appearance_available_hz"] == 8.0
    assert profile.maximum_frame_age_ms == 250.0


def test_higher_cadence_rejects_a_slower_phase_faster_than_local_tracking() -> None:
    with pytest.raises(XR02CadenceError, match="cannot exceed"):
        HigherCadenceProfile(local_tracking_hz=5.0, global_association_hz=6.0)


def test_publication_worker_drops_intermediate_state_without_queue_debt() -> None:
    entered = threading.Event()
    release = threading.Event()
    completed: list[int] = []

    def operation(value: int) -> None:
        entered.set()
        assert release.wait(timeout=1.0)
        completed.append(value)

    worker = ZeroQueueWorker(operation, "cadence-test")
    assert worker.try_submit(1)
    assert entered.wait(timeout=1.0)
    assert not worker.try_submit(2)
    release.set()
    worker.close(wait=True)
    assert completed == [1]
    assert worker.telemetry().busy_dropped_items == 1


def test_latest_pending_worker_processes_only_running_and_newest_pending() -> None:
    entered = threading.Event()
    release = threading.Event()
    completed: list[int] = []
    timestamps = {1: monotonic_ns(), 2: monotonic_ns(), 3: monotonic_ns()}

    def operation(value: int) -> None:
        if value == 1:
            entered.set()
            assert release.wait(timeout=1.0)
        completed.append(value)

    worker = LatestPendingWorker(
        operation,
        "latest-pending-test",
        item_monotonic_ns=timestamps.__getitem__,
        maximum_pending_age_ms=150.0,
    )
    assert worker.try_submit(1)
    assert entered.wait(timeout=1.0)
    assert worker.try_submit(2)
    assert not worker.try_submit(3)
    release.set()
    worker.close(wait=True)
    telemetry = worker.telemetry()
    assert completed == [1, 3]
    assert telemetry.submitted_ticks == 2
    assert telemetry.completed_ticks == 2
    assert telemetry.busy_dropped_ticks == 1
    assert telemetry.pending_replaced_ticks == 1
    assert telemetry.pending_consumed_ticks == 1
    assert not telemetry.busy
    assert not telemetry.pending


def test_latest_pending_worker_rejects_stale_or_invalidated_pending_work() -> None:
    entered = threading.Event()
    release = threading.Event()
    completed: list[int] = []
    timestamps = {1: monotonic_ns(), 2: 0, 3: monotonic_ns()}

    def operation(value: int) -> None:
        entered.set()
        assert release.wait(timeout=1.0)
        completed.append(value)

    worker = LatestPendingWorker(
        operation,
        "stale-pending-test",
        item_monotonic_ns=timestamps.__getitem__,
        maximum_pending_age_ms=150.0,
    )
    assert worker.try_submit(1)
    assert entered.wait(timeout=1.0)
    assert worker.try_submit(2)
    release.set()
    worker.close(wait=True)
    assert completed == [1]
    assert worker.telemetry().pending_stale_dropped_ticks == 1

    entered.clear()
    release.clear()
    completed.clear()
    worker = LatestPendingWorker(
        operation,
        "invalidated-pending-test",
        item_monotonic_ns=timestamps.__getitem__,
        maximum_pending_age_ms=150.0,
    )
    assert worker.try_submit(1)
    assert entered.wait(timeout=1.0)
    assert worker.try_submit(3)
    assert worker.invalidate_pending()
    release.set()
    worker.close(wait=True)
    assert completed == [1]
    assert worker.telemetry().pending_invalidated_ticks == 1
