"""Deterministic multi-rate scheduling for XR02 live tracking.

The scheduler contains no threads or wall-clock sleeps.  It decides which slower
phases are due on each admitted local-tracking tick, so replay and live runs use
the same bounded policy.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from time import monotonic_ns
from typing import Generic, TypeVar


class XR02CadenceError(ValueError):
    """Raised when a multi-rate live profile is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class HigherCadenceProfile:
    """One zero-debt cadence envelope for the office pilot.

    Frequencies are requested rates. Fresh appearance is available to global
    association on every admitted local tick; ``appearance_refresh_hz`` controls
    durable gallery/evidence persistence only.
    """

    local_tracking_hz: float = 8.0
    appearance_refresh_hz: float = 2.0
    global_association_hz: float = 8.0
    publication_hz: float = 2.0
    maximum_frame_age_ms: float = 250.0
    cached_embedding_max_age_seconds: float = 2.0
    local_track_buffer_seconds: float = 1.5
    profile_id: str = "wp4-sustainable-8hz-selective-continuity-v3"

    def __post_init__(self) -> None:
        rates = (
            self.local_tracking_hz,
            self.appearance_refresh_hz,
            self.global_association_hz,
            self.publication_hz,
        )
        if not all(math.isfinite(value) and value > 0 for value in rates):
            raise XR02CadenceError("all live frequencies must be finite and positive")
        if not 1.0 <= self.local_tracking_hz <= 20.0:
            raise XR02CadenceError("local tracking frequency must be within 1..20 Hz")
        if any(value > self.local_tracking_hz for value in rates[1:]):
            raise XR02CadenceError("slower phase frequency cannot exceed local tracking")
        if not 100.0 <= self.maximum_frame_age_ms <= 2_000.0:
            raise XR02CadenceError("maximum frame age must be within 100..2000 ms")
        if not 0.1 <= self.cached_embedding_max_age_seconds <= 10.0:
            raise XR02CadenceError("cached embedding age must be within 0.1..10 seconds")
        if not 0.1 <= self.local_track_buffer_seconds <= 10.0:
            raise XR02CadenceError("local track buffer must be within 0.1..10 seconds")
        if not self.profile_id:
            raise XR02CadenceError("cadence profile identity is required")

    @property
    def appearance_every_n_local_frames(self) -> int:
        return max(1, int(round(self.local_tracking_hz / self.appearance_refresh_hz)))

    @property
    def effective_appearance_hz(self) -> float:
        return self.local_tracking_hz / self.appearance_every_n_local_frames

    @property
    def local_track_buffer_frames(self) -> int:
        return max(1, int(round(self.local_tracking_hz * self.local_track_buffer_seconds)))

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.update(
            {
                "appearance_every_n_local_frames": self.appearance_every_n_local_frames,
                "effective_appearance_hz": self.effective_appearance_hz,
                "appearance_available_hz": self.local_tracking_hz,
                "appearance_persistence_hz": self.effective_appearance_hz,
                "local_track_buffer_frames": self.local_track_buffer_frames,
                "queue_policy": (
                    "one running local tick; one overwriteable latest pending tick; no FIFO debt"
                ),
                "publication_policy": "rate-limited same-state projection; no tracking authority",
            }
        )
        return value


@dataclass(frozen=True, slots=True)
class CadenceDecision:
    local_tick_index: int
    appearance_persistence_due: bool
    association_due: bool
    publication_due: bool


class DeterministicMultiRateCadence:
    """Map one local tick stream onto deterministic slower phase clocks."""

    def __init__(self, profile: HigherCadenceProfile) -> None:
        self.profile = profile

    def decision(self, local_tick_index: int) -> CadenceDecision:
        if local_tick_index < 0:
            raise XR02CadenceError("local tick index must be non-negative")
        return CadenceDecision(
            local_tick_index=local_tick_index,
            appearance_persistence_due=_is_due(
                local_tick_index,
                self.profile.local_tracking_hz,
                self.profile.appearance_refresh_hz,
            ),
            association_due=_is_due(
                local_tick_index,
                self.profile.local_tracking_hz,
                self.profile.global_association_hz,
            ),
            publication_due=_is_due(
                local_tick_index,
                self.profile.local_tracking_hz,
                self.profile.publication_hz,
            ),
        )


def _is_due(tick_index: int, source_hz: float, target_hz: float) -> bool:
    if tick_index == 0:
        return True
    previous_bucket = math.floor((tick_index - 1) * target_hz / source_hz)
    current_bucket = math.floor(tick_index * target_hz / source_hz)
    return current_bucket > previous_bucket


@dataclass(frozen=True, slots=True)
class ZeroQueueTelemetry:
    submitted_items: int
    busy_dropped_items: int
    completed_items: int
    failed_items: int
    busy: bool


@dataclass(frozen=True, slots=True)
class LatestPendingTelemetry:
    """Accounting for one running item plus one overwriteable latest pending item."""

    submitted_ticks: int
    busy_dropped_ticks: int
    completed_ticks: int
    failed_ticks: int
    busy: bool
    pending: bool
    pending_stored_ticks: int
    pending_replaced_ticks: int
    pending_consumed_ticks: int
    pending_stale_dropped_ticks: int
    pending_invalidated_ticks: int
    maximum_pending_age_ms: float


_T = TypeVar("_T")


class LatestPendingWorker(Generic[_T]):
    """Run one item and retain only the newest bounded-age pending item.

    This is deliberately not a FIFO. A busy submission occupies the sole
    pending slot, and every later submission replaces that pending item. The
    worker drains the latest pending item immediately after the running item
    only while it remains inside the configured age bound.
    """

    def __init__(
        self,
        operation: Callable[[_T], None],
        name: str,
        *,
        item_monotonic_ns: Callable[[_T], int],
        maximum_pending_age_ms: float,
    ) -> None:
        if (
            not math.isfinite(maximum_pending_age_ms)
            or not 1.0 <= maximum_pending_age_ms <= 1_000.0
        ):
            raise XR02CadenceError("pending-item age bound must be within 1..1000 ms")
        self._operation = operation
        self._item_monotonic_ns = item_monotonic_ns
        self._maximum_pending_age_ns = int(round(maximum_pending_age_ms * 1_000_000.0))
        self._maximum_pending_age_ms = maximum_pending_age_ms
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._pending: tuple[_T, int, int] | None = None
        self._invalidation_generation = 0
        self._submitted = 0
        self._dropped = 0
        self._completed = 0
        self._failed = 0
        self._pending_stored = 0
        self._pending_replaced = 0
        self._pending_consumed = 0
        self._pending_stale_dropped = 0
        self._pending_invalidated = 0

    def try_submit(self, item: _T) -> bool:
        """Submit immediately or replace the one pending item.

        Returns ``False`` only when this submission replaces an older pending
        item. The new item is still retained as the latest pending work.
        """

        item_timestamp_ns = self._item_monotonic_ns(item)
        if item_timestamp_ns < 0:
            raise XR02CadenceError("pending-item monotonic timestamp must be non-negative")
        with self._lock:
            if self._closed:
                raise XR02CadenceError("latest-pending worker is closed")
            if self._busy:
                replaced = self._pending is not None
                if replaced:
                    self._dropped += 1
                    self._pending_replaced += 1
                self._pending = (
                    item,
                    item_timestamp_ns,
                    self._invalidation_generation,
                )
                self._pending_stored += 1
                return not replaced
            self._busy = True
            self._submitted += 1
            submission_generation = self._invalidation_generation
        self._executor.submit(self._run, item, submission_generation)
        return True

    def invalidate_pending(self) -> bool:
        """Discard pending work after a capture/scene discontinuity."""

        with self._lock:
            self._invalidation_generation += 1
            if self._pending is None:
                return False
            self._pending = None
            self._dropped += 1
            self._pending_invalidated += 1
            return True

    def _run(self, item: _T, submission_generation: int) -> None:
        current = item
        current_generation = submission_generation
        while True:
            with self._lock:
                if current_generation != self._invalidation_generation:
                    self._dropped += 1
                    self._pending_invalidated += 1
                    pending = self._pending
                    self._pending = None
                    if pending is None:
                        self._busy = False
                        return
                    current, pending_timestamp_ns, current_generation = pending
                    age_ns = monotonic_ns() - pending_timestamp_ns
                    if age_ns < 0 or age_ns > self._maximum_pending_age_ns:
                        self._dropped += 1
                        self._pending_stale_dropped += 1
                        self._busy = False
                        return
                    self._pending_consumed += 1
                    self._submitted += 1
            try:
                self._operation(current)
            except Exception:
                with self._lock:
                    self._failed += 1
            else:
                with self._lock:
                    self._completed += 1
            with self._lock:
                pending = self._pending
                self._pending = None
                if pending is None:
                    self._busy = False
                    return
                pending_item, pending_timestamp_ns, pending_generation = pending
                age_ns = monotonic_ns() - pending_timestamp_ns
                if age_ns < 0 or age_ns > self._maximum_pending_age_ns:
                    self._dropped += 1
                    self._pending_stale_dropped += 1
                    self._busy = False
                    return
                self._pending_consumed += 1
                self._submitted += 1
                current = pending_item
                current_generation = pending_generation

    def telemetry(self) -> LatestPendingTelemetry:
        with self._lock:
            return LatestPendingTelemetry(
                self._submitted,
                self._dropped,
                self._completed,
                self._failed,
                self._busy,
                self._pending is not None,
                self._pending_stored,
                self._pending_replaced,
                self._pending_consumed,
                self._pending_stale_dropped,
                self._pending_invalidated,
                self._maximum_pending_age_ms,
            )

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)


class ZeroQueueWorker(Generic[_T]):
    """One running operation and no queue; slow consumers drop intermediate state."""

    def __init__(self, operation: Callable[[_T], None], name: str) -> None:
        self._operation = operation
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._submitted = 0
        self._dropped = 0
        self._completed = 0
        self._failed = 0

    def try_submit(self, item: _T) -> bool:
        with self._lock:
            if self._closed:
                raise XR02CadenceError("zero-queue worker is closed")
            if self._busy:
                self._dropped += 1
                return False
            self._busy = True
            self._submitted += 1
        self._executor.submit(self._run, item)
        return True

    def _run(self, item: _T) -> None:
        try:
            self._operation(item)
        except Exception:
            with self._lock:
                self._failed += 1
            raise
        else:
            with self._lock:
                self._completed += 1
        finally:
            with self._lock:
                self._busy = False

    def telemetry(self) -> ZeroQueueTelemetry:
        with self._lock:
            return ZeroQueueTelemetry(
                self._submitted,
                self._dropped,
                self._completed,
                self._failed,
                self._busy,
            )

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)
