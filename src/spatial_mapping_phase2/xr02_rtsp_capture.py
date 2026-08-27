"""Process-isolated RTSP capture with explicit freshness and discontinuity epochs.

This module deliberately keeps capture authority separate from tracking.  A camera
worker may decode quickly, stall inside a native backend, exit, or reconnect, but it
can never silently extend a previous tracker epoch across a hard discontinuity.
"""

from __future__ import annotations

import importlib
import math
import multiprocessing as mp
import queue
import site
import struct
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from time import monotonic_ns, sleep, time_ns
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.p09_live_runtime import (
    CapturedFrame,
    LatestFrameSlot,
    SlotTelemetry,
    paired_source_timing,
)
from spatial_mapping_phase2.p09_tracking_domain import LiveFrameIdentity


class XR02CaptureError(RuntimeError):
    """Raised when the supervised capture contract cannot be maintained."""


class CaptureBackend(StrEnum):
    PYAV = "pyav"
    GSTREAMER = "gstreamer"


class CaptureProcessState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    CURRENT = "current"
    STALE = "stale"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class CaptureEventKind(StrEnum):
    GENERATION_STARTED = "generation_started"
    FIRST_FRAME = "first_frame"
    FRAME_STALE = "frame_stale"
    WORKER_EXITED = "worker_exited"
    WATCHDOG_RESTART = "watchdog_restart"
    RECOVERED = "recovered"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class SupervisedDecoderPolicy:
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 2.0
    heartbeat_interval_seconds: float = 0.25
    heartbeat_timeout_seconds: float = 1.5
    first_frame_timeout_seconds: float = 12.0
    frame_stall_timeout_seconds: float = 2.5
    graceful_stop_seconds: float = 0.75
    restart_delay_seconds: float = 0.50
    maximum_restart_delay_seconds: float = 4.0
    gstreamer_latency_ms: int = 250
    shared_frame_capacity_bytes: int = 1920 * 1080 * 3

    def __post_init__(self) -> None:
        seconds = (
            self.connect_timeout_seconds,
            self.read_timeout_seconds,
            self.heartbeat_interval_seconds,
            self.heartbeat_timeout_seconds,
            self.first_frame_timeout_seconds,
            self.frame_stall_timeout_seconds,
            self.graceful_stop_seconds,
            self.restart_delay_seconds,
            self.maximum_restart_delay_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in seconds):
            raise XR02CaptureError("capture timeouts must be finite and positive")
        if self.heartbeat_timeout_seconds <= self.heartbeat_interval_seconds:
            raise XR02CaptureError("heartbeat timeout must exceed its interval")
        if self.frame_stall_timeout_seconds <= self.read_timeout_seconds:
            raise XR02CaptureError("frame-stall timeout must exceed backend read timeout")
        if self.maximum_restart_delay_seconds < self.restart_delay_seconds:
            raise XR02CaptureError("maximum restart delay cannot be shorter than its base")
        if not 0 <= self.gstreamer_latency_ms <= 5_000:
            raise XR02CaptureError("GStreamer latency must be within 0..5000 ms")
        if self.shared_frame_capacity_bytes < 640 * 480 * 3:
            raise XR02CaptureError("shared-frame capacity is too small for a live camera")


@dataclass(frozen=True, slots=True)
class CaptureEvent:
    camera_id: str
    generation: int
    tracker_epoch: int
    kind: CaptureEventKind
    occurred_monotonic_ns: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "generation": self.generation,
            "tracker_epoch": self.tracker_epoch,
            "kind": self.kind.value,
            "occurred_monotonic_ns": self.occurred_monotonic_ns,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SupervisedDecoderTelemetry:
    camera_id: str
    backend: CaptureBackend
    state: CaptureProcessState
    generation: int
    tracker_epoch: int
    decoded_frames: int
    delivered_frames: int
    replaced_before_delivery: int
    reconnects: int
    watchdog_restarts: int
    failure_class: str | None
    failure_detail: str | None
    restart_reason: str | None
    last_process_heartbeat_monotonic_ns: int | None
    last_acquisition_monotonic_ns: int | None
    running: bool


class GenerationAwareLatestFrameSlot(LatestFrameSlot):
    """P09-compatible latest slot that can discard a disconnected generation."""

    def __init__(self, camera_id: str) -> None:
        super().__init__(camera_id)
        self._invalidations = 0

    def invalidate(self) -> None:
        with self._lock:
            self._latest = None
            self._invalidations += 1

    @property
    def invalidations(self) -> int:
        with self._lock:
            return self._invalidations

    def telemetry(self) -> SlotTelemetry:
        return super().telemetry()


class _ProcessLike(Protocol):
    exitcode: int | None

    def start(self) -> None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


WorkerTarget = Callable[[str, str, int, dict[str, object], Any, Any, Any], None]
EpochCallback = Callable[[str, int, int, str], None]

_SHARED_METADATA_FORMAT = "<10q"
_SHARED_HEADER_BYTES = 8 + struct.calcsize(_SHARED_METADATA_FORMAT)
_MISSING_INT64 = -(2**63)


class _SharedFrameWriter:
    """Child-owned seqlock writer for one native BGR latest-frame buffer."""

    def __init__(self, memory: Any, capacity_bytes: int, ready_event: Any | None = None) -> None:
        self._memory = memory
        self._capacity_bytes = capacity_bytes
        self._ready_event = ready_event

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> _SharedFrameWriter | None:
        name = payload.get("shared_frame_name")
        if not isinstance(name, str) or not name:
            return None
        from multiprocessing import shared_memory

        capacity = _required_int(payload, "shared_frame_capacity_bytes")
        return cls(
            shared_memory.SharedMemory(name=name),
            capacity,
            payload.get("shared_frame_ready_event"),
        )

    def publish(self, message: dict[str, object]) -> None:
        frame = message.get("frame_bgr")
        if not isinstance(frame, np.ndarray):
            raise XR02CaptureError("shared-frame publication requires an ndarray")
        image = np.ascontiguousarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise XR02CaptureError("shared-frame publication requires BGR image shape")
        if image.nbytes > self._capacity_bytes:
            raise XR02CaptureError("decoded frame exceeds configured shared-memory capacity")
        source_pts = message.get("source_pts")
        encoded_pts = source_pts if isinstance(source_pts, int) else _MISSING_INT64
        time_base_num, time_base_den = _time_base_parts(message.get("source_time_base"))
        current = struct.unpack_from("<q", self._memory.buf, 0)[0]
        odd_version = current + 1 if current % 2 == 0 else current + 2
        struct.pack_into("<q", self._memory.buf, 0, odd_version)
        struct.pack_into(
            _SHARED_METADATA_FORMAT,
            self._memory.buf,
            8,
            _required_int(message, "generation"),
            _required_int(message, "sequence"),
            _required_int(message, "acquisition_monotonic_ns"),
            _required_int(message, "observed_at_utc_ns"),
            _required_int(message, "decoded_frames"),
            int(image.shape[0]),
            int(image.shape[1]),
            encoded_pts,
            time_base_num,
            time_base_den,
        )
        target: NDArray[np.uint8] = np.ndarray(
            (image.nbytes,),
            dtype=np.uint8,
            buffer=self._memory.buf,
            offset=_SHARED_HEADER_BYTES,
        )
        target[:] = image.reshape(-1)
        struct.pack_into("<q", self._memory.buf, 0, odd_version + 1)
        if self._ready_event is not None:
            self._ready_event.set()

    def close(self) -> None:
        self._memory.close()


def _read_shared_frame(memory: Any) -> dict[str, object] | None:
    """Copy one consistent seqlock snapshot or return no newly stable frame."""

    for _attempt in range(3):
        version_before = struct.unpack_from("<q", memory.buf, 0)[0]
        if version_before <= 0 or version_before % 2:
            return None
        values = struct.unpack_from(_SHARED_METADATA_FORMAT, memory.buf, 8)
        (
            generation,
            sequence,
            acquired,
            observed_utc_ns,
            decoded_frames,
            height,
            width,
            source_pts,
            time_base_num,
            time_base_den,
        ) = values
        frame_bytes = height * width * 3
        if height <= 0 or width <= 0 or frame_bytes > len(memory.buf) - _SHARED_HEADER_BYTES:
            raise XR02CaptureError("shared-frame metadata exceeds its allocated buffer")
        frame: NDArray[np.uint8] = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=memory.buf,
            offset=_SHARED_HEADER_BYTES,
        ).copy()
        version_after = struct.unpack_from("<q", memory.buf, 0)[0]
        if version_before != version_after or version_after % 2:
            continue
        observed = datetime.fromtimestamp(observed_utc_ns // 1_000_000_000, UTC) + timedelta(
            microseconds=(observed_utc_ns % 1_000_000_000) // 1_000
        )
        return {
            "generation": generation,
            "sequence": sequence,
            "acquisition_monotonic_ns": acquired,
            "observed_at_utc": observed.isoformat(),
            "source_pts": None if source_pts == _MISSING_INT64 else source_pts,
            "source_time_base": (
                None if time_base_den == 0 else f"{time_base_num}/{time_base_den}"
            ),
            "decoded_frames": decoded_frames,
            "frame_bgr": frame,
        }
    return None


def _time_base_parts(value: object) -> tuple[int, int]:
    if value is None:
        return 0, 0
    if not isinstance(value, str) or "/" not in value:
        raise XR02CaptureError("source time base must be a rational string")
    numerator, denominator = value.split("/", 1)
    try:
        result = (int(numerator), int(denominator))
    except ValueError as error:
        raise XR02CaptureError("source time base is malformed") from error
    if result[1] == 0:
        raise XR02CaptureError("source time-base denominator cannot be zero")
    return result


@dataclass(frozen=True, slots=True, repr=False)
class _WorkerSettings:
    camera_id: str
    endpoint_url: str
    backend: CaptureBackend
    generation: int
    policy: SupervisedDecoderPolicy
    gstreamer_overlay_path: str | None

    def child_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "connect_timeout_seconds": self.policy.connect_timeout_seconds,
            "read_timeout_seconds": self.policy.read_timeout_seconds,
            "heartbeat_interval_seconds": self.policy.heartbeat_interval_seconds,
            "gstreamer_latency_ms": self.policy.gstreamer_latency_ms,
            "gstreamer_overlay_path": self.gstreamer_overlay_path,
            "shared_frame_capacity_bytes": self.policy.shared_frame_capacity_bytes,
        }


class SupervisedRtspDecoder:
    """Supervise one credential-safe decoder process and publish only its latest frame."""

    def __init__(
        self,
        endpoint: LocalRtspEndpoint,
        slot: GenerationAwareLatestFrameSlot,
        policy: SupervisedDecoderPolicy,
        backend: CaptureBackend = CaptureBackend.PYAV,
        *,
        gstreamer_overlay_path: Path | None = None,
        epoch_callback: EpochCallback | None = None,
        worker_target: WorkerTarget | None = None,
    ) -> None:
        if endpoint.camera_id != slot.camera_id:
            raise XR02CaptureError("capture endpoint and latest-frame slot disagree")
        if backend is CaptureBackend.GSTREAMER and gstreamer_overlay_path is None:
            raise XR02CaptureError("GStreamer backend requires an explicit isolated overlay")
        if gstreamer_overlay_path is not None and not gstreamer_overlay_path.is_dir():
            raise XR02CaptureError("GStreamer overlay is unavailable")
        self._endpoint = endpoint
        self._slot = slot
        self._policy = policy
        self._backend = backend
        self._gstreamer_overlay = (
            None if gstreamer_overlay_path is None else str(gstreamer_overlay_path.resolve())
        )
        self._epoch_callback = epoch_callback
        self._worker_target = worker_target or _capture_worker_entry
        self._shared_memory_enabled = worker_target is None
        self._context = mp.get_context("spawn")
        self._supervisor_stop = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._child_stop: Any | None = None
        self._frame_queue: Any | None = None
        self._shared_frame: Any | None = None
        self._shared_frame_ready: Any | None = None
        self._event_queue: Any | None = None
        self._process: _ProcessLike | None = None
        self._lock = threading.Lock()
        self._state = CaptureProcessState.STOPPED
        self._generation = 0
        self._tracker_epoch = 0
        self._decoded_frames = 0
        self._delivered_frames = 0
        self._replaced_before_delivery = 0
        self._reconnects = 0
        self._watchdog_restarts = 0
        self._failure_class: str | None = None
        self._failure_detail: str | None = None
        self._restart_reason: str | None = None
        self._last_heartbeat_ns: int | None = None
        self._last_acquisition_ns: int | None = None
        self._generation_started_ns: int | None = None
        self._generation_first_frame = False
        self._generation_decoded_base = 0
        self._last_transport_sequence = -1
        self._events: list[CaptureEvent] = []

    def start(self) -> None:
        if self._supervisor is not None:
            raise XR02CaptureError("capture supervisor is already started")
        self._supervisor_stop.clear()
        self._supervisor = threading.Thread(
            target=self._supervise,
            name=f"xr02-capture-supervisor-{self._endpoint.camera_id}",
            daemon=True,
        )
        self._supervisor.start()

    def close(self) -> None:
        self._supervisor_stop.set()
        supervisor = self._supervisor
        if supervisor is not None:
            supervisor.join(
                timeout=self._policy.graceful_stop_seconds
                + self._policy.heartbeat_timeout_seconds
                + 2.0
            )
            if supervisor.is_alive():
                raise XR02CaptureError("capture supervisor exceeded bounded shutdown")
        self._supervisor = None

    def telemetry(self) -> SupervisedDecoderTelemetry:
        with self._lock:
            return SupervisedDecoderTelemetry(
                camera_id=self._endpoint.camera_id,
                backend=self._backend,
                state=self._state,
                generation=self._generation,
                tracker_epoch=self._tracker_epoch,
                decoded_frames=self._decoded_frames,
                delivered_frames=self._delivered_frames,
                replaced_before_delivery=self._replaced_before_delivery,
                reconnects=self._reconnects,
                watchdog_restarts=self._watchdog_restarts,
                failure_class=self._failure_class,
                failure_detail=self._failure_detail,
                restart_reason=self._restart_reason,
                last_process_heartbeat_monotonic_ns=self._last_heartbeat_ns,
                last_acquisition_monotonic_ns=self._last_acquisition_ns,
                running=self._supervisor is not None and self._supervisor.is_alive(),
            )

    def events(self) -> tuple[CaptureEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def frame_transport(self) -> str:
        return "shared_memory_seqlock" if self._shared_memory_enabled else "queue_test_adapter"

    def _supervise(self) -> None:
        restart_attempt = 0
        reason = "initial_start"
        try:
            while not self._supervisor_stop.is_set():
                self._start_generation(reason)
                restart_reason = self._monitor_generation()
                if restart_reason is None:
                    break
                with self._lock:
                    generation_was_current = self._generation_first_frame
                if generation_was_current:
                    restart_attempt = 0
                restart_attempt += 1
                self._slot.invalidate()
                self._stop_generation()
                with self._lock:
                    self._state = CaptureProcessState.RECONNECTING
                    self._reconnects += 1
                    self._restart_reason = restart_reason
                    if restart_reason.startswith("watchdog_"):
                        self._watchdog_restarts += 1
                self._record_event(
                    CaptureEventKind.WATCHDOG_RESTART
                    if restart_reason.startswith("watchdog_")
                    else CaptureEventKind.WORKER_EXITED,
                    restart_reason,
                )
                delay = min(
                    self._policy.maximum_restart_delay_seconds,
                    self._policy.restart_delay_seconds * (2 ** min(restart_attempt - 1, 3)),
                )
                if self._supervisor_stop.wait(delay):
                    break
                reason = restart_reason
        except Exception as error:
            with self._lock:
                self._state = CaptureProcessState.FAILED
                self._failure_class = type(error).__name__
        finally:
            self._stop_generation()
            self._slot.invalidate()
            with self._lock:
                if self._state is not CaptureProcessState.FAILED:
                    self._state = CaptureProcessState.STOPPED
            self._record_event(CaptureEventKind.STOPPED, "supervisor_stopped")

    def _start_generation(self, reason: str) -> None:
        self._generation += 1
        self._tracker_epoch += 1
        self._slot.invalidate()
        self._child_stop = self._context.Event()
        if self._shared_memory_enabled:
            from multiprocessing import shared_memory

            self._shared_frame = shared_memory.SharedMemory(
                create=True,
                size=_SHARED_HEADER_BYTES + self._policy.shared_frame_capacity_bytes,
            )
            struct.pack_into("<q", cast(bytearray, self._shared_frame.buf), 0, 0)
            self._shared_frame_ready = self._context.Event()
            self._frame_queue = None
        else:
            self._shared_frame_ready = None
            self._frame_queue = self._context.Queue(maxsize=1)
        self._event_queue = self._context.Queue(maxsize=64)
        settings = _WorkerSettings(
            self._endpoint.camera_id,
            self._endpoint.for_read_only_adapter(),
            self._backend,
            self._generation,
            self._policy,
            self._gstreamer_overlay,
        )
        payload = settings.child_payload()
        if self._shared_frame is not None:
            payload["shared_frame_name"] = self._shared_frame.name
            payload["shared_frame_ready_event"] = self._shared_frame_ready
        process = cast(
            _ProcessLike,
            self._context.Process(
                target=self._worker_target,
                args=(
                    settings.camera_id,
                    settings.endpoint_url,
                    settings.generation,
                    payload,
                    self._child_stop,
                    self._frame_queue,
                    self._event_queue,
                ),
                name=f"xr02-{self._backend.value}-{self._endpoint.camera_id}",
                daemon=True,
            ),
        )
        now = monotonic_ns()
        with self._lock:
            self._process = process
            self._state = CaptureProcessState.STARTING
            self._generation_started_ns = now
            self._generation_first_frame = False
            self._generation_decoded_base = self._decoded_frames
            self._last_transport_sequence = -1
            self._last_heartbeat_ns = now
        process.start()
        self._record_event(CaptureEventKind.GENERATION_STARTED, reason)
        if self._epoch_callback is not None:
            self._epoch_callback(
                self._endpoint.camera_id,
                self._generation,
                self._tracker_epoch,
                reason,
            )

    def _monitor_generation(self) -> str | None:
        while not self._supervisor_stop.is_set():
            ready = self._shared_frame_ready
            if ready is None:
                if self._supervisor_stop.wait(0.025):
                    break
            else:
                ready.wait(0.025)
                ready.clear()
                if self._supervisor_stop.is_set():
                    break
            self._drain_events()
            self._drain_latest_frame()
            now = monotonic_ns()
            with self._lock:
                process = self._process
                heartbeat = self._last_heartbeat_ns
                generation_started = self._generation_started_ns
                last_frame = self._last_acquisition_ns
                first_frame = self._generation_first_frame
            if process is None or not process.is_alive():
                exitcode = None if process is None else process.exitcode
                with self._lock:
                    failure_class = self._failure_class
                if failure_class is not None:
                    return f"backend_failure_{failure_class}"
                return f"worker_exit_{exitcode}"
            if (
                heartbeat is None
                or (now - heartbeat) / 1_000_000_000.0 > self._policy.heartbeat_timeout_seconds
            ):
                return "watchdog_process_heartbeat_timeout"
            if not first_frame:
                assert generation_started is not None
                if (
                    now - generation_started
                ) / 1_000_000_000.0 > self._policy.first_frame_timeout_seconds:
                    return "watchdog_first_frame_timeout"
            elif (
                last_frame is not None
                and (now - last_frame) / 1_000_000_000.0 > self._policy.frame_stall_timeout_seconds
            ):
                with self._lock:
                    self._state = CaptureProcessState.STALE
                self._record_event(CaptureEventKind.FRAME_STALE, "watchdog_frame_stall")
                return "watchdog_frame_stall_timeout"
        return None

    def _drain_events(self) -> None:
        event_queue = self._event_queue
        if event_queue is None:
            return
        while True:
            try:
                message = event_queue.get_nowait()
            except queue.Empty:
                return
            if not isinstance(message, dict):
                continue
            now = monotonic_ns()
            kind = message.get("kind")
            with self._lock:
                if kind == "heartbeat":
                    self._last_heartbeat_ns = now
                    self._decoded_frames = max(
                        self._decoded_frames,
                        self._generation_decoded_base
                        + _optional_int(message, "decoded_frames", 0),
                    )
                elif kind == "failure":
                    failure = message.get("failure_class")
                    self._failure_class = failure if isinstance(failure, str) else "BackendError"
                    detail = message.get("failure_detail")
                    self._failure_detail = detail if isinstance(detail, str) else None

    def _drain_latest_frame(self) -> None:
        if self._shared_frame is not None:
            message = _read_shared_frame(self._shared_frame)
            if message is None:
                return
            sequence = _required_int(message, "sequence")
            if sequence <= self._last_transport_sequence:
                return
            removed = max(0, sequence - self._last_transport_sequence - 1)
            self._last_transport_sequence = sequence
            self._deliver_frame_message(message, removed)
            return
        frame_queue = self._frame_queue
        if frame_queue is None:
            return
        latest: dict[str, object] | None = None
        removed = 0
        while True:
            try:
                candidate = frame_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(candidate, dict):
                if latest is not None:
                    removed += 1
                latest = candidate
        if latest is None:
            return
        self._deliver_frame_message(latest, removed)

    def _deliver_frame_message(self, latest: dict[str, object], removed: int) -> None:
        frame = latest.get("frame_bgr")
        if not isinstance(frame, np.ndarray):
            return
        generation = _required_int(latest, "generation")
        sequence = _required_int(latest, "sequence")
        acquired = _required_int(latest, "acquisition_monotonic_ns")
        height, width = frame.shape[:2]
        identity = LiveFrameIdentity(
            self._endpoint.camera_id,
            f"{self._endpoint.camera_id}-g{generation:06d}-{sequence:012d}",
            acquired,
            str(latest["observed_at_utc"]),
            _required_int(latest, "source_pts") if latest.get("source_pts") is not None else None,
            str(latest["source_time_base"])
            if latest.get("source_time_base") is not None
            else None,
            width,
            height,
        )
        was_first = False
        with self._lock:
            if generation != self._generation:
                return
            was_first = not self._generation_first_frame
            self._generation_first_frame = True
            self._state = CaptureProcessState.CURRENT
            self._last_acquisition_ns = acquired
            self._delivered_frames += 1
            self._replaced_before_delivery += removed
            self._decoded_frames = max(
                self._decoded_frames,
                self._generation_decoded_base
                + _optional_int(latest, "decoded_frames", sequence + 1),
            )
            self._failure_class = None
            self._failure_detail = None
        self._slot.publish(CapturedFrame(identity, frame))
        if was_first:
            self._record_event(CaptureEventKind.FIRST_FRAME, "first_fresh_frame")
            if generation > 1:
                self._record_event(CaptureEventKind.RECOVERED, "new_generation_current")

    def _stop_generation(self) -> None:
        child_stop = self._child_stop
        process = self._process
        if child_stop is not None:
            child_stop.set()
        if process is not None:
            process.join(timeout=self._policy.graceful_stop_seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=self._policy.graceful_stop_seconds)
            if process.is_alive():
                process.kill()
                process.join(timeout=self._policy.graceful_stop_seconds)
        for owned_queue in (self._frame_queue, self._event_queue):
            if owned_queue is not None:
                owned_queue.close()
                owned_queue.join_thread()
        shared_frame = self._shared_frame
        if shared_frame is not None:
            shared_frame.close()
            try:
                shared_frame.unlink()
            except FileNotFoundError:
                pass
        with self._lock:
            self._process = None
        self._child_stop = None
        self._frame_queue = None
        self._event_queue = None
        self._shared_frame = None
        self._shared_frame_ready = None

    def _record_event(self, kind: CaptureEventKind, reason: str) -> None:
        with self._lock:
            self._events.append(
                CaptureEvent(
                    self._endpoint.camera_id,
                    self._generation,
                    self._tracker_epoch,
                    kind,
                    monotonic_ns(),
                    reason,
                )
            )


def _capture_worker_entry(
    camera_id: str,
    endpoint_url: str,
    generation: int,
    payload: dict[str, object],
    stop_event: Any,
    frame_queue: Any,
    event_queue: Any,
) -> None:
    shared_writer = _SharedFrameWriter.from_payload(payload)
    frame_transport = shared_writer if shared_writer is not None else frame_queue
    heartbeat_stop = threading.Event()
    decoded_counter = [0]
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(heartbeat_stop, stop_event, event_queue, decoded_counter, payload),
        daemon=True,
    )
    heartbeat.start()
    try:
        backend = CaptureBackend(str(payload["backend"]))
        if backend is CaptureBackend.PYAV:
            _pyav_capture(
                camera_id,
                endpoint_url,
                generation,
                payload,
                stop_event,
                frame_transport,
                decoded_counter,
            )
        else:
            _gstreamer_capture(
                camera_id,
                endpoint_url,
                generation,
                payload,
                stop_event,
                frame_transport,
                decoded_counter,
            )
    except Exception as error:
        location = traceback.extract_tb(error.__traceback__)[-1]
        safe_code = str(error) if isinstance(error, XR02CaptureError) else ""
        errno = getattr(error, "errno", None)
        if isinstance(errno, int):
            safe_code = f"errno={errno}"
        _put_event(
            event_queue,
            {
                "kind": "failure",
                "failure_class": type(error).__name__,
                "failure_detail": ":".join(
                    value for value in (f"{location.name}:{location.lineno}", safe_code) if value
                ),
            },
        )
        sleep(0.10)
        raise SystemExit(2) from None
    finally:
        if shared_writer is not None:
            shared_writer.close()
        heartbeat_stop.set()
        heartbeat.join(timeout=0.5)


def _heartbeat_loop(
    local_stop: threading.Event,
    process_stop: Any,
    event_queue: Any,
    decoded_counter: list[int],
    payload: dict[str, object],
) -> None:
    interval = _required_float(payload, "heartbeat_interval_seconds")
    while not local_stop.is_set() and not process_stop.is_set():
        _put_event(
            event_queue,
            {"kind": "heartbeat", "decoded_frames": decoded_counter[0]},
        )
        local_stop.wait(interval)


def _pyav_capture(
    camera_id: str,
    endpoint_url: str,
    generation: int,
    payload: dict[str, object],
    stop_event: Any,
    frame_queue: Any,
    decoded_counter: list[int],
) -> None:
    av: Any = importlib.import_module("av")

    with av.open(
        endpoint_url,
        mode="r",
        options={"rtsp_transport": "tcp"},
        timeout=(
            _required_float(payload, "connect_timeout_seconds"),
            _required_float(payload, "read_timeout_seconds"),
        ),
    ) as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            raise XR02CaptureError("RTSP source has no video stream")
        for decoded in container.decode(stream):
            if stop_event.is_set():
                return
            frame_bgr = decoded.to_ndarray(format="bgr24")
            acquired = monotonic_ns()
            source_pts, time_base = paired_source_timing(decoded.pts, stream.time_base)
            decoded_counter[0] += 1
            _put_latest_frame(
                frame_queue,
                _frame_message(
                    generation,
                    decoded_counter[0] - 1,
                    acquired,
                    frame_bgr,
                    source_pts,
                    time_base,
                    decoded_counter[0],
                ),
            )


def _gstreamer_capture(
    camera_id: str,
    endpoint_url: str,
    generation: int,
    payload: dict[str, object],
    stop_event: Any,
    frame_queue: Any,
    decoded_counter: list[int],
) -> None:
    overlay = payload.get("gstreamer_overlay_path")
    if not isinstance(overlay, str) or not overlay:
        raise XR02CaptureError("GStreamer overlay is not configured")
    site.addsitedir(overlay)
    gi: Any = importlib.import_module("gi")

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gi.require_version("GstRtsp", "1.0")
    gi.require_version("GstVideo", "1.0")
    Gst: Any = importlib.import_module("gi.repository.Gst")
    GstRtsp: Any = importlib.import_module("gi.repository.GstRtsp")
    GstVideo: Any = importlib.import_module("gi.repository.GstVideo")

    Gst.init(None)
    pipeline = Gst.Pipeline.new(f"xr02-{camera_id}")
    source = Gst.ElementFactory.make("uridecodebin", "source")
    convert = Gst.ElementFactory.make("videoconvert", "convert")
    caps_filter = Gst.ElementFactory.make("capsfilter", "bgr-caps")
    sink = Gst.ElementFactory.make("appsink", "sink")
    if any(element is None for element in (pipeline, source, convert, caps_filter, sink)):
        raise XR02CaptureError("required GStreamer element is unavailable")
    source.set_property("uri", endpoint_url)
    caps_filter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
    sink.set_property("emit-signals", False)
    sink.set_property("sync", False)
    sink.set_property("max-buffers", 1)
    if sink.find_property("leaky-type") is not None:
        sink.set_property("leaky-type", 2)
    elif sink.find_property("drop") is not None:
        sink.set_property("drop", True)
    source.connect(
        "source-setup",
        _configure_gstreamer_rtsp_source,
        payload,
        GstRtsp,
    )
    source.connect("pad-added", _link_decode_pad, convert)
    pipeline.add(source)
    pipeline.add(convert)
    pipeline.add(caps_filter)
    pipeline.add(sink)
    if not convert.link(caps_filter) or not caps_filter.link(sink):
        raise XR02CaptureError("GStreamer BGR pipeline could not be linked")
    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        while not stop_event.is_set():
            message = bus.timed_pop_filtered(
                0,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if message is not None:
                if message.type == Gst.MessageType.ERROR:
                    error, _debug = message.parse_error()
                    raise XR02CaptureError(f"gstreamer_bus_error={error.domain}:{error.code}")
                raise XR02CaptureError("GStreamer RTSP stream ended")
            sample = sink.emit("try-pull-sample", 100_000_000)
            if sample is None:
                continue
            caps = sample.get_caps()
            info = GstVideo.VideoInfo.new_from_caps(caps)
            buffer = sample.get_buffer()
            payload_bytes = buffer.extract_dup(0, buffer.get_size())
            stride = int(info.stride[0])
            width = int(info.width)
            height = int(info.height)
            raw = np.frombuffer(payload_bytes, dtype=np.uint8)
            expected = height * stride
            if raw.size < expected:
                raise XR02CaptureError("GStreamer BGR buffer is shorter than its caps")
            frame_bgr = (
                raw[:expected]
                .reshape(height, stride)[:, : width * 3]
                .reshape(height, width, 3)
                .copy()
            )
            acquired = monotonic_ns()
            source_pts = None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
            source_time_base = None if source_pts is None else "1/1000000000"
            decoded_counter[0] += 1
            _put_latest_frame(
                frame_queue,
                _frame_message(
                    generation,
                    decoded_counter[0] - 1,
                    acquired,
                    frame_bgr,
                    source_pts,
                    source_time_base,
                    decoded_counter[0],
                ),
            )
    finally:
        pipeline.set_state(Gst.State.NULL)


def _configure_gstreamer_rtsp_source(
    _decodebin: Any,
    source: Any,
    payload: dict[str, object],
    gst_rtsp: Any,
) -> None:
    properties = {
        "latency": _required_int(payload, "gstreamer_latency_ms"),
        "drop-on-latency": True,
        "tcp-timeout": int(_required_float(payload, "read_timeout_seconds") * 1_000_000),
        "tcp-timestamp": True,
        "do-rtsp-keep-alive": True,
        "protocols": gst_rtsp.RTSPLowerTrans.TCP,
    }
    for name, value in properties.items():
        if source.find_property(name) is not None:
            source.set_property(name, value)


def _link_decode_pad(_source: Any, pad: Any, convert: Any) -> None:
    caps = pad.get_current_caps() or pad.query_caps(None)
    if caps is None or caps.get_size() == 0:
        return
    media_type = caps.get_structure(0).get_name()
    if not str(media_type).startswith("video/"):
        return
    sink_pad = convert.get_static_pad("sink")
    if not sink_pad.is_linked():
        try:
            pad.link(sink_pad)
        except Exception:
            return


def _frame_message(
    generation: int,
    sequence: int,
    acquired: int,
    frame_bgr: np.ndarray[Any, Any],
    source_pts: int | None,
    source_time_base: str | None,
    decoded_frames: int,
) -> dict[str, object]:
    return {
        "generation": generation,
        "sequence": sequence,
        "acquisition_monotonic_ns": acquired,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "observed_at_utc_ns": time_ns(),
        "source_pts": source_pts,
        "source_time_base": source_time_base,
        "decoded_frames": decoded_frames,
        "frame_bgr": frame_bgr,
    }


def _put_latest_frame(frame_queue: Any, message: dict[str, object]) -> None:
    if isinstance(frame_queue, _SharedFrameWriter):
        frame_queue.publish(message)
        return
    try:
        frame_queue.put_nowait(message)
        return
    except queue.Full:
        pass
    try:
        frame_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        frame_queue.put_nowait(message)
    except queue.Full:
        pass


def _put_event(event_queue: Any, message: dict[str, object]) -> None:
    try:
        event_queue.put(message, timeout=0.10)
    except queue.Full:
        pass


def _required_int(value: dict[str, object], key: str) -> int:
    selected = value.get(key)
    if not isinstance(selected, int):
        raise XR02CaptureError(f"capture message {key} must be an integer")
    return selected


def _optional_int(value: dict[str, object], key: str, default: int) -> int:
    selected = value.get(key, default)
    if not isinstance(selected, int):
        raise XR02CaptureError(f"capture message {key} must be an integer")
    return selected


def _required_float(value: dict[str, object], key: str) -> float:
    selected = value.get(key)
    if not isinstance(selected, int | float):
        raise XR02CaptureError(f"capture setting {key} must be numeric")
    return float(selected)
