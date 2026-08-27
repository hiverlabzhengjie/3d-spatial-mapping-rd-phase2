"""Latest-frame capture and single-slot inference admission for P09 live operation."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS, LiveFrameIdentity

Array = NDArray[Any]


class P09LiveRuntimeError(RuntimeError):
    """Raised for bounded live-runtime lifecycle or frame-contract failures."""


@dataclass(frozen=True, slots=True, repr=False)
class CapturedFrame:
    identity: LiveFrameIdentity
    frame_bgr: Array

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame_bgr, dtype=np.uint8).copy()
        expected = (self.identity.height_pixels, self.identity.width_pixels, 3)
        if frame.shape != expected:
            raise P09LiveRuntimeError("captured frame dimensions disagree with identity")
        frame.setflags(write=False)
        object.__setattr__(self, "frame_bgr", frame)


@dataclass(frozen=True, slots=True)
class LatestFrameSnapshot:
    frames: tuple[CapturedFrame, ...]
    stale_camera_ids: tuple[str, ...]
    missing_camera_ids: tuple[str, ...]
    cross_camera_skew_ms: float | None
    snapshot_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class SlotTelemetry:
    published_frames: int
    replaced_frames: int
    snapshot_count: int
    stale_snapshot_count: int


class LatestFrameSlot:
    """Thread-safe capacity-one slot: publication always replaces older unconsumed work."""

    def __init__(self, camera_id: str) -> None:
        if camera_id not in CAMERA_IDS:
            raise P09LiveRuntimeError("latest-frame slot camera is not in the P09 roster")
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._latest: CapturedFrame | None = None
        self._published = 0
        self._replaced = 0
        self._snapshots = 0
        self._stale = 0

    def publish(self, frame: CapturedFrame) -> None:
        if frame.identity.camera_id != self.camera_id:
            raise P09LiveRuntimeError("published frame camera disagrees with slot")
        with self._lock:
            if self._latest is not None:
                if (
                    frame.identity.acquisition_monotonic_ns
                    < self._latest.identity.acquisition_monotonic_ns
                ):
                    raise P09LiveRuntimeError("latest-frame acquisition time moved backwards")
                self._replaced += 1
            self._latest = frame
            self._published += 1

    def snapshot(self, now_monotonic_ns: int, maximum_age_ms: float) -> CapturedFrame | None:
        if now_monotonic_ns < 0 or not math.isfinite(maximum_age_ms) or maximum_age_ms <= 0:
            raise P09LiveRuntimeError("snapshot time and maximum age must be positive")
        with self._lock:
            self._snapshots += 1
            if self._latest is None:
                return None
            age_ns = now_monotonic_ns - self._latest.identity.acquisition_monotonic_ns
            if age_ns < 0:
                raise P09LiveRuntimeError("snapshot time precedes frame acquisition")
            if age_ns / 1_000_000.0 > maximum_age_ms:
                self._stale += 1
                return None
            return self._latest

    def telemetry(self) -> SlotTelemetry:
        with self._lock:
            return SlotTelemetry(self._published, self._replaced, self._snapshots, self._stale)


class LatestFrameCoordinator:
    def __init__(self, slots: Mapping[str, LatestFrameSlot]) -> None:
        if tuple(slots) != CAMERA_IDS or any(slots[key].camera_id != key for key in CAMERA_IDS):
            raise P09LiveRuntimeError("coordinator requires four slots in fixed camera order")
        self._slots = dict(slots)

    def snapshot(self, now_monotonic_ns: int, maximum_age_ms: float) -> LatestFrameSnapshot:
        frames: list[CapturedFrame] = []
        stale: list[str] = []
        missing: list[str] = []
        for camera_id in CAMERA_IDS:
            slot = self._slots[camera_id]
            before = slot.telemetry()
            frame = slot.snapshot(now_monotonic_ns, maximum_age_ms)
            if frame is not None:
                frames.append(frame)
                continue
            after = slot.telemetry()
            if after.stale_snapshot_count > before.stale_snapshot_count:
                stale.append(camera_id)
            else:
                missing.append(camera_id)
        times = [frame.identity.acquisition_monotonic_ns for frame in frames]
        skew = (max(times) - min(times)) / 1_000_000.0 if len(times) >= 2 else None
        return LatestFrameSnapshot(
            tuple(frames), tuple(stale), tuple(missing), skew, now_monotonic_ns
        )


@dataclass(frozen=True, slots=True)
class InferenceAdmissionTelemetry:
    submitted_ticks: int
    busy_dropped_ticks: int
    completed_ticks: int
    failed_ticks: int
    busy: bool


class BoundedInferenceWorker:
    """One running job and zero queued jobs; a busy tick is measured and dropped."""

    def __init__(
        self,
        operation: Callable[[LatestFrameSnapshot], None],
        name: str = "p09-inference",
    ) -> None:
        self._operation = operation
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._submitted = 0
        self._dropped = 0
        self._completed = 0
        self._failed = 0

    def try_submit(self, snapshot: LatestFrameSnapshot) -> bool:
        with self._lock:
            if self._closed:
                raise P09LiveRuntimeError("bounded inference worker is closed")
            if self._busy:
                self._dropped += 1
                return False
            self._busy = True
            self._submitted += 1
        self._executor.submit(self._run, snapshot)
        return True

    def _run(self, snapshot: LatestFrameSnapshot) -> None:
        try:
            self._operation(snapshot)
        except Exception:
            with self._lock:
                self._failed += 1
        else:
            with self._lock:
                self._completed += 1
        finally:
            with self._lock:
                self._busy = False

    def telemetry(self) -> InferenceAdmissionTelemetry:
        with self._lock:
            return InferenceAdmissionTelemetry(
                self._submitted,
                self._dropped,
                self._completed,
                self._failed,
                self._busy,
            )

    def close(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)


@dataclass(frozen=True, slots=True)
class DecoderPolicy:
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 2.0
    reconnect_delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.connect_timeout_seconds,
                self.read_timeout_seconds,
                self.reconnect_delay_seconds,
            )
        ):
            raise P09LiveRuntimeError("decoder timeouts must be finite and positive")


@dataclass(frozen=True, slots=True)
class DecoderTelemetry:
    camera_id: str
    decoded_frames: int
    reconnects: int
    failure_class: str | None
    last_acquisition_monotonic_ns: int | None
    running: bool


class PersistentPyAvDecoder:
    """One credential-safe, reconnecting, read-only decoder feeding one latest-frame slot."""

    def __init__(
        self,
        endpoint: LocalRtspEndpoint,
        slot: LatestFrameSlot,
        policy: DecoderPolicy,
    ) -> None:
        if endpoint.camera_id != slot.camera_id:
            raise P09LiveRuntimeError("decoder endpoint and slot camera IDs disagree")
        self._endpoint = endpoint
        self._slot = slot
        self._policy = policy
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._decoded = 0
        self._reconnects = 0
        self._failure_class: str | None = None
        self._last_acquisition: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise P09LiveRuntimeError("decoder is already started")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"p09-decoder-{self._endpoint.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(
                timeout=self._policy.connect_timeout_seconds
                + self._policy.read_timeout_seconds
                + 1.0
            )
            if thread.is_alive():
                raise P09LiveRuntimeError("decoder did not stop within its bounded timeout")
        self._thread = None

    def telemetry(self) -> DecoderTelemetry:
        with self._lock:
            return DecoderTelemetry(
                self._endpoint.camera_id,
                self._decoded,
                self._reconnects,
                self._failure_class,
                self._last_acquisition,
                self._thread is not None and self._thread.is_alive(),
            )

    def _loop(self) -> None:
        sequence = 0
        while not self._stop.is_set():
            try:
                av = _av()
                with av.open(
                    self._endpoint.for_read_only_adapter(),
                    mode="r",
                    options={"rtsp_transport": "tcp"},
                    timeout=(
                        self._policy.connect_timeout_seconds,
                        self._policy.read_timeout_seconds,
                    ),
                ) as container:
                    stream = next(iter(container.streams.video), None)
                    if stream is None:
                        raise P09LiveRuntimeError("RTSP source has no video stream")
                    for decoded in container.decode(stream):
                        if self._stop.is_set():
                            return
                        frame_bgr = decoded.to_ndarray(format="bgr24")
                        acquired = monotonic_ns()
                        height, width = frame_bgr.shape[:2]
                        source_pts, time_base = paired_source_timing(decoded.pts, stream.time_base)
                        identity = LiveFrameIdentity(
                            self._endpoint.camera_id,
                            f"{self._endpoint.camera_id}-live-{sequence:012d}",
                            acquired,
                            datetime.now(UTC).isoformat(),
                            source_pts if time_base is not None else None,
                            time_base,
                            width,
                            height,
                        )
                        self._slot.publish(CapturedFrame(identity, frame_bgr))
                        sequence += 1
                        with self._lock:
                            self._decoded += 1
                            self._failure_class = None
                            self._last_acquisition = acquired
            except Exception as error:
                with self._lock:
                    self._reconnects += 1
                    self._failure_class = type(error).__name__
                if self._stop.wait(self._policy.reconnect_delay_seconds):
                    return


def _av() -> Any:
    try:
        import av
    except ImportError as error:
        raise P09LiveRuntimeError("PyAV is unavailable in the P09 runtime") from error
    return av


def paired_source_timing(
    source_pts: int | None, stream_time_base: object | None
) -> tuple[int | None, str | None]:
    """Keep source PTS/time-base evidence atomic when cameras omit frame PTS."""

    if source_pts is None or stream_time_base is None:
        return None, None
    return int(source_pts), str(stream_time_base)
