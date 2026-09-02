"""Bounded partial-fleet lifecycle service for XR02 WP4."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic_ns

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.p09_live_runtime import (
    LatestFrameCoordinator,
    LatestFrameSnapshot,
)
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS
from spatial_mapping_phase2.xr02_cadence import (
    HigherCadenceProfile,
    LatestPendingWorker,
    ZeroQueueWorker,
)
from spatial_mapping_phase2.xr02_compact_telemetry import CompactLiveTelemetryJournal
from spatial_mapping_phase2.xr02_global_domain import GlobalTrackState
from spatial_mapping_phase2.xr02_live_domain import (
    AdoptedSceneSelection,
    CameraHealthEvidence,
    CameraLiveState,
    LiveServiceState,
    XR02LiveContractError,
)
from spatial_mapping_phase2.xr02_live_pipeline import LiveAssociationTick, XR02LivePipeline
from spatial_mapping_phase2.xr02_live_rerun import XR02LiveRerunLogger
from spatial_mapping_phase2.xr02_rtsp_capture import (
    CaptureBackend,
    CaptureProcessState,
    GenerationAwareLatestFrameSlot,
    SupervisedDecoderPolicy,
    SupervisedRtspDecoder,
)


@dataclass(frozen=True, slots=True)
class XR02LiveServiceConfig:
    local_tracking_hz: float = 8.0
    appearance_refresh_hz: float = 2.0
    association_hz: float = 8.0
    publication_hz: float = 2.0
    maximum_frame_age_ms: float = 250.0
    cached_embedding_max_age_seconds: float = 2.0
    local_track_buffer_seconds: float = 1.5
    inference_pending_max_age_ms: float = 150.0
    scene_check_interval_ticks: int = 20
    capture_backend: CaptureBackend = CaptureBackend.PYAV
    gstreamer_overlay_path: str | None = None

    def __post_init__(self) -> None:
        try:
            _ = self.cadence_profile
        except ValueError as error:
            raise XR02LiveContractError(str(error)) from error
        if self.scene_check_interval_ticks <= 0:
            raise XR02LiveContractError("scene check interval must be positive")
        if not 1.0 <= self.inference_pending_max_age_ms <= 1_000.0:
            raise XR02LiveContractError("inference pending age bound must be within 1..1000 ms")
        if self.capture_backend is CaptureBackend.GSTREAMER and not self.gstreamer_overlay_path:
            raise XR02LiveContractError(
                "GStreamer capture requires an explicit isolated overlay path"
            )

    @property
    def cadence_profile(self) -> HigherCadenceProfile:
        return HigherCadenceProfile(
            local_tracking_hz=self.local_tracking_hz,
            appearance_refresh_hz=self.appearance_refresh_hz,
            global_association_hz=self.association_hz,
            publication_hz=self.publication_hz,
            maximum_frame_age_ms=self.maximum_frame_age_ms,
            cached_embedding_max_age_seconds=self.cached_embedding_max_age_seconds,
            local_track_buffer_seconds=self.local_track_buffer_seconds,
        )


class XR02LiveService:
    """Four decoders, a latest-pending GPU actor and a decoupled Rerun publisher."""

    def __init__(
        self,
        endpoints: tuple[LocalRtspEndpoint, ...],
        scene: AdoptedSceneSelection,
        pipeline: XR02LivePipeline,
        logger: XR02LiveRerunLogger,
        scene_resolver: Callable[[], AdoptedSceneSelection],
        config: XR02LiveServiceConfig | None = None,
        decoder_policy: SupervisedDecoderPolicy | None = None,
        *,
        compact_telemetry: CompactLiveTelemetryJournal | None = None,
        retain_full_evidence: bool = True,
    ) -> None:
        if tuple(item.camera_id for item in endpoints) != CAMERA_IDS:
            raise XR02LiveContractError("WP4 requires the exact ordered office camera roster")
        self.scene = scene
        self._pipeline = pipeline
        self._logger = logger
        self._scene_resolver = scene_resolver
        self._config = config or XR02LiveServiceConfig()
        self._compact_telemetry = compact_telemetry
        self._retain_full_evidence = retain_full_evidence
        policy = decoder_policy or SupervisedDecoderPolicy()
        self._decoder_policy = policy
        self._slots = {
            camera_id: GenerationAwareLatestFrameSlot(camera_id) for camera_id in CAMERA_IDS
        }
        self._coordinator = LatestFrameCoordinator(self._slots)
        self._pending_capture_epochs: dict[str, tuple[int, int, str]] = {}
        overlay = (
            None
            if self._config.gstreamer_overlay_path is None
            else Path(self._config.gstreamer_overlay_path)
        )
        self._decoders = tuple(
            SupervisedRtspDecoder(
                endpoint,
                self._slots[endpoint.camera_id],
                policy,
                self._config.capture_backend,
                gstreamer_overlay_path=overlay,
                epoch_callback=self._on_capture_epoch,
            )
            for endpoint in endpoints
        )
        self._worker = LatestPendingWorker(
            self._process,
            name="xr02-wp4-inference",
            item_monotonic_ns=lambda snapshot: snapshot.snapshot_monotonic_ns,
            maximum_pending_age_ms=self._config.inference_pending_max_age_ms,
        )
        self._publisher = ZeroQueueWorker(
            self._publish,
            name="xr02-wp4-publication",
        )
        self._stop = threading.Event()
        self._clock_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = LiveServiceState.STOPPED
        self._started_ns: int | None = None
        self._stopped_ns: int | None = None
        self._latest_tick: LiveAssociationTick | None = None
        self._latest_snapshot: LatestFrameSnapshot | None = None
        self._worker_failure_class: str | None = None
        self._publisher_failure_class: str | None = None
        self._scene_update: AdoptedSceneSelection | None = None
        self._tick_summaries: list[dict[str, object]] | deque[dict[str, object]] = (
            [] if retain_full_evidence else deque(maxlen=32)
        )
        self._health_samples: list[dict[str, object]] | deque[dict[str, object]] = (
            [] if retain_full_evidence else deque(maxlen=32)
        )
        self._clock_ticks = 0
        self._scheduler_missed_deadlines = 0
        self._last_published_tick_index: int | None = None
        self._forced_final_publications = 0

    def start(self) -> None:
        with self._lock:
            if self._state is not LiveServiceState.STOPPED:
                raise XR02LiveContractError("live service is not stopped")
            self._state = LiveServiceState.STARTING
            self._started_ns = monotonic_ns()
            self._stopped_ns = None
        self._pipeline.warmup()
        self._stop.clear()
        for decoder in self._decoders:
            decoder.start()
        self._clock_thread = threading.Thread(
            target=self._clock,
            name="xr02-wp4-clock",
            daemon=True,
        )
        self._clock_thread.start()
        with self._lock:
            self._state = LiveServiceState.RUNNING
        self._logger.log_service_state("running", "Frozen scene actor started.")

    def stop(self) -> None:
        with self._lock:
            if self._state is LiveServiceState.STOPPED:
                return
            self._state = LiveServiceState.STOPPING
        self._stop.set()
        clock = self._clock_thread
        if clock is not None:
            clock.join(timeout=2.0 + 1.0 / self._config.local_tracking_hz)
            if clock.is_alive():
                raise XR02LiveContractError("association clock exceeded bounded shutdown")
        self._clock_thread = None
        self._worker.close(wait=True)
        self._pipeline.finalize_evidence()
        self._publisher.close(wait=True)
        with self._lock:
            latest = self._latest_tick
            last_published = self._last_published_tick_index
        if latest is not None and latest.tick_index != last_published:
            self._logger.log_tick(latest)
            with self._lock:
                self._last_published_tick_index = latest.tick_index
                self._forced_final_publications += 1
        decoder_errors: list[Exception] = []
        for decoder in self._decoders:
            try:
                decoder.close()
            except Exception as error:
                decoder_errors.append(error)
        self._stopped_ns = monotonic_ns()
        self._logger.log_service_state(
            "stopped",
            "Run finalized; the overwriteable pending-frame slot is empty.",
        )
        try:
            self._logger.close()
        finally:
            if self._compact_telemetry is not None:
                self._compact_telemetry.close()
        with self._lock:
            failed = bool(decoder_errors) or self._publisher_failure_class is not None
            self._state = LiveServiceState.FAILED if failed else LiveServiceState.STOPPED
        if decoder_errors:
            raise XR02LiveContractError(
                "one or more decoders exceeded bounded shutdown"
            ) from decoder_errors[0]
        if self._publisher_failure_class is not None:
            raise XR02LiveContractError("Rerun publisher failed during the live run")

    def open_viewer(self) -> None:
        self._logger.open_viewer()

    def close_viewer(self) -> None:
        self._logger.close_viewer()

    def reset_trails(self) -> None:
        self._logger.reset_trails()

    def status(self) -> dict[str, object]:
        with self._lock:
            state = self._state
            latest = self._latest_tick
            failure = self._worker_failure_class
            publisher_failure = self._publisher_failure_class
            scene_update = self._scene_update
        health = self._camera_health(monotonic_ns())
        return {
            "schema": "xr02.wp4.live_status.v3",
            "state": state.value,
            "running": state
            in {LiveServiceState.RUNNING, LiveServiceState.SCENE_UPDATE_AVAILABLE},
            "scene_context_sha256": self.scene.scene.context_sha256,
            "scene_selection_signature_sha256": self.scene.selection_signature_sha256,
            "scene_update_available": scene_update is not None,
            "new_scene_context_sha256": (
                None if scene_update is None else scene_update.scene.context_sha256
            ),
            "worker_failure_class": failure,
            "publisher_failure_class": publisher_failure,
            "worker": asdict(self._worker.telemetry()),
            "inference_admission": {
                "profile": "one-running-one-overwriteable-latest-pending-v1",
                "fifo": False,
                "maximum_pending_items": 1,
                "maximum_pending_age_ms": self._config.inference_pending_max_age_ms,
            },
            "publication_worker": asdict(self._publisher.telemetry()),
            "cadence": self._config.cadence_profile.as_dict(),
            "camera_health": [item.as_dict() for item in health],
            "capture_transport": {
                "profile": getattr(self._decoders[0], "frame_transport", "queue_test_adapter"),
                "capacity_bytes_per_camera": self._decoder_policy.shared_frame_capacity_bytes,
                "semantics": "generation-scoped capacity-one latest native frame",
            },
            "latest_tick": None if latest is None else latest.summary(),
            "global_tracks": (
                []
                if latest is None
                else [
                    item.as_dict()
                    for item in latest.association.tracks
                    if item.state is not GlobalTrackState.ENDED
                ]
            ),
        }

    def evidence(self) -> dict[str, object]:
        with self._lock:
            summaries = list(self._tick_summaries)
            health = list(self._health_samples)
        return {
            "schema": "xr02.wp4.live_service_evidence.v3",
            "config": asdict(self._config),
            "capture_transport": {
                "profile": getattr(self._decoders[0], "frame_transport", "queue_test_adapter"),
                "capacity_bytes_per_camera": self._decoder_policy.shared_frame_capacity_bytes,
                "pickle_full_frames": False,
            },
            "started_monotonic_ns": self._started_ns,
            "stopped_monotonic_ns": self._stopped_ns,
            "clock_ticks": self._clock_ticks,
            "scheduler_missed_deadlines": self._scheduler_missed_deadlines,
            "forced_final_publications": self._forced_final_publications,
            "status": self.status(),
            "pipeline_profile": self._pipeline.profile_identity,
            "rerun": self._logger.evidence(),
            "compact_telemetry": (
                None if self._compact_telemetry is None else self._compact_telemetry.evidence()
            ),
            "full_tick_history_retained": self._retain_full_evidence,
            "tick_summaries": summaries,
            "camera_health_samples": health,
            "capture_events": [
                event.as_dict() for decoder in self._decoders for event in decoder.events()
            ],
        }

    def _clock(self) -> None:
        period_ns = max(1, int(round(1_000_000_000 / self._config.local_tracking_hz)))
        next_deadline_ns = monotonic_ns()
        while not self._stop.is_set():
            wait_ns = next_deadline_ns - monotonic_ns()
            if wait_ns > 0 and self._stop.wait(wait_ns / 1_000_000_000.0):
                break
            tick_ns = monotonic_ns()
            snapshot = self._coordinator.snapshot(
                tick_ns,
                self._config.maximum_frame_age_ms,
            )
            with self._lock:
                self._latest_snapshot = snapshot
            self._worker.try_submit(snapshot)
            self._clock_ticks += 1
            if self._clock_ticks % self._config.scene_check_interval_ticks == 0:
                self._check_scene_update()
            next_deadline_ns += period_ns
            now_ns = monotonic_ns()
            if now_ns >= next_deadline_ns + period_ns:
                missed = (now_ns - next_deadline_ns) // period_ns
                self._scheduler_missed_deadlines += int(missed)
                next_deadline_ns += int(missed) * period_ns

    def _process(self, snapshot: LatestFrameSnapshot) -> None:
        try:
            self._apply_pending_capture_epochs()
            result = self._pipeline.process(snapshot)
            if result.publication_due:
                self._publisher.try_submit(result)
        except Exception as error:
            with self._lock:
                self._worker_failure_class = type(error).__name__
                self._state = LiveServiceState.FAILED
            self._logger.log_service_state("worker failed", type(error).__name__)
            raise
        health = self._camera_health(result.completed_monotonic_ns)
        if self._compact_telemetry is not None:
            self._compact_telemetry.record(result)
        with self._lock:
            self._latest_tick = result
            self._worker_failure_class = None
            if self._scene_update is not None:
                self._state = LiveServiceState.SCENE_UPDATE_AVAILABLE
            elif self._state is not LiveServiceState.STOPPING:
                self._state = LiveServiceState.RUNNING
            self._tick_summaries.append(result.summary())
            self._health_samples.append(
                {
                    "tick_index": result.tick_index,
                    "cameras": [item.as_dict() for item in health],
                }
            )

    def _publish(self, result: LiveAssociationTick) -> None:
        try:
            self._logger.log_tick(result)
        except Exception as error:
            with self._lock:
                self._publisher_failure_class = type(error).__name__
                self._state = LiveServiceState.FAILED
            raise
        with self._lock:
            self._last_published_tick_index = result.tick_index

    def _check_scene_update(self) -> None:
        try:
            selected = self._scene_resolver()
        except Exception as error:
            with self._lock:
                self._worker_failure_class = f"SceneResolver{type(error).__name__}"
            return
        if selected.selection_signature_sha256 == self.scene.selection_signature_sha256:
            return
        with self._lock:
            self._scene_update = selected
            if self._state is LiveServiceState.RUNNING:
                self._state = LiveServiceState.SCENE_UPDATE_AVAILABLE
        self._logger.log_service_state(
            "scene_update_available",
            "Stop and start a controlled new scene epoch; active geometry was not hot-swapped.",
        )

    def _on_capture_epoch(
        self,
        camera_id: str,
        generation: int,
        tracker_epoch: int,
        reason: str,
    ) -> None:
        accepted = False
        with self._lock:
            previous = self._pending_capture_epochs.get(camera_id)
            if previous is None or generation > previous[0]:
                self._pending_capture_epochs[camera_id] = (
                    generation,
                    tracker_epoch,
                    reason,
                )
                accepted = True
        if accepted:
            self._worker.invalidate_pending()

    def _apply_pending_capture_epochs(self) -> None:
        with self._lock:
            pending = dict(self._pending_capture_epochs)
            self._pending_capture_epochs.clear()
        for camera_id in CAMERA_IDS:
            epoch = pending.get(camera_id)
            if epoch is None:
                continue
            generation, tracker_epoch, reason = epoch
            self._pipeline.begin_camera_epoch(
                camera_id,
                generation,
                tracker_epoch,
                reason,
            )

    def _camera_health(self, now_ns: int) -> tuple[CameraHealthEvidence, ...]:
        snapshot = self._latest_snapshot
        current = (
            set() if snapshot is None else {item.identity.camera_id for item in snapshot.frames}
        )
        stale = set() if snapshot is None else set(snapshot.stale_camera_ids)
        missing = set(CAMERA_IDS) if snapshot is None else set(snapshot.missing_camera_ids)
        decoder_by_camera = {
            item.camera_id: item for item in (d.telemetry() for d in self._decoders)
        }
        result: list[CameraHealthEvidence] = []
        for camera_id in CAMERA_IDS:
            decoder = decoder_by_camera[camera_id]
            slot = self._slots[camera_id].telemetry()
            if decoder.state is CaptureProcessState.STOPPED:
                state = CameraLiveState.STOPPED
            elif camera_id in current:
                state = CameraLiveState.CURRENT
            elif camera_id in stale:
                state = CameraLiveState.STALE
            elif decoder.state is CaptureProcessState.RECONNECTING:
                state = CameraLiveState.RECONNECTING
            elif decoder.state is CaptureProcessState.STARTING:
                state = CameraLiveState.STARTING
            elif decoder.failure_class is not None or decoder.state is CaptureProcessState.FAILED:
                state = CameraLiveState.FAILED
            elif camera_id in missing:
                state = CameraLiveState.MISSING
            else:
                state = CameraLiveState.MISSING
            age = (
                None
                if decoder.last_acquisition_monotonic_ns is None
                else max(0.0, (now_ns - decoder.last_acquisition_monotonic_ns) / 1_000_000.0)
            )
            heartbeat_age = (
                None
                if decoder.last_process_heartbeat_monotonic_ns is None
                else max(
                    0.0,
                    (now_ns - decoder.last_process_heartbeat_monotonic_ns) / 1_000_000.0,
                )
            )
            result.append(
                CameraHealthEvidence(
                    camera_id=camera_id,
                    state=state,
                    generation=decoder.generation,
                    decoded_frames=decoder.decoded_frames,
                    reconnects=decoder.reconnects,
                    replaced_frames=slot.replaced_frames,
                    stale_snapshots=slot.stale_snapshot_count,
                    failure_class=decoder.failure_class,
                    frame_age_ms=age,
                    failure_detail=decoder.failure_detail,
                    capture_backend=decoder.backend.value,
                    capture_process_state=decoder.state.value,
                    tracker_epoch=decoder.tracker_epoch,
                    delivered_frames=decoder.delivered_frames,
                    supervisor_restarts=decoder.watchdog_restarts,
                    restart_reason=decoder.restart_reason,
                    process_heartbeat_age_ms=heartbeat_age,
                )
            )
        return tuple(result)
