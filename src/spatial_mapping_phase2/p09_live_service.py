"""Lifecycle service for the bounded P09 live four-camera demonstrator."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from time import monotonic_ns
from typing import Any

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.p09_live_runtime import (
    BoundedInferenceWorker,
    DecoderPolicy,
    LatestFrameCoordinator,
    LatestFrameSlot,
    LatestFrameSnapshot,
    PersistentPyAvDecoder,
)
from spatial_mapping_phase2.p09_pipeline import (
    CameraTickResult,
    P09TrackingPipeline,
    PipelineTickResult,
)
from spatial_mapping_phase2.p09_rerun import P09RerunLogger
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS


class P09LiveServiceError(RuntimeError):
    """Raised for duplicate workers or a failed bounded live lifecycle."""


@dataclass(frozen=True, slots=True)
class LiveServiceConfig:
    inference_hz: float = 2.0
    maximum_frame_age_ms: float = 750.0

    def __post_init__(self) -> None:
        if not 0.2 <= self.inference_hz <= 10.0:
            raise P09LiveServiceError("inference clock must be within 0.2..10 Hz")
        if not 100.0 <= self.maximum_frame_age_ms <= 5000.0:
            raise P09LiveServiceError("maximum frame age must be within 100..5000 ms")


class P09LiveService:
    """Exactly four decoder threads, one zero-queue worker and one inference clock."""

    def __init__(
        self,
        endpoints: tuple[LocalRtspEndpoint, ...],
        pipeline: P09TrackingPipeline,
        logger: P09RerunLogger,
        config: LiveServiceConfig,
        decoder_policy: DecoderPolicy | None = None,
    ) -> None:
        if tuple(endpoint.camera_id for endpoint in endpoints) != CAMERA_IDS:
            raise P09LiveServiceError("live service requires exact ordered four-camera endpoints")
        self._config = config
        policy = decoder_policy or DecoderPolicy()
        self._logger = logger
        self._slots = {camera_id: LatestFrameSlot(camera_id) for camera_id in CAMERA_IDS}
        self._coordinator = LatestFrameCoordinator(self._slots)
        self._decoders = tuple(
            PersistentPyAvDecoder(endpoint, self._slots[endpoint.camera_id], policy)
            for endpoint in endpoints
        )
        self._pipeline = pipeline
        self._worker = BoundedInferenceWorker(self._process)
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: PipelineTickResult | None = None
        self._failure_class: str | None = None
        self._started_ns: int | None = None
        self._stopped_ns: int | None = None
        self._tick_records: list[dict[str, Any]] = []

    def start(self) -> None:
        if self._ticker is not None:
            raise P09LiveServiceError("live tracking is already started")
        self._stop.clear()
        self._started_ns = monotonic_ns()
        self._stopped_ns = None
        for decoder in self._decoders:
            decoder.start()
        self._ticker = threading.Thread(target=self._clock, name="p09-clock", daemon=True)
        self._ticker.start()

    def _clock(self) -> None:
        period_seconds = 1.0 / self._config.inference_hz
        while not self._stop.is_set():
            tick = monotonic_ns()
            snapshot = self._coordinator.snapshot(tick, self._config.maximum_frame_age_ms)
            self._worker.try_submit(snapshot)
            elapsed = (monotonic_ns() - tick) / 1e9
            self._stop.wait(max(0.0, period_seconds - elapsed))

    def _process(self, snapshot: LatestFrameSnapshot) -> None:
        try:
            result = self._pipeline.process(snapshot)
            self._logger.log_tick(result)
        except Exception as error:
            with self._lock:
                self._failure_class = type(error).__name__
            raise
        with self._lock:
            self._latest = result
            self._failure_class = None
            self._tick_records.append(
                {
                    "tick_monotonic_ns": result.tick_monotonic_ns,
                    "completed_monotonic_ns": result.completed_monotonic_ns,
                    "state": result.tracking.state.value,
                    "reason": result.tracking.reason,
                    "current_xy_metres": result.tracking.current_xy_metres,
                    "contributing_camera_ids": result.tracking.contributing_camera_ids,
                    "rejected_camera_ids": result.tracking.rejected_camera_ids,
                    "last_known_xy_metres": result.tracking.last_known_xy_metres,
                    "last_known_age_ms": result.tracking.last_known_age_ms,
                    "stale_camera_ids": result.stale_camera_ids,
                    "missing_camera_ids": result.missing_camera_ids,
                    "cross_camera_skew_ms": result.cross_camera_skew_ms,
                    "processing_latency_ms": result.processing_latency_ms,
                    "cameras": [_camera_tick_evidence(camera) for camera in result.camera_results],
                }
            )

    def stop(self) -> None:
        if self._ticker is None:
            return
        self._stop.set()
        self._ticker.join(timeout=2.0 + 1.0 / self._config.inference_hz)
        if self._ticker.is_alive():
            raise P09LiveServiceError("inference clock did not stop within its bound")
        self._ticker = None
        self._worker.close(wait=True)
        errors: list[Exception] = []
        for decoder in self._decoders:
            try:
                decoder.close()
            except Exception as error:
                errors.append(error)
        self._stopped_ns = monotonic_ns()
        self._logger.close()
        if errors:
            raise P09LiveServiceError("one or more decoders failed bounded shutdown") from errors[
                0
            ]

    def open_viewer(self) -> None:
        self._logger.open_viewer()

    def reset_trail(self) -> None:
        self._logger.reset_trail()

    def status(self) -> dict[str, Any]:
        worker = self._worker.telemetry()
        decoder_records = [asdict(decoder.telemetry()) for decoder in self._decoders]
        with self._lock:
            latest = self._latest
            failure_class = self._failure_class
        return {
            "running": self._ticker is not None and self._ticker.is_alive(),
            "global_state": latest.tracking.state.value if latest is not None else "starting",
            "global_reason": latest.tracking.reason if latest is not None else "no completed tick",
            "failure_class": failure_class,
            "worker": asdict(worker),
            "decoders": decoder_records,
            "slots": {
                camera_id: asdict(self._slots[camera_id].telemetry()) for camera_id in CAMERA_IDS
            },
        }

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._tick_records)
        return {
            "config": asdict(self._config),
            "started_monotonic_ns": self._started_ns,
            "stopped_monotonic_ns": self._stopped_ns,
            "status": self.status(),
            "rerun_stream": self._logger.evidence(),
            "ticks": records,
        }


def _camera_tick_evidence(camera: CameraTickResult) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    for detection, projection in zip(camera.detections, camera.projections, strict=True):
        observation = projection.observation
        detections.append(
            {
                "detection_index": detection.detection_index,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.bbox_xyxy,
                "image_point_uv": detection.image_point_uv,
                "footpoint_kind": detection.footpoint_kind.value,
                "clipped_at_image_bottom": detection.clipped_at_image_bottom,
                "projection_status": projection.status.value,
                "projection_reason": projection.reason,
                "xy_metres": observation.xy_metres if observation is not None else None,
                "ray_parameter_metres": (
                    observation.ray_parameter_metres if observation is not None else None
                ),
                "ray_floor_incidence": (
                    observation.ray_floor_incidence if observation is not None else None
                ),
                "quality_weight": observation.quality_weight if observation is not None else None,
            }
        )
    return {
        "camera_id": camera.camera_id,
        "frame_id": camera.frame.frame_id,
        "acquisition_monotonic_ns": camera.frame.acquisition_monotonic_ns,
        "observed_at_utc": camera.frame.observed_at_utc,
        "source_pts": camera.frame.source_pts,
        "source_time_base": camera.frame.source_time_base,
        "detector_ms": camera.total_detector_ms,
        "detections": detections,
    }
