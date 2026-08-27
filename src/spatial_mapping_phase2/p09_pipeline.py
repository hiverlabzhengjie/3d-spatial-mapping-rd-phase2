"""Pure orchestration for one bounded P09 four-camera inference tick."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_detector import DetectorBatchResult, DetectorProtocol
from spatial_mapping_phase2.p09_fusion import AnonymousWorldTracker
from spatial_mapping_phase2.p09_live_runtime import LatestFrameSnapshot
from spatial_mapping_phase2.p09_projection import (
    CameraProjectionCalibration,
    FloorRectangle,
    LiveFrameRectifier,
    project_detection_to_floor,
)
from spatial_mapping_phase2.p09_tracking_domain import (
    CAMERA_IDS,
    LiveFrameIdentity,
    PersonDetection,
    ProjectionResult,
    TrackingEstimate,
    WorldFloorObservation,
)

Array = NDArray[Any]


class P09PipelineError(RuntimeError):
    """Raised when a live tick violates frozen P09 orchestration contracts."""


@dataclass(frozen=True, slots=True)
class CameraTickResult:
    camera_id: str
    frame: LiveFrameIdentity
    calibrated_frame_bgr: Array
    detections: tuple[PersonDetection, ...]
    projections: tuple[ProjectionResult, ...]
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float

    def __post_init__(self) -> None:
        if self.camera_id != self.frame.camera_id or self.camera_id not in CAMERA_IDS:
            raise P09PipelineError("camera tick identity mismatch")
        image = np.asarray(self.calibrated_frame_bgr, dtype=np.uint8).copy()
        if image.shape != (280, 504, 3):
            raise P09PipelineError("camera tick image is not calibrated 504x280 BGR")
        if len(self.detections) != len(self.projections):
            raise P09PipelineError("detections and projections are misaligned")
        image.setflags(write=False)
        object.__setattr__(self, "calibrated_frame_bgr", image)

    @property
    def total_detector_ms(self) -> float:
        return self.preprocessing_ms + self.inference_ms + self.postprocessing_ms


@dataclass(frozen=True, slots=True)
class PipelineTickResult:
    tick_monotonic_ns: int
    completed_monotonic_ns: int
    camera_results: tuple[CameraTickResult, ...]
    stale_camera_ids: tuple[str, ...]
    missing_camera_ids: tuple[str, ...]
    cross_camera_skew_ms: float | None
    tracking: TrackingEstimate
    projection_ms: float
    fusion_ms: float

    def __post_init__(self) -> None:
        if self.completed_monotonic_ns < self.tick_monotonic_ns:
            raise P09PipelineError("pipeline completion precedes tick")
        camera_ids = tuple(result.camera_id for result in self.camera_results)
        if len(set(camera_ids)) != len(camera_ids):
            raise P09PipelineError("pipeline result duplicates a camera")
        if any(
            not math.isfinite(value) or value < 0 for value in (self.projection_ms, self.fusion_ms)
        ):
            raise P09PipelineError("pipeline timings must be finite and non-negative")

    @property
    def processing_latency_ms(self) -> float:
        return (self.completed_monotonic_ns - self.tick_monotonic_ns) / 1_000_000.0


class P09TrackingPipeline:
    """Rectify, detect, project and fuse only the current capacity-one snapshot."""

    def __init__(
        self,
        calibrations: dict[str, CameraProjectionCalibration],
        floor: FloorRectangle,
        detector: DetectorProtocol,
        tracker: AnonymousWorldTracker,
    ) -> None:
        if tuple(calibrations) != CAMERA_IDS:
            raise P09PipelineError("pipeline requires frozen calibrations in camera order")
        self._calibrations = calibrations
        self._rectifiers = {
            camera_id: LiveFrameRectifier(calibrations[camera_id]) for camera_id in CAMERA_IDS
        }
        self._floor = floor
        self._detector = detector
        self._tracker = tracker

    def process(self, snapshot: LatestFrameSnapshot) -> PipelineTickResult:
        camera_results: list[CameraTickResult] = []
        observations: list[WorldFloorObservation] = []
        person_counts = {camera_id: 0 for camera_id in CAMERA_IDS}
        projection_ms = 0.0
        for captured in snapshot.frames:
            camera_id = captured.identity.camera_id
            calibrated = self._rectifiers[camera_id].rectify(captured.frame_bgr)
            calibrated_identity = LiveFrameIdentity(
                camera_id=camera_id,
                frame_id=captured.identity.frame_id,
                acquisition_monotonic_ns=captured.identity.acquisition_monotonic_ns,
                observed_at_utc=captured.identity.observed_at_utc,
                source_pts=captured.identity.source_pts,
                source_time_base=captured.identity.source_time_base,
                width_pixels=504,
                height_pixels=280,
            )
            detection_result = self._detector.detect(calibrated_identity, calibrated)
            person_counts[camera_id] = len(detection_result.detections)
            projection_started = monotonic_ns()
            projections = tuple(
                project_detection_to_floor(
                    detection,
                    self._calibrations[camera_id],
                    self._floor,
                    snapshot.snapshot_monotonic_ns,
                )
                for detection in detection_result.detections
            )
            projection_ms += (monotonic_ns() - projection_started) / 1_000_000.0
            observations.extend(
                projection.observation
                for projection in projections
                if projection.observation is not None
            )
            camera_results.append(
                _camera_result(
                    camera_id, calibrated_identity, calibrated, detection_result, projections
                )
            )
        fusion_started = monotonic_ns()
        tracking = self._tracker.evaluate(
            tuple(observations), person_counts, snapshot.snapshot_monotonic_ns
        )
        fusion_ms = (monotonic_ns() - fusion_started) / 1_000_000.0
        return PipelineTickResult(
            snapshot.snapshot_monotonic_ns,
            monotonic_ns(),
            tuple(camera_results),
            snapshot.stale_camera_ids,
            snapshot.missing_camera_ids,
            snapshot.cross_camera_skew_ms,
            tracking,
            projection_ms,
            fusion_ms,
        )


def _camera_result(
    camera_id: str,
    frame: LiveFrameIdentity,
    calibrated: Array,
    detector: DetectorBatchResult,
    projections: tuple[ProjectionResult, ...],
) -> CameraTickResult:
    return CameraTickResult(
        camera_id,
        frame,
        calibrated,
        detector.detections,
        projections,
        detector.preprocessing_ms,
        detector.inference_ms,
        detector.postprocessing_ms,
    )
