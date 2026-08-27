from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_detector import DetectorBatchResult
from spatial_mapping_phase2.p09_fusion import AnonymousWorldTracker, FusionConfig
from spatial_mapping_phase2.p09_live_runtime import CapturedFrame, LatestFrameSnapshot
from spatial_mapping_phase2.p09_pipeline import P09TrackingPipeline
from spatial_mapping_phase2.p09_projection import CameraProjectionCalibration, FloorRectangle
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS, LiveFrameIdentity, TrackingState


class EmptyDetector:
    def detect(
        self, frame: LiveFrameIdentity, calibrated: NDArray[np.uint8]
    ) -> DetectorBatchResult:
        assert calibrated.shape == (280, 504, 3)
        return DetectorBatchResult((), 1.0, 2.0, 0.5, "test")


def _calibrations() -> dict[str, CameraProjectionCalibration]:
    result: dict[str, CameraProjectionCalibration] = {}
    for camera_id in CAMERA_IDS:
        result[camera_id] = CameraProjectionCalibration(
            camera_id,
            np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
            0.0,
            np.array([[300.0, 0.0, 252.0], [0.0, 300.0, 140.0], [0.0, 0.0, 1.0]]),
            np.eye(4),
        )
    return result


def test_empty_current_snapshot_fails_closed_without_position() -> None:
    tracker = AnonymousWorldTracker(FusionConfig(500.0, 250.0, 1.0, 3.0, 0.5))
    pipeline = P09TrackingPipeline(
        _calibrations(), FloorRectangle((-1.0, -1.0), (1.0, 1.0)), EmptyDetector(), tracker
    )
    now = 10_000_000_000
    identity = LiveFrameIdentity(
        CAMERA_IDS[0],
        "office-cam-01-test",
        now,
        datetime.now(UTC).isoformat(),
        None,
        None,
        1920,
        1080,
    )
    captured = CapturedFrame(identity, np.zeros((1080, 1920, 3), dtype=np.uint8))
    result = pipeline.process(LatestFrameSnapshot((captured,), (), CAMERA_IDS[1:], None, now))
    assert result.tracking.state is TrackingState.UNKNOWN
    assert result.tracking.current_xy_metres is None
    assert result.camera_results[0].total_detector_ms == 3.5
    assert result.missing_camera_ids == CAMERA_IDS[1:]
