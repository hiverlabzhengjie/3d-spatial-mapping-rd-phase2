"""XR02 camera-local observation assembly over the accepted P08 floor authority."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_projection import (
    CameraProjectionCalibration,
    FloorRectangle,
    project_detection_to_floor,
)
from spatial_mapping_phase2.p09_tracking_domain import (
    FootpointKind,
    LiveFrameIdentity,
    PersonDetection,
)
from spatial_mapping_phase2.xr02_boxmot import LocalTrackRows
from spatial_mapping_phase2.xr02_journal import EmbeddingRepository, XR02JournalError
from spatial_mapping_phase2.xr02_local_domain import (
    CropQuality,
    EmbeddingReference,
    EmbeddingStatus,
    FootpointSource,
    FrameKey,
    LocalTrackKey,
    LocalTrackObservation,
    SceneContextKey,
    WorldProjectionStatus,
    XR02ContractError,
)
from spatial_mapping_phase2.xr02_supervision import CanonicalDetections, bottom_center_points


@dataclass(frozen=True, slots=True)
class CropQualityPolicy:
    minimum_area_pixels: float = 1_500.0
    minimum_visible_fraction: float = 0.80
    boundary_margin_pixels: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_area_pixels) or self.minimum_area_pixels <= 0:
            raise XR02ContractError("minimum crop area must be finite and positive")
        if not 0 < self.minimum_visible_fraction <= 1:
            raise XR02ContractError("minimum visible fraction must be within (0, 1]")
        if not math.isfinite(self.boundary_margin_pixels) or self.boundary_margin_pixels < 0:
            raise XR02ContractError("crop boundary margin must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EmbeddingCadence:
    every_n_tracking_frames: int = 1

    def __post_init__(self) -> None:
        if self.every_n_tracking_frames <= 0:
            raise XR02ContractError("embedding cadence must be positive")

    def is_due(self, frame_sequence: int) -> bool:
        return frame_sequence % self.every_n_tracking_frames == 0


def build_scene_context(
    scene_id: str,
    scene_epoch_id: str,
    geometry_sha256: str,
    floor_sha256: str,
    calibration_authority: dict[str, str],
    camera_policy_sha256: str | None = None,
) -> SceneContextKey:
    """Bind a scene epoch to exact immutable calibration source identities."""

    calibration_payload = json.dumps(
        calibration_authority, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    calibration_sha256 = hashlib.sha256(calibration_payload).hexdigest()
    return SceneContextKey(
        scene_id=scene_id,
        scene_epoch_id=scene_epoch_id,
        geometry_sha256=geometry_sha256,
        floor_sha256=floor_sha256,
        calibration_sha256=calibration_sha256,
        camera_policy_sha256=camera_policy_sha256,
    )


class P08ProjectionAdapter:
    """Delegate without approximation to the accepted, tested P09/P08 projection."""

    def __init__(
        self,
        calibrations: dict[str, CameraProjectionCalibration],
        floor: FloorRectangle,
    ) -> None:
        self._calibrations = dict(calibrations)
        self._floor = floor

    def project(
        self,
        frame: FrameKey,
        detection_index: int,
        confidence: float,
        bbox_xyxy: tuple[float, float, float, float],
        footpoint_uv: tuple[float, float],
        evaluated_monotonic_ns: int,
    ) -> tuple[WorldProjectionStatus, tuple[float, float] | None, str]:
        calibration = self._calibrations.get(frame.camera_id)
        if calibration is None:
            raise XR02ContractError("camera calibration is unavailable for this scene")
        p09_frame = LiveFrameIdentity(
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            acquisition_monotonic_ns=frame.acquisition_monotonic_ns,
            observed_at_utc=frame.observed_at_utc,
            source_pts=None,
            source_time_base=None,
            width_pixels=frame.width_pixels,
            height_pixels=frame.height_pixels,
        )
        p09_detection = PersonDetection(
            frame=p09_frame,
            detection_index=detection_index,
            confidence=confidence,
            bbox_xyxy=bbox_xyxy,
            image_point_uv=footpoint_uv,
            footpoint_kind=FootpointKind.BBOX_BOTTOM_CENTER,
        )
        result = project_detection_to_floor(
            p09_detection,
            calibration,
            self._floor,
            evaluated_monotonic_ns,
        )
        status = WorldProjectionStatus(result.status.value)
        xy = None if result.observation is None else result.observation.xy_metres
        return status, xy, result.reason


class LocalObservationAssembler:
    """Join tracker rows, crop evidence, embeddings and exact world projection."""

    def __init__(
        self,
        tracker_profile: str,
        projection: P08ProjectionAdapter,
        embedding_store: EmbeddingRepository,
        quality_policy: CropQualityPolicy | None = None,
        cadence: EmbeddingCadence | None = None,
    ) -> None:
        self._tracker_profile = tracker_profile
        self._projection = projection
        self._embedding_store = embedding_store
        self._quality_policy = quality_policy or CropQualityPolicy()
        self._cadence = cadence or EmbeddingCadence()

    @property
    def quality_policy(self) -> CropQualityPolicy:
        """Expose the immutable gate shared by durable and transient appearance."""

        return self._quality_policy

    def assemble(
        self,
        frame: FrameKey,
        detections: CanonicalDetections,
        tracks: LocalTrackRows,
        embeddings: NDArray[np.float32] | None,
        evaluated_monotonic_ns: int,
    ) -> tuple[LocalTrackObservation, ...]:
        if embeddings is not None:
            vectors = np.asarray(embeddings, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != detections.count:
                raise XR02ContractError("embedding batch must match canonical detections")
        else:
            vectors = None
        footpoints = bottom_center_points(detections)
        observations: list[LocalTrackObservation] = []
        for row_index in range(tracks.count):
            detection_index = int(tracks.detection_indices[row_index])
            if detection_index >= detections.count:
                raise XR02ContractError("tracker detection index is out of range")
            box = tuple(float(value) for value in detections.xyxy[detection_index])
            bbox_xyxy = (box[0], box[1], box[2], box[3])
            footpoint = tuple(float(value) for value in footpoints[detection_index])
            footpoint_uv = _inside_image_point(
                (footpoint[0], footpoint[1]), frame.width_pixels, frame.height_pixels
            )
            quality = evaluate_crop_quality(
                bbox_xyxy,
                frame.width_pixels,
                frame.height_pixels,
                self._quality_policy.boundary_margin_pixels,
            )
            status, reference = self._embedding_evidence(
                frame.frame_sequence, detection_index, quality, vectors
            )
            projection_status, xy, reason = self._projection.project(
                frame=frame,
                detection_index=detection_index,
                confidence=float(detections.confidence[detection_index]),
                bbox_xyxy=bbox_xyxy,
                footpoint_uv=footpoint_uv,
                evaluated_monotonic_ns=evaluated_monotonic_ns,
            )
            observations.append(
                LocalTrackObservation(
                    frame=frame,
                    track=LocalTrackKey(
                        scene_context_sha256=frame.scene.context_sha256,
                        camera_id=frame.camera_id,
                        tracker_profile=self._tracker_profile,
                        local_track_id=int(tracks.local_track_ids[row_index]),
                    ),
                    detection_index=detection_index,
                    confidence=float(detections.confidence[detection_index]),
                    bbox_xyxy=bbox_xyxy,
                    footpoint_uv=footpoint_uv,
                    footpoint_source=FootpointSource.BBOX_BOTTOM_CENTER,
                    crop_quality=quality,
                    embedding_status=status,
                    embedding=reference,
                    projection_status=projection_status,
                    world_xy_metres=xy,
                    projection_reason=reason,
                )
            )
        return tuple(observations)

    def _embedding_evidence(
        self,
        frame_sequence: int,
        detection_index: int,
        quality: CropQuality,
        embeddings: NDArray[np.float32] | None,
    ) -> tuple[EmbeddingStatus, EmbeddingReference | None]:
        if not self._cadence.is_due(frame_sequence):
            return EmbeddingStatus.NOT_DUE, None
        if (
            quality.area_pixels < self._quality_policy.minimum_area_pixels
            or quality.visible_fraction < self._quality_policy.minimum_visible_fraction
        ):
            return EmbeddingStatus.LOW_QUALITY, None
        if embeddings is None:
            return EmbeddingStatus.MODEL_UNAVAILABLE, None
        try:
            reference = self._embedding_store.put(embeddings[detection_index])
        except XR02JournalError:
            return EmbeddingStatus.EXTRACTION_FAILED, None
        return EmbeddingStatus.AVAILABLE, reference


def evaluate_crop_quality(
    bbox_xyxy: tuple[float, float, float, float],
    width_pixels: int,
    height_pixels: int,
    boundary_margin_pixels: float,
) -> CropQuality:
    x1, y1, x2, y2 = bbox_xyxy
    raw_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    clipped_x1 = min(max(x1, 0.0), float(width_pixels))
    clipped_y1 = min(max(y1, 0.0), float(height_pixels))
    clipped_x2 = min(max(x2, 0.0), float(width_pixels))
    clipped_y2 = min(max(y2, 0.0), float(height_pixels))
    visible_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    if raw_area <= 0 or visible_area <= 0:
        raise XR02ContractError("crop must overlap the frame with positive area")
    sides: list[str] = []
    if x1 <= boundary_margin_pixels:
        sides.append("left")
    if y1 <= boundary_margin_pixels:
        sides.append("top")
    if x2 >= width_pixels - boundary_margin_pixels:
        sides.append("right")
    if y2 >= height_pixels - boundary_margin_pixels:
        sides.append("bottom")
    return CropQuality(
        visible_fraction=visible_area / raw_area,
        area_pixels=visible_area,
        aspect_ratio=(clipped_x2 - clipped_x1) / (clipped_y2 - clipped_y1),
        clipped_sides=tuple(sides),
    )


def _inside_image_point(
    point_uv: tuple[float, float], width_pixels: int, height_pixels: int
) -> tuple[float, float]:
    return (
        min(max(point_uv[0], 0.0), math.nextafter(float(width_pixels), 0.0)),
        min(max(point_uv[1], 0.0), math.nextafter(float(height_pixels), 0.0)),
    )
