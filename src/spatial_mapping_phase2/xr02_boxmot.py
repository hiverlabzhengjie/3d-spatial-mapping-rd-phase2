"""Auditable BoxMOT v22 adapter for XR02 camera-local tracker comparisons."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.xr02_supervision import CanonicalDetections


class XR02BoxMotError(RuntimeError):
    """Raised when the BoxMOT boundary changes or receives invalid inputs."""


class TrackerProtocol(Protocol):
    def update(
        self,
        dets: NDArray[np.float32],
        img: NDArray[np.uint8],
        embs: NDArray[np.float32] | None = None,
    ) -> object: ...


class TrackResultsProtocol(Protocol):
    xyxy: object
    id: object
    conf: object
    cls: object
    det_ind: object


@dataclass(frozen=True, slots=True)
class BoxMotProfile:
    profile_id: str
    tracker_type: str
    tracker_kwargs: dict[str, object]

    def __post_init__(self) -> None:
        if self.tracker_type not in {"botsort", "deepocsort"}:
            raise XR02BoxMotError("WP2 permits only BoT-SORT and Deep-OC-SORT")
        if self.tracker_type == "botsort" and self.tracker_kwargs.get("use_cmc") is not False:
            raise XR02BoxMotError("fixed-camera BoT-SORT profile must disable CMC")
        if self.tracker_type == "deepocsort" and self.tracker_kwargs.get("cmc_off") is not True:
            raise XR02BoxMotError("fixed-camera Deep-OC-SORT profile must disable CMC")


@dataclass(frozen=True, slots=True)
class LocalTrackRows:
    xyxy: NDArray[np.float32]
    local_track_ids: NDArray[np.int64]
    confidence: NDArray[np.float32]
    class_id: NDArray[np.int32]
    detection_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        arrays = (
            self.local_track_ids,
            self.confidence,
            self.class_id,
            self.detection_indices,
        )
        if self.xyxy.ndim != 2 or self.xyxy.shape[1:] != (4,):
            raise XR02BoxMotError("track boxes must have shape (N, 4)")
        if any(value.shape != (self.xyxy.shape[0],) for value in arrays):
            raise XR02BoxMotError("track result columns must have equal length")
        if np.any(self.local_track_ids < 0) or np.any(self.detection_indices < 0):
            raise XR02BoxMotError("track and detection indices must be non-negative")

    @property
    def count(self) -> int:
        return int(self.local_track_ids.size)


TrackerFactory = Callable[[BoxMotProfile], TrackerProtocol]


class CameraLocalTracker:
    """One BoxMOT tracker instance bound to exactly one scene camera."""

    def __init__(
        self,
        camera_id: str,
        profile: BoxMotProfile,
        factory: TrackerFactory | None = None,
    ) -> None:
        if not camera_id:
            raise XR02BoxMotError("camera ID is required")
        self.camera_id = camera_id
        self.profile = profile
        self._tracker = (factory or create_boxmot_tracker)(profile)
        self._last_frame_sequence: int | None = None

    def update(
        self,
        frame_sequence: int,
        image_bgr: NDArray[np.uint8],
        detections: CanonicalDetections,
        embeddings: NDArray[np.float32] | None,
    ) -> LocalTrackRows:
        if self._last_frame_sequence is not None and frame_sequence <= self._last_frame_sequence:
            raise XR02BoxMotError("camera-local frame sequence must increase strictly")
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise XR02BoxMotError("tracker image must be uint8 BGR")
        if embeddings is not None:
            embedding_array = np.asarray(embeddings, dtype=np.float32)
            if embedding_array.ndim != 2 or embedding_array.shape[0] != detections.count:
                raise XR02BoxMotError("precomputed embeddings must match detection count")
            if not np.all(np.isfinite(embedding_array)):
                raise XR02BoxMotError("precomputed embeddings must be finite")
        else:
            embedding_array = None
        det_rows = np.column_stack(
            (detections.xyxy, detections.confidence, detections.class_id)
        ).astype(np.float32, copy=False)
        raw = self._tracker.update(det_rows, image, embs=embedding_array)
        tracks = _normalize_track_results(raw)
        if np.any(tracks.detection_indices >= detections.count):
            raise XR02BoxMotError("BoxMOT returned an invalid detection index")
        self._last_frame_sequence = frame_sequence
        return tracks


def create_boxmot_tracker(profile: BoxMotProfile) -> TrackerProtocol:
    """Lazily create a pure-Python BoxMOT tracker with caller-supplied embeddings."""

    try:
        registry = importlib.import_module("boxmot.trackers.registry")
    except ModuleNotFoundError as error:
        raise XR02BoxMotError("pinned BoxMOT v22 runtime is unavailable") from error
    create_tracker: Any = getattr(registry, "create_tracker", None)
    if create_tracker is None:
        raise XR02BoxMotError("BoxMOT v22 create_tracker API is unavailable")
    tracker: Any = create_tracker(
        tracker_type=profile.tracker_type,
        per_class=False,
        class_ids=[0],
        class_names={0: "person"},
        tracker_kwargs=dict(profile.tracker_kwargs),
        tracker_backend="python",
        precomputed_reid=True,
    )
    return cast(TrackerProtocol, tracker)


def fixed_camera_profiles() -> tuple[BoxMotProfile, BoxMotProfile]:
    """Return the frozen initial comparison profiles; results decide the WP2 default."""

    botsort = BoxMotProfile(
        profile_id="botsort-fixed-v1",
        tracker_type="botsort",
        tracker_kwargs={
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.80,
            "use_cmc": False,
            "with_reid": True,
            "frame_rate": 25,
            "min_hits": 1,
        },
    )
    deepocsort = BoxMotProfile(
        profile_id="deepocsort-fixed-v1",
        tracker_type="deepocsort",
        tracker_kwargs={
            "det_thresh": 0.25,
            "max_age": 30,
            "min_hits": 1,
            "iou_thresh": 0.30,
            "delta_t": 3,
            "inertia": 0.20,
            "w_association_emb": 0.75,
            "alpha_fixed_emb": 0.95,
            "aw_param": 0.50,
            "embedding_off": False,
            "cmc_off": True,
            "aw_off": False,
            "Q_xy_scaling": 0.01,
            "Q_s_scaling": 0.0001,
        },
    )
    return botsort, deepocsort


def live_cadence_profile(
    source: BoxMotProfile,
    *,
    local_tracking_hz: float,
    track_buffer_frames: int,
    new_track_confidence: float | None = None,
    minimum_confirmation_hits: int | None = None,
) -> BoxMotProfile:
    """Bind frame-count tracker settings to the measured live clock.

    The WP2 profile remains immutable.  This creates an auditable WP4-derived
    profile whose frame-rate and loss buffer no longer inherit a misleading
    25-fps assumption while the service runs at another cadence.
    """

    if not math.isfinite(local_tracking_hz) or local_tracking_hz <= 0:
        raise XR02BoxMotError("live tracking frequency must be finite and positive")
    if track_buffer_frames <= 0:
        raise XR02BoxMotError("live track buffer must be positive")
    if new_track_confidence is not None and (
        not math.isfinite(new_track_confidence) or not 0 < new_track_confidence <= 1
    ):
        raise XR02BoxMotError("new-track confidence must be within (0, 1]")
    if minimum_confirmation_hits is not None and minimum_confirmation_hits <= 0:
        raise XR02BoxMotError("minimum confirmation hits must be positive")
    kwargs = dict(source.tracker_kwargs)
    kwargs["frame_rate"] = int(round(local_tracking_hz))
    if source.tracker_type == "botsort":
        kwargs["track_buffer"] = track_buffer_frames
        if new_track_confidence is not None:
            kwargs["new_track_thresh"] = new_track_confidence
        if minimum_confirmation_hits is not None:
            kwargs["min_hits"] = minimum_confirmation_hits
    elif source.tracker_type == "deepocsort":
        kwargs["max_age"] = track_buffer_frames
    return BoxMotProfile(
        profile_id=(
            f"{source.profile_id}.live-{local_tracking_hz:g}hz-buffer-{track_buffer_frames}f"
        ),
        tracker_type=source.tracker_type,
        tracker_kwargs=kwargs,
    )


def _normalize_track_results(value: object) -> LocalTrackRows:
    result = cast(TrackResultsProtocol, value)
    try:
        xyxy = np.asarray(result.xyxy, dtype=np.float32).reshape(-1, 4)
        local_track_ids = np.asarray(result.id, dtype=np.int64).reshape(-1)
        confidence = np.asarray(result.conf, dtype=np.float32).reshape(-1)
        class_id = np.asarray(result.cls, dtype=np.int32).reshape(-1)
        detection_indices = np.asarray(result.det_ind, dtype=np.int64).reshape(-1)
    except (AttributeError, TypeError, ValueError) as error:
        raise XR02BoxMotError("BoxMOT TrackResults contract changed") from error
    return LocalTrackRows(xyxy, local_track_ids, confidence, class_id, detection_indices)
