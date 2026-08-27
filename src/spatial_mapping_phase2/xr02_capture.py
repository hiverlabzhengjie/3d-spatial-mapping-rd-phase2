"""Deployment-scalable latest-frame admission independent of TrackStudio."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.xr02_local_domain import (
    CameraAvailability,
    FrameKey,
    SceneContextKey,
    XR02ContractError,
)


@dataclass(frozen=True, slots=True)
class SceneFrame:
    identity: FrameKey
    image_bgr: NDArray[np.uint8]

    def __post_init__(self) -> None:
        image = np.asarray(self.image_bgr, dtype=np.uint8).copy()
        if image.shape != (self.identity.height_pixels, self.identity.width_pixels, 3):
            raise XR02ContractError("frame image and identity dimensions disagree")
        image.setflags(write=False)
        object.__setattr__(self, "image_bgr", image)


@dataclass(frozen=True, slots=True)
class CameraFrameState:
    camera_id: str
    availability: CameraAvailability
    frame: SceneFrame | None
    age_ms: float | None
    published_frames: int
    replaced_frames: int


@dataclass(frozen=True, slots=True)
class SceneFrameSnapshot:
    scene: SceneContextKey
    evaluated_monotonic_ns: int
    camera_states: tuple[CameraFrameState, ...]
    cross_camera_skew_ms: float | None


class _LatestFrameSlot:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._latest: SceneFrame | None = None
        self._published = 0
        self._replaced = 0

    def publish(self, frame: SceneFrame) -> None:
        if frame.identity.camera_id != self.camera_id:
            raise XR02ContractError("published frame camera disagrees with slot")
        with self._lock:
            if self._latest is not None:
                if (
                    frame.identity.acquisition_monotonic_ns
                    <= self._latest.identity.acquisition_monotonic_ns
                ):
                    raise XR02ContractError("camera acquisition time must increase strictly")
                self._replaced += 1
            self._latest = frame
            self._published += 1

    def inspect(self, now_ns: int, maximum_age_ms: float) -> CameraFrameState:
        with self._lock:
            frame = self._latest
            published = self._published
            replaced = self._replaced
        if frame is None:
            return CameraFrameState(
                self.camera_id, CameraAvailability.MISSING, None, None, published, replaced
            )
        age_ms = (now_ns - frame.identity.acquisition_monotonic_ns) / 1_000_000.0
        if age_ms < 0:
            raise XR02ContractError("snapshot time precedes camera acquisition")
        if age_ms > maximum_age_ms:
            return CameraFrameState(
                self.camera_id, CameraAvailability.STALE, None, age_ms, published, replaced
            )
        return CameraFrameState(
            self.camera_id, CameraAvailability.CURRENT, frame, age_ms, published, replaced
        )


class SceneLatestFrameBuffer:
    """Capacity-one latest-frame slots for an arbitrary validated camera roster."""

    def __init__(self, scene: SceneContextKey, camera_ids: tuple[str, ...]) -> None:
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise XR02ContractError("scene camera roster must be non-empty and unique")
        self.scene = scene
        slots = {camera_id: _LatestFrameSlot(camera_id) for camera_id in camera_ids}
        self._slots = MappingProxyType(slots)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._slots)

    def publish(self, frame: SceneFrame) -> None:
        if frame.identity.scene != self.scene:
            raise XR02ContractError("frame belongs to another scene context")
        try:
            slot = self._slots[frame.identity.camera_id]
        except KeyError as error:
            raise XR02ContractError("frame camera is absent from the scene roster") from error
        slot.publish(frame)

    def snapshot(self, now_ns: int, maximum_age_ms: float) -> SceneFrameSnapshot:
        if now_ns < 0 or not math.isfinite(maximum_age_ms) or maximum_age_ms <= 0:
            raise XR02ContractError("snapshot time and maximum age must be positive")
        states = tuple(slot.inspect(now_ns, maximum_age_ms) for slot in self._slots.values())
        times = [
            state.frame.identity.acquisition_monotonic_ns
            for state in states
            if state.frame is not None
        ]
        skew = (max(times) - min(times)) / 1_000_000.0 if len(times) >= 2 else None
        return SceneFrameSnapshot(self.scene, now_ns, states, skew)
