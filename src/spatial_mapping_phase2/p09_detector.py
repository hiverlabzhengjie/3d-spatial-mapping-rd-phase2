"""Detector-neutral contracts for the P09 anonymous tracking pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from numpy.typing import NDArray

from spatial_mapping_phase2.p09_tracking_domain import LiveFrameIdentity, PersonDetection

Array = NDArray[Any]


class P09DetectorError(RuntimeError):
    """Raised when the P09 detector contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class DetectorBatchResult:
    detections: tuple[PersonDetection, ...]
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    backend: str

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                self.preprocessing_ms,
                self.inference_ms,
                self.postprocessing_ms,
            )
        ):
            raise P09DetectorError("detector timings must be finite and non-negative")
        if not self.backend:
            raise P09DetectorError("detector backend label is required")


class DetectorProtocol(Protocol):
    def detect(
        self, frame: LiveFrameIdentity, calibrated_frame_bgr: Array
    ) -> DetectorBatchResult: ...
