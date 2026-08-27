"""Optional, replaceable Supervision utilities at the XR02 presentation boundary."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray


class XR02SupervisionError(RuntimeError):
    """Raised when the optional Supervision boundary cannot preserve canonical data."""


@dataclass(frozen=True, slots=True)
class CanonicalDetections:
    """Library-neutral detector output used by the XR02 core."""

    xyxy: NDArray[np.float32]
    confidence: NDArray[np.float32]
    class_id: NDArray[np.int32]

    def __post_init__(self) -> None:
        boxes = np.asarray(self.xyxy, dtype=np.float32).copy()
        confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1).copy()
        class_id = np.asarray(self.class_id, dtype=np.int32).reshape(-1).copy()
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise XR02SupervisionError("canonical boxes must have shape (N, 4)")
        if boxes.shape[0] != confidence.size or confidence.size != class_id.size:
            raise XR02SupervisionError("canonical detection arrays must have equal length")
        if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(confidence)):
            raise XR02SupervisionError("canonical detections must be finite")
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise XR02SupervisionError("canonical boxes must have positive area")
        if np.any((confidence < 0) | (confidence > 1)):
            raise XR02SupervisionError("canonical confidence must be within [0, 1]")
        for value in (boxes, confidence, class_id):
            value.setflags(write=False)
        object.__setattr__(self, "xyxy", boxes)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "class_id", class_id)

    @property
    def count(self) -> int:
        return int(self.confidence.size)


class SupervisionAdapter:
    """Convert canonical detections for optional annotation; own no CV policy."""

    def __init__(self, supervision: ModuleType | None = None) -> None:
        try:
            self._sv = supervision or importlib.import_module("supervision")
        except ModuleNotFoundError as error:
            raise XR02SupervisionError(
                "Supervision is optional and unavailable; core tracking remains usable"
            ) from error
        if not hasattr(self._sv, "Detections"):
            raise XR02SupervisionError("Supervision lacks the required Detections contract")

    @property
    def version(self) -> str:
        return str(getattr(self._sv, "__version__", "unknown"))

    def to_supervision(self, detections: CanonicalDetections) -> Any:
        return self._sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy(),
            class_id=detections.class_id.copy(),
        )


def bottom_center_points(detections: CanonicalDetections) -> NDArray[np.float32]:
    """Return deterministic bbox-bottom centres without requiring Supervision."""

    if detections.count == 0:
        return np.empty((0, 2), dtype=np.float32)
    result = np.column_stack(
        (
            (detections.xyxy[:, 0] + detections.xyxy[:, 2]) * 0.5,
            detections.xyxy[:, 3],
        )
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(result)


def person_detections_from_boxmot(
    raw_detections: NDArray[np.floating[Any]],
) -> CanonicalDetections:
    """Normalize BoxMOT detector rows [x1,y1,x2,y2,conf,class] to the core."""

    rows = np.asarray(raw_detections, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise XR02SupervisionError("BoxMOT detector output must have at least six columns")
    people = rows[rows[:, 5].astype(np.int32) == 0]
    return CanonicalDetections(
        xyxy=people[:, :4],
        confidence=people[:, 4],
        class_id=people[:, 5].astype(np.int32),
    )
