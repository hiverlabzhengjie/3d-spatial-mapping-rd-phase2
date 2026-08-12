"""Explicit capture-time mapping for future source-frame ingestion.

This small, dependency-free contract is adapted from Phase 1 source commit
``e9e038c498c17d83bcfa65a71e1c355e3f4aa8d7``. It maps source PTS seconds into
the capture timeline; it does not decode media, infer synchronization, or
represent model-completion time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimestampTransform:
    """Affine mapping from source PTS seconds to capture-time seconds.

    ``scale`` and ``offset_seconds`` must be established by a versioned capture
    session or stream-profile record in P03. The result is not a processing or
    model-completion timestamp.
    """

    scale: float = 1.0
    offset_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("timestamp scale must be finite and positive")
        if not math.isfinite(self.offset_seconds):
            raise ValueError("timestamp offset must be finite")

    def apply(self, source_timestamp_seconds: float) -> float:
        """Map one finite source PTS to a finite, non-negative capture time."""
        if not math.isfinite(source_timestamp_seconds):
            raise ValueError("source timestamp must be finite")
        capture_timestamp = self.scale * source_timestamp_seconds + self.offset_seconds
        if not math.isfinite(capture_timestamp) or capture_timestamp < 0:
            raise ValueError("mapped capture timestamp must be finite and non-negative")
        return capture_timestamp

