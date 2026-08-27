"""Small, dependency-free contracts for the XR02 WP1 feasibility benchmark.

The GPU benchmark itself remains a thin script because its vendor dependencies live in the
isolated XR02 worker runtime.  This module owns the reproducibility checks and statistics that can
be tested in the normal project runtime.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar


class XR02WP1Error(RuntimeError):
    """Raised when a WP1 benchmark boundary or immutable input is invalid."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Immutable local input identity."""

    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Milliseconds summary for one measured operation."""

    count: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    if chunk_bytes <= 0:
        raise XR02WP1Error("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_file(path: Path, *, expected_sha256: str | None = None) -> FileIdentity:
    """Return a local identity and fail closed on absence or hash mismatch."""

    if not path.is_file():
        raise XR02WP1Error(f"Required local asset is missing: {path}")
    identity = FileIdentity(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))
    if expected_sha256 is not None and identity.sha256 != expected_sha256.lower():
        raise XR02WP1Error(
            f"SHA-256 mismatch for {path}: expected {expected_sha256.lower()}, "
            f"observed {identity.sha256}"
        )
    return identity


def _linear_percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise XR02WP1Error("quantile must be between zero and one")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_ms(values: Sequence[float]) -> TimingSummary:
    """Summarize non-negative finite timing observations."""

    if not values:
        raise XR02WP1Error("At least one timing observation is required")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise XR02WP1Error("Timing observations must be finite and non-negative")
    ordered = sorted(numeric)
    return TimingSummary(
        count=len(ordered),
        median_ms=_linear_percentile(ordered, 0.5),
        p95_ms=_linear_percentile(ordered, 0.95),
        mean_ms=sum(ordered) / len(ordered),
        minimum_ms=ordered[0],
        maximum_ms=ordered[-1],
    )


T = TypeVar("T")


def batches(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield deterministic contiguous batches."""

    if batch_size <= 0:
        raise XR02WP1Error("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
