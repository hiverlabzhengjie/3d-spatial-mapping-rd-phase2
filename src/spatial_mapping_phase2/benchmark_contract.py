"""Stable, model-agnostic records for P00 runtime comparison runs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

CHECKPOINT_ID = "DA3NESTED-GIANT-LARGE-1.1"
ALLOWED_VIEW_COUNTS = frozenset({1, 2, 3})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkContractError(ValueError):
    """Raised when a run cannot be compared or provenance-bound."""


@dataclass(frozen=True)
class BenchmarkCase:
    """One controlled DA3 inference case, independent of runtime implementation."""

    case_id: str
    view_count: int
    width_px: int
    height_px: int
    batch_size: int
    input_manifest_sha256: str

    def validate(self) -> None:
        if not self.case_id:
            raise BenchmarkContractError("Benchmark case ID must not be empty.")
        if self.view_count not in ALLOWED_VIEW_COUNTS:
            raise BenchmarkContractError("Benchmark view count must be one of: 1, 2, 3.")
        if self.width_px <= 0 or self.height_px <= 0:
            raise BenchmarkContractError("Benchmark resolution must be positive in pixels.")
        if self.batch_size != 1:
            raise BenchmarkContractError("P00 benchmark batch size must be 1 for comparability.")
        if not _SHA256_PATTERN.fullmatch(self.input_manifest_sha256):
            raise BenchmarkContractError("Input manifest must be a lowercase SHA-256 digest.")


@dataclass(frozen=True)
class BenchmarkRunManifest:
    """Minimum provenance that every runtime must write for a benchmark result."""

    runtime_id: str
    code_revision: str
    dependency_lock_sha256: str
    checkpoint_sha256: str
    case: BenchmarkCase
    started_at_utc: str

    @classmethod
    def create(
        cls,
        *,
        runtime_id: str,
        code_revision: str,
        dependency_lock_sha256: str,
        checkpoint_sha256: str,
        case: BenchmarkCase,
    ) -> BenchmarkRunManifest:
        return cls(
            runtime_id=runtime_id,
            code_revision=code_revision,
            dependency_lock_sha256=dependency_lock_sha256,
            checkpoint_sha256=checkpoint_sha256,
            case=case,
            started_at_utc=datetime.now(UTC).isoformat(),
        )

    def validate(self) -> None:
        if not self.runtime_id or not self.code_revision:
            raise BenchmarkContractError("Runtime ID and code revision must not be empty.")
        for name, value in (
            ("Dependency lock", self.dependency_lock_sha256),
            ("Checkpoint", self.checkpoint_sha256),
        ):
            if not _SHA256_PATTERN.fullmatch(value):
                raise BenchmarkContractError(f"{name} must be a lowercase SHA-256 digest.")
        self.case.validate()
        try:
            datetime.fromisoformat(self.started_at_utc)
        except ValueError as error:
            raise BenchmarkContractError("Run start time must be ISO-8601.") from error

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation after complete validation."""
        self.validate()
        return asdict(self)
