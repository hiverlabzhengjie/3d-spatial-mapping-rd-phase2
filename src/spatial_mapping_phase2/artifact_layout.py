"""Validated locations for untracked Phase 2 artifacts.

The project repository holds code and lightweight evidence. Large or sensitive runtime artifacts
live under one configured root outside the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ARTIFACT_SUBDIRECTORIES: tuple[str, ...] = (
    "model_weights",
    "captures",
    "cache",
    "runs",
    "exports",
)


class ArtifactLayoutError(ValueError):
    """Raised when an artifact-root configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class ArtifactLayout:
    """Project-owned directories below one existing absolute artifact root."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> ArtifactLayout:
        """Validate an existing absolute root without creating or modifying it."""
        candidate = Path(root)
        if not candidate.is_absolute():
            raise ArtifactLayoutError("Artifact root must be an absolute path.")
        if not candidate.exists():
            raise ArtifactLayoutError(f"Artifact root does not exist: {candidate}")
        if not candidate.is_dir():
            raise ArtifactLayoutError(f"Artifact root is not a directory: {candidate}")
        return cls(root=candidate)

    def path_for(self, category: str) -> Path:
        """Return the isolated path for a permitted artifact category."""
        if category not in ARTIFACT_SUBDIRECTORIES:
            allowed = ", ".join(ARTIFACT_SUBDIRECTORIES)
            raise ArtifactLayoutError(
                f"Unknown artifact category {category!r}; expected one of: {allowed}."
            )
        return self.root / category

    def required_paths(self) -> tuple[Path, ...]:
        """Return the complete required layout in stable policy order."""
        return tuple(self.path_for(category) for category in ARTIFACT_SUBDIRECTORIES)

    def validate_complete(self) -> None:
        """Reject a root whose required category directories are missing or invalid."""
        missing_or_invalid = [
            path for path in self.required_paths() if not path.is_dir()
        ]
        if missing_or_invalid:
            names = ", ".join(str(path) for path in missing_or_invalid)
            raise ArtifactLayoutError(f"Artifact layout is incomplete: {names}")
