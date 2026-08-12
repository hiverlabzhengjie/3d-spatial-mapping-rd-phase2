from pathlib import Path

import pytest

from spatial_mapping_phase2.artifact_layout import (
    ARTIFACT_SUBDIRECTORIES,
    ArtifactLayout,
    ArtifactLayoutError,
)


def test_layout_returns_paths_in_policy_order(tmp_path: Path) -> None:
    layout = ArtifactLayout.from_root(tmp_path)

    assert layout.required_paths() == tuple(
        tmp_path / category for category in ARTIFACT_SUBDIRECTORIES
    )


def test_layout_rejects_unknown_category(tmp_path: Path) -> None:
    layout = ArtifactLayout.from_root(tmp_path)

    with pytest.raises(ArtifactLayoutError, match="Unknown artifact category"):
        layout.path_for("secrets")


def test_layout_rejects_missing_required_directories(tmp_path: Path) -> None:
    layout = ArtifactLayout.from_root(tmp_path)

    with pytest.raises(ArtifactLayoutError, match="Artifact layout is incomplete"):
        layout.validate_complete()


def test_layout_validates_complete_directory_set(tmp_path: Path) -> None:
    for category in ARTIFACT_SUBDIRECTORIES:
        (tmp_path / category).mkdir()
    layout = ArtifactLayout.from_root(tmp_path)

    layout.validate_complete()


def test_layout_rejects_relative_or_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ArtifactLayoutError, match="absolute"):
        ArtifactLayout.from_root("relative-root")
    with pytest.raises(ArtifactLayoutError, match="does not exist"):
        ArtifactLayout.from_root(tmp_path / "missing")
