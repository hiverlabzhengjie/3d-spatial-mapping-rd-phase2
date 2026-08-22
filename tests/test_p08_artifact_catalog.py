from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from spatial_mapping_phase2.p08_artifact_catalog import (
    ArtifactCatalogError,
    SceneArtifactCatalog,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register(
    catalog: SceneArtifactCatalog,
    path: Path,
    artifact_id: str,
    milestone_key: str,
    *,
    selected: bool = False,
    parents: tuple[str, ...] = (),
) -> None:
    catalog.register(
        artifact_id=artifact_id,
        milestone_key=milestone_key,
        phase_id="P02" if milestone_key == "floor-plan-source" else "P07",
        kind="floor-plan-pdf" if milestone_key == "floor-plan-source" else "point-cloud-npz",
        path=path,
        sha256=_sha256(path),
        display_name=path.name,
        significance="test milestone",
        selected=selected,
        metadata={"selectable": True},
        parent_artifact_ids=parents,
    )


def test_catalog_versions_selection_impacts_and_preserves_files(tmp_path: Path) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    plan_one = tmp_path / "plan-one.pdf"
    plan_two = tmp_path / "plan-two.pdf"
    geometry = tmp_path / "geometry.npz"
    plan_one.write_bytes(b"plan-one")
    plan_two.write_bytes(b"plan-two")
    geometry.write_bytes(b"geometry")
    _register(catalog, plan_one, "plan-one", "floor-plan-source", selected=True)
    _register(catalog, plan_two, "plan-two", "floor-plan-source")
    _register(catalog, geometry, "geometry-one", "reconstructed-geometry", selected=True)

    impact = catalog.selection_impact("plan-two")
    assert impact["requires_confirmation"] is True
    assert "reconstructed-geometry" in impact["downstream_selections"]
    with pytest.raises(ArtifactCatalogError, match="confirm"):
        catalog.select("plan-two", confirm_impacts=False)

    selected = catalog.select("plan-two", confirm_impacts=True)
    assert selected["cleared_downstream_selections"] == ["reconstructed-geometry"]
    assert catalog.selected("floor-plan-source")["artifact_id"] == "plan-two"  # type: ignore[index]
    assert catalog.selected("reconstructed-geometry") is None
    assert geometry.read_bytes() == b"geometry"


def test_catalog_verify_archive_restore_and_selected_protection(tmp_path: Path) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    _register(catalog, first, "first", "floor-plan-source", selected=True)
    _register(catalog, second, "second", "floor-plan-source")

    with pytest.raises(ArtifactCatalogError, match="select another"):
        catalog.set_archived("first", True)
    assert catalog.set_archived("second", True)["lifecycle"] == "archived"
    assert second.is_file()
    assert catalog.set_archived("second", False)["lifecycle"] == "available"
    second.write_bytes(b"changed")
    assert catalog.verify("second")["lifecycle"] == "corrupt"
    second.unlink()
    assert catalog.verify("second")["lifecycle"] == "missing"


def test_catalog_rejects_supporting_record_selection(tmp_path: Path) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    record = tmp_path / "record.json"
    record.write_text("{}", encoding="utf-8")
    catalog.register(
        artifact_id="supporting-record",
        milestone_key="reconstructed-geometry",
        phase_id="P07",
        kind="geometry-adoption-manifest",
        path=record,
        sha256=_sha256(record),
        display_name=record.name,
        significance="supporting record",
        metadata={"selectable": False},
    )
    with pytest.raises(ArtifactCatalogError, match="supporting"):
        catalog.select("supporting-record", confirm_impacts=True)


def test_catalog_prevents_archiving_parent_of_current_result(tmp_path: Path) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    plan = tmp_path / "plan.pdf"
    geometry = tmp_path / "geometry.npz"
    plan.write_bytes(b"plan")
    geometry.write_bytes(b"geometry")
    _register(catalog, plan, "plan", "floor-plan-source")
    _register(
        catalog,
        geometry,
        "geometry",
        "reconstructed-geometry",
        selected=True,
        parents=("plan",),
    )
    with pytest.raises(ArtifactCatalogError, match="current later result"):
        catalog.set_archived("plan", True)
    assert plan.is_file()


def test_catalog_groups_internal_milestones_into_operator_workflow_sections(
    tmp_path: Path,
) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    plan = tmp_path / "plan.pdf"
    plan.write_bytes(b"plan")
    _register(catalog, plan, "plan", "floor-plan-source", selected=True)

    status = catalog.status()
    assert status["schema_version"] == "p08-scene-artifact-catalog-status-v3"
    assert [item["section_key"] for item in status["workflow_sections"]] == [
        "setup",
        "capture",
        "calibration",
        "reconstruction",
        "floor",
        "final",
    ]
    assert status["workflow_sections"][0]["current_items"][0]["artifact_id"] == "plan"
    assert status["storage"]["current_version_count"] == 1


def test_permanent_deletion_requires_fresh_button_token_and_hides_deleted_version(
    tmp_path: Path,
) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    old = tmp_path / "old.pdf"
    current = tmp_path / "current.pdf"
    old.write_bytes(b"old")
    current.write_bytes(b"current")
    _register(catalog, old, "old", "floor-plan-source")
    _register(catalog, current, "current", "floor-plan-source", selected=True)

    impact = catalog.deletion_impact("old")
    assert impact["allowed"] is True
    assert len(impact["deletion_token"]) == 64
    with pytest.raises(ArtifactCatalogError, match="stale"):
        catalog.delete_permanently("old", "wrong-token")
    assert old.is_file()

    result = catalog.delete_permanently("old", impact["deletion_token"])
    assert result["status"] == "deleted"
    assert result["deleted_byte_count"] == 3
    assert not old.exists()
    deleted = next(
        version
        for milestone in catalog.status()["milestones"]
        for version in milestone["versions"]
        if version["artifact_id"] == "old"
    )
    assert deleted["lifecycle"] == "deleted"
    assert catalog.verify("old")["lifecycle"] == "deleted"
    setup = catalog.status()["workflow_sections"][0]
    assert [version["artifact_id"] for version in setup["past_items"]] == []
    assert catalog.status()["storage"]["past_version_count"] == 0
    assert catalog.status()["storage"]["deleted_version_count"] == 1


def test_permanent_deletion_blocks_current_and_dependency_parent(tmp_path: Path) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    plan = tmp_path / "plan.pdf"
    geometry = tmp_path / "geometry.npz"
    plan.write_bytes(b"plan")
    geometry.write_bytes(b"geometry")
    _register(catalog, plan, "plan", "floor-plan-source")
    _register(
        catalog,
        geometry,
        "geometry",
        "reconstructed-geometry",
        selected=True,
        parents=("plan",),
    )

    parent_impact = catalog.deletion_impact("plan")
    assert parent_impact["allowed"] is False
    assert parent_impact["dependents"][0]["artifact_id"] == "geometry"
    assert "depend" in " ".join(parent_impact["blockers"]).lower()
    current_impact = catalog.deletion_impact("geometry")
    assert current_impact["allowed"] is False
    assert "current workflow" in " ".join(current_impact["blockers"]).lower()
    assert plan.is_file() and geometry.is_file()


def test_batch_deletion_allows_a_selected_dependency_chain_and_deletes_leaf_first(
    tmp_path: Path,
) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    parent = tmp_path / "parent.pdf"
    child = tmp_path / "child.npz"
    current = tmp_path / "current.npz"
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    current.write_bytes(b"current")
    _register(catalog, parent, "parent", "floor-plan-source")
    _register(
        catalog,
        child,
        "child",
        "reconstructed-geometry",
        parents=("parent",),
    )
    _register(
        catalog,
        current,
        "current",
        "reconstructed-geometry",
        selected=True,
    )

    parent_only = catalog.batch_deletion_impact(("parent",))
    assert parent_only["all_allowed"] is False
    batch = catalog.batch_deletion_impact(("parent", "child"))
    assert batch["all_allowed"] is True
    tokens = {str(item["artifact_id"]): str(item["deletion_token"]) for item in batch["items"]}
    result = catalog.delete_batch_permanently(tokens)
    assert result["deleted_artifact_count"] == 2
    assert result["deleted_byte_count"] == len(b"parent") + len(b"child")
    assert not parent.exists() and not child.exists()
    assert current.is_file()


@pytest.mark.parametrize(
    "retention_class",
    ["accepted-predecessor", "selected-authority", "required-rollback"],
)
def test_protected_retention_blocks_single_and_batch_deletion_after_deselection(
    tmp_path: Path, retention_class: str
) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    protected = tmp_path / "protected.npz"
    current = tmp_path / "current.npz"
    protected.write_bytes(b"protected")
    current.write_bytes(b"current")
    _register(
        catalog,
        protected,
        "protected",
        "reconstructed-geometry",
        selected=True,
    )
    catalog.protect_retention(
        "protected",
        retention_class,
        "authority-critical test record",
        source="test policy",
    )
    _register(catalog, current, "current", "reconstructed-geometry")
    catalog.select("current", confirm_impacts=True)

    impact = catalog.deletion_impact("protected")
    assert impact["allowed"] is False
    assert impact["deletion_token"] is None
    assert impact["protected_retention"]["class"] == retention_class
    assert "protected retention" in " ".join(impact["blockers"]).lower()
    batch = catalog.batch_deletion_impact(("protected",))
    assert batch["all_allowed"] is False
    with pytest.raises(ArtifactCatalogError, match="protected retention"):
        catalog.delete_batch_permanently({"protected": "not-issued"})
    assert protected.is_file()
    version = next(
        version
        for milestone in catalog.status()["milestones"]
        for version in milestone["versions"]
        if version["artifact_id"] == "protected"
    )
    assert version["retention"]["class"] == retention_class
    assert catalog.status()["storage"]["protected_version_count"] == 1


def test_protected_retention_is_idempotent_and_cannot_be_reclassified(tmp_path: Path) -> None:
    catalog = SceneArtifactCatalog(tmp_path / "catalog.sqlite3", "project", "scene")
    artifact = tmp_path / "rollback.json"
    artifact.write_text("{}", encoding="utf-8")
    _register(catalog, artifact, "rollback", "reconstructed-geometry")
    first = catalog.protect_retention(
        "rollback",
        "required-rollback",
        "exact rollback identity",
        source="test contract",
    )
    second = catalog.protect_retention(
        "rollback",
        "required-rollback",
        "exact rollback identity",
        source="test contract",
    )
    assert second == first
    with pytest.raises(ArtifactCatalogError, match="different"):
        catalog.protect_retention(
            "rollback",
            "selected-authority",
            "different policy",
            source="test contract",
        )
