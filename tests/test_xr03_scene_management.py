from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    CameraConfig,
    P08WorkflowError,
    PhaseRecord,
    PhaseState,
    SceneWorkspace,
    SceneWorkspaceRepository,
)
from spatial_mapping_phase2.xr03_scene_management import (
    SceneLifecycle,
    SceneReadiness,
    SceneRegistry,
    StorageOwnership,
)


def registry(tmp_path: Path) -> SceneRegistry:
    return SceneRegistry(tmp_path / "control" / "scenes.sqlite3", tmp_path / "managed")


def existing_workspace(tmp_path: Path) -> Path:
    root = tmp_path / "existing-office"
    scene = SceneWorkspace(
        project_id="phase2",
        scene_id="office",
        display_name="Existing office",
        artifact_root=(tmp_path / "existing-artifacts").resolve(),
        cameras=(CameraConfig("office-cam-01", "Office camera 1"),),
        phases=tuple(PhaseRecord(phase_id, PhaseState.READY, "Ready") for phase_id in PHASE_ORDER),
    )
    SceneWorkspaceRepository(root).create(scene)
    return root


def test_create_scene_is_isolated_and_credential_free(tmp_path: Path) -> None:
    store = registry(tmp_path)

    record = store.create_scene(" Second Office ", ("Entrance", "Workshop"))

    assert record.display_name == "Second Office"
    assert record.readiness is SceneReadiness.DRAFT
    assert record.storage_ownership is StorageOwnership.MANAGED
    assert record.camera_count == 2
    scene = SceneWorkspaceRepository(record.workspace_root).load()
    assert [camera.display_name for camera in scene.cameras] == ["Entrance", "Workshop"]
    assert all(camera.endpoint_environment_key for camera in scene.cameras)
    runtime = json.loads((record.workspace_root.parent / "scene-runtime.json").read_text())
    assert runtime["scene_uuid"] == record.scene_uuid
    assert "rtsp://" not in json.dumps(runtime).lower()
    assert store.list_scenes() == (record,)


def test_register_existing_preserves_identity_and_protects_storage(tmp_path: Path) -> None:
    store = registry(tmp_path)
    root = existing_workspace(tmp_path)
    original = (root / "scene.json").read_bytes()

    record = store.register_existing(root)
    repeated = store.register_existing(root)

    assert repeated.scene_uuid == record.scene_uuid
    assert record.readiness is SceneReadiness.READY
    assert record.storage_ownership is StorageOwnership.REFERENCED
    impact = store.delete_impact(record.scene_uuid)
    assert impact["can_delete_files"] is False
    assert impact["can_remove_from_list"] is True
    assert (root / "scene.json").read_bytes() == original


def test_archive_rename_and_optimistic_revision(tmp_path: Path) -> None:
    store = registry(tmp_path)
    record = store.create_scene("Office A", ("Camera 1",))

    renamed = store.rename(record.scene_uuid, "Office Alpha", record.revision)
    with pytest.raises(P08WorkflowError, match="scene changed"):
        store.set_archived(record.scene_uuid, True, record.revision)
    archived = store.set_archived(record.scene_uuid, True, renamed.revision)

    assert archived.lifecycle is SceneLifecycle.ARCHIVED
    assert store.list_scenes() == ()
    assert store.list_scenes(include_archived=True) == (archived,)
    restored = store.set_archived(record.scene_uuid, False, archived.revision)
    assert restored.lifecycle is SceneLifecycle.ACTIVE


def test_delete_managed_scene_requires_fresh_impact_and_retains_tombstone(
    tmp_path: Path,
) -> None:
    store = registry(tmp_path)
    record = store.create_scene("Temporary", ("Camera",))
    root = record.workspace_root.parent
    (root / "artifacts" / "sample.txt").write_text("evidence", encoding="utf-8")
    impact = store.delete_impact(record.scene_uuid)

    with pytest.raises(P08WorkflowError, match="review deletion impact"):
        store.delete(
            record.scene_uuid,
            deletion_token="stale",
            delete_files=True,
            expected_revision=record.revision,
        )
    result = store.delete(
        record.scene_uuid,
        deletion_token=impact["deletion_token"],
        delete_files=True,
        expected_revision=record.revision,
    )

    assert result == {
        "scene_uuid": record.scene_uuid,
        "status": "deleted",
        "files_deleted": True,
        "tombstone_retained": True,
    }
    assert not root.exists()
    with pytest.raises(P08WorkflowError, match="deleted"):
        store.require(record.scene_uuid, include_archived=True)


def test_existing_scene_can_be_removed_without_deleting_its_files(tmp_path: Path) -> None:
    store = registry(tmp_path)
    root = existing_workspace(tmp_path)
    record = store.register_existing(root)
    impact = store.delete_impact(record.scene_uuid)

    with pytest.raises(P08WorkflowError, match="files are protected"):
        store.delete(
            record.scene_uuid,
            deletion_token=impact["deletion_token"],
            delete_files=True,
            expected_revision=record.revision,
        )
    result = store.delete(
        record.scene_uuid,
        deletion_token=impact["deletion_token"],
        delete_files=False,
        expected_revision=record.revision,
    )

    assert result["files_deleted"] is False
    assert (root / "scene.json").is_file()


def test_managed_scene_removal_failure_is_truthful_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = registry(tmp_path)
    record = store.create_scene("Retry removal", ("Camera",))
    root = record.workspace_root.parent
    impact = store.delete_impact(record.scene_uuid)
    real_rmtree = shutil.rmtree

    def fail_removal(_root: Path) -> None:
        raise PermissionError("forced Windows sharing violation")

    monkeypatch.setattr(shutil, "rmtree", fail_removal)
    with pytest.raises(P08WorkflowError, match="review and retry"):
        store.delete(
            record.scene_uuid,
            deletion_token=impact["deletion_token"],
            delete_files=True,
            expected_revision=record.revision,
        )

    failed = store.require(record.scene_uuid, include_archived=True)
    assert failed.lifecycle is SceneLifecycle.DELETION_FAILED
    assert root.is_dir()
    assert store.list_scenes() == (failed,)
    with sqlite3.connect(store.database_path) as connection:
        row = connection.execute(
            "SELECT files_deleted, details_json FROM scene_tombstones WHERE scene_uuid = ?",
            (record.scene_uuid,),
        ).fetchone()
    assert row is not None
    assert row[0] == 0
    assert json.loads(row[1])["deletion_state"] == "failed"

    retry_impact = store.delete_impact(record.scene_uuid)
    assert retry_impact["deletion_token"] != impact["deletion_token"]

    def leave_root_in_place(_root: Path) -> None:
        return None

    monkeypatch.setattr(shutil, "rmtree", leave_root_in_place)
    with pytest.raises(P08WorkflowError, match="review and retry"):
        store.delete(
            record.scene_uuid,
            deletion_token=retry_impact["deletion_token"],
            delete_files=True,
            expected_revision=failed.revision,
        )
    still_failed = store.require(record.scene_uuid, include_archived=True)
    assert still_failed.lifecycle is SceneLifecycle.DELETION_FAILED
    assert root.is_dir()

    final_impact = store.delete_impact(record.scene_uuid)
    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    result = store.delete(
        record.scene_uuid,
        deletion_token=final_impact["deletion_token"],
        delete_files=True,
        expected_revision=still_failed.revision,
    )

    assert result["files_deleted"] is True
    assert not root.exists()


def test_heavy_operations_queue_across_scenes_in_fifo_order(tmp_path: Path) -> None:
    store = registry(tmp_path)
    first = store.create_scene("First", ("Camera",))
    second = store.create_scene("Second", ("Camera",))
    release_first = threading.Event()
    entered_first = threading.Event()
    completed: list[str] = []

    def run_first() -> None:
        store.run_with_resource(
            first.scene_uuid,
            "first-job",
            "reconstruction",
            threading.Event(),
            lambda: _held_operation(entered_first, release_first, completed, "first"),
        )

    def run_second() -> None:
        store.run_with_resource(
            second.scene_uuid,
            "second-job",
            "floor",
            threading.Event(),
            lambda: _completed_operation(completed, "second"),
        )

    first_thread = threading.Thread(target=run_first)
    second_thread = threading.Thread(target=run_second)
    first_thread.start()
    assert entered_first.wait(timeout=2)
    second_thread.start()
    deadline = time.monotonic() + 2
    while not store.resource_status()["queue"] and time.monotonic() < deadline:
        time.sleep(0.01)

    status = store.resource_status()
    assert status["active"]["scene_uuid"] == first.scene_uuid
    assert status["queue"][0]["scene_uuid"] == second.scene_uuid
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert completed == ["first", "second"]
    assert store.resource_status() == {"active": None, "queue": []}


def test_live_lease_refuses_a_second_scene_until_released(tmp_path: Path) -> None:
    store = registry(tmp_path)
    first = store.create_scene("First", ("Camera",))
    second = store.create_scene("Second", ("Camera",))

    lease = store.acquire_resource_now(first.scene_uuid, "live")
    with pytest.raises(P08WorkflowError, match="busy"):
        store.acquire_resource_now(second.scene_uuid, "recording")
    store.release_resource(lease)
    second_lease = store.acquire_resource_now(second.scene_uuid, "recording")
    store.release_resource(second_lease)


def _held_operation(
    entered: threading.Event,
    release: threading.Event,
    completed: list[str],
    value: str,
) -> dict[str, bool]:
    entered.set()
    assert release.wait(timeout=2)
    completed.append(value)
    return {"complete": True}


def _completed_operation(completed: list[str], value: str) -> dict[str, bool]:
    completed.append(value)
    return {"complete": True}
