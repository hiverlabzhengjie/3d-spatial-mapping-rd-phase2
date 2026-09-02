"""Scene registry, lifecycle, ownership, and single-accelerator coordination.

The registry is a small control plane.  Detailed workflow history remains in each
scene workspace and is never copied into this database.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    CameraConfig,
    P08WorkflowError,
    PhaseRecord,
    PhaseState,
    SceneWorkspace,
    SceneWorkspaceRepository,
)

SCENE_REGISTRY_SCHEMA = "xr03-scene-registry-v2"
_PREVIOUS_SCENE_REGISTRY_SCHEMA = "xr03-scene-registry-v1"


class SceneLifecycle(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETING = "deleting"
    DELETION_FAILED = "deletion_failed"
    DELETED = "deleted"


class SceneReadiness(StrEnum):
    DRAFT = "draft"
    READY = "ready"


class StorageOwnership(StrEnum):
    MANAGED = "managed"
    REFERENCED = "referenced"


@dataclass(frozen=True)
class SceneRecord:
    scene_uuid: str
    scene_key: str
    display_name: str
    workspace_root: Path
    readiness: SceneReadiness
    lifecycle: SceneLifecycle
    storage_ownership: StorageOwnership
    camera_count: int
    created_at_utc: str
    updated_at_utc: str
    last_opened_at_utc: str | None
    revision: int

    @property
    def can_delete_files(self) -> bool:
        return (
            self.storage_ownership is StorageOwnership.MANAGED
            and self.lifecycle is not SceneLifecycle.DELETED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_uuid": self.scene_uuid,
            "scene_key": self.scene_key,
            "display_name": self.display_name,
            "workspace_root": str(self.workspace_root),
            "readiness": self.readiness.value,
            "lifecycle": self.lifecycle.value,
            "storage_ownership": self.storage_ownership.value,
            "camera_count": self.camera_count,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "last_opened_at_utc": self.last_opened_at_utc,
            "revision": self.revision,
            "can_delete_files": self.can_delete_files,
        }


class SceneRegistry:
    """Versioned local registry of scene workspaces and exact storage ownership."""

    def __init__(self, database_path: Path, managed_storage_root: Path) -> None:
        self.database_path = database_path.resolve()
        self.managed_storage_root = managed_storage_root.resolve()
        self._lock = threading.RLock()
        self._resource_condition = threading.Condition(self._lock)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.managed_storage_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registry_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_uuid TEXT PRIMARY KEY,
                    scene_key TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    workspace_root TEXT NOT NULL UNIQUE,
                    readiness TEXT NOT NULL,
                    lifecycle TEXT NOT NULL,
                    storage_ownership TEXT NOT NULL,
                    camera_count INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    last_opened_at_utc TEXT,
                    revision INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scene_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_uuid TEXT NOT NULL,
                    action TEXT NOT NULL,
                    occurred_at_utc TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scene_tombstones (
                    scene_uuid TEXT PRIMARY KEY,
                    scene_key TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    deleted_at_utc TEXT NOT NULL,
                    files_deleted INTEGER NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resource_lease (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    scene_uuid TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    acquired_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resource_queue (
                    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    scene_uuid TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    queued_at_utc TEXT NOT NULL
                );
                """
            )
            existing = connection.execute(
                "SELECT value FROM registry_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO registry_metadata(key, value) VALUES ('schema_version', ?)",
                    (SCENE_REGISTRY_SCHEMA,),
                )
            elif existing["value"] == _PREVIOUS_SCENE_REGISTRY_SCHEMA:
                connection.execute(
                    "UPDATE registry_metadata SET value = ? WHERE key = 'schema_version'",
                    (SCENE_REGISTRY_SCHEMA,),
                )
            elif existing["value"] != SCENE_REGISTRY_SCHEMA:
                raise P08WorkflowError("unsupported scene registry schema")
            self._reconcile_interrupted_deletions(connection)

    def _reconcile_interrupted_deletions(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT scenes.*, scene_tombstones.details_json AS tombstone_details "
            "FROM scenes LEFT JOIN scene_tombstones USING(scene_uuid) WHERE lifecycle = ?",
            (SceneLifecycle.DELETING.value,),
        ).fetchall()
        for row in rows:
            record = _scene_record(row)
            root = record.workspace_root.parent.resolve()
            exact_root = _is_exact_managed_root(record, self.managed_storage_root)
            files_deleted = bool(exact_root and not root.exists())
            lifecycle = SceneLifecycle.DELETED if files_deleted else SceneLifecycle.DELETION_FAILED
            now = _now()
            prior_details = (
                json.loads(str(row["tombstone_details"]))
                if row["tombstone_details"] is not None
                else {}
            )
            details = {
                **prior_details,
                "deletion_state": ("complete" if files_deleted else "failed"),
                "failure": (
                    None
                    if files_deleted
                    else {
                        "type": "InterruptedDeletion",
                        "message": (
                            "console restarted before managed storage removal completed"
                            if exact_root
                            else "managed storage boundary could not be revalidated"
                        ),
                    }
                ),
                "reconciled_at_utc": now,
            }
            connection.execute(
                "UPDATE scenes SET lifecycle = ?, updated_at_utc = ?, revision = revision + 1 "
                "WHERE scene_uuid = ?",
                (lifecycle.value, now, record.scene_uuid),
            )
            connection.execute(
                "UPDATE scene_tombstones SET files_deleted = ?, details_json = ? "
                "WHERE scene_uuid = ?",
                (int(files_deleted), json.dumps(details, sort_keys=True), record.scene_uuid),
            )
            self._event(
                connection,
                record.scene_uuid,
                "deletion-reconciled",
                {"files_deleted": files_deleted, "lifecycle": lifecycle.value},
            )

    def list_scenes(self, *, include_archived: bool = False) -> tuple[SceneRecord, ...]:
        lifecycles = (
            (SceneLifecycle.ACTIVE.value, SceneLifecycle.ARCHIVED.value)
            if include_archived
            else (SceneLifecycle.ACTIVE.value,)
        )
        values = (*lifecycles, SceneLifecycle.DELETION_FAILED.value)
        placeholders = ",".join("?" for _ in values)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM scenes WHERE lifecycle IN ({placeholders}) "  # noqa: S608
                "ORDER BY COALESCE(last_opened_at_utc, created_at_utc) DESC, display_name",
                values,
            ).fetchall()
        return tuple(_scene_record(row) for row in rows)

    def require(self, scene_uuid: str, *, include_archived: bool = False) -> SceneRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scenes WHERE scene_uuid = ?", (scene_uuid,)
            ).fetchone()
        if row is None:
            raise P08WorkflowError("unknown scene")
        record = _scene_record(row)
        if record.lifecycle is SceneLifecycle.DELETED:
            raise P08WorkflowError("scene has been deleted")
        if record.lifecycle is SceneLifecycle.DELETING:
            raise P08WorkflowError("scene deletion is still being reconciled")
        if record.lifecycle is SceneLifecycle.DELETION_FAILED and not include_archived:
            raise P08WorkflowError("scene deletion failed; open Manage to retry")
        if record.lifecycle is SceneLifecycle.ARCHIVED and not include_archived:
            raise P08WorkflowError("scene is archived")
        return record

    def register_existing(
        self,
        workspace_root: Path,
        *,
        readiness: SceneReadiness = SceneReadiness.READY,
        preferred_uuid: str | None = None,
    ) -> SceneRecord:
        root = workspace_root.resolve()
        repository = SceneWorkspaceRepository(root)
        scene = repository.load()
        scene_uuid = preferred_uuid or str(uuid.uuid5(uuid.NAMESPACE_URL, str(root).lower()))
        now = _now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scenes WHERE workspace_root = ? OR scene_uuid = ?",
                (str(root), scene_uuid),
            ).fetchone()
            if row is not None:
                if row["lifecycle"] == SceneLifecycle.DELETED.value:
                    if row["storage_ownership"] != StorageOwnership.REFERENCED.value:
                        raise P08WorkflowError(
                            "deleted managed scene storage cannot be re-registered"
                        )
                    connection.execute(
                        "UPDATE scenes SET lifecycle = ?, readiness = ?, updated_at_utc = ?, "
                        "revision = revision + 1 WHERE scene_uuid = ?",
                        (
                            SceneLifecycle.ACTIVE.value,
                            readiness.value,
                            now,
                            row["scene_uuid"],
                        ),
                    )
                    self._event(connection, str(row["scene_uuid"]), "re-registered", {})
                    refreshed = connection.execute(
                        "SELECT * FROM scenes WHERE scene_uuid = ?", (row["scene_uuid"],)
                    ).fetchone()
                    assert refreshed is not None
                    return _scene_record(refreshed)
                if (
                    readiness is SceneReadiness.READY
                    and row["readiness"] != SceneReadiness.READY.value
                ):
                    connection.execute(
                        "UPDATE scenes SET readiness = ?, updated_at_utc = ?, "
                        "revision = revision + 1 WHERE scene_uuid = ?",
                        (SceneReadiness.READY.value, now, row["scene_uuid"]),
                    )
                    refreshed = connection.execute(
                        "SELECT * FROM scenes WHERE scene_uuid = ?", (row["scene_uuid"],)
                    ).fetchone()
                    assert refreshed is not None
                    return _scene_record(refreshed)
                return _scene_record(row)
            scene_key = scene.scene_id
            conflict = connection.execute(
                "SELECT scene_uuid FROM scenes WHERE scene_key = ?", (scene_key,)
            ).fetchone()
            if conflict is not None:
                scene_key = f"{scene.scene_id}-{scene_uuid.split('-')[0]}"
            connection.execute(
                """
                INSERT INTO scenes(
                    scene_uuid, scene_key, display_name, workspace_root, readiness, lifecycle,
                    storage_ownership, camera_count, created_at_utc, updated_at_utc,
                    last_opened_at_utc, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    scene_uuid,
                    scene_key,
                    scene.display_name,
                    str(root),
                    readiness.value,
                    SceneLifecycle.ACTIVE.value,
                    StorageOwnership.REFERENCED.value,
                    len(scene.cameras),
                    now,
                    now,
                    now,
                ),
            )
            self._event(
                connection,
                scene_uuid,
                "registered-existing",
                {"workspace_root": str(root)},
            )
        return self.require(scene_uuid)

    def create_scene(self, display_name: str, camera_names: Sequence[str]) -> SceneRecord:
        name = _display_name(display_name)
        cleaned_camera_names = tuple(_display_name(value) for value in camera_names)
        if not cleaned_camera_names:
            raise P08WorkflowError("add at least one camera")
        if len(cleaned_camera_names) > 64:
            raise P08WorkflowError("a scene may contain at most 64 cameras")
        scene_uuid = str(uuid.uuid4())
        short_id = scene_uuid.split("-")[0]
        scene_key = f"scene-{short_id}"
        root = (self.managed_storage_root / scene_uuid).resolve()
        if not root.is_relative_to(self.managed_storage_root):
            raise P08WorkflowError("managed scene path escaped its storage root")
        workspace_root = root / "workspace"
        artifact_root = root / "artifacts"
        cameras = tuple(
            CameraConfig(
                camera_id=f"{scene_key}-cam-{index:02d}",
                display_name=camera_name,
                endpoint_environment_key=f"XR03_{short_id.upper()}_CAMERA_{index:02d}_RTSP",
            )
            for index, camera_name in enumerate(cleaned_camera_names, start=1)
        )
        phases = tuple(
            PhaseRecord(
                phase_id=phase_id,
                state=PhaseState.READY if phase_id == "P02" else PhaseState.UNAVAILABLE,
                message=(
                    "Ready to configure the facility and cameras"
                    if phase_id == "P02"
                    else "Complete the previous workflow step first"
                ),
                prerequisites=() if index == 0 else (PHASE_ORDER[index - 1],),
            )
            for index, phase_id in enumerate(PHASE_ORDER)
        )
        scene = SceneWorkspace(
            project_id="spatial-mapping",
            scene_id=scene_key,
            display_name=name,
            artifact_root=artifact_root,
            cameras=cameras,
            phases=phases,
        )
        now = _now()
        try:
            root.mkdir(parents=True, exist_ok=False)
            artifact_root.mkdir()
            for directory_name in (
                "capture",
                "calibration",
                "reconstruction",
                "operations",
            ):
                (root / directory_name).mkdir()
            SceneWorkspaceRepository(workspace_root).create(scene)
            _write_runtime_manifest(root, scene_uuid, scene)
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO scenes(
                        scene_uuid, scene_key, display_name, workspace_root, readiness,
                        lifecycle, storage_ownership, camera_count, created_at_utc,
                        updated_at_utc, last_opened_at_utc, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        scene_uuid,
                        scene_key,
                        name,
                        str(workspace_root),
                        SceneReadiness.DRAFT.value,
                        SceneLifecycle.ACTIVE.value,
                        StorageOwnership.MANAGED.value,
                        len(cameras),
                        now,
                        now,
                        now,
                    ),
                )
                self._event(
                    connection,
                    scene_uuid,
                    "created",
                    {"camera_count": len(cameras), "managed_root": str(root)},
                )
        except Exception:
            if root.exists():
                shutil.rmtree(root)
            raise
        return self.require(scene_uuid)

    def mark_opened(self, scene_uuid: str) -> SceneRecord:
        record = self.require(scene_uuid)
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE scenes SET last_opened_at_utc = ?, updated_at_utc = ? "
                "WHERE scene_uuid = ?",
                (now, now, scene_uuid),
            )
            self._event(connection, scene_uuid, "opened", {})
        return self.require(record.scene_uuid)

    def rename(self, scene_uuid: str, display_name: str, expected_revision: int) -> SceneRecord:
        record = self.require(scene_uuid, include_archived=True)
        _expect_revision(record, expected_revision)
        name = _display_name(display_name)
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE scenes SET display_name = ?, updated_at_utc = ?, revision = revision + 1 "
                "WHERE scene_uuid = ?",
                (name, now, scene_uuid),
            )
            self._event(connection, scene_uuid, "renamed", {"display_name": name})
        return self.require(scene_uuid, include_archived=True)

    def set_archived(self, scene_uuid: str, archived: bool, expected_revision: int) -> SceneRecord:
        record = self.require(scene_uuid, include_archived=True)
        _expect_revision(record, expected_revision)
        if record.lifecycle is SceneLifecycle.DELETION_FAILED:
            raise P08WorkflowError("review and retry the failed deletion before archiving")
        if archived:
            resources = self.resource_status()
            active = resources["active"]
            queued = resources["queue"]
            if (active and active["scene_uuid"] == scene_uuid) or any(
                item["scene_uuid"] == scene_uuid for item in queued
            ):
                raise P08WorkflowError("wait for this scene's processing before archiving it")
        lifecycle = SceneLifecycle.ARCHIVED if archived else SceneLifecycle.ACTIVE
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE scenes SET lifecycle = ?, updated_at_utc = ?, revision = revision + 1 "
                "WHERE scene_uuid = ?",
                (lifecycle.value, now, scene_uuid),
            )
            self._event(connection, scene_uuid, lifecycle.value, {})
        return self.require(scene_uuid, include_archived=True)

    def delete_impact(self, scene_uuid: str) -> dict[str, Any]:
        record = self.require(scene_uuid, include_archived=True)
        managed_root = record.workspace_root.parent.resolve()
        owned = record.storage_ownership is StorageOwnership.MANAGED
        exact_root = _is_exact_managed_root(record, self.managed_storage_root)
        active_lease = self.resource_status()["active"]
        busy = bool(active_lease and active_lease["scene_uuid"] == scene_uuid)
        reasons: list[str] = []
        if not owned:
            reasons.append("This scene uses pre-existing storage, so its files are protected.")
        if owned and not exact_root:
            reasons.append("The managed storage boundary could not be verified.")
        if busy:
            reasons.append("Stop the scene's active processing before deleting it.")
        files = _storage_summary(managed_root) if owned and managed_root.exists() else None
        token_payload = f"{scene_uuid}:{record.revision}:{managed_root}"
        return {
            "scene": record.to_dict(),
            "can_delete_files": owned and exact_root and not busy,
            "can_remove_from_list": not busy,
            "protected_reasons": reasons,
            "exact_managed_root": str(managed_root) if owned else None,
            "storage": files,
            "deletion_token": hashlib.sha256(token_payload.encode("utf-8")).hexdigest(),
        }

    def delete(
        self,
        scene_uuid: str,
        *,
        deletion_token: str,
        delete_files: bool,
        expected_revision: int,
    ) -> dict[str, Any]:
        record = self.require(scene_uuid, include_archived=True)
        _expect_revision(record, expected_revision)
        impact = self.delete_impact(scene_uuid)
        if deletion_token != impact["deletion_token"]:
            raise P08WorkflowError("scene changed; review deletion impact again")
        if not impact["can_remove_from_list"]:
            raise P08WorkflowError("stop active scene processing before deletion")
        if delete_files and not impact["can_delete_files"]:
            raise P08WorkflowError("this scene's files are protected")
        root = record.workspace_root.parent.resolve()
        if not delete_files:
            now = _now()
            details = {
                "deletion_state": "complete",
                "delete_files": False,
                "storage": impact["storage"],
            }
            with self._connect() as connection:
                connection.execute(
                    "UPDATE scenes SET lifecycle = ?, updated_at_utc = ?, "
                    "revision = revision + 1 WHERE scene_uuid = ?",
                    (SceneLifecycle.DELETED.value, now, scene_uuid),
                )
                self._upsert_tombstone(
                    connection,
                    record,
                    deleted_at_utc=now,
                    files_deleted=False,
                    details=details,
                )
                self._event(connection, scene_uuid, "deleted", {"files_deleted": False})
            return {
                "scene_uuid": scene_uuid,
                "status": "deleted",
                "files_deleted": False,
                "tombstone_retained": True,
            }

        started_at = _now()
        removing_details = {
            "deletion_state": "removing",
            "delete_files": True,
            "storage": impact["storage"],
        }
        with self._connect() as connection:
            connection.execute(
                "UPDATE scenes SET lifecycle = ?, updated_at_utc = ?, revision = revision + 1 "
                "WHERE scene_uuid = ?",
                (SceneLifecycle.DELETING.value, started_at, scene_uuid),
            )
            self._upsert_tombstone(
                connection,
                record,
                deleted_at_utc=started_at,
                files_deleted=False,
                details=removing_details,
            )
            self._event(connection, scene_uuid, "deletion-started", {"files_deleted": False})

        try:
            # The resolved exact target was checked above and is a UUID child of the managed root.
            shutil.rmtree(root)
            if root.exists():
                raise OSError("managed scene root still exists after filesystem removal")
        except OSError as error:
            if root.exists():
                failed_at = _now()
                failure = {
                    "type": type(error).__name__,
                    "message": str(error)[:500],
                }
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE scenes SET lifecycle = ?, updated_at_utc = ?, "
                        "revision = revision + 1 WHERE scene_uuid = ?",
                        (SceneLifecycle.DELETION_FAILED.value, failed_at, scene_uuid),
                    )
                    self._upsert_tombstone(
                        connection,
                        record,
                        deleted_at_utc=started_at,
                        files_deleted=False,
                        details={
                            **removing_details,
                            "deletion_state": "failed",
                            "failed_at_utc": failed_at,
                            "failure": failure,
                        },
                    )
                    self._event(connection, scene_uuid, "deletion-failed", failure)
                raise P08WorkflowError(
                    "managed scene files could not be removed; review and retry deletion"
                ) from error

        completed_at = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE scenes SET lifecycle = ?, updated_at_utc = ?, revision = revision + 1 "
                "WHERE scene_uuid = ?",
                (SceneLifecycle.DELETED.value, completed_at, scene_uuid),
            )
            self._upsert_tombstone(
                connection,
                record,
                deleted_at_utc=completed_at,
                files_deleted=True,
                details={
                    **removing_details,
                    "deletion_state": "complete",
                    "completed_at_utc": completed_at,
                },
            )
            self._event(connection, scene_uuid, "deleted", {"files_deleted": True})
        return {
            "scene_uuid": scene_uuid,
            "status": "deleted",
            "files_deleted": True,
            "tombstone_retained": True,
        }

    @staticmethod
    def _upsert_tombstone(
        connection: sqlite3.Connection,
        record: SceneRecord,
        *,
        deleted_at_utc: str,
        files_deleted: bool,
        details: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO scene_tombstones(
                scene_uuid, scene_key, display_name, workspace_root, deleted_at_utc,
                files_deleted, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scene_uuid) DO UPDATE SET
                deleted_at_utc = excluded.deleted_at_utc,
                files_deleted = excluded.files_deleted,
                details_json = excluded.details_json
            """,
            (
                record.scene_uuid,
                record.scene_key,
                record.display_name,
                str(record.workspace_root),
                deleted_at_utc,
                int(files_deleted),
                json.dumps(dict(details), sort_keys=True),
            ),
        )

    def resource_status(self) -> dict[str, Any]:
        with self._connect() as connection:
            active = connection.execute(
                "SELECT * FROM resource_lease WHERE singleton = 1"
            ).fetchone()
            queued = connection.execute(
                "SELECT request_id, scene_uuid, operation, queued_at_utc "
                "FROM resource_queue ORDER BY position_id"
            ).fetchall()
        return {
            "active": dict(active) if active is not None else None,
            "queue": [dict(row) for row in queued],
        }

    def clear_stale_resources(self) -> None:
        """Reconcile process-local leases after a full console restart."""

        with self._connect() as connection:
            connection.execute("DELETE FROM resource_lease")
            connection.execute("DELETE FROM resource_queue")
        with self._resource_condition:
            self._resource_condition.notify_all()

    def run_with_resource(
        self,
        scene_uuid: str,
        request_id: str,
        operation: str,
        cancel_event: threading.Event,
        callback: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Queue a heavy batch operation and run it under the single local lease."""

        self.require(scene_uuid)
        lease_id = str(uuid.uuid4())
        with self._resource_condition:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO resource_queue(request_id, scene_uuid, operation, queued_at_utc) "
                    "VALUES (?, ?, ?, ?)",
                    (request_id, scene_uuid, operation, _now()),
                )
            while True:
                if cancel_event.is_set():
                    with self._connect() as connection:
                        connection.execute(
                            "DELETE FROM resource_queue WHERE request_id = ?", (request_id,)
                        )
                    self._resource_condition.notify_all()
                    raise P08WorkflowError("processing cancelled while waiting for this computer")
                with self._connect() as connection:
                    active = connection.execute(
                        "SELECT scene_uuid FROM resource_lease WHERE singleton = 1"
                    ).fetchone()
                    first = connection.execute(
                        "SELECT request_id FROM resource_queue ORDER BY position_id LIMIT 1"
                    ).fetchone()
                    if active is None and first is not None and first["request_id"] == request_id:
                        connection.execute(
                            "DELETE FROM resource_queue WHERE request_id = ?", (request_id,)
                        )
                        connection.execute(
                            "INSERT INTO resource_lease(singleton, scene_uuid, operation, "
                            "lease_id, acquired_at_utc) VALUES (1, ?, ?, ?, ?)",
                            (scene_uuid, operation, lease_id, _now()),
                        )
                        break
                self._resource_condition.wait(timeout=0.2)
        try:
            return callback()
        finally:
            self.release_resource(lease_id)

    def acquire_resource_now(self, scene_uuid: str, operation: str) -> str:
        """Acquire the local heavy-operation lease for an interactive Live session."""

        self.require(scene_uuid)
        lease_id = str(uuid.uuid4())
        with self._resource_condition, self._connect() as connection:
            active = connection.execute(
                "SELECT scene_uuid, operation FROM resource_lease WHERE singleton = 1"
            ).fetchone()
            queued = connection.execute("SELECT COUNT(*) AS count FROM resource_queue").fetchone()
            if active is not None or (queued is not None and int(queued["count"]) > 0):
                owner = "another scene" if active is None else str(active["scene_uuid"])
                raise P08WorkflowError(
                    f"This computer is busy with {owner}; wait for that operation to finish"
                )
            connection.execute(
                "INSERT INTO resource_lease(singleton, scene_uuid, operation, lease_id, "
                "acquired_at_utc) VALUES (1, ?, ?, ?, ?)",
                (scene_uuid, operation, lease_id, _now()),
            )
        return lease_id

    def release_resource(self, lease_id: str) -> None:
        with self._resource_condition:
            with self._connect() as connection:
                connection.execute("DELETE FROM resource_lease WHERE lease_id = ?", (lease_id,))
            self._resource_condition.notify_all()

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        scene_uuid: str,
        action: str,
        details: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO scene_events(scene_uuid, action, occurred_at_utc, details_json) "
            "VALUES (?, ?, ?, ?)",
            (scene_uuid, action, _now(), json.dumps(dict(details), sort_keys=True)),
        )


def _scene_record(row: sqlite3.Row) -> SceneRecord:
    return SceneRecord(
        scene_uuid=str(row["scene_uuid"]),
        scene_key=str(row["scene_key"]),
        display_name=str(row["display_name"]),
        workspace_root=Path(str(row["workspace_root"])),
        readiness=SceneReadiness(str(row["readiness"])),
        lifecycle=SceneLifecycle(str(row["lifecycle"])),
        storage_ownership=StorageOwnership(str(row["storage_ownership"])),
        camera_count=int(row["camera_count"]),
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
        last_opened_at_utc=(
            str(row["last_opened_at_utc"]) if row["last_opened_at_utc"] is not None else None
        ),
        revision=int(row["revision"]),
    )


def _is_exact_managed_root(record: SceneRecord, managed_storage_root: Path) -> bool:
    managed_root = record.workspace_root.parent.resolve()
    return (
        record.storage_ownership is StorageOwnership.MANAGED
        and managed_root.is_relative_to(managed_storage_root)
        and managed_root.parent == managed_storage_root
        and managed_root.name == record.scene_uuid
        and record.workspace_root.name == "workspace"
    )


def _write_runtime_manifest(root: Path, scene_uuid: str, scene: SceneWorkspace) -> None:
    payload = {
        "schema_version": "xr03-scene-runtime-v1",
        "scene_uuid": scene_uuid,
        "scene_key": scene.scene_id,
        "workspace_root": str((root / "workspace").resolve()),
        "artifact_root": str(scene.artifact_root.resolve()),
        "secret_file": str((root / "secrets.env").resolve()),
        "capture_root": str((root / "capture").resolve()),
        "calibration_root": str((root / "calibration").resolve()),
        "reconstruction_root": str((root / "reconstruction").resolve()),
        "operations_root": str((root / "operations").resolve()),
        "recording_catalog": str((root / "operations" / "xr02-recordings.sqlite3").resolve()),
        "camera_ids": [camera.camera_id for camera in scene.cameras],
        "camera_endpoint_environment_keys": {
            camera.camera_id: camera.endpoint_environment_key for camera in scene.cameras
        },
    }
    path = root / "scene-runtime.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _storage_summary(root: Path) -> dict[str, int]:
    file_count = 0
    byte_count = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            byte_count += path.stat().st_size
    return {"file_count": file_count, "byte_count": byte_count}


def _display_name(value: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise P08WorkflowError("scene and camera names must not be blank")
    if len(cleaned) > 80:
        raise P08WorkflowError("scene and camera names may contain at most 80 characters")
    return cleaned


def _expect_revision(record: SceneRecord, expected_revision: int) -> None:
    if expected_revision != record.revision:
        raise P08WorkflowError("scene changed; refresh and try again")


def _now() -> str:
    return datetime.now(UTC).isoformat()
