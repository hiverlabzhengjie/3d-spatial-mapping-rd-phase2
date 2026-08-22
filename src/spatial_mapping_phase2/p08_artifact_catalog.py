"""Scene-local catalog for immutable workflow artifact versions and selections."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


class ArtifactCatalogError(ValueError):
    """Raised when a catalog action would violate version-control rules."""


@dataclass(frozen=True)
class MilestoneDefinition:
    key: str
    title: str
    significance: str


@dataclass(frozen=True)
class WorkflowSectionDefinition:
    key: str
    title: str
    description: str
    milestone_keys: tuple[str, ...]


MILESTONES: tuple[MilestoneDefinition, ...] = (
    MilestoneDefinition(
        "floor-plan-source",
        "Floor plan source",
        "Defines the physical environment represented by this scene.",
    ),
    MilestoneDefinition(
        "facility-registration",
        "Facility registration",
        "Defines world origin, scale, floor level and camera mounting references.",
    ),
    MilestoneDefinition(
        "capture-bundle",
        "Camera capture bundle",
        "Provides the exact synchronized camera frames used by later processing.",
    ),
    MilestoneDefinition(
        "calibration-correspondence",
        "Calibration correspondence",
        "Records the reviewed image-to-floor-plan correspondence work.",
    ),
    MilestoneDefinition(
        "calibration-pose-registry",
        "Camera calibration and poses",
        "Provides the selected intrinsics and world-frame pose for every camera.",
    ),
    MilestoneDefinition(
        "reconstruction-input",
        "Reconstruction input set",
        "Freezes the exact camera frames, calibration and poses supplied to reconstruction.",
    ),
    MilestoneDefinition(
        "reconstructed-geometry",
        "Combined point cloud",
        "Provides the combined geometric source used for visual review and floor refinement.",
    ),
    MilestoneDefinition(
        "geometry-review",
        "Geometry review recording",
        "Shows the selected combined point cloud and camera context in Rerun.",
    ),
    MilestoneDefinition(
        "floor-refined-geometry",
        "Floor-refined result",
        "Adds the authoritative floor representation without changing source colour samples.",
    ),
    MilestoneDefinition(
        "floor-verification",
        "Floor verification",
        "Checks the floor derivative and its unchanged source identity.",
    ),
    MilestoneDefinition(
        "final-review",
        "Final Rerun review",
        "Shows the selected floor-refined result for final human review.",
    ),
)
MILESTONE_INDEX = {item.key: index for index, item in enumerate(MILESTONES)}

WORKFLOW_SECTIONS: tuple[WorkflowSectionDefinition, ...] = (
    WorkflowSectionDefinition(
        "setup",
        "Floor plan & setup",
        "The plan, scale, world origin and fixed camera locations for this scene.",
        ("floor-plan-source", "facility-registration"),
    ),
    WorkflowSectionDefinition(
        "capture",
        "Camera capture",
        "The camera frames selected for processing.",
        ("capture-bundle",),
    ),
    WorkflowSectionDefinition(
        "calibration",
        "Calibration & camera poses",
        "The reviewed correspondences, intrinsics and world-frame camera poses.",
        ("calibration-correspondence", "calibration-pose-registry"),
    ),
    WorkflowSectionDefinition(
        "reconstruction",
        "Static reconstruction",
        "The reconstruction inputs, combined point cloud and geometry preview.",
        ("reconstruction-input", "reconstructed-geometry", "geometry-review"),
    ),
    WorkflowSectionDefinition(
        "floor",
        "Floor refinement",
        "The floor-completed result and its verification record.",
        ("floor-refined-geometry", "floor-verification"),
    ),
    WorkflowSectionDefinition(
        "final",
        "Final review",
        "The Rerun recording used for the final visual check.",
        ("final-review",),
    ),
)

RETENTION_CLASSES = frozenset({"accepted-predecessor", "selected-authority", "required-rollback"})


class SceneArtifactCatalog:
    """SQLite-backed index over immutable artifacts belonging to exactly one scene."""

    def __init__(
        self,
        path: Path,
        project_id: str,
        scene_id: str,
        allowed_artifact_roots: Sequence[Path] = (),
    ) -> None:
        self.path = path.resolve()
        self.project_id = project_id
        self.scene_id = scene_id
        roots = tuple(root.resolve() for root in allowed_artifact_roots)
        self.allowed_artifact_roots = roots or (self.path.parent,)
        self._lock = threading.RLock()
        self._registered: dict[tuple[str, str, str], str] = {}
        self._known_by_path: dict[tuple[str, str], str] = {}
        self._selections: dict[str, str] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._load_cache()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_versions (
                    artifact_id TEXT PRIMARY KEY,
                    milestone_key TEXT NOT NULL,
                    phase_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_count INTEGER,
                    display_name TEXT NOT NULL,
                    significance TEXT NOT NULL,
                    lifecycle TEXT NOT NULL CHECK(
                        lifecycle IN ('available','archived','missing','corrupt')
                    ),
                    created_at TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS artifact_identity
                    ON artifact_versions(milestone_key, path, sha256);
                CREATE TABLE IF NOT EXISTS milestone_selections (
                    milestone_key TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL REFERENCES artifact_versions(artifact_id),
                    selected_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_dependencies (
                    parent_artifact_id TEXT NOT NULL REFERENCES artifact_versions(artifact_id),
                    child_artifact_id TEXT NOT NULL REFERENCES artifact_versions(artifact_id),
                    relation TEXT NOT NULL,
                    PRIMARY KEY(parent_artifact_id, child_artifact_id, relation)
                );
                CREATE TABLE IF NOT EXISTS artifact_events (
                    event_id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    artifact_id TEXT,
                    milestone_key TEXT,
                    detail_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_deletions (
                    artifact_id TEXT PRIMARY KEY REFERENCES artifact_versions(artifact_id),
                    deleted_at TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    original_sha256 TEXT NOT NULL,
                    original_byte_count INTEGER NOT NULL,
                    confirmation_phrase TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_retention (
                    artifact_id TEXT PRIMARY KEY REFERENCES artifact_versions(artifact_id),
                    retention_class TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    protected_at TEXT NOT NULL,
                    source TEXT NOT NULL
                );
                """
            )
            expected = {"project_id": self.project_id, "scene_id": self.scene_id}
            existing = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM catalog_meta")
            }
            schema_version = existing.get("schema_version")
            if schema_version not in {
                None,
                "p08-scene-artifact-catalog-v1",
                "p08-scene-artifact-catalog-v2",
                "p08-scene-artifact-catalog-v3",
            }:
                raise ArtifactCatalogError("artifact catalog schema is not supported")
            if existing and any(existing.get(key) != value for key, value in expected.items()):
                raise ArtifactCatalogError("artifact catalog belongs to a different scene")
            for key, value in expected.items():
                connection.execute(
                    "INSERT OR IGNORE INTO catalog_meta(key, value) VALUES (?, ?)",
                    (key, value),
                )
            connection.execute(
                "INSERT INTO catalog_meta(key, value) "
                "VALUES ('schema_version', 'p08-scene-artifact-catalog-v3') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )

    def _load_cache(self) -> None:
        with self._lock, self._connect() as connection:
            for row in connection.execute(
                "SELECT artifact_id, milestone_key, path, sha256 "
                "FROM artifact_versions ORDER BY discovered_at"
            ):
                milestone_key = str(row["milestone_key"])
                path = str(row["path"])
                sha256 = str(row["sha256"])
                self._registered[(milestone_key, path, sha256)] = str(row["artifact_id"])
                self._known_by_path[(milestone_key, path)] = sha256
            self._selections = {
                str(row["milestone_key"]): str(row["artifact_id"])
                for row in connection.execute(
                    "SELECT milestone_key, artifact_id FROM milestone_selections"
                )
            }

    def register(
        self,
        *,
        artifact_id: str,
        milestone_key: str,
        phase_id: str,
        kind: str,
        path: Path,
        sha256: str,
        display_name: str,
        significance: str,
        selected: bool = False,
        metadata: Mapping[str, Any] | None = None,
        parent_artifact_ids: Iterable[str] = (),
    ) -> str:
        if milestone_key not in MILESTONE_INDEX:
            raise ArtifactCatalogError(f"unknown artifact milestone: {milestone_key}")
        resolved = path.resolve()
        cache_key = (milestone_key, str(resolved), sha256)
        with self._lock:
            cached_id = self._registered.get(cache_key)
            if cached_id is not None and (
                not selected or self._selections.get(milestone_key) == cached_id
            ):
                return cached_id
        byte_count = resolved.stat().st_size if resolved.is_file() else None
        now = _now()
        with self._lock, self._connect() as connection:
            identity = connection.execute(
                "SELECT artifact_id FROM artifact_versions "
                "WHERE milestone_key=? AND path=? AND sha256=?",
                (milestone_key, str(resolved), sha256),
            ).fetchone()
            canonical_id = str(identity["artifact_id"]) if identity else artifact_id
            previous = connection.execute(
                "SELECT lifecycle FROM artifact_versions WHERE artifact_id=?",
                (canonical_id,),
            ).fetchone()
            if not resolved.is_file():
                lifecycle = "missing"
            elif previous is not None:
                lifecycle = str(previous["lifecycle"])
            else:
                lifecycle = "available"
            connection.execute(
                """
                INSERT INTO artifact_versions(
                    artifact_id, milestone_key, phase_id, kind, path, sha256, byte_count,
                    display_name, significance, lifecycle, created_at, discovered_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    milestone_key=excluded.milestone_key, phase_id=excluded.phase_id,
                    kind=excluded.kind, path=excluded.path, sha256=excluded.sha256,
                    byte_count=excluded.byte_count, display_name=excluded.display_name,
                    significance=excluded.significance,
                    lifecycle=CASE WHEN artifact_versions.lifecycle='archived'
                        THEN 'archived' ELSE excluded.lifecycle END,
                    metadata_json=excluded.metadata_json
                """,
                (
                    canonical_id,
                    milestone_key,
                    phase_id,
                    kind,
                    str(resolved),
                    sha256,
                    byte_count,
                    display_name,
                    significance,
                    lifecycle,
                    _file_time(resolved),
                    now,
                    json.dumps(dict(metadata or {}), sort_keys=True),
                ),
            )
            for parent_id in parent_artifact_ids:
                if connection.execute(
                    "SELECT 1 FROM artifact_versions WHERE artifact_id=?", (parent_id,)
                ).fetchone():
                    connection.execute(
                        "INSERT OR IGNORE INTO artifact_dependencies "
                        "VALUES (?, ?, 'derived-from')",
                        (parent_id, canonical_id),
                    )
            if selected:
                self._select_in_connection(connection, milestone_key, canonical_id, now)
            self._registered[cache_key] = canonical_id
            self._known_by_path[(milestone_key, str(resolved))] = sha256
            return canonical_id

    def status(self, *, event_limit: int = 40) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            selections = {
                row["milestone_key"]: row["artifact_id"]
                for row in connection.execute(
                    "SELECT milestone_key, artifact_id FROM milestone_selections"
                )
            }
            rows = list(
                connection.execute(
                    "SELECT a.*, d.deleted_at, r.retention_class, r.reason AS retention_reason, "
                    "r.protected_at, r.source AS retention_source FROM artifact_versions a "
                    "LEFT JOIN artifact_deletions d ON d.artifact_id=a.artifact_id "
                    "LEFT JOIN artifact_retention r ON r.artifact_id=a.artifact_id "
                    "ORDER BY a.created_at DESC, a.discovered_at DESC"
                )
            )
            grouped: dict[str, list[dict[str, Any]]] = {item.key: [] for item in MILESTONES}
            for row in rows:
                value = _row_to_dict(row)
                if row["retention_class"] is not None:
                    value["retention"] = {
                        "class": row["retention_class"],
                        "reason": row["retention_reason"],
                        "protected_at": row["protected_at"],
                        "source": row["retention_source"],
                    }
                if row["deleted_at"] is not None:
                    value["lifecycle"] = "deleted"
                    value["deleted_at"] = row["deleted_at"]
                value["selected"] = selections.get(value["milestone_key"]) == value["artifact_id"]
                grouped[value["milestone_key"]].append(value)
            events = [
                {
                    "event_id": row["event_id"],
                    "occurred_at": row["occurred_at"],
                    "action": row["action"],
                    "artifact_id": row["artifact_id"],
                    "milestone_key": row["milestone_key"],
                    "detail": json.loads(row["detail_json"]),
                }
                for row in connection.execute(
                    "SELECT * FROM artifact_events ORDER BY occurred_at DESC LIMIT ?",
                    (event_limit,),
                )
            ]
        milestones: list[dict[str, Any]] = []
        for definition in MILESTONES:
            versions = grouped[definition.key]
            milestones.append(
                {
                    "milestone_key": definition.key,
                    "title": definition.title,
                    "significance": definition.significance,
                    "selected_artifact_id": selections.get(definition.key),
                    "version_count": len(versions),
                    "versions": versions,
                }
            )
        milestone_by_key = {item["milestone_key"]: item for item in milestones}
        workflow_sections: list[dict[str, Any]] = []
        for section in WORKFLOW_SECTIONS:
            section_milestones = [milestone_by_key[key] for key in section.milestone_keys]
            current_items = [
                version
                for milestone in section_milestones
                for version in milestone["versions"]
                if version["selected"] and version["metadata"].get("selectable") is not False
            ]
            past_items = [
                version
                for milestone in section_milestones
                for version in milestone["versions"]
                if not version["selected"]
                and version["lifecycle"] != "deleted"
                and version["metadata"].get("selectable") is not False
            ]
            workflow_sections.append(
                {
                    "section_key": section.key,
                    "title": section.title,
                    "description": section.description,
                    "milestone_keys": list(section.milestone_keys),
                    "current_items": current_items,
                    "past_items": past_items,
                    "past_count": len(past_items),
                }
            )
        operator_versions: list[dict[str, Any]] = [
            version
            for milestone in milestones
            for version in milestone["versions"]
            if version["metadata"].get("selectable") is not False
        ]
        storage = {
            "current_version_count": sum(bool(item["selected"]) for item in operator_versions),
            "past_version_count": sum(
                not bool(item["selected"]) and item["lifecycle"] != "deleted"
                for item in operator_versions
            ),
            "archived_version_count": sum(
                item["lifecycle"] == "archived" for item in operator_versions
            ),
            "deleted_version_count": sum(
                item["lifecycle"] == "deleted" for item in operator_versions
            ),
            "retained_byte_count": sum(
                int(item["byte_count"] or 0)
                for item in operator_versions
                if item["lifecycle"] != "deleted"
            ),
            "past_retained_byte_count": sum(
                int(item["byte_count"] or 0)
                for item in operator_versions
                if not item["selected"] and item["lifecycle"] != "deleted"
            ),
            "protected_version_count": sum("retention" in item for item in operator_versions),
        }
        return {
            "schema_version": "p08-scene-artifact-catalog-status-v3",
            "project_id": self.project_id,
            "scene_id": self.scene_id,
            "database_path": str(self.path),
            "milestones": milestones,
            "workflow_sections": workflow_sections,
            "storage": storage,
            "events": events,
        }

    def has_versions(self) -> bool:
        with self._lock, self._connect() as connection:
            return (
                connection.execute("SELECT 1 FROM artifact_versions LIMIT 1").fetchone()
                is not None
            )

    def known_sha256(self, milestone_key: str, path: Path) -> str | None:
        with self._lock:
            return self._known_by_path.get((milestone_key, str(path.resolve())))

    def selection_impact(self, artifact_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            artifact = self._artifact(connection, artifact_id)
            milestone_key = str(artifact["milestone_key"])
            current = connection.execute(
                "SELECT artifact_id FROM milestone_selections WHERE milestone_key=?",
                (milestone_key,),
            ).fetchone()
            downstream = [
                definition.key
                for definition in MILESTONES[MILESTONE_INDEX[milestone_key] + 1 :]
                if connection.execute(
                    "SELECT 1 FROM milestone_selections WHERE milestone_key=?",
                    (definition.key,),
                ).fetchone()
            ]
        changing = current is None or current["artifact_id"] != artifact_id
        return {
            "artifact_id": artifact_id,
            "milestone_key": milestone_key,
            "changes_selection": changing,
            "downstream_selections": downstream if changing else [],
            "downstream_titles": (
                [milestone_definition(key).title for key in downstream] if changing else []
            ),
            "requires_confirmation": bool(changing and downstream),
            "message": (
                "Selecting this earlier version will clear downstream current selections "
                "and review approvals. Files remain preserved."
                if changing and downstream
                else "This version can be selected without clearing a downstream selection."
            ),
        }

    def select(self, artifact_id: str, *, confirm_impacts: bool) -> dict[str, Any]:
        impact = self.selection_impact(artifact_id)
        if impact["requires_confirmation"] and not confirm_impacts:
            raise ArtifactCatalogError("confirm the listed downstream reset before selecting")
        now = _now()
        with self._lock, self._connect() as connection:
            artifact = self._artifact(connection, artifact_id)
            if self._is_deleted(connection, artifact_id):
                raise ArtifactCatalogError("a permanently deleted version cannot be selected")
            if artifact["lifecycle"] != "available":
                raise ArtifactCatalogError("only an available, verified artifact can be selected")
            metadata = json.loads(artifact["metadata_json"])
            if metadata.get("selectable") is False:
                raise ArtifactCatalogError(
                    "this supporting record is not a selectable workflow input"
                )
            milestone_key = str(artifact["milestone_key"])
            cleared = list(impact["downstream_selections"])
            for key in cleared:
                connection.execute(
                    "DELETE FROM milestone_selections WHERE milestone_key=?", (key,)
                )
                self._selections.pop(key, None)
            self._select_in_connection(connection, milestone_key, artifact_id, now)
            self._event(
                connection,
                "selected",
                artifact_id,
                milestone_key,
                {"cleared_downstream_selections": cleared},
            )
        return {**impact, "status": "selected", "cleared_downstream_selections": cleared}

    def verify(self, artifact_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            artifact = self._artifact(connection, artifact_id)
            if self._is_deleted(connection, artifact_id):
                return {
                    "artifact_id": artifact_id,
                    "lifecycle": "deleted",
                    "byte_count": artifact["byte_count"],
                }
            lifecycle = _health(Path(artifact["path"]), str(artifact["sha256"]))
            if artifact["lifecycle"] == "archived" and lifecycle == "available":
                lifecycle = "archived"
            artifact_path = Path(artifact["path"])
            byte_count = artifact_path.stat().st_size if artifact_path.is_file() else None
            connection.execute(
                "UPDATE artifact_versions SET lifecycle=?, byte_count=? WHERE artifact_id=?",
                (lifecycle, byte_count, artifact_id),
            )
            self._event(
                connection,
                "verified",
                artifact_id,
                str(artifact["milestone_key"]),
                {"lifecycle": lifecycle},
            )
        return {"artifact_id": artifact_id, "lifecycle": lifecycle, "byte_count": byte_count}

    def set_archived(self, artifact_id: str, archived: bool) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            artifact = self._artifact(connection, artifact_id)
            if self._is_deleted(connection, artifact_id):
                raise ArtifactCatalogError("a permanently deleted version cannot be archived")
            selected = connection.execute(
                "SELECT 1 FROM milestone_selections WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
            if archived and selected:
                raise ArtifactCatalogError(
                    "select another version before archiving the current one"
                )
            downstream = connection.execute(
                """
                WITH RECURSIVE descendants(artifact_id) AS (
                    SELECT child_artifact_id FROM artifact_dependencies
                    WHERE parent_artifact_id=?
                    UNION
                    SELECT d.child_artifact_id FROM artifact_dependencies d
                    JOIN descendants p ON d.parent_artifact_id=p.artifact_id
                )
                SELECT s.milestone_key FROM descendants d
                JOIN milestone_selections s ON s.artifact_id=d.artifact_id
                LIMIT 1
                """,
                (artifact_id,),
            ).fetchone()
            if archived and downstream:
                raise ArtifactCatalogError(
                    "this version is used by a current later result; "
                    "select a compatible chain first"
                )
            lifecycle = (
                "archived"
                if archived
                else _health(Path(artifact["path"]), str(artifact["sha256"]))
            )
            connection.execute(
                "UPDATE artifact_versions SET lifecycle=? WHERE artifact_id=?",
                (lifecycle, artifact_id),
            )
            self._event(
                connection,
                "archived" if archived else "restored",
                artifact_id,
                str(artifact["milestone_key"]),
                {"lifecycle": lifecycle, "physical_file_preserved": True},
            )
        return {"artifact_id": artifact_id, "lifecycle": lifecycle}

    def deletion_impact(
        self,
        artifact_id: str,
        *,
        planned_deletions: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """Describe exact permanent-deletion effects without changing the filesystem."""

        with self._lock, self._connect() as connection:
            artifact = self._artifact(connection, artifact_id)
            deleted = self._is_deleted(connection, artifact_id)
            selected = connection.execute(
                "SELECT milestone_key FROM milestone_selections WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            retention = connection.execute(
                "SELECT retention_class, reason, protected_at, source "
                "FROM artifact_retention WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            descendants = list(
                connection.execute(
                    """
                    WITH RECURSIVE descendants(artifact_id, depth) AS (
                        SELECT child_artifact_id, 1 FROM artifact_dependencies
                        WHERE parent_artifact_id=?
                        UNION
                        SELECT d.child_artifact_id, p.depth + 1
                        FROM artifact_dependencies d
                        JOIN descendants p ON d.parent_artifact_id=p.artifact_id
                    )
                    SELECT DISTINCT a.artifact_id, a.milestone_key, a.display_name,
                        a.lifecycle, descendants.depth,
                        CASE WHEN s.artifact_id IS NULL THEN 0 ELSE 1 END AS selected
                    FROM descendants
                    JOIN artifact_versions a ON a.artifact_id=descendants.artifact_id
                    LEFT JOIN milestone_selections s ON s.artifact_id=a.artifact_id
                    LEFT JOIN artifact_deletions x ON x.artifact_id=a.artifact_id
                    WHERE x.artifact_id IS NULL
                    ORDER BY descendants.depth, a.created_at
                    """,
                    (artifact_id,),
                )
            )
            shared_records = connection.execute(
                """
                SELECT COUNT(*) AS count FROM artifact_versions a
                LEFT JOIN artifact_deletions d ON d.artifact_id=a.artifact_id
                WHERE a.path=? AND d.artifact_id IS NULL
                """,
                (artifact["path"],),
            ).fetchone()
        path = Path(str(artifact["path"])).resolve()
        inside_allowed_root = any(
            path.is_relative_to(root) for root in self.allowed_artifact_roots
        )
        blockers: list[str] = []
        if deleted:
            blockers.append("This file was already permanently deleted.")
        if selected:
            blockers.append(
                "This is part of the current workflow. Make another version current first."
            )
        if retention is not None:
            blockers.append(
                "This file has protected retention as "
                f"{retention['retention_class']}: {retention['reason']} "
                "Control-tower disposition is required before deletion."
            )
        retained_descendant_ids = {
            str(row["artifact_id"])
            for row in descendants
            if str(row["artifact_id"]) not in planned_deletions
        }
        if retained_descendant_ids:
            blockers.append(
                "Other retained results depend on this file. Delete those later results first."
            )
        if shared_records is not None and int(shared_records["count"]) > 1:
            blockers.append("More than one catalog record points to this file.")
        if not inside_allowed_root:
            blockers.append("The file is outside this scene's managed storage roots.")
        if path == self.path:
            blockers.append("The scene catalog database cannot delete itself.")
        lifecycle = "deleted" if deleted else str(artifact["lifecycle"])
        if lifecycle not in {"available", "archived"}:
            blockers.append("The file must exist and match its recorded identity before deletion.")
        deletion_token = self._deletion_token(artifact_id, path, str(artifact["sha256"]))
        return {
            "artifact_id": artifact_id,
            "milestone_key": artifact["milestone_key"],
            "display_name": artifact["display_name"],
            "path": str(path),
            "sha256": artifact["sha256"],
            "short_version": str(artifact["sha256"])[:12],
            "byte_count": artifact["byte_count"],
            "lifecycle": lifecycle,
            "selected": selected is not None,
            "protected_retention": (
                {
                    "class": retention["retention_class"],
                    "reason": retention["reason"],
                    "protected_at": retention["protected_at"],
                    "source": retention["source"],
                }
                if retention is not None
                else None
            ),
            "dependents": [
                {
                    "artifact_id": row["artifact_id"],
                    "milestone_key": row["milestone_key"],
                    "milestone_title": milestone_definition(str(row["milestone_key"])).title,
                    "display_name": row["display_name"],
                    "selected": bool(row["selected"]),
                }
                for row in descendants
            ],
            "allowed": not blockers,
            "blockers": blockers,
            "warning": (
                "Permanent deletion removes this exact file from disk. It cannot be restored "
                "from this console. The catalog keeps only its identity and deletion record."
            ),
            "deletion_token": deletion_token if not blockers else None,
        }

    def batch_deletion_impact(self, artifact_ids: Sequence[str]) -> dict[str, Any]:
        """Preview a bounded batch deletion, including dependencies within the batch."""

        unique_ids = tuple(dict.fromkeys(artifact_ids))
        if not unique_ids:
            raise ArtifactCatalogError("select at least one past file to delete")
        if len(unique_ids) > 100:
            raise ArtifactCatalogError("a maximum of 100 files can be deleted in one batch")
        planned = frozenset(unique_ids)
        impacts = [
            self.deletion_impact(artifact_id, planned_deletions=planned)
            for artifact_id in unique_ids
        ]
        return {
            "artifact_count": len(impacts),
            "total_byte_count": sum(int(item["byte_count"] or 0) for item in impacts),
            "all_allowed": all(bool(item["allowed"]) for item in impacts),
            "items": impacts,
            "warning": (
                "Permanent deletion removes every listed file from disk. The files cannot be "
                "restored from this console. Only their identities and deletion audit remain."
            ),
        }

    def delete_permanently(self, artifact_id: str, deletion_token: str) -> dict[str, Any]:
        """Permanently unlink one exact, dependency-free, non-current artifact file."""

        with self._lock:
            impact = self.deletion_impact(artifact_id)
            if not impact["allowed"]:
                raise ArtifactCatalogError(" ".join(impact["blockers"]))
            expected_token = str(impact["deletion_token"])
            if deletion_token != expected_token:
                raise ArtifactCatalogError("the deletion check is stale; review the warning again")
            path = Path(str(impact["path"])).resolve()
            if not path.is_file():
                raise ArtifactCatalogError("the exact artifact file no longer exists")
            if _sha256(path) != impact["sha256"]:
                raise ArtifactCatalogError("the file changed; permanent deletion was cancelled")
            byte_count = path.stat().st_size
            path.unlink()
            if path.exists():
                raise ArtifactCatalogError("the artifact file could not be removed")
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO artifact_deletions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        artifact_id,
                        _now(),
                        str(path),
                        impact["sha256"],
                        byte_count,
                        f"button-confirmed:{expected_token[:16]}",
                    ),
                )
                connection.execute(
                    "UPDATE artifact_versions SET lifecycle='missing' WHERE artifact_id=?",
                    (artifact_id,),
                )
                self._event(
                    connection,
                    "deleted-permanently",
                    artifact_id,
                    str(impact["milestone_key"]),
                    {
                        "path": str(path),
                        "sha256": impact["sha256"],
                        "byte_count": byte_count,
                        "recoverable_from_console": False,
                    },
                )
        return {
            "artifact_id": artifact_id,
            "status": "deleted",
            "path": str(path),
            "sha256": impact["sha256"],
            "deleted_byte_count": byte_count,
            "recoverable_from_console": False,
        }

    def delete_batch_permanently(self, deletion_tokens: Mapping[str, str]) -> dict[str, Any]:
        """Delete a preflighted set leaf-first so selected dependency chains are safe."""

        with self._lock:
            impact = self.batch_deletion_impact(tuple(deletion_tokens))
            if not impact["all_allowed"]:
                blockers = [blocker for item in impact["items"] for blocker in item["blockers"]]
                raise ArtifactCatalogError(" ".join(dict.fromkeys(blockers)))
            for item in impact["items"]:
                provided = deletion_tokens.get(str(item["artifact_id"]))
                if provided != item["deletion_token"]:
                    raise ArtifactCatalogError(
                        "the batch deletion check is stale; review the warning again"
                    )
                path = Path(str(item["path"])).resolve()
                if not path.is_file() or _sha256(path) != item["sha256"]:
                    raise ArtifactCatalogError(
                        "a selected file changed; batch deletion was cancelled"
                    )
            ordered = sorted(impact["items"], key=lambda item: len(item["dependents"]))
            results = [
                self.delete_permanently(
                    str(item["artifact_id"]),
                    str(deletion_tokens[str(item["artifact_id"])]),
                )
                for item in ordered
            ]
        return {
            "status": "deleted",
            "deleted_artifact_count": len(results),
            "deleted_byte_count": sum(int(item["deleted_byte_count"]) for item in results),
            "recoverable_from_console": False,
            "items": results,
        }

    def _deletion_token(self, artifact_id: str, path: Path, sha256: str) -> str:
        material = (
            f"p08-delete-button-v1|{self.project_id}|{self.scene_id}|{artifact_id}|{path}|{sha256}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def selected(self, milestone_key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.* FROM artifact_versions a
                JOIN milestone_selections s ON s.artifact_id=a.artifact_id
                WHERE s.milestone_key=?
                """,
                (milestone_key,),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def record_event(
        self,
        action: str,
        *,
        artifact_id: str | None = None,
        milestone_key: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            self._event(
                connection,
                action,
                artifact_id,
                milestone_key,
                dict(detail or {}),
            )

    def protect_retention(
        self,
        artifact_id: str,
        retention_class: str,
        reason: str,
        *,
        source: str,
    ) -> dict[str, str]:
        """Apply an idempotent deletion barrier to an authority-critical record."""

        if retention_class not in RETENTION_CLASSES:
            raise ArtifactCatalogError("unknown artifact retention class")
        if not reason.strip() or not source.strip():
            raise ArtifactCatalogError("retention reason and source are required")
        now = _now()
        with self._lock, self._connect() as connection:
            artifact = self._artifact(connection, artifact_id)
            existing = connection.execute(
                "SELECT retention_class, reason, protected_at, source "
                "FROM artifact_retention WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO artifact_retention VALUES (?, ?, ?, ?, ?)",
                    (artifact_id, retention_class, reason.strip(), now, source.strip()),
                )
                self._event(
                    connection,
                    "retention-protected",
                    artifact_id,
                    str(artifact["milestone_key"]),
                    {
                        "retention_class": retention_class,
                        "reason": reason.strip(),
                        "source": source.strip(),
                    },
                )
                protected_at = now
            else:
                if (
                    str(existing["retention_class"]) != retention_class
                    or str(existing["reason"]) != reason.strip()
                    or str(existing["source"]) != source.strip()
                ):
                    raise ArtifactCatalogError(
                        "artifact already has a different protected-retention policy"
                    )
                protected_at = str(existing["protected_at"])
        return {
            "artifact_id": artifact_id,
            "retention_class": retention_class,
            "reason": reason.strip(),
            "protected_at": protected_at,
            "source": source.strip(),
        }

    def _artifact(self, connection: sqlite3.Connection, artifact_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM artifact_versions WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise ArtifactCatalogError("unknown artifact version")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _is_deleted(connection: sqlite3.Connection, artifact_id: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM artifact_deletions WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
            is not None
        )

    def _select_in_connection(
        self, connection: sqlite3.Connection, milestone_key: str, artifact_id: str, now: str
    ) -> None:
        connection.execute(
            """
            INSERT INTO milestone_selections(milestone_key, artifact_id, selected_at)
            VALUES (?, ?, ?)
            ON CONFLICT(milestone_key) DO UPDATE SET artifact_id=excluded.artifact_id,
                selected_at=excluded.selected_at
            """,
            (milestone_key, artifact_id, now),
        )
        self._selections[milestone_key] = artifact_id

    def _event(
        self,
        connection: sqlite3.Connection,
        action: str,
        artifact_id: str | None,
        milestone_key: str | None,
        detail: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO artifact_events VALUES (?, ?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                _now(),
                action,
                artifact_id,
                milestone_key,
                json.dumps(dict(detail), sort_keys=True),
            ),
        )


def milestone_for_artifact(artifact_id: str, phase_id: str, kind: str) -> str:
    """Map established workflow artifact kinds to a human workflow milestone."""

    if kind == "floor-plan-pdf":
        return "floor-plan-source"
    mapping = {
        "facility-registration": "facility-registration",
        "capture-bundle": "capture-bundle",
        "pose-closeout-manifest": "calibration-correspondence",
        "camera-pose-registry": "calibration-pose-registry",
        "da3-input-manifest": "reconstruction-input",
        "da3-run-manifest": "reconstruction-input",
        "geometry-adoption-manifest": "reconstructed-geometry",
        "point-cloud-npz": "reconstructed-geometry",
        "floor-completion-manifest": "floor-refined-geometry",
        "floor-plane-npz": "floor-refined-geometry",
        "floor-verification": "floor-verification",
    }
    if kind in mapping:
        return mapping[kind]
    if kind == "rerun-recording":
        return (
            "final-review"
            if phase_id == "P08" or artifact_id.startswith("floor-")
            else "geometry-review"
        )
    raise ArtifactCatalogError(f"artifact kind is not a managed milestone: {kind}")


def milestone_definition(key: str) -> MilestoneDefinition:
    return MILESTONES[MILESTONE_INDEX[key]]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "artifact_id": row["artifact_id"],
        "milestone_key": row["milestone_key"],
        "phase_id": row["phase_id"],
        "kind": row["kind"],
        "path": row["path"],
        "sha256": row["sha256"],
        "short_version": str(row["sha256"])[:12],
        "byte_count": row["byte_count"],
        "display_name": row["display_name"],
        "significance": row["significance"],
        "lifecycle": row["lifecycle"],
        "created_at": row["created_at"],
        "discovered_at": row["discovered_at"],
        "metadata": json.loads(row["metadata_json"]),
    }


def _health(path: Path, sha256: str) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "available" if digest == sha256 else "corrupt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_time(path: Path) -> str:
    if not path.exists():
        return _now()
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat().replace("+00:00", "Z")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
