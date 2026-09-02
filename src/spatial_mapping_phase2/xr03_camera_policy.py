"""Versioned, scene-scoped camera relationship policy for XR03.

The policy is deliberately separate from immutable ``scene.json`` and artifact files.  SQLite
revisions are append-only; selecting or rolling back a revision writes an additional audit event.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class CameraPolicyError(ValueError):
    """Raised when a camera policy action is malformed, stale, or unsafe."""


class OverlapVerdict(StrEnum):
    OVERLAP = "overlap"
    NO_OVERLAP = "no_overlap"
    UNREVIEWED = "unreviewed"


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class IntrinsicGroup:
    group_id: str
    lens_model: str
    camera_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IntrinsicGroup:
        camera_ids = _string_sequence(value.get("camera_ids"), "intrinsic group camera_ids")
        return cls(
            group_id=_string(value.get("group_id"), "intrinsic group_id"),
            lens_model=_string(value.get("lens_model"), "lens model"),
            camera_ids=camera_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "lens_model": self.lens_model,
            "camera_ids": list(self.camera_ids),
        }


@dataclass(frozen=True)
class OverlapPairReview:
    camera_id_a: str
    camera_id_b: str
    verdict: OverlapVerdict

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OverlapPairReview:
        return cls(
            camera_id_a=_string(value.get("camera_id_a"), "overlap camera_id_a"),
            camera_id_b=_string(value.get("camera_id_b"), "overlap camera_id_b"),
            verdict=OverlapVerdict(_string(value.get("verdict"), "overlap verdict")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "camera_id_a": self.camera_id_a,
            "camera_id_b": self.camera_id_b,
            "verdict": self.verdict.value,
        }


@dataclass(frozen=True)
class SceneCameraPolicy:
    project_id: str
    scene_id: str
    camera_ids: tuple[str, ...]
    intrinsic_groups: tuple[IntrinsicGroup, ...]
    overlap_pair_reviews: tuple[OverlapPairReview, ...]
    schema_version: str = "xr03-camera-policy-v1"

    @classmethod
    def build(
        cls,
        project_id: str,
        scene_id: str,
        camera_ids: Sequence[str],
        intrinsic_groups: Sequence[Mapping[str, Any]],
        overlap_pair_reviews: Sequence[Mapping[str, Any]],
    ) -> SceneCameraPolicy:
        roster = tuple(camera_ids)
        roster_index = {camera_id: index for index, camera_id in enumerate(roster)}
        groups = tuple(IntrinsicGroup.from_dict(value) for value in intrinsic_groups)
        reviews = tuple(OverlapPairReview.from_dict(value) for value in overlap_pair_reviews)
        value = cls(project_id, scene_id, roster, groups, reviews)
        value._validate(roster_index)
        return value._canonical(roster_index)

    @classmethod
    def blank(cls, project_id: str, scene_id: str, camera_ids: Sequence[str]) -> SceneCameraPolicy:
        roster = tuple(camera_ids)
        pairs = [
            {
                "camera_id_a": left,
                "camera_id_b": right,
                "verdict": OverlapVerdict.NO_OVERLAP.value,
            }
            for left, right in itertools.combinations(roster, 2)
        ]
        return cls.build(project_id, scene_id, roster, (), pairs)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneCameraPolicy:
        if value.get("schema_version") != "xr03-camera-policy-v1":
            raise CameraPolicyError("unsupported camera policy schema")
        groups = _mapping_sequence(value.get("intrinsic_groups"), "intrinsic_groups")
        reviews = _mapping_sequence(value.get("overlap_pair_reviews"), "overlap_pair_reviews")
        return cls.build(
            _string(value.get("project_id"), "project_id"),
            _string(value.get("scene_id"), "scene_id"),
            _string_sequence(value.get("camera_ids"), "camera_ids"),
            groups,
            reviews,
        )

    def _validate(self, roster_index: Mapping[str, int]) -> None:
        if not self.camera_ids:
            raise CameraPolicyError("camera policy roster must not be empty")
        if len(roster_index) != len(self.camera_ids):
            raise CameraPolicyError("camera policy roster contains duplicate camera IDs")
        for label, value in (("project_id", self.project_id), ("scene_id", self.scene_id)):
            if not _IDENTIFIER.fullmatch(value):
                raise CameraPolicyError(f"{label} is malformed")
        seen_groups: set[str] = set()
        assigned: set[str] = set()
        for group in self.intrinsic_groups:
            if not _IDENTIFIER.fullmatch(group.group_id):
                raise CameraPolicyError("intrinsic group_id is malformed")
            if group.group_id in seen_groups:
                raise CameraPolicyError("intrinsic group IDs must be unique")
            seen_groups.add(group.group_id)
            if not group.lens_model.strip():
                raise CameraPolicyError("lens model must not be blank")
            if not group.camera_ids:
                raise CameraPolicyError("intrinsic groups must contain at least one camera")
            if len(set(group.camera_ids)) != len(group.camera_ids):
                raise CameraPolicyError("an intrinsic group contains duplicate cameras")
            for camera_id in group.camera_ids:
                if camera_id not in roster_index:
                    raise CameraPolicyError("intrinsic group contains a camera outside the scene")
                if camera_id in assigned:
                    raise CameraPolicyError("each camera may belong to only one intrinsic group")
                assigned.add(camera_id)
        expected_pairs = set(itertools.combinations(self.camera_ids, 2))
        actual_pairs: set[tuple[str, str]] = set()
        for review in self.overlap_pair_reviews:
            pair = (review.camera_id_a, review.camera_id_b)
            if pair[0] not in roster_index or pair[1] not in roster_index:
                raise CameraPolicyError("overlap review contains a camera outside the scene")
            if roster_index[pair[0]] >= roster_index[pair[1]]:
                raise CameraPolicyError("overlap pairs must follow canonical scene-roster order")
            if pair in actual_pairs:
                raise CameraPolicyError("overlap pairs must be unique")
            actual_pairs.add(pair)
        if actual_pairs != expected_pairs:
            raise CameraPolicyError("overlap reviews must include every unordered camera pair")

    def _canonical(self, roster_index: Mapping[str, int]) -> SceneCameraPolicy:
        groups = tuple(
            IntrinsicGroup(
                group.group_id,
                group.lens_model.strip(),
                tuple(sorted(group.camera_ids, key=roster_index.__getitem__)),
            )
            for group in sorted(self.intrinsic_groups, key=lambda item: item.group_id)
        )
        reviews = tuple(
            sorted(
                self.overlap_pair_reviews,
                key=lambda item: (
                    roster_index[item.camera_id_a],
                    roster_index[item.camera_id_b],
                ),
            )
        )
        return SceneCameraPolicy(
            self.project_id,
            self.scene_id,
            self.camera_ids,
            groups,
            reviews,
        )

    @property
    def assigned_camera_ids(self) -> frozenset[str]:
        return frozenset(
            camera_id for group in self.intrinsic_groups for camera_id in group.camera_ids
        )

    @property
    def lens_complete(self) -> bool:
        return self.assigned_camera_ids == frozenset(self.camera_ids)

    @property
    def overlap_complete(self) -> bool:
        return all(
            review.verdict is not OverlapVerdict.UNREVIEWED for review in self.overlap_pair_reviews
        )

    @property
    def overlap_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (review.camera_id_a, review.camera_id_b)
            for review in self.overlap_pair_reviews
            if review.verdict is OverlapVerdict.OVERLAP
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "scene_id": self.scene_id,
            "camera_ids": list(self.camera_ids),
            "intrinsic_groups": [group.to_dict() for group in self.intrinsic_groups],
            "overlap_pair_reviews": [review.to_dict() for review in self.overlap_pair_reviews],
            "static_reconstruction_policy": "all-enabled-cameras-per-scene-joint",
            "overlap_usage": "xr02-genuine-overlap-deduplication-only",
        }

    def intrinsic_policy_payload(self) -> dict[str, Any]:
        """Return the policy subset that can materially change camera calibration."""

        return {
            "schema_version": "xr03-intrinsic-policy-identity-v1",
            "project_id": self.project_id,
            "scene_id": self.scene_id,
            "camera_ids": list(self.camera_ids),
            "intrinsic_groups": [group.to_dict() for group in self.intrinsic_groups],
        }

    @property
    def intrinsic_policy_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.intrinsic_policy_payload()).encode("utf-8")
        ).hexdigest()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload(),
            "policy_sha256": self.sha256,
            "intrinsic_policy_sha256": self.intrinsic_policy_sha256,
            "lens_complete": self.lens_complete,
            "overlap_complete": self.overlap_complete,
            "unassigned_camera_ids": [
                camera_id
                for camera_id in self.camera_ids
                if camera_id not in self.assigned_camera_ids
            ],
            "overlap_edges": [list(edge) for edge in self.overlap_edges],
        }


class CameraPolicyRepository:
    """Append-only policy revisions plus auditable active-revision selections."""

    def __init__(
        self, path: Path, project_id: str, scene_id: str, camera_ids: Sequence[str]
    ) -> None:
        self.path = path.resolve()
        self.project_id = project_id
        self.scene_id = scene_id
        self.camera_ids = tuple(camera_ids)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def open(cls, path: Path) -> CameraPolicyRepository:
        """Open an existing store using its immutable scene/roster binding metadata."""

        resolved = path.resolve()
        if not resolved.is_file():
            raise CameraPolicyError(f"camera policy database is unavailable: {resolved}")
        connection = sqlite3.connect(resolved, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT project_id, scene_id, camera_ids_json FROM policy_metadata "
                "WHERE singleton=1"
            ).fetchone()
        except sqlite3.Error as error:
            raise CameraPolicyError("camera policy database metadata is malformed") from error
        finally:
            connection.close()
        if row is None:
            raise CameraPolicyError("camera policy database metadata is missing")
        camera_ids = json.loads(row["camera_ids_json"])
        if not isinstance(camera_ids, list) or not all(
            isinstance(item, str) for item in camera_ids
        ):
            raise CameraPolicyError("camera policy database roster is malformed")
        return cls(resolved, row["project_id"], row["scene_id"], camera_ids)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS policy_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    camera_ids_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policy_revisions (
                    revision INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_sha256 TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    action_id TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS policy_selections (
                    selection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision INTEGER NOT NULL REFERENCES policy_revisions(revision),
                    action_id TEXT NOT NULL UNIQUE,
                    selected_at_utc TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_policy (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    revision INTEGER NOT NULL REFERENCES policy_revisions(revision),
                    selection_id INTEGER NOT NULL REFERENCES policy_selections(selection_id)
                );
                """
            )
            row = connection.execute("SELECT * FROM policy_metadata WHERE singleton=1").fetchone()
            expected = (
                "xr03-camera-policy-store-v1",
                self.project_id,
                self.scene_id,
                _canonical_json(list(self.camera_ids)),
            )
            if row is None:
                connection.execute("INSERT INTO policy_metadata VALUES (1, ?, ?, ?, ?)", expected)
            elif (
                tuple(
                    row[key]
                    for key in ("schema_version", "project_id", "scene_id", "camera_ids_json")
                )
                != expected
            ):
                raise CameraPolicyError(
                    "camera policy database does not match this scene and enabled roster"
                )

    def status(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            active = connection.execute(
                "SELECT r.*, s.selection_id, s.selected_at_utc, s.reason "
                "FROM active_policy a JOIN policy_revisions r ON r.revision=a.revision "
                "JOIN policy_selections s ON s.selection_id=a.selection_id WHERE a.singleton=1"
            ).fetchone()
            rows = list(
                connection.execute(
                    "SELECT r.*, MAX(s.selected_at_utc) AS last_selected_at_utc, "
                    "COUNT(s.selection_id) AS selection_count FROM policy_revisions r "
                    "LEFT JOIN policy_selections s ON s.revision=r.revision "
                    "GROUP BY r.revision ORDER BY r.revision DESC"
                )
            )
        active_revision = int(active["revision"]) if active is not None else None
        revisions = [
            {
                "revision": int(row["revision"]),
                "policy_sha256": row["policy_sha256"],
                "created_at_utc": row["created_at_utc"],
                "action_id": row["action_id"],
                "last_selected_at_utc": row["last_selected_at_utc"],
                "selection_count": int(row["selection_count"]),
                "active": int(row["revision"]) == active_revision,
                "policy": SceneCameraPolicy.from_dict(json.loads(row["payload_json"])).to_dict(),
            }
            for row in rows
        ]
        proposal = SceneCameraPolicy.blank(
            self.project_id, self.scene_id, self.camera_ids
        ).to_dict()
        return {
            "schema_version": "xr03-camera-policy-status-v1",
            "database_path": str(self.path),
            "project_id": self.project_id,
            "scene_id": self.scene_id,
            "camera_ids": list(self.camera_ids),
            "active_revision": active_revision,
            "active_selection_id": int(active["selection_id"]) if active is not None else None,
            "selected_at_utc": active["selected_at_utc"] if active is not None else None,
            "selection_reason": active["reason"] if active is not None else None,
            "active_policy": (
                SceneCameraPolicy.from_dict(json.loads(active["payload_json"])).to_dict()
                if active is not None
                else None
            ),
            "proposal": proposal,
            "revisions": revisions,
        }

    def active(
        self, *, require_lens: bool = False, require_overlap: bool = False
    ) -> SceneCameraPolicy:
        status = self.status()
        value = status["active_policy"]
        if value is None:
            raise CameraPolicyError("no camera policy revision is active")
        policy = SceneCameraPolicy.from_dict(value)
        if require_lens and not policy.lens_complete:
            raise CameraPolicyError("lens-model grouping is incomplete for the enabled cameras")
        if require_overlap and not policy.overlap_complete:
            raise CameraPolicyError("view-overlap review is incomplete for the enabled cameras")
        return policy

    def by_sha256(self, policy_sha256: str) -> SceneCameraPolicy | None:
        """Return one retained policy revision by immutable identity."""

        if not re.fullmatch(r"[0-9a-f]{64}", policy_sha256):
            raise CameraPolicyError("camera policy SHA-256 is malformed")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM policy_revisions WHERE policy_sha256=?",
                (policy_sha256,),
            ).fetchone()
        if row is None:
            return None
        return SceneCameraPolicy.from_dict(json.loads(row["payload_json"]))

    def impact(self, proposed: SceneCameraPolicy) -> dict[str, Any]:
        self._assert_binding(proposed)
        status = self.status()
        current_value = status["active_policy"]
        current = SceneCameraPolicy.from_dict(current_value) if current_value else None
        old_membership = _group_membership(current) if current is not None else {}
        new_membership = _group_membership(proposed)
        lens_changed = [
            camera_id
            for camera_id in self.camera_ids
            if old_membership.get(camera_id) != new_membership.get(camera_id)
        ]
        old_overlap = _overlap_map(current) if current is not None else {}
        new_overlap = _overlap_map(proposed)
        overlap_changed = [
            {"camera_id_a": pair[0], "camera_id_b": pair[1]}
            for pair in itertools.combinations(self.camera_ids, 2)
            if old_overlap.get(pair) != new_overlap.get(pair)
        ]
        changed = current is None or current.sha256 != proposed.sha256
        return {
            "changed": changed,
            "current_revision": status["active_revision"],
            "current_policy_sha256": current.sha256 if current is not None else None,
            "proposed_policy_sha256": proposed.sha256,
            "lens_membership_changed_camera_ids": lens_changed,
            "overlap_changed_pairs": overlap_changed,
            "intrinsic_reprocessing_required": bool(current is not None and lens_changed),
            "xr02_new_epoch_required": bool(current is not None and overlap_changed),
            "static_reconstruction_cohort_changed": False,
            "requires_confirmation": bool(current is not None and changed),
        }

    def apply(
        self,
        action_id: str,
        proposed: SceneCameraPolicy,
        *,
        expected_revision: int | None,
        confirm_impacts: bool,
    ) -> dict[str, Any]:
        _require_identifier(action_id, "camera policy action_id")
        self._assert_binding(proposed)
        impact = self.impact(proposed)
        if impact["requires_confirmation"] and not confirm_impacts:
            raise CameraPolicyError("confirm camera-policy downstream impacts before applying")
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT revision FROM active_policy WHERE singleton=1"
            ).fetchone()
            current_revision = int(current["revision"]) if current is not None else None
            if current_revision != expected_revision:
                raise CameraPolicyError("camera policy revision is stale; reload before saving")
            existing = connection.execute(
                "SELECT revision FROM policy_revisions WHERE policy_sha256=?",
                (proposed.sha256,),
            ).fetchone()
            if existing is not None:
                revision = int(existing["revision"])
                if revision == current_revision:
                    raise CameraPolicyError("this camera policy revision is already active")
                selection_id = self._select(
                    connection, revision, f"{action_id}-selection", now, "reapplied"
                )
                return {
                    **impact,
                    "revision": revision,
                    "selection_id": selection_id,
                    "policy": proposed.to_dict(),
                }
            cursor = connection.execute(
                "INSERT INTO policy_revisions "
                "(policy_sha256, payload_json, created_at_utc, action_id) VALUES (?, ?, ?, ?)",
                (proposed.sha256, _canonical_json(proposed.payload()), now, action_id),
            )
            if cursor.lastrowid is None:
                raise CameraPolicyError("camera policy revision identity was not created")
            revision = int(cursor.lastrowid)
            selection_id = self._select(
                connection, revision, f"{action_id}-selection", now, "applied"
            )
        return {
            **impact,
            "revision": revision,
            "selection_id": selection_id,
            "policy": proposed.to_dict(),
        }

    def rollback(
        self,
        action_id: str,
        target_revision: int,
        *,
        expected_revision: int,
        confirm_impacts: bool,
    ) -> dict[str, Any]:
        _require_identifier(action_id, "camera policy rollback action_id")
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT revision FROM active_policy WHERE singleton=1"
            ).fetchone()
            current_revision = int(current["revision"]) if current is not None else None
            if current_revision != expected_revision:
                raise CameraPolicyError("camera policy revision is stale; reload before rollback")
            if target_revision == current_revision:
                raise CameraPolicyError("target camera policy revision is already active")
            target = connection.execute(
                "SELECT payload_json FROM policy_revisions WHERE revision=?", (target_revision,)
            ).fetchone()
            if target is None:
                raise CameraPolicyError("target camera policy revision does not exist")
            policy = SceneCameraPolicy.from_dict(json.loads(target["payload_json"]))
            impact = self.impact(policy)
            if impact["requires_confirmation"] and not confirm_impacts:
                raise CameraPolicyError("confirm camera-policy downstream impacts before rollback")
            selection_id = self._select(connection, target_revision, action_id, now, "rollback")
        return {
            **impact,
            "revision": target_revision,
            "selection_id": selection_id,
            "policy": policy.to_dict(),
        }

    def _select(
        self,
        connection: sqlite3.Connection,
        revision: int,
        action_id: str,
        at_utc: str,
        reason: str,
    ) -> int:
        cursor = connection.execute(
            "INSERT INTO policy_selections (revision, action_id, selected_at_utc, reason) "
            "VALUES (?, ?, ?, ?)",
            (revision, action_id, at_utc, reason),
        )
        if cursor.lastrowid is None:
            raise CameraPolicyError("camera policy selection identity was not created")
        selection_id = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO active_policy VALUES (1, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET revision=excluded.revision, "
            "selection_id=excluded.selection_id",
            (revision, selection_id),
        )
        return selection_id

    def _assert_binding(self, policy: SceneCameraPolicy) -> None:
        if (
            policy.project_id != self.project_id
            or policy.scene_id != self.scene_id
            or policy.camera_ids != self.camera_ids
        ):
            raise CameraPolicyError("camera policy does not match this scene and enabled roster")


def policy_from_changes(
    repository: CameraPolicyRepository, value: Mapping[str, Any]
) -> SceneCameraPolicy:
    return SceneCameraPolicy.build(
        repository.project_id,
        repository.scene_id,
        repository.camera_ids,
        _mapping_sequence(value.get("intrinsic_groups"), "intrinsic_groups"),
        _mapping_sequence(value.get("overlap_pair_reviews"), "overlap_pair_reviews"),
    )


def _group_membership(policy: SceneCameraPolicy | None) -> dict[str, tuple[str, str]]:
    if policy is None:
        return {}
    return {
        camera_id: (group.group_id, group.lens_model)
        for group in policy.intrinsic_groups
        for camera_id in group.camera_ids
    }


def _overlap_map(
    policy: SceneCameraPolicy | None,
) -> dict[tuple[str, str], OverlapVerdict]:
    if policy is None:
        return {}
    return {
        (review.camera_id_a, review.camera_id_b): review.verdict
        for review in policy.overlap_pair_reviews
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraPolicyError(f"{label} must be a non-empty string")
    return value.strip()


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) and not isinstance(value, tuple):
        raise CameraPolicyError(f"{label} must be a list")
    return tuple(_string(item, label) for item in value)


def _mapping_sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) and not isinstance(value, tuple):
        raise CameraPolicyError(f"{label} must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise CameraPolicyError(f"{label} entries must be objects")
    return tuple(value)


def _require_identifier(value: str, label: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise CameraPolicyError(f"{label} is malformed")
