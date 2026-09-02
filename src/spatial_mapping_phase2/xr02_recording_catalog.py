"""Persistent XR02 operator-run lifecycle and saved-recording catalogue."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4


class XR02RunMode(StrEnum):
    LIVE = "live"
    RECORDING = "recording"


class XR02RunState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    FINALIZING = "finalizing"
    STOPPED = "stopped"
    AWAITING_DISPOSITION = "awaiting_disposition"
    SAVED = "saved"
    DELETED = "deleted"
    RECOVERY_REQUIRED = "recovery_required"


_TRANSIENT = (
    XR02RunState.STARTING.value,
    XR02RunState.RUNNING.value,
    XR02RunState.FINALIZING.value,
)
_BLOCKING = (
    *_TRANSIENT,
    XR02RunState.AWAITING_DISPOSITION.value,
    XR02RunState.RECOVERY_REQUIRED.value,
)


class RecordingCatalogError(RuntimeError):
    """Raised when an operator-run transition would violate the lifecycle contract."""


@dataclass(frozen=True, slots=True)
class OperatorRun:
    session_id: str
    mode: XR02RunMode
    state: XR02RunState
    run_directory: Path
    started_at_utc: str
    stopped_at_utc: str | None
    label: str | None
    manifest_path: Path | None
    telemetry_path: Path | None
    byte_count: int | None
    error_detail: str | None
    scene_context_sha256: str | None = None
    scene_binding_sha256: str | None = None
    stop_reason: str | None = None
    resumed_from_session_id: str | None = None
    scene_update_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "state": self.state.value,
            "run_directory": str(self.run_directory),
            "started_at_utc": self.started_at_utc,
            "stopped_at_utc": self.stopped_at_utc,
            "label": self.label,
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "telemetry_path": None if self.telemetry_path is None else str(self.telemetry_path),
            "byte_count": self.byte_count,
            "error_detail": self.error_detail,
            "scene_context_sha256": self.scene_context_sha256,
            "scene_binding_sha256": self.scene_binding_sha256,
            "stop_reason": self.stop_reason,
            "resumed_from_session_id": self.resumed_from_session_id,
            "scene_update_id": self.scene_update_id,
        }


class XR02RecordingCatalog:
    """SQLite-owned lifecycle state survives browser and console-process restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operator_runs (
                    session_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL CHECK(mode IN ('live','recording')),
                    state TEXT NOT NULL,
                    run_directory TEXT NOT NULL UNIQUE,
                    started_at_utc TEXT NOT NULL,
                    stopped_at_utc TEXT,
                    label TEXT,
                    manifest_path TEXT,
                    telemetry_path TEXT,
                    byte_count INTEGER,
                    error_detail TEXT
                );
                CREATE INDEX IF NOT EXISTS operator_runs_state
                    ON operator_runs(state, started_at_utc);
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(operator_runs)").fetchall()
            }
            additions = {
                "scene_context_sha256": "TEXT",
                "scene_binding_sha256": "TEXT",
                "stop_reason": "TEXT",
                "resumed_from_session_id": "TEXT",
                "scene_update_id": "TEXT",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE operator_runs ADD COLUMN {name} {declaration}"
                    )
            placeholders = ",".join("?" for _ in _TRANSIENT)
            connection.execute(
                f"""UPDATE operator_runs
                    SET state = ?, error_detail = COALESCE(error_detail, ?)
                    WHERE state IN ({placeholders})""",
                (
                    XR02RunState.RECOVERY_REQUIRED.value,
                    "console process ended before this transition completed",
                    *_TRANSIENT,
                ),
            )

    def begin(
        self,
        mode: XR02RunMode,
        run_directory: Path,
        *,
        scene_context_sha256: str | None = None,
        scene_binding_sha256: str | None = None,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> OperatorRun:
        session_id = f"xr02-{uuid4().hex}"
        started = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            if self._blocking_row(connection) is not None:
                raise RecordingCatalogError(
                    "resolve the staged or interrupted run before starting another run"
                )
            connection.execute(
                """INSERT INTO operator_runs
                    (session_id, mode, state, run_directory, started_at_utc,
                     scene_context_sha256, scene_binding_sha256,
                     resumed_from_session_id, scene_update_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    mode.value,
                    XR02RunState.STARTING.value,
                    str(run_directory.resolve()),
                    started,
                    scene_context_sha256,
                    scene_binding_sha256,
                    resumed_from_session_id,
                    scene_update_id,
                ),
            )
        return self.require(session_id)

    def mark_running(self, session_id: str) -> OperatorRun:
        return self._transition(session_id, XR02RunState.STARTING, XR02RunState.RUNNING)

    def mark_finalizing(self, session_id: str) -> OperatorRun:
        return self._transition(session_id, XR02RunState.RUNNING, XR02RunState.FINALIZING)

    def finalize(
        self,
        session_id: str,
        *,
        manifest_path: Path | None,
        telemetry_path: Path | None,
        byte_count: int,
        stop_reason: str = "operator",
    ) -> OperatorRun:
        current = self.require(session_id)
        target = (
            XR02RunState.STOPPED
            if current.mode is XR02RunMode.LIVE
            else XR02RunState.AWAITING_DISPOSITION
        )
        stopped = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE operator_runs
                    SET state = ?, stopped_at_utc = ?, manifest_path = ?, telemetry_path = ?,
                        byte_count = ?, error_detail = NULL, stop_reason = ?
                    WHERE session_id = ? AND state = ?""",
                (
                    target.value,
                    stopped,
                    None if manifest_path is None else str(manifest_path.resolve()),
                    None if telemetry_path is None else str(telemetry_path.resolve()),
                    byte_count,
                    stop_reason,
                    session_id,
                    XR02RunState.FINALIZING.value,
                ),
            ).rowcount
        if changed != 1:
            raise RecordingCatalogError("run was not in finalizing state")
        return self.require(session_id)

    def require_recovery(self, session_id: str, detail: str) -> OperatorRun:
        if not detail:
            raise RecordingCatalogError("recovery detail is required")
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE operator_runs SET state = ?, error_detail = ?
                    WHERE session_id = ? AND state NOT IN (?, ?, ?)""",
                (
                    XR02RunState.RECOVERY_REQUIRED.value,
                    detail,
                    session_id,
                    XR02RunState.SAVED.value,
                    XR02RunState.DELETED.value,
                    XR02RunState.STOPPED.value,
                ),
            ).rowcount
        if changed != 1:
            raise RecordingCatalogError("completed run cannot enter recovery")
        return self.require(session_id)

    def save(self, session_id: str, label: str) -> OperatorRun:
        normalized = " ".join(label.split())
        if not normalized or len(normalized) > 80:
            raise RecordingCatalogError("recording label must contain 1..80 visible characters")
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE operator_runs SET state = ?, label = ?
                    WHERE session_id = ? AND mode = ? AND state = ?""",
                (
                    XR02RunState.SAVED.value,
                    normalized,
                    session_id,
                    XR02RunMode.RECORDING.value,
                    XR02RunState.AWAITING_DISPOSITION.value,
                ),
            ).rowcount
        if changed != 1:
            raise RecordingCatalogError("only a finalized staged recording can be saved")
        return self.require(session_id)

    def mark_deleted(self, session_id: str) -> OperatorRun:
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE operator_runs SET state = ?, label = NULL
                    WHERE session_id = ? AND (
                        state = ? OR (mode = ? AND state = ?)
                    )""",
                (
                    XR02RunState.DELETED.value,
                    session_id,
                    XR02RunState.RECOVERY_REQUIRED.value,
                    XR02RunMode.RECORDING.value,
                    XR02RunState.AWAITING_DISPOSITION.value,
                ),
            ).rowcount
        if changed != 1:
            raise RecordingCatalogError("recording is not eligible for deletion")
        return self.require(session_id)

    def require(self, session_id: str) -> OperatorRun:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM operator_runs WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise RecordingCatalogError("operator run does not exist")
        return _row(row)

    def blocking(self) -> OperatorRun | None:
        with self._connect() as connection:
            row = self._blocking_row(connection)
        return None if row is None else _row(row)

    def saved(self) -> tuple[OperatorRun, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM operator_runs WHERE state = ?
                    ORDER BY started_at_utc DESC""",
                (XR02RunState.SAVED.value,),
            ).fetchall()
        return tuple(_row(row) for row in rows)

    def recent_live(self, limit: int = 10) -> tuple[OperatorRun, ...]:
        if limit <= 0:
            raise RecordingCatalogError("recent Live limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM operator_runs WHERE mode = ? AND state = ?
                    ORDER BY started_at_utc DESC LIMIT ?""",
                (XR02RunMode.LIVE.value, XR02RunState.STOPPED.value, limit),
            ).fetchall()
        return tuple(_row(row) for row in rows)

    def _transition(
        self, session_id: str, current: XR02RunState, target: XR02RunState
    ) -> OperatorRun:
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE operator_runs SET state = ? WHERE session_id = ? AND state = ?",
                (target.value, session_id, current.value),
            ).rowcount
        if changed != 1:
            raise RecordingCatalogError(f"run was not in {current.value} state")
        return self.require(session_id)

    @staticmethod
    def _blocking_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
        placeholders = ",".join("?" for _ in _BLOCKING)
        rows = connection.execute(
            f"""SELECT * FROM operator_runs WHERE state IN ({placeholders})
                ORDER BY started_at_utc DESC LIMIT 2""",
            _BLOCKING,
        ).fetchall()
        if len(rows) > 1:
            raise RecordingCatalogError("multiple unresolved operator runs require manual review")
        return None if not rows else rows[0]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection


def _row(row: sqlite3.Row) -> OperatorRun:
    return OperatorRun(
        session_id=str(row["session_id"]),
        mode=XR02RunMode(str(row["mode"])),
        state=XR02RunState(str(row["state"])),
        run_directory=Path(str(row["run_directory"])),
        started_at_utc=str(row["started_at_utc"]),
        stopped_at_utc=(None if row["stopped_at_utc"] is None else str(row["stopped_at_utc"])),
        label=None if row["label"] is None else str(row["label"]),
        manifest_path=(None if row["manifest_path"] is None else Path(str(row["manifest_path"]))),
        telemetry_path=(
            None if row["telemetry_path"] is None else Path(str(row["telemetry_path"]))
        ),
        byte_count=None if row["byte_count"] is None else int(row["byte_count"]),
        error_detail=None if row["error_detail"] is None else str(row["error_detail"]),
        scene_context_sha256=(
            None if row["scene_context_sha256"] is None else str(row["scene_context_sha256"])
        ),
        scene_binding_sha256=(
            None if row["scene_binding_sha256"] is None else str(row["scene_binding_sha256"])
        ),
        stop_reason=None if row["stop_reason"] is None else str(row["stop_reason"]),
        resumed_from_session_id=(
            None if row["resumed_from_session_id"] is None else str(row["resumed_from_session_id"])
        ),
        scene_update_id=(None if row["scene_update_id"] is None else str(row["scene_update_id"])),
    )
