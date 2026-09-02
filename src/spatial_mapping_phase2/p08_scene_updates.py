"""Persistent post-review static-scene update scheduling and adoption contracts."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class SceneUpdateError(ValueError):
    """Raised when scene-update configuration or state is unsafe or malformed."""


class UpdateMode(StrEnum):
    MANUAL = "manual"
    DAILY = "daily"
    INTERVAL = "interval"


ALLOWED_INTERVAL_SECONDS = (1800, 3600, 10_800, 21_600, 43_200, 86_400, 604_800)


@dataclass(frozen=True)
class SceneUpdateSchedule:
    enabled: bool
    mode: UpdateMode
    timezone: str
    daily_time: str = "09:00"
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    interval_seconds: int = 3600
    median_frame_count: int = 5
    median_spacing_seconds: int = 120

    def __post_init__(self) -> None:
        if self.mode is UpdateMode.MANUAL and self.enabled:
            raise SceneUpdateError("manual mode cannot enable background scheduling")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise SceneUpdateError("schedule timezone is unknown") from error
        _parse_local_time(self.daily_time)
        if not self.weekdays or len(set(self.weekdays)) != len(self.weekdays):
            raise SceneUpdateError("daily schedule requires distinct selected weekdays")
        if any(day < 0 or day > 6 for day in self.weekdays):
            raise SceneUpdateError("weekday values must be between Monday 0 and Sunday 6")
        if self.interval_seconds not in ALLOWED_INTERVAL_SECONDS:
            raise SceneUpdateError("interval is not one of the supported bounded choices")
        if (
            self.median_frame_count < 3
            or self.median_frame_count > 11
            or self.median_frame_count % 2 == 0
        ):
            raise SceneUpdateError("median frame count must be an odd value from 3 to 11")
        if self.median_spacing_seconds < 10 or self.median_spacing_seconds > 600:
            raise SceneUpdateError("median frame spacing must be between 10 and 600 seconds")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneUpdateSchedule:
        try:
            mode = UpdateMode(str(value.get("mode", UpdateMode.MANUAL.value)))
        except ValueError as error:
            raise SceneUpdateError("scene update mode is unknown") from error
        weekdays = value.get("weekdays", list(range(7)))
        if not isinstance(weekdays, list) or not all(
            isinstance(day, int) and not isinstance(day, bool) for day in weekdays
        ):
            raise SceneUpdateError("weekdays must be an integer list")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise SceneUpdateError("schedule enabled must be boolean")
        return cls(
            enabled=enabled,
            mode=mode,
            timezone=_required_string(value, "timezone", default="Asia/Singapore"),
            daily_time=_required_string(value, "daily_time", default="09:00"),
            weekdays=tuple(weekdays),
            interval_seconds=_required_int(value, "interval_seconds", 3600),
            median_frame_count=_required_int(value, "median_frame_count", 5),
            median_spacing_seconds=_required_int(value, "median_spacing_seconds", 120),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "timezone": self.timezone,
            "daily_time": self.daily_time,
            "weekdays": list(self.weekdays),
            "interval_seconds": self.interval_seconds,
            "median_frame_count": self.median_frame_count,
            "median_spacing_seconds": self.median_spacing_seconds,
        }

    @property
    def capture_frame_count(self) -> int:
        return self.median_frame_count if self.mode is UpdateMode.INTERVAL else 1

    @property
    def capture_spacing_seconds(self) -> int:
        return self.median_spacing_seconds if self.mode is UpdateMode.INTERVAL else 0

    def next_due_after(self, now_utc: datetime, *, anchor_utc: datetime | None = None) -> datetime:
        now = _aware_utc(now_utc)
        if self.mode is UpdateMode.MANUAL:
            raise SceneUpdateError("manual mode has no scheduled occurrence")
        if self.mode is UpdateMode.INTERVAL:
            anchor = _aware_utc(anchor_utc or now)
            interval = timedelta(seconds=self.interval_seconds)
            candidate = anchor + interval
            if candidate <= now:
                elapsed = (now - anchor).total_seconds()
                candidate = anchor + interval * (int(elapsed // self.interval_seconds) + 1)
            return candidate
        zone = ZoneInfo(self.timezone)
        local_now = now.astimezone(zone)
        local_clock = _parse_local_time(self.daily_time)
        for offset in range(8):
            day = local_now.date() + timedelta(days=offset)
            if day.weekday() not in self.weekdays:
                continue
            candidate = datetime.combine(day, local_clock, tzinfo=zone)
            if candidate > local_now:
                return candidate.astimezone(UTC)
        raise SceneUpdateError("daily schedule could not produce a future occurrence")


@dataclass(frozen=True)
class AdoptedSceneResult:
    result_id: str
    adopted_at_utc: str
    trigger_mode: UpdateMode
    geometry_artifact: dict[str, Any]
    final_artifact: dict[str, Any]
    geometry_source: dict[str, Any]
    floor_job_id: str
    floor_output_directory: str
    initial_manual_result: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdoptedSceneResult:
        try:
            mode = UpdateMode(_required_string(value, "trigger_mode"))
        except ValueError as error:
            raise SceneUpdateError("adopted result trigger mode is unknown") from error
        return cls(
            result_id=_required_string(value, "result_id"),
            adopted_at_utc=_required_string(value, "adopted_at_utc"),
            trigger_mode=mode,
            geometry_artifact=_required_object(value, "geometry_artifact"),
            final_artifact=_required_object(value, "final_artifact"),
            geometry_source=_required_object(value, "geometry_source"),
            floor_job_id=_required_string(value, "floor_job_id"),
            floor_output_directory=_required_string(value, "floor_output_directory"),
            initial_manual_result=bool(value.get("initial_manual_result", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "adopted_at_utc": self.adopted_at_utc,
            "trigger_mode": self.trigger_mode.value,
            "geometry_artifact": dict(self.geometry_artifact),
            "final_artifact": dict(self.final_artifact),
            "geometry_source": dict(self.geometry_source),
            "floor_job_id": self.floor_job_id,
            "floor_output_directory": self.floor_output_directory,
            "initial_manual_result": self.initial_manual_result,
        }


class SceneUpdatePipeline(Protocol):
    def run(
        self,
        update_id: str,
        schedule: SceneUpdateSchedule,
        cancel_event: threading.Event,
    ) -> Mapping[str, Any]: ...


UpdateSubmitter = Callable[[str, UpdateMode], None]
BusyCheck = Callable[[], bool]


class SceneUpdateRepository:
    """Atomic scene-local state; it contains no RTSP endpoints or credentials."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def read(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SceneUpdateError("scene update state is unreadable") from error
        if not isinstance(value, dict) or value.get("schema_version") != "xr01-scene-updates-v1":
            raise SceneUpdateError("scene update state schema is unsupported")
        return value

    def write(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(self.path)


class SceneUpdateScheduler:
    """One daemon timer that skips missed/busy work and never queues catch-up runs."""

    def __init__(
        self,
        repository: SceneUpdateRepository,
        submit: UpdateSubmitter,
        is_busy: BusyCheck,
        *,
        poll_seconds: float = 5.0,
        now: Callable[[], datetime] | None = None,
        start_thread: bool = True,
    ) -> None:
        self.repository = repository
        self.submit = submit
        self.is_busy = is_busy
        self.poll_seconds = poll_seconds
        self.now = now or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._state = self._load_or_default()
        self._normalize_after_startup()
        self._thread: threading.Thread | None = None
        if start_thread:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._loop, name="xr01-scene-update-scheduler", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.poll_seconds * 2))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return cast(dict[str, Any], json.loads(json.dumps(self._state)))

    def defer_for_recording(self, update_id: str, mode: UpdateMode) -> None:
        """Collapse one or more due occurrences into one post-recording update."""

        with self._lock:
            existing = self._state.get("deferred_update")
            if isinstance(existing, dict):
                self._event(
                    "scheduled-update-collapsed-recording",
                    "Collapsed another due occurrence into the deferred scene update",
                )
            else:
                self._state["deferred_update"] = {
                    "update_id": update_id,
                    "mode": mode.value,
                    "deferred_at_utc": _iso(self.now()),
                }
                self._event(
                    "scheduled-update-deferred-recording",
                    "Scheduled scene update deferred until Replayable Recording stops",
                )
                self._warning(
                    "recording-update-deferred",
                    "A scheduled scene update was deferred because Replayable Recording is "
                    "active. It will run automatically after recording stops.",
                )
            self._persist()

    def deferred_update(self) -> tuple[str, UpdateMode] | None:
        with self._lock:
            value = self._state.get("deferred_update")
            if not isinstance(value, dict):
                return None
            return (
                _required_string(value, "update_id"),
                UpdateMode(_required_string(value, "mode")),
            )

    def clear_deferred_update(self, update_id: str) -> None:
        with self._lock:
            value = self._state.get("deferred_update")
            if isinstance(value, dict) and value.get("update_id") == update_id:
                self._state["deferred_update"] = None
                self._persist()

    def set_live_coordination(
        self,
        state: str,
        *,
        update_id: str | None = None,
        previous_session_id: str | None = None,
        resume_requested: bool = False,
        message: str | None = None,
    ) -> None:
        with self._lock:
            self._state["live_coordination"] = {
                "state": state,
                "update_id": update_id,
                "previous_session_id": previous_session_id,
                "resume_requested": resume_requested,
                "message": message,
                "updated_at_utc": _iso(self.now()),
            }
            self._persist()

    def cancel_live_resume(self) -> None:
        with self._lock:
            value = self._state.get("live_coordination")
            if not isinstance(value, dict) or value.get("resume_requested") is not True:
                raise SceneUpdateError("no automatic Live resume is pending")
            value["resume_requested"] = False
            value["message"] = "Automatic Live restart was cancelled by the operator"
            value["updated_at_utc"] = _iso(self.now())
            self._event("live-auto-resume-cancelled", value["message"])
            self._persist()

    def live_resume_requested(self, update_id: str) -> bool:
        with self._lock:
            value = self._state.get("live_coordination")
            return bool(
                isinstance(value, dict)
                and value.get("update_id") == update_id
                and value.get("resume_requested") is True
            )

    def warn_update_failed_live_resumed(self, update_id: str) -> None:
        with self._lock:
            self._event(
                "scheduled-update-failed-live-resumed",
                f"{update_id} failed; Live restarted on the previous accepted scene",
            )
            self._warning(
                "update-failed-live-resumed",
                "The scheduled scene update failed. Live Service restarted using the previous "
                "accepted scene.",
            )
            self._persist()

    def warn_live_paused_for_update(self, update_id: str) -> None:
        with self._lock:
            self._event(
                "live-paused-for-scheduled-update",
                f"Live Service paused for scheduled update {update_id}",
            )
            self._warning(
                "live-paused-for-update",
                "Live Service was stopped automatically for a scheduled scene update. It will "
                "restart automatically when the update finishes.",
            )
            self._persist()

    def warn_live_coordination_failure(self, update_id: str, message: str) -> None:
        with self._lock:
            self._event("live-update-coordination-failed", f"{update_id}: {message[:300]}")
            self._warning("live-update-coordination-failed", message)
            self._persist()

    def warn_deferred_update_start_failed(self, update_id: str) -> None:
        with self._lock:
            self._event(
                "deferred-update-start-failed",
                f"Deferred update {update_id} could not start after recording stopped",
            )
            self._warning(
                "deferred-update-start-failed",
                "Replayable Recording stopped, but the deferred scheduled scene update could "
                "not start. It remains pending for recovery.",
            )
            self._persist()

    def acknowledge_warning(self, warning_id: str) -> None:
        with self._lock:
            warning = self._state.get("operator_warning")
            if not isinstance(warning, dict) or warning.get("warning_id") != warning_id:
                return
            self._state["operator_warning"] = None
            self._event(
                "operator-warning-acknowledged",
                f"Operator acknowledged {warning.get('kind', 'live-operations-warning')}",
            )
            self._persist()

    def unlock(self, initial_result: AdoptedSceneResult) -> None:
        with self._lock:
            if self._state["unlocked_at_utc"] is not None:
                return
            self._state["unlocked_at_utc"] = initial_result.adopted_at_utc
            self._state["adopted_results"] = [initial_result.to_dict()]
            self._state["active_result_id"] = initial_result.result_id
            self._event("unlocked", "First manually approved final result unlocked scene updates")
            self._persist()

    def configure(self, schedule: SceneUpdateSchedule) -> dict[str, Any]:
        with self._lock:
            self._require_unlocked()
            self._state["schedule"] = schedule.to_dict()
            now = self.now()
            self._state["configured_at_utc"] = _iso(now)
            self._state["next_due_at_utc"] = (
                _iso(schedule.next_due_after(now, anchor_utc=now)) if schedule.enabled else None
            )
            self._event(
                "schedule-enabled" if schedule.enabled else "schedule-disabled",
                f"Scene update mode set to {schedule.mode.value}",
            )
            self._persist()
            return self.status()

    def record_candidate(self, result: AdoptedSceneResult) -> None:
        with self._lock:
            self._state["manual_candidate"] = result.to_dict()
            self._state["manual_candidate_previewed"] = False
            self._event("manual-candidate-ready", f"Manual candidate {result.result_id} is ready")
            self._persist()

    def mark_candidate_previewed(self, result_id: str) -> None:
        with self._lock:
            candidate = self._candidate(result_id)
            self._state["manual_candidate_previewed"] = True
            self._event("manual-candidate-previewed", f"Opened {candidate.result_id} for review")
            self._persist()

    def adopt_candidate(self, result_id: str) -> AdoptedSceneResult:
        with self._lock:
            candidate = self._candidate(result_id)
            if self._state["manual_candidate_previewed"] is not True:
                raise SceneUpdateError("open the manual candidate preview before adopting it")
            self._adopt(candidate)
            self._state["manual_candidate"] = None
            self._state["manual_candidate_previewed"] = False
            self._event("manual-candidate-adopted", f"Adopted manual result {result_id}")
            self._persist()
            return candidate

    def adopt_scheduled(self, result: AdoptedSceneResult) -> None:
        with self._lock:
            self._adopt(result)
            self._event("scheduled-result-adopted", f"Automatically adopted {result.result_id}")
            self._persist()

    def rollback_choices(self) -> tuple[AdoptedSceneResult, ...]:
        with self._lock:
            active = self._state["active_result_id"]
            results = [
                AdoptedSceneResult.from_dict(value)
                for value in self._state["adopted_results"]
                if value["result_id"] != active
            ]
            initial = [result for result in results if result.initial_manual_result]
            recent = [result for result in results if not result.initial_manual_result][-3:]
            ordered = initial + [item for item in recent if item not in initial]
            return tuple(reversed(ordered))

    def rollback(self, result_id: str) -> AdoptedSceneResult:
        with self._lock:
            choices = {result.result_id: result for result in self.rollback_choices()}
            try:
                selected = choices[result_id]
            except KeyError as error:
                raise SceneUpdateError("result is not in the bounded rollback choices") from error
            self._state["active_result_id"] = result_id
            self._event("result-rollback", f"Restored accepted result {result_id}")
            self._persist()
            return selected

    def record_failure(self, update_id: str, message: str) -> None:
        with self._lock:
            self._event("update-failed", f"{update_id}: {message[:300]}")
            self._persist()

    def tick(self) -> None:
        with self._lock:
            schedule = SceneUpdateSchedule.from_dict(self._state["schedule"])
            raw_due = self._state["next_due_at_utc"]
            if not schedule.enabled or raw_due is None:
                return
            now = _aware_utc(self.now())
            due = datetime.fromisoformat(raw_due)
            if now < due:
                return
            self._state["last_due_at_utc"] = _iso(due)
            self._state["next_due_at_utc"] = _iso(schedule.next_due_after(now, anchor_utc=due))
            if self.is_busy():
                self._event("scheduled-run-skipped-busy", f"Skipped occurrence due at {_iso(due)}")
                self._persist()
                return
            update_id = f"auto-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
            self._event("scheduled-run-started", f"Started {update_id}")
            self._persist()
        try:
            self.submit(update_id, schedule.mode)
        except Exception as error:
            with self._lock:
                self._event("scheduled-run-rejected", str(error)[:300])
                self._persist()

    def _normalize_after_startup(self) -> None:
        with self._lock:
            self._state.setdefault("deferred_update", None)
            self._state.setdefault(
                "live_coordination",
                {
                    "state": "idle",
                    "update_id": None,
                    "previous_session_id": None,
                    "resume_requested": False,
                    "message": None,
                    "updated_at_utc": None,
                },
            )
            self._state.setdefault("operator_warning", None)
            schedule = SceneUpdateSchedule.from_dict(self._state["schedule"])
            raw_due = self._state.get("next_due_at_utc")
            now = _aware_utc(self.now())
            if schedule.enabled and raw_due is not None and datetime.fromisoformat(raw_due) <= now:
                self._event("scheduled-run-missed-offline", f"Missed occurrence due at {raw_due}")
                self._state["next_due_at_utc"] = _iso(
                    schedule.next_due_after(now, anchor_utc=datetime.fromisoformat(raw_due))
                )
                self._persist()

    def _load_or_default(self) -> dict[str, Any]:
        state = self.repository.read()
        if state is not None:
            SceneUpdateSchedule.from_dict(_required_object(state, "schedule"))
            return state
        return {
            "schema_version": "xr01-scene-updates-v1",
            "unlocked_at_utc": None,
            "configured_at_utc": None,
            "schedule": SceneUpdateSchedule(
                enabled=False, mode=UpdateMode.MANUAL, timezone="Asia/Singapore"
            ).to_dict(),
            "next_due_at_utc": None,
            "last_due_at_utc": None,
            "active_result_id": None,
            "adopted_results": [],
            "manual_candidate": None,
            "manual_candidate_previewed": False,
            "deferred_update": None,
            "live_coordination": {
                "state": "idle",
                "update_id": None,
                "previous_session_id": None,
                "resume_requested": False,
                "message": None,
                "updated_at_utc": None,
            },
            "operator_warning": None,
            "events": [],
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.tick()
            except Exception as error:
                with self._lock:
                    self._event("scheduler-error", str(error)[:300])
                    self._persist()

    def _candidate(self, result_id: str) -> AdoptedSceneResult:
        value = self._state["manual_candidate"]
        if not isinstance(value, dict):
            raise SceneUpdateError("no manual scene-update candidate is ready")
        candidate = AdoptedSceneResult.from_dict(value)
        if candidate.result_id != result_id:
            raise SceneUpdateError("manual candidate identity changed")
        return candidate

    def _adopt(self, result: AdoptedSceneResult) -> None:
        values = self._state["adopted_results"]
        values[:] = [value for value in values if value["result_id"] != result.result_id]
        values.append(result.to_dict())
        self._state["active_result_id"] = result.result_id

    def _event(self, kind: str, message: str) -> None:
        events = self._state["events"]
        events.append({"at_utc": _iso(self.now()), "kind": kind, "message": message})
        del events[:-100]

    def _warning(self, kind: str, message: str) -> None:
        self._state["operator_warning"] = {
            "warning_id": f"{kind}-{uuid.uuid4().hex}",
            "kind": kind,
            "message": message,
            "at_utc": _iso(self.now()),
        }

    def _persist(self) -> None:
        self.repository.write(self._state)

    def _require_unlocked(self) -> None:
        if self._state["unlocked_at_utc"] is None:
            raise SceneUpdateError("scene updates unlock after the first manual final approval")


def _parse_local_time(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as error:
        raise SceneUpdateError("daily time must use HH:MM local time") from error
    if parsed.second or parsed.microsecond or parsed.tzinfo is not None:
        raise SceneUpdateError("daily time must use HH:MM local time")
    return parsed


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise SceneUpdateError("scheduler timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _aware_utc(value).isoformat()


def _required_string(value: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    raw = value.get(key, default)
    if not isinstance(raw, str) or not raw.strip():
        raise SceneUpdateError(f"{key} must be a non-blank string")
    return raw.strip()


def _required_int(value: Mapping[str, Any], key: str, default: int) -> int:
    raw = value.get(key, default)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise SceneUpdateError(f"{key} must be an integer")
    return raw


def _required_object(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = value.get(key)
    if not isinstance(raw, dict):
        raise SceneUpdateError(f"{key} must be an object")
    return dict(raw)
