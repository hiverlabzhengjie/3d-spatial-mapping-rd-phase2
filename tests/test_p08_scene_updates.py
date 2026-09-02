from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from spatial_mapping_phase2.p08_scene_updates import (
    AdoptedSceneResult,
    SceneUpdateError,
    SceneUpdateRepository,
    SceneUpdateSchedule,
    SceneUpdateScheduler,
    UpdateMode,
)


def _result(result_id: str, *, initial: bool = False) -> AdoptedSceneResult:
    identity = {"path": f"C:\\runs\\{result_id}.rrd", "sha256": "a" * 64}
    return AdoptedSceneResult(
        result_id=result_id,
        adopted_at_utc="2026-08-22T01:00:00+00:00",
        trigger_mode=UpdateMode.MANUAL if initial else UpdateMode.INTERVAL,
        geometry_artifact={**identity, "artifact_id": f"{result_id}-geometry"},
        final_artifact={**identity, "artifact_id": f"{result_id}-final"},
        geometry_source={"path": f"C:\\runs\\{result_id}.npz", "sha256": "b" * 64},
        floor_job_id=f"{result_id}-floor",
        floor_output_directory=f"C:\\runs\\{result_id}-floor",
        initial_manual_result=initial,
    )


def test_daily_schedule_uses_selected_local_weekdays() -> None:
    schedule = SceneUpdateSchedule(
        enabled=True,
        mode=UpdateMode.DAILY,
        timezone="Asia/Singapore",
        daily_time="09:00",
        weekdays=(0, 2, 4),
    )
    # Friday 10:00 Singapore -> next selected day is Monday 09:00 Singapore.
    now = datetime(2026, 8, 21, 2, 0, tzinfo=UTC)
    assert schedule.next_due_after(now) == datetime(2026, 8, 24, 1, 0, tzinfo=UTC)


def test_interval_schedule_skips_missed_occurrences_without_catchup() -> None:
    schedule = SceneUpdateSchedule(
        enabled=True,
        mode=UpdateMode.INTERVAL,
        timezone="Asia/Singapore",
        interval_seconds=1800,
    )
    anchor = datetime(2026, 8, 22, 0, 0, tzinfo=UTC)
    now = anchor + timedelta(hours=2, minutes=7)
    assert schedule.next_due_after(now, anchor_utc=anchor) == anchor + timedelta(
        hours=2, minutes=30
    )


def test_schedule_validation_rejects_unsafe_or_ambiguous_values() -> None:
    with pytest.raises(SceneUpdateError, match="background"):
        SceneUpdateSchedule(True, UpdateMode.MANUAL, "Asia/Singapore")
    with pytest.raises(SceneUpdateError, match="odd"):
        SceneUpdateSchedule(True, UpdateMode.INTERVAL, "Asia/Singapore", median_frame_count=4)
    with pytest.raises(SceneUpdateError, match="selected weekdays"):
        SceneUpdateSchedule(True, UpdateMode.DAILY, "Asia/Singapore", weekdays=())


def test_scheduler_unlock_persists_and_busy_occurrence_is_skipped(tmp_path: Path) -> None:
    clock = [datetime(2026, 8, 22, 0, 0, tzinfo=UTC)]
    submitted: list[tuple[str, UpdateMode]] = []
    scheduler = SceneUpdateScheduler(
        SceneUpdateRepository(tmp_path / "scene-updates.json"),
        lambda update_id, mode: submitted.append((update_id, mode)),
        lambda: True,
        now=lambda: clock[0],
        start_thread=False,
    )
    scheduler.unlock(_result("initial", initial=True))
    scheduler.configure(
        SceneUpdateSchedule(
            True,
            UpdateMode.INTERVAL,
            "Asia/Singapore",
            interval_seconds=1800,
        )
    )
    clock[0] += timedelta(minutes=31)
    scheduler.tick()

    status = scheduler.status()
    assert submitted == []
    assert status["unlocked_at_utc"] is not None
    assert status["events"][-1]["kind"] == "scheduled-run-skipped-busy"
    assert datetime.fromisoformat(status["next_due_at_utc"]) > clock[0]

    restored = SceneUpdateScheduler(
        SceneUpdateRepository(tmp_path / "scene-updates.json"),
        lambda _update_id, _mode: None,
        lambda: False,
        now=lambda: clock[0],
        start_thread=False,
    )
    assert restored.status()["unlocked_at_utc"] == status["unlocked_at_utc"]


def test_manual_candidate_requires_preview_and_rollback_is_bounded(tmp_path: Path) -> None:
    scheduler = SceneUpdateScheduler(
        SceneUpdateRepository(tmp_path / "scene-updates.json"),
        lambda _update_id, _mode: None,
        lambda: False,
        start_thread=False,
    )
    scheduler.unlock(_result("initial", initial=True))
    scheduler.record_candidate(_result("manual-1"))
    with pytest.raises(SceneUpdateError, match="open"):
        scheduler.adopt_candidate("manual-1")
    scheduler.mark_candidate_previewed("manual-1")
    scheduler.adopt_candidate("manual-1")
    for result_id in ("auto-1", "auto-2", "auto-3", "auto-4"):
        scheduler.adopt_scheduled(_result(result_id))

    choices = scheduler.rollback_choices()
    assert {result.result_id for result in choices} == {
        "initial",
        "auto-1",
        "auto-2",
        "auto-3",
    }
    assert len(choices) == 4
    scheduler.rollback("initial")
    assert scheduler.status()["active_result_id"] == "initial"


def test_recording_deferral_collapses_due_occurrences_and_warns_once(tmp_path: Path) -> None:
    scheduler = SceneUpdateScheduler(
        SceneUpdateRepository(tmp_path / "scene-updates.json"),
        lambda _update_id, _mode: None,
        lambda: False,
        start_thread=False,
    )

    scheduler.defer_for_recording("auto-first", UpdateMode.INTERVAL)
    first = scheduler.status()
    scheduler.defer_for_recording("auto-second", UpdateMode.INTERVAL)
    collapsed = scheduler.status()

    assert scheduler.deferred_update() == ("auto-first", UpdateMode.INTERVAL)
    assert collapsed["operator_warning"]["warning_id"] == first["operator_warning"]["warning_id"]
    assert collapsed["events"][-1]["kind"] == "scheduled-update-collapsed-recording"


def test_live_coordination_warning_and_resume_cancel_are_persistent(tmp_path: Path) -> None:
    path = tmp_path / "scene-updates.json"
    scheduler = SceneUpdateScheduler(
        SceneUpdateRepository(path),
        lambda _update_id, _mode: None,
        lambda: False,
        start_thread=False,
    )
    scheduler.set_live_coordination(
        "scene_update_running",
        update_id="auto-1",
        previous_session_id="xr02-old",
        resume_requested=True,
    )
    scheduler.warn_live_paused_for_update("auto-1")
    scheduler.cancel_live_resume()

    restored = SceneUpdateScheduler(
        SceneUpdateRepository(path),
        lambda _update_id, _mode: None,
        lambda: False,
        start_thread=False,
    )
    state = restored.status()
    assert state["operator_warning"]["kind"] == "live-paused-for-update"
    assert state["live_coordination"]["resume_requested"] is False
    assert state["events"][-1]["kind"] == "live-auto-resume-cancelled"
    restored.acknowledge_warning(state["operator_warning"]["warning_id"])
    assert restored.status()["operator_warning"] is None
