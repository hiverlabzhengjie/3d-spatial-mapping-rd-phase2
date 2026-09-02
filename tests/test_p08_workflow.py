from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from spatial_mapping_phase2.p08_scene_updates import UpdateMode
from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    ArtifactReference,
    BoundedJobManager,
    CameraConfig,
    JobState,
    P08WorkflowError,
    PhaseRecord,
    PhaseState,
    SafeRerunLauncher,
    SceneWorkspace,
    SceneWorkspaceRepository,
    WorkflowService,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene(tmp_path: Path, artifact_path: Path | None = None) -> SceneWorkspace:
    root = tmp_path / "artifacts"
    root.mkdir(exist_ok=True)
    rerun = artifact_path or root / "selected.rrd"
    if artifact_path is None:
        rerun.write_bytes(b"selected-rerun")
    states = {
        "P02": PhaseState.PROVISIONAL,
        "P03": PhaseState.COMPLETE,
        "P04": PhaseState.PROVISIONAL,
        "P05": PhaseState.REJECTED,
        "P06": PhaseState.PROVISIONAL,
        "P07": PhaseState.COMPLETE,
        "P08": PhaseState.READY,
    }
    phases = tuple(
        PhaseRecord(
            phase_id=phase_id,
            state=states[phase_id],
            message=f"{phase_id} configured status",
            prerequisites=() if index == 0 else (PHASE_ORDER[index - 1],),
            artifact_ids=("selected-rerun",) if phase_id == "P07" else (),
        )
        for index, phase_id in enumerate(PHASE_ORDER)
    )
    return SceneWorkspace(
        project_id="test-project",
        scene_id="variable-roster-scene",
        display_name="Variable roster test",
        artifact_root=root.resolve(),
        cameras=(
            CameraConfig("north-camera", "North", "RTSP_NORTH"),
            CameraConfig("south-camera", "South", "RTSP_SOUTH"),
        ),
        phases=phases,
        artifacts=(
            ArtifactReference(
                artifact_id="selected-rerun",
                phase_id="P07",
                kind="rerun-recording",
                path=rerun.resolve(),
                sha256=_sha256(rerun),
                authority="working-usability-visualization",
                selected=True,
            ),
        ),
    )


def _wait_for_terminal(manager: BoundedJobManager, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = manager.status(job_id)
        if status["state"] in {
            JobState.COMPLETE.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        }:
            return status
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_scene_round_trip_supports_variable_camera_roster_and_phase_states(
    tmp_path: Path,
) -> None:
    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    scene = _scene(tmp_path)
    repository.create(scene)
    loaded = repository.load()
    assert [camera.camera_id for camera in loaded.cameras] == [
        "north-camera",
        "south-camera",
    ]
    assert loaded.to_dict() == scene.to_dict()
    with pytest.raises(P08WorkflowError, match="already exists"):
        repository.create(scene)


def test_status_detects_stale_artifact_and_blocks_downstream_phase(tmp_path: Path) -> None:
    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    scene = _scene(tmp_path)
    repository.create(scene)
    jobs = BoundedJobManager()
    try:
        service = WorkflowService(repository, jobs)
        status = service.status()
        assert len(status["camera_roster"]) == 2
        phase_by_id = {item["phase_id"]: item for item in status["phases"]}
        assert phase_by_id["P07"]["state"] == "complete"
        assert phase_by_id["P08"]["state"] == "ready"
        scene.artifacts[0].path.write_bytes(b"changed")
        stale = service.status()
        stale_by_id = {item["phase_id"]: item for item in stale["phases"]}
        assert stale_by_id["P07"]["state"] == "stale"
        assert stale_by_id["P08"]["state"] == "unavailable"
    finally:
        jobs.close()


def test_unprovisioned_floor_remains_blocked_after_geometry_approval(tmp_path: Path) -> None:
    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    repository.create(_scene(tmp_path))
    service = WorkflowService(repository, BoundedJobManager())
    try:
        service._geometry_approved = True

        operator = service.operator_status()
        steps = {step["step_id"]: step["state"] for step in operator["steps"]}

        assert steps["floor"] == "blocked"
        assert operator["workflow_capabilities"]["floor"]["state"] == "not_provisioned"
    finally:
        service.close()


def test_camera_policy_history_is_scene_scoped_without_mutating_scene_json(
    tmp_path: Path,
) -> None:
    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    repository.create(_scene(tmp_path))
    scene_before = repository.scene_path.read_bytes()
    service = WorkflowService(repository, BoundedJobManager())
    try:
        payload = {
            "intrinsic_groups": [
                {
                    "group_id": "same-lens",
                    "lens_model": "Model A",
                    "camera_ids": ["north-camera", "south-camera"],
                }
            ],
            "overlap_pair_reviews": [
                {
                    "camera_id_a": "north-camera",
                    "camera_id_b": "south-camera",
                    "verdict": "no_overlap",
                }
            ],
        }
        result = service.apply_camera_policy(
            "camera-policy-first",
            payload,
            expected_revision=None,
            confirm_impacts=False,
        )

        assert result["revision"] == 1
        assert repository.camera_policy_path.is_file()
        assert repository.scene_path.read_bytes() == scene_before
        history = service.artifact_catalog_status()["camera_policy"]
        assert history["active_revision"] == 1
        assert history["active_policy"]["lens_complete"] is True
        assert (repository.runs_directory / "camera-policy-first.json").is_file()
    finally:
        service.close()


def test_operator_status_deduplicates_shared_camera_policy_issue(tmp_path: Path) -> None:
    class Calibration:
        def status(self, _policy: object) -> dict[str, object]:
            raise AssertionError("an incomplete camera policy must block the adapter call")

    class Reconstruction:
        supports_scene_camera_policy = True

        def readiness_errors(self) -> tuple[str, ...]:
            return ()

    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    repository.create(_scene(tmp_path))
    service = WorkflowService(repository, BoundedJobManager())
    service.calibration_adapter = Calibration()
    service.reconstruction_adapter = Reconstruction()
    try:
        issues = service.operator_status()["input_issues"]

        assert issues == ["no camera policy revision is active"]
    finally:
        service.close()


class _FakeLiveOperations:
    def __init__(self, mode: str) -> None:
        self.active_mode: str | None = mode
        self.active_session_id: str | None = "xr02-original"
        self.stop_reasons: list[str] = []
        self.resume_links: list[tuple[str | None, str | None]] = []

    def status(self) -> dict[str, Any]:
        return {
            "active": self.active_mode is not None,
            "active_mode": self.active_mode,
            "active_session_id": self.active_session_id,
            "operator_state": "ready"
            if self.active_mode is None
            else f"{self.active_mode}_running",
            "pending_run": None,
            "saved_recordings": [],
            "recent_live_runs": [],
        }

    def stop(self, *, reason: str = "operator") -> dict[str, Any]:
        self.stop_reasons.append(reason)
        self.active_mode = None
        self.active_session_id = None
        return self.status()

    def start_live(
        self,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, Any]:
        self.resume_links.append((resumed_from_session_id, scene_update_id))
        self.active_mode = "live"
        self.active_session_id = "xr02-resumed"
        return self.status()


def _coordination_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    *,
    fail_job: bool = False,
) -> tuple[WorkflowService, _FakeLiveOperations]:
    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    repository.create(_scene(tmp_path))
    service = WorkflowService(repository, BoundedJobManager())
    live = _FakeLiveOperations(mode)
    service.enable_live_operations(live)

    def start_update(update_id: str, _mode: Any) -> dict[str, Any]:
        def operation(_cancel: threading.Event) -> dict[str, Any]:
            if fail_job:
                raise RuntimeError("fixture update failed")
            return {"complete_chain": True}

        return service.jobs.submit(update_id, "P08", "full-scene-update", operation)

    monkeypatch.setattr(service, "start_scene_update", start_update)
    return service, live


def _wait_for_live_resume(
    live: _FakeLiveOperations,
    *,
    service: WorkflowService | None = None,
    expected_warning: str | None = None,
) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if live.resume_links:
            if expected_warning is None:
                return
            if service is None:
                raise AssertionError("service is required when waiting for an operator warning")
            warning = service.scene_update_status().get("operator_warning")
            if isinstance(warning, dict) and warning.get("kind") == expected_warning:
                return
        time.sleep(0.01)
    raise AssertionError("Live Service did not reach the expected resumed state")


def test_scheduled_update_pauses_and_resumes_live_with_popup_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, live = _coordination_service(tmp_path, monkeypatch, "live")
    try:
        service._submit_scheduled_scene_update("auto-success", UpdateMode.INTERVAL)
        _wait_for_live_resume(live)

        state = service.scene_update_status()
        assert live.stop_reasons == ["scheduled_scene_update"]
        assert live.resume_links == [("xr02-original", "auto-success")]
        assert state["live_coordination"]["state"] == "live_resumed"
        assert state["operator_warning"]["kind"] == "live-paused-for-update"
    finally:
        service.close()


def test_failed_scheduled_update_resumes_previous_scene_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, live = _coordination_service(tmp_path, monkeypatch, "live", fail_job=True)
    try:
        service._submit_scheduled_scene_update("auto-failed", UpdateMode.INTERVAL)
        _wait_for_live_resume(
            live,
            service=service,
            expected_warning="update-failed-live-resumed",
        )

        state = service.scene_update_status()
        assert live.resume_links == [("xr02-original", "auto-failed")]
        assert state["live_coordination"]["message"].endswith("previous accepted scene")
        assert state["operator_warning"]["kind"] == "update-failed-live-resumed"
    finally:
        service.close()


def test_recording_defers_one_update_then_runs_it_after_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, live = _coordination_service(tmp_path, monkeypatch, "recording")
    try:
        service._submit_scheduled_scene_update("auto-deferred", UpdateMode.INTERVAL)
        deferred = service.scene_update_status()
        assert deferred["deferred_update"]["update_id"] == "auto-deferred"
        assert deferred["operator_warning"]["kind"] == "recording-update-deferred"
        assert service.jobs.list() == ()

        service.stop_live_operations()
        completed = _wait_for_terminal(service.jobs, "auto-deferred")
        assert completed["state"] == JobState.COMPLETE.value
        assert service.scene_update_status()["deferred_update"] is None
        assert live.stop_reasons == ["operator"]
    finally:
        service.close()


def test_bounded_jobs_report_success_redacted_failure_and_cancellation() -> None:
    manager = BoundedJobManager(maximum_workers=1, maximum_outstanding_jobs=2)
    started = threading.Event()
    try:
        submitted = manager.submit("success", "P08", "test", lambda _cancel: {"ok": True})
        assert submitted["submitted_at_utc"].endswith("+00:00")
        completed = _wait_for_terminal(manager, "success")
        assert completed["result"] == {"ok": True}
        assert completed["completed_at_utc"].endswith("+00:00")

        def fail(_cancel: threading.Event) -> dict[str, Any]:
            raise RuntimeError("rtsp://user:password@example/live token=private")

        manager.submit("failure", "P08", "test", fail)
        failed = _wait_for_terminal(manager, "failure")
        assert failed["state"] == "failed"
        assert failed["completed_at_utc"].endswith("+00:00")
        assert "password@example" not in failed["error_message"]
        assert "token=<redacted>" in failed["error_message"]

        def wait_for_cancel(cancel: threading.Event) -> dict[str, Any]:
            started.set()
            cancel.wait(1.0)
            raise RuntimeError("cancelled")

        manager.submit("cancel", "P08", "test", wait_for_cancel)
        assert started.wait(1.0)
        manager.cancel("cancel")
        assert _wait_for_terminal(manager, "cancel")["state"] == "cancelled"
    finally:
        manager.close()


def test_bounded_jobs_reject_duplicate_unknown_and_excess_capacity() -> None:
    manager = BoundedJobManager(maximum_workers=1, maximum_outstanding_jobs=1)
    release = threading.Event()
    try:
        manager.submit("one", "P08", "wait", lambda _cancel: {"released": release.wait(1.0)})
        with pytest.raises(P08WorkflowError, match="already exists"):
            manager.submit("one", "P08", "wait", lambda _cancel: {})
        with pytest.raises(P08WorkflowError, match="capacity"):
            manager.submit("two", "P08", "wait", lambda _cancel: {})
        with pytest.raises(P08WorkflowError, match="unknown"):
            manager.status("missing")
        with pytest.raises(P08WorkflowError, match="active"):
            manager.clear_terminal()
        release.set()
        _wait_for_terminal(manager, "one")
        assert [job["job_id"] for job in manager.clear_terminal()] == ["one"]
        assert manager.list() == ()
    finally:
        manager.close()


def test_safe_rerun_launch_uses_only_fixed_viewer_and_selected_allowed_artifact(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    viewer = tmp_path / "rerun-viewer.exe"
    viewer.write_bytes(b"viewer")
    calls: list[tuple[str, ...]] = []

    class Process:
        pid = 4321

    def launch(arguments: object) -> Process:
        assert isinstance(arguments, tuple)
        calls.append(arguments)
        return Process()

    launcher = SafeRerunLauncher(viewer, (scene.artifact_root,), launch)
    result = launcher.launch_selected(scene, "selected-rerun")
    assert calls == [
        (
            str(viewer.resolve()),
            "--port",
            "0",
            str(scene.artifacts[0].path.resolve()),
        )
    ]
    assert result["process_id"] == 4321
    assert result["status"] == "launched"

    unselected = replace(
        scene,
        artifacts=(replace(scene.artifacts[0], selected=False),),
    )
    with pytest.raises(P08WorkflowError, match="selected"):
        launcher.launch_selected(unselected, "selected-rerun")


def test_safe_rerun_launch_rejects_outside_root_wrong_suffix_and_stale_hash(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    viewer = tmp_path / "rerun-viewer.exe"
    viewer.write_bytes(b"viewer")
    launcher = SafeRerunLauncher(viewer, (scene.artifact_root,), lambda _arguments: object())
    outside = tmp_path / "outside.rrd"
    outside.write_bytes(b"outside")
    outside_scene = replace(
        scene,
        artifacts=(replace(scene.artifacts[0], path=outside, sha256=_sha256(outside)),),
    )
    with pytest.raises(P08WorkflowError, match="outside"):
        launcher.launch_selected(outside_scene, "selected-rerun")
    wrong_suffix = scene.artifacts[0].path.with_suffix(".txt")
    wrong_suffix.write_bytes(b"wrong")
    wrong_scene = replace(
        scene,
        artifacts=(replace(scene.artifacts[0], path=wrong_suffix, sha256=_sha256(wrong_suffix)),),
    )
    with pytest.raises(P08WorkflowError, match=r"\.rrd"):
        launcher.launch_selected(wrong_scene, "selected-rerun")
    scene.artifacts[0].path.write_bytes(b"stale")
    with pytest.raises(P08WorkflowError, match="stale"):
        launcher.launch_selected(scene, "selected-rerun")


def test_safe_rerun_launcher_resolves_windows_environment_shim(tmp_path: Path) -> None:
    environment = tmp_path / "runtime"
    wrapper = environment / "Scripts" / "rerun.exe"
    native = environment / "Lib" / "site-packages" / "rerun_sdk" / "rerun_cli" / "rerun.exe"
    wrapper.parent.mkdir(parents=True)
    native.parent.mkdir(parents=True)
    wrapper.write_bytes(b"shim")
    native.write_bytes(b"native")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    launcher = SafeRerunLauncher(wrapper, (artifact_root,))

    assert launcher.viewer_executable == native.resolve()


def test_workflow_launch_writes_immutable_action_manifest_before_duplicate(
    tmp_path: Path,
) -> None:
    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    scene = _scene(tmp_path)
    repository.create(scene)
    viewer = tmp_path / "viewer.exe"
    viewer.write_bytes(b"viewer")
    launches: list[object] = []
    launcher = SafeRerunLauncher(
        viewer,
        (scene.artifact_root,),
        lambda arguments: launches.append(arguments),
    )
    jobs = BoundedJobManager()
    try:
        service = WorkflowService(repository, jobs, rerun_launcher=launcher)
        result = service.launch_rerun("launch-one", "selected-rerun")
        assert result["workflow_run_manifest"]["sha256"]
        with pytest.raises(P08WorkflowError, match="run_id"):
            service.launch_rerun("launch-one", "selected-rerun")
        assert len(launches) == 1
    finally:
        jobs.close()
