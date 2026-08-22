from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

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


def test_bounded_jobs_report_success_redacted_failure_and_cancellation() -> None:
    manager = BoundedJobManager(maximum_workers=1, maximum_outstanding_jobs=2)
    started = threading.Event()
    try:
        manager.submit("success", "P08", "test", lambda _cancel: {"ok": True})
        assert _wait_for_terminal(manager, "success")["result"] == {"ok": True}

        def fail(_cancel: threading.Event) -> dict[str, Any]:
            raise RuntimeError("rtsp://user:password@example/live token=private")

        manager.submit("failure", "P08", "test", fail)
        failed = _wait_for_terminal(manager, "failure")
        assert failed["state"] == "failed"
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
