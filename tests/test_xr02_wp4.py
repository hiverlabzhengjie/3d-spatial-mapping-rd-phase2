from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import spatial_mapping_phase2.xr02_live_service as live_service_module
import spatial_mapping_phase2.xr02_trial_recording as trial_recording_module
import spatial_mapping_phase2.xr02_wp4 as wp4_module
from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p09_live_runtime import (
    CapturedFrame,
    InferenceAdmissionTelemetry,
    LatestFrameSnapshot,
)
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS, LiveFrameIdentity
from spatial_mapping_phase2.xr02_global_domain import (
    AssociationTickResult,
    GlobalTrackSnapshot,
    GlobalTrackState,
    MemberAssignment,
)
from spatial_mapping_phase2.xr02_live_domain import (
    AdoptedSceneSelection,
    XR02LiveContractError,
    resolve_adopted_scene,
)
from spatial_mapping_phase2.xr02_live_pipeline import (
    LiveAssociationTick,
    LiveModelProfile,
    XR02LivePipeline,
)
from spatial_mapping_phase2.xr02_live_rerun import (
    XR02LiveRerunLogger,
    _active_tracks_markdown,
    _camera_feed_status,
    _camera_observation_presentation,
    _identity_color,
    _trail_break_reason,
    _TrailState,
)
from spatial_mapping_phase2.xr02_live_service import (
    XR02LiveService,
    XR02LiveServiceConfig,
)
from spatial_mapping_phase2.xr02_operator_web import XR02OperatorServer
from spatial_mapping_phase2.xr02_rtsp_capture import (
    CaptureBackend,
    CaptureProcessState,
    SupervisedDecoderTelemetry,
)
from spatial_mapping_phase2.xr02_trial_recording import (
    MediaMtxTrialRecorder,
    TrialRecordingError,
    TrialRecordingPolicy,
    _record_command,
    _validate_credential_free_local_endpoint,
)
from spatial_mapping_phase2.xr02_wp4 import (
    WP4_REQUIRED_RUNTIME,
    _environment,
    _write_json,
    apply_offline_model_controls,
    validate_wp4_runtime,
)
from spatial_mapping_phase2.xr02_wp4_verification import _verify_trial_recording
from spatial_mapping_phase2.xr03_live_operations import XR02WorkerClient


def test_scene_resolution_binds_exact_approved_inputs(tmp_path: Path) -> None:
    selection = _scene_fixture(tmp_path)
    assert selection.scene.scene_id == "office"
    assert selection.geometry.sha256 == _sha256(tmp_path / "geometry.npz")
    assert selection.floor.sha256 == _sha256(tmp_path / "floor" / "authoritative_floor_plane.npz")
    assert len(selection.selection_signature_sha256) == 64


def test_scene_resolution_rejects_unapproved_or_changed_geometry(tmp_path: Path) -> None:
    selection = _scene_fixture(tmp_path)
    operator_path = Path(selection.operator_state.path)
    operator = json.loads(operator_path.read_text(encoding="utf-8"))
    operator["geometry_approved"] = False
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    with pytest.raises(XR02LiveContractError, match="operator-approved"):
        resolve_adopted_scene(
            operator_path,
            Path(selection.p06_calibration.path),
            Path(selection.p07_pose.path),
            Path(selection.p08_floor_manifest.path),
        )
    operator["geometry_approved"] = True
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    Path(selection.geometry.path).write_bytes(b"changed")
    with pytest.raises(XR02LiveContractError, match="identity changed"):
        resolve_adopted_scene(
            operator_path,
            Path(selection.p06_calibration.path),
            Path(selection.p07_pose.path),
            Path(selection.p08_floor_manifest.path),
        )


def test_environment_and_json_evidence_never_emit_rtsp_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("KEY=value=with=equals\n# ignored\n", encoding="utf-8")
    assert _environment(env) == {"KEY": "value=with=equals"}
    with pytest.raises(RuntimeError, match="credential-safe"):
        _write_json(tmp_path / "unsafe.json", {"source": "rtsp://secret@example.test"})
    assert not (tmp_path / "unsafe.json").exists()


def test_wp4_runtime_preflight_rejects_wrong_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("spatial_mapping_phase2.xr02_wp4.sys.version_info", (3, 10, 0))
    with pytest.raises(RuntimeError, match="Python 3.11 is required"):
        validate_wp4_runtime()


def test_offline_model_controls_override_unsafe_inherited_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("YOLO_OFFLINE", "0")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "wrong"))

    apply_offline_model_controls(tmp_path / "ultralytics")

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["YOLO_OFFLINE"] == "1"
    assert os.environ["YOLO_CONFIG_DIR"] == str((tmp_path / "ultralytics").resolve())


def test_wp4_runtime_preflight_rejects_missing_boxmot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "boxmot":
            raise PackageNotFoundError(distribution)
        return dict((name, expected) for name, _module, expected in WP4_REQUIRED_RUNTIME)[
            distribution
        ]

    monkeypatch.setattr(wp4_module, "version", fake_version)
    monkeypatch.setattr(
        wp4_module,
        "import_module",
        lambda module: (
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            if module == "torch"
            else SimpleNamespace()
        ),
    )
    with pytest.raises(RuntimeError, match="boxmot is not installed"):
        validate_wp4_runtime()


def test_operator_console_success_and_failure_routes() -> None:
    controller = _FakeController()
    server = XR02OperatorServer(controller, port=0)
    server.start(open_browser=False)
    try:
        with urllib.request.urlopen(server.url + "api/status") as response:
            status = json.loads(response.read())
        assert status["service"]["state"] == "stopped"
        request = urllib.request.Request(server.url + "api/start", method="POST")
        with urllib.request.urlopen(request) as response:
            started = json.loads(response.read())
        assert started["service"]["state"] == "running"
        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(urllib.request.Request(server.url + "api/start", method="POST"))
        assert captured.value.code == 409
        urllib.request.urlopen(urllib.request.Request(server.url + "api/stop", method="POST"))
        save = urllib.request.Request(
            server.url + "api/save-recording",
            data=json.dumps({"session_id": "fixture", "label": "Shift 1"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(save) as response:
            assert json.loads(response.read())["operator_state"] == "ready"
        with pytest.raises(urllib.error.HTTPError) as missing_body:
            urllib.request.urlopen(
                urllib.request.Request(server.url + "api/save-recording", method="POST")
            )
        assert missing_body.value.code == 409
        page = urllib.request.urlopen(server.url).read().decode("utf-8")
        assert "Open Rerun 3D" in page
        assert "Start Live Service" in page
        assert "Start Replayable Recording" in page
        assert "compact 1 Hz count/track/XY history" in page
        assert "Diagnostics &amp; engineering controls" in page
        assert "?'\nNew scene" not in page
        assert "?'\\nNew scene" in page
        assert "rtsp://" not in page.lower()
    finally:
        server.close()


def test_api_only_worker_requires_token_and_does_not_serve_operator_page() -> None:
    controller = _FakeController()
    server = XR02OperatorServer(
        controller,
        port=0,
        api_token="fixture-token",
        serve_page=False,
    )
    server.start(open_browser=False)
    try:
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(server.url + "api/status")
        assert unauthorized.value.code == 401

        request = urllib.request.Request(
            server.url + "api/status",
            headers={"X-XR02-Worker-Token": "fixture-token"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["operator_state"] == "ready"

        with pytest.raises(urllib.error.HTTPError) as page:
            urllib.request.urlopen(server.url)
        assert page.value.code == 404
    finally:
        server.close()


def test_integrated_worker_client_uses_authenticated_control_routes() -> None:
    controller = _FakeController()
    server = XR02OperatorServer(
        controller,
        port=0,
        api_token="fixture-token",
        serve_page=False,
    )
    server.start(open_browser=False)
    client = XR02WorkerClient(server.url, "fixture-token")
    try:
        assert client.status()["operator_state"] == "ready"
        assert (
            client.start_live(
                resumed_from_session_id="xr02-previous",
                scene_update_id="auto-1",
            )["active"]
            is True
        )
        assert client.stop(reason="scheduled_scene_update")["operator_state"] == "ready"
    finally:
        server.close()


def test_live_service_survives_one_missing_camera_and_has_zero_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _scene_fixture(tmp_path)
    monkeypatch.setattr(live_service_module, "SupervisedRtspDecoder", _FakeDecoder)
    monkeypatch.setattr(live_service_module, "LatestPendingWorker", _SynchronousWorker)
    pipeline = _FakePipeline(selection)
    logger = _FakeLogger()
    endpoints = tuple(
        LocalRtspEndpoint(camera_id, CAMERA_ENDPOINT_KEYS[camera_id], "rtsp://127.0.0.1/test")
        for camera_id in CAMERA_IDS
    )
    service = XR02LiveService(
        endpoints,
        selection,
        cast(XR02LivePipeline, pipeline),
        cast(XR02LiveRerunLogger, logger),
        lambda: selection,
        XR02LiveServiceConfig(association_hz=8.0, maximum_frame_age_ms=1000.0),
    )
    service.start()
    time.sleep(0.14)
    service.stop()
    evidence: Any = service.evidence()
    samples = evidence["camera_health_samples"]
    assert isinstance(samples, list) and samples
    last = {item["camera_id"]: item for item in samples[-1]["cameras"]}
    assert last["office-cam-04"]["state"] == "starting"
    assert all(last[camera]["state"] == "current" for camera in CAMERA_IDS[:3])
    assert evidence["status"]["state"] == "stopped"
    assert all(item["state"] == "stopped" for item in evidence["status"]["camera_health"])
    assert evidence["status"]["worker"]["busy_dropped_ticks"] == 0
    assert logger.closed


def test_rerun_presentation_separates_identity_color_from_lifecycle() -> None:
    colors = [_identity_color(f"g:{index:06d}") for index in range(1, 25)]
    assert len({tuple(color) for color in colors}) == 24
    track = GlobalTrackSnapshot(
        "g:000003",
        GlobalTrackState.CONFIRMED,
        (1.0, 2.0),
        10,
        ("office-cam-01",),
        4,
        "fixture",
    )
    confirmed = MemberAssignment(
        "obs-1",
        "local-1",
        "office-cam-01",
        "g:000003",
        GlobalTrackState.CONFIRMED,
        "fixture",
    )
    label, color = _camera_observation_presentation(
        7,
        0.8,
        "available",
        confirmed,
        {"g:000003": track},
    )
    assert "G000003 · CONFIRMED · L7" in label
    assert color == _identity_color("g:000003")
    ambiguous = MemberAssignment(
        "obs-2",
        "local-2",
        "office-cam-02",
        None,
        GlobalTrackState.AMBIGUOUS,
        "candidate_tie",
    )
    ambiguous_label, ambiguous_color = _camera_observation_presentation(
        8,
        0.7,
        "available",
        ambiguous,
        {"g:000003": track},
    )
    assert "AMBIGUOUS · no global ID" in ambiguous_label
    assert ambiguous_color != color


def test_people_summary_reports_deduplicated_visible_count_and_track_details() -> None:
    confirmed = GlobalTrackSnapshot(
        "g:000003",
        GlobalTrackState.CONFIRMED,
        (1.25, 2.5),
        10,
        ("office-cam-01", "office-cam-02"),
        4,
        "fixture",
    )
    lost = GlobalTrackSnapshot(
        "g:000008",
        GlobalTrackState.LOST,
        (3.0, 4.0),
        9,
        ("office-cam-03",),
        2,
        "fixture",
    )
    primary = MemberAssignment(
        "obs-1",
        "local-1",
        "office-cam-01",
        confirmed.global_track_id,
        GlobalTrackState.CONFIRMED,
        "fixture",
    )
    duplicate = MemberAssignment(
        "obs-2",
        "local-2",
        "office-cam-02",
        confirmed.global_track_id,
        GlobalTrackState.DUPLICATE,
        "fixture",
    )
    association = AssociationTickResult(
        "a" * 64,
        0,
        1,
        "fixture",
        (primary, duplicate),
        (confirmed, lost),
        (),
    )
    tick = LiveAssociationTick(
        0,
        1,
        2,
        (),
        (),
        (),
        None,
        association,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0,
        0,
        0,
    )
    markdown = _active_tracks_markdown(tick, {confirmed.global_track_id})
    confirmed_hex = "#" + "".join(
        f"{value:02X}" for value in _identity_color(confirmed.global_track_id)
    )
    assert "People currently observed: 1" in markdown
    assert confirmed_hex in markdown
    assert "G000003" in markdown
    assert "CONFIRMED · OBSERVED" in markdown
    assert "(1.25, 2.50)" in markdown
    assert "G000008" in markdown
    assert "LOST · NOT CURRENTLY VISIBLE" in markdown
    assert "Tracked IDs retained: **2**" in markdown


def test_trial_four_detector_and_new_track_confidence_are_separated(tmp_path: Path) -> None:
    detector = tmp_path / "detector.pt"
    reid = tmp_path / "reid.pt"
    detector.write_bytes(b"detector")
    reid.write_bytes(b"reid")
    profile = LiveModelProfile(detector, _sha256(detector), reid, _sha256(reid))
    assert profile.detector_confidence == 0.15
    assert profile.new_track_confidence == 0.65
    assert "c015-new065" in profile.profile_id
    with pytest.raises(XR02LiveContractError, match="confidence"):
        LiveModelProfile(
            detector,
            _sha256(detector),
            reid,
            _sha256(reid),
            detector_confidence=0.0,
        )
    with pytest.raises(XR02LiveContractError, match="must not exceed"):
        LiveModelProfile(
            detector,
            _sha256(detector),
            reid,
            _sha256(reid),
            detector_confidence=0.8,
            new_track_confidence=0.65,
        )


def test_rerun_trail_splits_without_breaking_global_identity() -> None:
    state = _TrailState(
        points=[[1.0, 1.0, 0.08]],
        last_observed_monotonic_ns=1_000_000_000,
        last_camera_ids=("office-cam-01",),
    )
    assert _trail_break_reason(state, [1.4, 1.0, 0.08], 1_500_000_000, ("office-cam-01",)) is None
    assert (
        _trail_break_reason(state, [2.2, 1.0, 0.08], 1_500_000_000, ("office-cam-03",))
        == "camera_handoff_discontinuity"
    )
    assert (
        _trail_break_reason(state, [3.1, 1.0, 0.08], 1_500_000_000, ("office-cam-01",))
        == "position_discontinuity"
    )
    state.break_pending = True
    assert (
        _trail_break_reason(state, [1.1, 1.0, 0.08], 1_500_000_000, ("office-cam-01",))
        == "not_observed_gap"
    )


def test_camera_feed_status_is_explicit_without_erasing_last_good_image() -> None:
    association = AssociationTickResult("a" * 64, 0, 1, "fixture", (), (), ())
    stale = LiveAssociationTick(
        0,
        1,
        2,
        (),
        ("office-cam-01",),
        (),
        None,
        association,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
    )
    label, color = _camera_feed_status("office-cam-01", stale)
    assert label.startswith("STALE")
    assert color == [255, 176, 48]
    missing = LiveAssociationTick(
        1,
        2,
        3,
        (),
        (),
        ("office-cam-02",),
        None,
        association,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
    )
    label, color = _camera_feed_status("office-cam-02", missing)
    assert label.startswith("OFFLINE/MISSING")
    assert color == [255, 72, 72]


def test_trial_recording_accepts_only_credential_free_mediamtx_fanout(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "ffmpeg.exe"
    binary.write_bytes(b"fixture")
    policy = TrialRecordingPolicy(binary)
    endpoints = tuple(
        LocalRtspEndpoint(
            camera_id,
            CAMERA_ENDPOINT_KEYS[camera_id],
            f"rtsp://127.0.0.1:8554/officecam{index:02d}",
        )
        for index, camera_id in enumerate(CAMERA_IDS, start=1)
    )
    recorder = MediaMtxTrialRecorder(endpoints, policy, tmp_path / "capture")
    serialized = json.dumps(recorder.evidence()).lower()
    assert "rtsp://" not in serialized
    assert recorder.evidence()["credentials_persisted"] is False
    assert recorder.status()["configured"] is True
    assert recorder.status()["active"] is False
    with pytest.raises(TrialRecordingError, match="credential-free"):
        _validate_credential_free_local_endpoint("rtsp://user:secret@camera.test/live")
    command = _record_command(
        binary,
        "rtsp://127.0.0.1:8554/officecam01",
        tmp_path / "office-cam-01.mkv",
    )
    assert command[command.index("-c:v") + 1] == "copy"
    assert "-an" in command


def test_trial_recording_verifier_accepts_segmented_camera_inventory(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    video_root = run / "trial-video"
    video_root.mkdir(parents=True)
    binary = tmp_path / "ffmpeg.exe"
    binary.write_bytes(b"ffmpeg")
    captures: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for camera_id in CAMERA_IDS:
        video = video_root / f"{camera_id}-g0001.mkv"
        video.write_bytes(camera_id.encode("ascii"))
        artifact = {
            "path": str(video.resolve()),
            "present": True,
            "bytes": video.stat().st_size,
            "sha256": _sha256(video),
        }
        captures.append(
            {
                "camera_id": camera_id,
                "generation": 1,
                "return_code": 0,
                "artifact": artifact,
            }
        )
        artifacts.append(artifact)
    manifest = {
        "client_frames_persisted_separately": True,
        "trial_recording": {
            "schema": "xr02.wp4.trial_recording.v1",
            "enabled": True,
            "credentials_persisted": False,
            "video_mode": "encoded-stream-copy",
            "reconnect_policy": "supervised per-camera segmented restart",
            "ffmpeg_binary_path": str(binary.resolve()),
            "ffmpeg_binary_sha256": _sha256(binary),
            "ffmpeg_version": "ffmpeg version fixture",
            "captures": captures,
        },
    }
    assert _verify_trial_recording(
        run / "wp4-live-manifest.json",
        manifest,
        {"trial_video": artifacts},
    )
    captures[0]["artifact"] = {
        "path": str((tmp_path / "escaped.mkv").resolve()),
        "present": False,
    }
    with pytest.raises(ValueError, match="escaped"):
        _verify_trial_recording(
            run / "wp4-live-manifest.json",
            manifest,
            {"trial_video": artifacts},
        )


def test_trial_recording_supervises_segment_restart_and_bounded_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "ffmpeg.exe"
    binary.write_bytes(b"fixture")
    launches_by_endpoint: dict[str, int] = {}

    def fake_popen(*args: Any, **_kwargs: object) -> Any:
        command = cast(list[str], args[0])
        endpoint = command[command.index("-i") + 1]
        launches_by_endpoint[endpoint] = launches_by_endpoint.get(endpoint, 0) + 1
        return _FakeRecorderProcess(exit_immediately=launches_by_endpoint[endpoint] == 1)

    monkeypatch.setattr(trial_recording_module, "_spawn_ffmpeg", fake_popen)
    monkeypatch.setattr(
        trial_recording_module,
        "_ffmpeg_version",
        lambda _binary: "ffmpeg version fixture",
    )
    endpoints = tuple(
        LocalRtspEndpoint(
            camera_id,
            CAMERA_ENDPOINT_KEYS[camera_id],
            f"rtsp://127.0.0.1:8554/officecam{index:02d}",
        )
        for index, camera_id in enumerate(CAMERA_IDS, start=1)
    )
    recorder = MediaMtxTrialRecorder(
        endpoints,
        TrialRecordingPolicy(binary, reconnect_delay_seconds=0.1),
        tmp_path / "segments",
    )
    recorder.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (
        len(launches_by_endpoint) < 4 or min(launches_by_endpoint.values()) < 2
    ):
        time.sleep(0.01)
    recorder.stop()
    assert len(launches_by_endpoint) == 4
    assert min(launches_by_endpoint.values()) >= 2
    assert recorder.status()["active"] is False
    generations_by_camera: dict[str, set[int]] = {}
    for item in recorder.capture_artifacts():
        generations_by_camera.setdefault(str(item["camera_id"]), set()).add(
            int(cast(int, item["generation"]))
        )
    assert all(len(generations_by_camera[camera_id]) >= 2 for camera_id in CAMERA_IDS)


def _scene_fixture(root: Path) -> AdoptedSceneSelection:
    geometry = root / "geometry.npz"
    np.savez_compressed(
        geometry,
        points=np.zeros((1, 3), dtype=np.float32),
        colors_rgb=np.zeros((1, 3), dtype=np.uint8),
    )
    floor_root = root / "floor"
    floor_root.mkdir()
    floor = floor_root / "authoritative_floor_plane.npz"
    np.savez_compressed(floor, minimum_xy_metres=[0.0, 0.0], maximum_xy_metres=[1.0, 1.0])
    static = root / "static.rrd"
    static.write_bytes(b"rrd")
    p06 = root / "p06.json"
    p07 = root / "p07.json"
    p08 = root / "floor-manifest.json"
    for path in (p06, p07, p08):
        path.write_text("{}", encoding="utf-8")
    operator = root / "operator-state.json"
    operator.write_text(
        json.dumps(
            {
                "geometry_approved": True,
                "floor_approved": True,
                "active_geometry_artifact_id": "geometry-1",
                "active_floor_artifact_id": "floor-1",
                "geometry_source_path": str(geometry),
                "geometry_source_sha256": _sha256(geometry),
                "current_floor_output_directory": str(floor_root),
                "runtime_artifacts": [
                    {
                        "artifact_id": "floor-1",
                        "selected": True,
                        "path": str(static),
                        "sha256": _sha256(static),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return resolve_adopted_scene(operator, p06, p07, p08)


class _FakeController:
    def __init__(self) -> None:
        self.running = False

    def start(self) -> dict[str, object]:
        if self.running:
            raise RuntimeError("already running")
        self.running = True
        return self.status()

    def start_live(
        self,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, object]:
        if (resumed_from_session_id is None) != (scene_update_id is None):
            raise RuntimeError("resume linkage must be complete")
        return self.start()

    def start_recording(self) -> dict[str, object]:
        return self.start()

    def stop(self, *, reason: str = "operator") -> dict[str, object]:
        if not reason:
            raise RuntimeError("stop reason required")
        self.running = False
        return self.status()

    def open_rerun(self) -> dict[str, object]:
        return self.status()

    def reset_trails(self) -> dict[str, object]:
        return self.status()

    def export_evidence_snapshot(self) -> dict[str, object]:
        return {"path": "evidence.json", "bytes": 1, "sha256": "a" * 64}

    def view_recording(self, session_id: str) -> dict[str, object]:
        if not session_id:
            raise RuntimeError("session required")
        return self.status()

    def save_recording(self, session_id: str, label: str) -> dict[str, object]:
        if not session_id or not label:
            raise RuntimeError("session and label required")
        return self.status()

    def delete_recording(self, session_id: str, confirmation: str) -> dict[str, object]:
        if confirmation != f"DELETE {session_id}":
            raise RuntimeError("confirmation mismatch")
        return self.status()

    def status(self) -> dict[str, object]:
        return {
            "active": self.running,
            "active_mode": "live" if self.running else None,
            "operator_state": "live_running" if self.running else "ready",
            "pending_run": None,
            "saved_recordings": [],
            "recent_live_runs": [],
            "recording_available": True,
            "storage": {"free_bytes": 1024},
            "viewer_auto_open": {"eligible": False, "opened": False},
            "service": {"state": "running" if self.running else "stopped"},
        }


class _FakeDecoder:
    def __init__(
        self,
        endpoint: LocalRtspEndpoint,
        slot: Any,
        policy: Any,
        backend: CaptureBackend,
        *,
        gstreamer_overlay_path: Path | None,
        epoch_callback: Any,
    ) -> None:
        self.camera_id = endpoint.camera_id
        self.slot = slot
        self.backend = backend
        self.epoch_callback = epoch_callback
        self.running = False
        self.frames = 0
        self.acquired: int | None = None

    def start(self) -> None:
        self.running = True
        self.epoch_callback(self.camera_id, 1, 1, "initial_start")
        if self.camera_id == "office-cam-04":
            return
        self.acquired = time.monotonic_ns()
        identity = LiveFrameIdentity(
            self.camera_id,
            f"{self.camera_id}-fixture-0",
            self.acquired,
            "2026-08-24T00:00:00+00:00",
            None,
            None,
            2,
            2,
        )
        self.slot.publish(CapturedFrame(identity, np.zeros((2, 2, 3), dtype=np.uint8)))
        self.frames = 1

    def close(self) -> None:
        self.running = False

    def telemetry(self) -> SupervisedDecoderTelemetry:
        return SupervisedDecoderTelemetry(
            camera_id=self.camera_id,
            backend=self.backend,
            state=(
                CaptureProcessState.STOPPED
                if not self.running
                else CaptureProcessState.CURRENT
                if self.frames
                else CaptureProcessState.STARTING
            ),
            generation=1,
            tracker_epoch=1,
            decoded_frames=self.frames,
            delivered_frames=self.frames,
            replaced_before_delivery=0,
            reconnects=0,
            watchdog_restarts=0,
            failure_class=None,
            failure_detail=None,
            restart_reason=None,
            last_process_heartbeat_monotonic_ns=self.acquired,
            last_acquisition_monotonic_ns=self.acquired,
            running=self.running,
        )

    def events(self) -> tuple[object, ...]:
        return ()


class _SynchronousWorker:
    def __init__(self, operation: Any, name: str, **_kwargs: object) -> None:
        self.operation = operation
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.closed = False

    def try_submit(self, snapshot: LatestFrameSnapshot) -> bool:
        self.submitted += 1
        try:
            self.operation(snapshot)
        except Exception:
            self.failed += 1
        else:
            self.completed += 1
        return True

    def close(self, wait: bool = True) -> None:
        self.closed = True

    def invalidate_pending(self) -> bool:
        return False

    def telemetry(self) -> InferenceAdmissionTelemetry:
        return InferenceAdmissionTelemetry(
            self.submitted,
            0,
            self.completed,
            self.failed,
            False,
        )


class _FakeRecorderStdin:
    def __init__(self, process: _FakeRecorderProcess) -> None:
        self._process = process

    def write(self, value: bytes) -> int:
        if value == b"q\n":
            self._process.returncode = 0
        return len(value)

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class _FakeRecorderProcess:
    def __init__(self, *, exit_immediately: bool) -> None:
        self.returncode: int | None = 1 if exit_immediately else None
        self.stdin = _FakeRecorderStdin(self)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _FakePipeline:
    def __init__(self, selection: AdoptedSceneSelection) -> None:
        self.selection = selection
        self.tick = 0

    @property
    def profile_identity(self) -> dict[str, object]:
        return {"profile": "fixture"}

    def warmup(self) -> None:
        return

    def finalize_evidence(self) -> None:
        return

    def begin_camera_epoch(
        self,
        camera_id: str,
        capture_generation: int,
        tracker_epoch: int,
        reason: str,
    ) -> None:
        assert camera_id in CAMERA_IDS
        assert capture_generation == tracker_epoch == 1
        assert reason == "initial_start"

    def process(self, snapshot: LatestFrameSnapshot) -> LiveAssociationTick:
        association = AssociationTickResult(
            self.selection.scene.context_sha256,
            self.tick,
            snapshot.snapshot_monotonic_ns,
            "fixture",
            (),
            (),
            (),
        )
        result = LiveAssociationTick(
            self.tick,
            snapshot.snapshot_monotonic_ns,
            time.monotonic_ns(),
            (),
            snapshot.stale_camera_ids,
            snapshot.missing_camera_ids,
            snapshot.cross_camera_skew_ms,
            association,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0,
            0,
            0,
        )
        self.tick += 1
        return result


@dataclass
class _FakeLogger:
    closed: bool = False

    def log_tick(self, tick: LiveAssociationTick) -> None:
        return

    def log_service_state(self, state: str, detail: str) -> None:
        return

    def open_viewer(self) -> None:
        return

    def reset_trails(self) -> None:
        return

    def close(self) -> None:
        self.closed = True

    def evidence(self) -> dict[str, object]:
        return {"closed": self.closed}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
