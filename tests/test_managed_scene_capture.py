from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from pathlib import Path
from time import sleep
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from spatial_mapping_phase2 import managed_scene_capture
from spatial_mapping_phase2.managed_scene_capture import (
    SCENE_CAPTURE_PROFILE_SCHEMA,
    PyAvSceneCaptureAdapter,
    SceneCameraBinding,
    SceneCaptureArtifact,
    SceneCaptureCancelledError,
    SceneCaptureError,
    SceneCapturePolicy,
    SceneCaptureRepository,
    SceneCaptureService,
    SceneCaptureSessionManifest,
    SceneCaptureStatus,
    SceneEndpointLoader,
    ScenePreviewFrame,
    SceneRationalTimeBase,
    SceneRtspEndpoint,
    SceneSourceFrame,
    SceneStorageMode,
    SceneStreamProfile,
)
from spatial_mapping_phase2.managed_scene_capture_app import create_scene_capture_app


def camera_bindings(count: int) -> dict[str, str]:
    return {
        f"scene-test-cam-{index:02d}": f"SCENE_TEST_CAMERA_{index:02d}_RTSP"
        for index in range(1, count + 1)
    }


def write_secrets(path: Path, bindings: Mapping[str, str], *, include: int | None = None) -> None:
    selected = list(bindings.items()) if include is None else list(bindings.items())[:include]
    path.write_text(
        "\n".join(f"{key}=rtsp://fixture.invalid/{camera_id}" for camera_id, key in selected)
        + "\n",
        encoding="utf-8",
    )


class FakeSceneCaptureAdapter:
    def __init__(self) -> None:
        self.seen_urls: list[str] = []
        self.capture_attempts: dict[str, int] = {}
        self.closed = False

    def profile(
        self, endpoint: SceneRtspEndpoint, policy: SceneCapturePolicy
    ) -> SceneStreamProfile:
        del policy
        self.seen_urls.append(endpoint.for_read_only_adapter())
        binding = endpoint.binding
        return SceneStreamProfile(
            SCENE_CAPTURE_PROFILE_SCHEMA,
            binding.camera_id,
            "scene-stream-profile-v1",
            binding.endpoint_environment_key,
            f"local-{binding.endpoint_environment_key.lower()}",
            640,
            360,
            "h264",
            SceneRationalTimeBase(1, 90_000),
            None,
            None,
            "2026-08-31T00:00:00Z",
        )

    def capture(
        self,
        endpoint: SceneRtspEndpoint,
        profile: SceneStreamProfile,
        output_path: Path,
        relative_path: str,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SceneSourceFrame, ...], SceneCaptureArtifact]:
        del policy
        if cancel.is_set():
            raise SceneCaptureCancelledError("synthetic cancellation")
        camera_id = endpoint.binding.camera_id
        self.capture_attempts[camera_id] = self.capture_attempts.get(camera_id, 0) + 1
        self.seen_urls.append(endpoint.for_read_only_adapter())
        payload = f"synthetic:{camera_id}:{self.capture_attempts[camera_id]}".encode()
        output_path.write_bytes(payload)
        offset = sum(ord(value) for value in camera_id) * 1_000_000
        frames = tuple(
            SceneSourceFrame(
                f"{camera_id}-f{index}",
                camera_id,
                profile.profile_version,
                index * 3_000,
                profile.time_base,
                offset + index * 33_000_000,
                "2026-08-31T00:00:00Z",
            )
            for index in range(3)
        )
        return frames, SceneCaptureArtifact(
            relative_path,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            SceneStorageMode.PACKET_PRESERVING_MP4,
            False,
            None,
        )

    def preview(
        self,
        endpoint: SceneRtspEndpoint,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> ScenePreviewFrame:
        del policy
        if cancel.is_set():
            raise SceneCaptureCancelledError("synthetic cancellation")
        self.seen_urls.append(endpoint.for_read_only_adapter())
        return ScenePreviewFrame(
            endpoint.binding.camera_id,
            "image/jpeg",
            f"preview:{endpoint.binding.camera_id}".encode(),
            "2026-08-31T00:00:00Z",
            0,
            "1/90000",
        )

    def close(self) -> None:
        self.closed = True


class BlockingSceneCaptureAdapter(FakeSceneCaptureAdapter):
    def capture(
        self,
        endpoint: SceneRtspEndpoint,
        profile: SceneStreamProfile,
        output_path: Path,
        relative_path: str,
        policy: SceneCapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SceneSourceFrame, ...], SceneCaptureArtifact]:
        del endpoint, profile, output_path, relative_path, policy
        cancel.wait(timeout=1.0)
        raise SceneCaptureCancelledError("synthetic cancellation")


def service(
    tmp_path: Path,
    bindings: Mapping[str, str],
    adapter: FakeSceneCaptureAdapter | None = None,
) -> SceneCaptureService:
    secret_file = tmp_path / "secrets.env"
    write_secrets(secret_file, bindings)
    return SceneCaptureService(
        SceneEndpointLoader(bindings, secret_file),
        adapter or FakeSceneCaptureAdapter(),
        SceneCaptureRepository(tmp_path / "artifacts"),
        "scene-test",
        "test-managed-capture-v1",
    )


@pytest.mark.parametrize("count", [1, 2, 4])
def test_variable_rosters_capture_immutable_bundle_without_credentials(
    tmp_path: Path, count: int
) -> None:
    bindings = camera_bindings(count)
    workflow = service(tmp_path, bindings)

    manifest = workflow.capture_session("session-1")

    assert tuple(result.camera_id for result in manifest.results) == tuple(bindings)
    assert all(result.status is SceneCaptureStatus.CAPTURED for result in manifest.results)
    assert workflow.repository.list_sessions() == ("session-1",)
    assert (
        tmp_path / "artifacts" / "captures" / "managed-scene" / "sessions" / "session-1"
    ).is_dir()
    assert "rtsp://" not in manifest.to_json()
    assert "fixture.invalid" not in manifest.to_json()

    bundle = workflow.select_bundle(manifest, "bundle-1")

    assert bundle.status.value == "complete-roster"
    assert tuple(frame.camera_id for frame in bundle.selected_frames) == tuple(bindings)
    assert len(bundle.pairwise_skew) == count * (count - 1) // 2
    assert "fixture.invalid" not in bundle.to_json()


def test_endpoint_loader_reloads_p02_secret_changes_and_reports_incomplete_config(
    tmp_path: Path,
) -> None:
    bindings = camera_bindings(2)
    secret_file = tmp_path / "secrets.env"
    write_secrets(secret_file, bindings, include=1)
    adapter = FakeSceneCaptureAdapter()
    workflow = SceneCaptureService(
        SceneEndpointLoader(bindings, secret_file),
        adapter,
        SceneCaptureRepository(tmp_path / "artifacts"),
        "scene-test",
    )

    before = workflow.health()

    assert before["scene-test-cam-01"]["state"] == "healthy"
    assert before["scene-test-cam-02"]["state"] == "not_configured"
    with pytest.raises(SceneCaptureError, match="not configured"):
        workflow.preview("scene-test-cam-02")

    write_secrets(secret_file, bindings)
    preview = workflow.preview("scene-test-cam-02")

    assert preview.camera_id == "scene-test-cam-02"
    assert adapter.seen_urls[-1].endswith("/scene-test-cam-02")
    assert workflow.health()["scene-test-cam-02"]["state"] == "healthy"


def test_pyav_profile_accepts_a_stream_opened_within_the_transport_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thread scheduling after a successful open must not create a false timeout."""

    class FakeOpen:
        def __init__(self) -> None:
            time_base = SimpleNamespace(numerator=1, denominator=90_000)
            stream = SimpleNamespace(
                time_base=time_base,
                metadata={},
                codec_context=SimpleNamespace(width=640, height=360, name="h264"),
            )
            self.streams = SimpleNamespace(video=[stream])

        def __enter__(self) -> FakeOpen:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    fake_av = SimpleNamespace(open=lambda *_args, **_kwargs: FakeOpen())
    elapsed = iter((0.0, 6.0))
    monkeypatch.setattr(managed_scene_capture, "_av", lambda: fake_av)
    monkeypatch.setattr(managed_scene_capture, "monotonic", lambda: next(elapsed))
    endpoint = SceneRtspEndpoint(
        SceneCameraBinding("scene-test-cam-01", "SCENE_TEST_CAMERA_01_RTSP"),
        "rtsp://fixture.invalid/camera-01",
    )

    profile = PyAvSceneCaptureAdapter().profile(endpoint, SceneCapturePolicy())

    assert profile.camera_id == "scene-test-cam-01"
    assert profile.width_pixels == 640


def test_incomplete_or_invalid_secret_never_leaks_credential_values(tmp_path: Path) -> None:
    bindings = camera_bindings(1)
    secret_file = tmp_path / "secrets.env"
    endpoint_key = next(iter(bindings.values()))
    secret_file.write_text(
        f"{endpoint_key}=rtsp://invalid host/live\n",
        encoding="utf-8",
    )
    workflow = SceneCaptureService(
        SceneEndpointLoader(bindings, secret_file),
        FakeSceneCaptureAdapter(),
        SceneCaptureRepository(tmp_path / "artifacts"),
        "scene-test",
    )

    health = workflow.health()
    manifest = workflow.capture_session("invalid-secret")

    assert health["scene-test-cam-01"]["configuration_state"] == "invalid"
    assert manifest.results[0].status is SceneCaptureStatus.FAILED
    assert "rtsp://" not in manifest.to_json().lower()
    assert "invalid host" not in manifest.to_json().lower()


def test_cancellation_and_close_preserve_failed_session_lifecycle(tmp_path: Path) -> None:
    bindings = camera_bindings(2)
    adapter = BlockingSceneCaptureAdapter()
    workflow = service(tmp_path, bindings, adapter)
    results: list[SceneCaptureSessionManifest] = []

    worker = threading.Thread(target=lambda: results.append(workflow.capture_session("cancelled")))
    worker.start()
    sleep(0.02)

    assert workflow.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    manifest = results[0]
    assert all(result.failure_class == "SceneCaptureCancelledError" for result in manifest.results)
    workflow.close()
    assert adapter.closed
    with pytest.raises(SceneCaptureError, match="closed"):
        workflow.health()


def test_fastapi_surface_reloads_secret_file_and_supports_capture_bundle(tmp_path: Path) -> None:
    bindings = camera_bindings(2)
    secret_file = tmp_path / "secrets.env"
    app = create_scene_capture_app(
        bindings,
        secret_file,
        tmp_path / "artifacts",
        "scene-test",
        adapter=FakeSceneCaptureAdapter(),
    )
    client = TestClient(app)

    assert "Capture current scene" in client.get("/").text
    assert client.get("/api/cameras").json() == {"cameras": list(bindings)}
    incomplete = client.get("/api/health")
    assert incomplete.status_code == 200
    assert incomplete.json()["scene-test-cam-01"]["state"] == "not_configured"

    write_secrets(secret_file, bindings)
    preview = client.get("/api/cameras/scene-test-cam-01/preview")
    assert preview.status_code == 200
    assert preview.headers["x-scene-capture-evidence"] == "ephemeral-preview-not-capture-evidence"
    assert "fixture.invalid" not in preview.text

    capture = client.post(
        "/api/capture", json={"session_id": "web-session", "duration_seconds": 0.2}
    )
    assert capture.status_code == 200
    assert "fixture.invalid" not in capture.text
    bundle = client.post("/api/sessions/web-session/bundles", json={"bundle_id": "web-bundle"})
    assert bundle.status_code == 200
    assert bundle.json()["status"] == "complete-roster"
    evidence = client.get("/api/evidence")
    assert evidence.status_code == 200
    assert evidence.json()["current_bundle"]["bundle_id"] == "web-bundle"
    assert evidence.json()["current_bundle"]["selection_source"] == "explicit-pointer"
    selection = tmp_path / "artifacts" / "captures" / "managed-scene" / "current-bundle.json"
    assert selection.is_file()
    assert "fixture.invalid" not in selection.read_text(encoding="utf-8")
    assert client.get("/api/cancel").status_code == 405
    assert client.post("/api/cancel").json() == {"state": "no-capture-active"}
