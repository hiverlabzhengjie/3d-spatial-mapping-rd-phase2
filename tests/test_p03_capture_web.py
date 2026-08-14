from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    CAMERA_IDS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p03_capture_service import (
    CapturePolicy,
    CaptureRepository,
    CaptureWorkflowService,
    SyntheticCaptureAdapter,
)
from spatial_mapping_phase2.p03_capture_web import create_p03_capture_app
from spatial_mapping_phase2.p03_temporal_capture import (
    SyntheticWarmFrameAdapter,
    TemporalBundleRepository,
    WarmTemporalCaptureService,
)


def service(tmp_path: Path) -> CaptureWorkflowService:
    endpoints = tuple(
        LocalRtspEndpoint(camera_id, CAMERA_ENDPOINT_KEYS[camera_id], "rtsp://fixture/live")
        for camera_id in CAMERA_IDS
    )
    return CaptureWorkflowService(
        endpoints, SyntheticCaptureAdapter(), CaptureRepository(tmp_path), "test-v1"
    )


def test_console_and_cli_service_surface_share_capture_and_bundle(tmp_path: Path) -> None:
    workflow = service(tmp_path)
    client = TestClient(create_p03_capture_app(workflow))
    page = client.get("/").text
    assert "Credential-safe health" in page
    assert "Select bundle and show skew" in page
    assert "img:not([src])" in page
    health = client.get("/api/health").json()
    assert list(health) == list(CAMERA_IDS)
    assert "rtsp://" not in str(health)
    preview = client.get(f"/api/cameras/{CAMERA_IDS[0]}/preview")
    assert preview.status_code == 200
    assert preview.headers["cache-control"] == "no-store"
    assert preview.headers["x-p03-evidence"] == "ephemeral-preview-not-capture-evidence"
    assert workflow.repository.list_sessions() == ()
    response = client.post(
        "/api/capture", json={"session_id": "web-session", "duration_seconds": 0.1}
    )
    assert response.status_code == 200
    assert client.get("/api/sessions").json() == {"sessions": ["web-session"]}
    bundle = client.post(
        "/api/sessions/web-session/bundles", json={"bundle_id": "web-bundle"}
    ).json()
    assert bundle["status"] == "complete-four-camera"
    assert bundle["overall_skew_ns"] == 3_000_000


def test_console_reports_validation_errors_without_endpoint_values(tmp_path: Path) -> None:
    client = TestClient(create_p03_capture_app(service(tmp_path)))
    response = client.post("/api/capture", json={"duration_seconds": 1})
    assert response.status_code == 422
    assert "rtsp" not in response.text.lower()


def test_console_temporal_capture_enforces_supplied_window(tmp_path: Path) -> None:
    workflow = service(tmp_path)
    timestamps: dict[str, tuple[int, ...]] = {
        camera_id: (1_000_000_000 + index * 10_000_000,)
        for index, camera_id in enumerate(CAMERA_IDS)
    }
    fixed_endpoints = tuple(
        LocalRtspEndpoint(camera_id, CAMERA_ENDPOINT_KEYS[camera_id], "rtsp://fixture/live")
        for camera_id in CAMERA_IDS
    )

    def temporal_factory() -> WarmTemporalCaptureService:
        return WarmTemporalCaptureService(
            fixed_endpoints,
            SyntheticWarmFrameAdapter(timestamps),
            TemporalBundleRepository(tmp_path),
            CapturePolicy(read_timeout_seconds=0.1, initial_backoff_seconds=0.001),
            "test-v1",
            clock_resolution_ns=1_000_000,
        )

    client = TestClient(create_p03_capture_app(workflow, temporal_factory))
    response = client.post(
        "/api/temporal-capture",
        json={"bundle_id": "web-temporal", "max_skew_ms": 50, "warmup_seconds": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["authority_status"] == "authoritative-local-acquisition-window"
    assert payload["overall_skew_ns"] == 30_000_000
    assert payload["conservative_overall_skew_ns"] == 32_000_000
