from __future__ import annotations

import threading
from pathlib import Path
from time import sleep

import pytest

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    CAMERA_IDS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p03_capture_domain import (
    BundleStatus,
    CaptureSessionManifest,
    CaptureStatus,
    P03ContractError,
)
from spatial_mapping_phase2.p03_capture_service import (
    CaptureCancelledError,
    CapturePolicy,
    CaptureRepository,
    CaptureWorkflowService,
    ConnectTimeoutError,
    SyntheticCaptureAdapter,
)


def endpoints() -> tuple[LocalRtspEndpoint, ...]:
    return tuple(
        LocalRtspEndpoint(camera_id, CAMERA_ENDPOINT_KEYS[camera_id], "rtsp://fixture/live")
        for camera_id in CAMERA_IDS
    )


def test_synthetic_four_camera_capture_reconnect_replay_and_bundle(tmp_path: Path) -> None:
    adapter = SyntheticCaptureAdapter({CAMERA_IDS[1]: 1})
    repository = CaptureRepository(tmp_path)
    service = CaptureWorkflowService(endpoints(), adapter, repository, "test-v1")
    manifest = service.capture_session("session-1", CapturePolicy(initial_backoff_seconds=0.001))
    assert all(result.status is CaptureStatus.CAPTURED for result in manifest.results)
    assert manifest.results[1].reconnect_events[-1].state == "reconnected"
    assert repository.read_session_payload("session-1")["session_id"] == "session-1"
    assert repository.read_session("session-1") == manifest
    bundle = service.select_bundle(manifest, "bundle-1")
    assert bundle.status is BundleStatus.COMPLETE
    assert bundle.overall_skew_ns == 3_000_000
    assert repository.list_sessions() == ("session-1",)
    service.close()
    assert adapter.closed


def test_exhausted_camera_is_preserved_as_partial_session(tmp_path: Path) -> None:
    adapter = SyntheticCaptureAdapter({CAMERA_IDS[2]: 9})
    service = CaptureWorkflowService(endpoints(), adapter, CaptureRepository(tmp_path), "test-v1")
    manifest = service.capture_session(
        "session-partial", CapturePolicy(retry_limit=1, initial_backoff_seconds=0.001)
    )
    failed = manifest.results[2]
    assert failed.status is CaptureStatus.FAILED
    assert failed.failure_message == "bounded capture failed"
    bundle = service.select_bundle(manifest, "bundle-partial")
    assert bundle.status is BundleStatus.PARTIAL
    assert bundle.missing_camera_ids == (CAMERA_IDS[2],)


def test_health_is_credential_safe_and_closed_service_rejects(tmp_path: Path) -> None:
    service = CaptureWorkflowService(
        endpoints(), SyntheticCaptureAdapter(), CaptureRepository(tmp_path), "test-v1"
    )
    payload = service.health(CapturePolicy())
    assert "rtsp://" not in str(payload)
    service.close()
    with pytest.raises(P03ContractError, match="closed"):
        service.health(CapturePolicy())


def test_repository_rejects_manifest_overwrite(tmp_path: Path) -> None:
    service = CaptureWorkflowService(
        endpoints(), SyntheticCaptureAdapter(), CaptureRepository(tmp_path), "test-v1"
    )
    service.capture_session("immutable", CapturePolicy())
    with pytest.raises(P03ContractError, match="already exists"):
        service.capture_session("immutable", CapturePolicy())


def test_preview_is_ephemeral_and_does_not_create_session(tmp_path: Path) -> None:
    service = CaptureWorkflowService(
        endpoints(), SyntheticCaptureAdapter(), CaptureRepository(tmp_path), "test-v1"
    )
    frame = service.preview(CAMERA_IDS[0], CapturePolicy())
    assert frame.media_type == "image/jpeg"
    assert service.repository.list_sessions() == ()
    assert "rtsp://" not in repr(frame)


def test_backpressure_is_explicit_and_preserves_failed_session(tmp_path: Path) -> None:
    service = CaptureWorkflowService(
        endpoints(), SyntheticCaptureAdapter(), CaptureRepository(tmp_path), "test-v1"
    )
    manifest = service.capture_session(
        "backpressure", CapturePolicy(queue_capacity=2, retry_limit=0)
    )
    assert {result.failure_class for result in manifest.results} == {"BackpressureError"}


class ConnectTimeoutAdapter(SyntheticCaptureAdapter):
    def profile(self, endpoint: LocalRtspEndpoint, policy: CapturePolicy):  # type: ignore[no-untyped-def]
        del endpoint, policy
        raise ConnectTimeoutError("synthetic connect timeout")


def test_connect_timeout_is_sanitized_per_camera(tmp_path: Path) -> None:
    service = CaptureWorkflowService(
        endpoints(), ConnectTimeoutAdapter(), CaptureRepository(tmp_path), "test-v1"
    )
    manifest = service.capture_session("connect-timeout", CapturePolicy(retry_limit=0))
    assert {result.failure_class for result in manifest.results} == {"ConnectTimeoutError"}
    assert "rtsp://" not in manifest.to_json()


class BlockingUntilCancelledAdapter(SyntheticCaptureAdapter):
    def capture(self, endpoint, profile, output_path, relative_path, policy, cancel):  # type: ignore[no-untyped-def]
        del endpoint, profile, output_path, relative_path, policy
        while not cancel.is_set():
            sleep(0.001)
        raise CaptureCancelledError("synthetic cancellation")


def test_cancellation_stops_workers_and_retains_failures(tmp_path: Path) -> None:
    service = CaptureWorkflowService(
        endpoints(), BlockingUntilCancelledAdapter(), CaptureRepository(tmp_path), "test-v1"
    )
    result: list[CaptureSessionManifest] = []
    worker = threading.Thread(
        target=lambda: result.append(
            service.capture_session("cancelled", CapturePolicy(retry_limit=0))
        )
    )
    worker.start()
    sleep(0.02)
    service.cancel()
    worker.join(timeout=2)
    assert not worker.is_alive()
    manifest = result[0]
    assert {item.failure_class for item in manifest.results} == {"CaptureCancelledError"}
