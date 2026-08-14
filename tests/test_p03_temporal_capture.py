from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path

import pytest

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    CAMERA_IDS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p03_capture_domain import P03ContractError
from spatial_mapping_phase2.p03_capture_service import CapturePolicy
from spatial_mapping_phase2.p03_temporal_capture import (
    BufferedFrame,
    SyntheticWarmFrameAdapter,
    TemporalAuthorityStatus,
    TemporalBundleRepository,
    WarmTemporalCaptureService,
)


def endpoints() -> tuple[LocalRtspEndpoint, ...]:
    return tuple(
        LocalRtspEndpoint(camera_id, CAMERA_ENDPOINT_KEYS[camera_id], "rtsp://fixture/live")
        for camera_id in CAMERA_IDS
    )


def service(tmp_path: Path, timestamps: dict[str, tuple[int, ...]]) -> WarmTemporalCaptureService:
    return WarmTemporalCaptureService(
        endpoints(),
        SyntheticWarmFrameAdapter(timestamps),
        TemporalBundleRepository(tmp_path),
        CapturePolicy(read_timeout_seconds=0.1, initial_backoff_seconds=0.001),
        "test-v1",
        ring_capacity=8,
        clock_resolution_ns=1_000_000,
    )


def test_warm_buffers_remove_late_fallback_and_authorize_within_window(tmp_path: Path) -> None:
    timestamps: dict[str, tuple[int, ...]] = {
        camera_id: (1_000_000_000 + index * 5_000_000, 1_040_000_000 + index * 5_000_000)
        for index, camera_id in enumerate(CAMERA_IDS)
    }
    collector = service(tmp_path, timestamps)
    collector.start()
    assert collector.wait_until_ready(1.0)
    manifest = collector.capture("authoritative-1", max_skew_ns=20_000_000)
    collector.close()
    assert manifest.authority_status is TemporalAuthorityStatus.AUTHORITATIVE
    assert manifest.overall_skew_ns == 15_000_000
    assert manifest.conservative_overall_skew_ns == 17_000_000
    assert manifest.clock_domain == "host-monotonic-acquisition"
    assert len(manifest.artifacts) == 4
    assert all((tmp_path / artifact.relative_path).is_file() for artifact in manifest.artifacts)


def test_outside_window_is_persisted_but_rejected(tmp_path: Path) -> None:
    timestamps: dict[str, tuple[int, ...]] = {
        CAMERA_IDS[0]: (1_000_000_000,),
        CAMERA_IDS[1]: (1_010_000_000,),
        CAMERA_IDS[2]: (1_020_000_000,),
        CAMERA_IDS[3]: (1_500_000_000,),
    }
    collector = service(tmp_path, timestamps)
    collector.start()
    assert collector.wait_until_ready(1.0)
    manifest = collector.capture("rejected-1", max_skew_ns=100_000_000)
    collector.close()
    assert manifest.authority_status is TemporalAuthorityStatus.REJECTED_SKEW
    assert manifest.overall_skew_ns == 500_000_000
    assert manifest.conservative_overall_skew_ns == 502_000_000


def test_selection_uses_closest_cross_camera_combination_deterministically(tmp_path: Path) -> None:
    timestamps: dict[str, tuple[int, ...]] = {
        CAMERA_IDS[0]: (100, 1_000),
        CAMERA_IDS[1]: (110, 1_100),
        CAMERA_IDS[2]: (120, 1_200),
        CAMERA_IDS[3]: (130, 5_000),
    }
    collector = service(tmp_path, timestamps)
    collector.start()
    assert collector.wait_until_ready(1.0)
    manifest = collector.capture("closest-1", max_skew_ns=50)
    collector.close()
    assert [artifact.acquisition_monotonic_ns for artifact in manifest.artifacts] == [
        100,
        110,
        120,
        130,
    ]
    assert manifest.bundle_acquisition_monotonic_ns == 110
    assert manifest.timestamp_uncertainty_ns == 1_000_020


def test_clock_quantization_can_reject_zero_measured_skew(tmp_path: Path) -> None:
    timestamps: dict[str, tuple[int, ...]] = {
        camera_id: (1_000_000_000,) for camera_id in CAMERA_IDS
    }
    collector = WarmTemporalCaptureService(
        endpoints(),
        SyntheticWarmFrameAdapter(timestamps),
        TemporalBundleRepository(tmp_path),
        CapturePolicy(read_timeout_seconds=0.1, initial_backoff_seconds=0.001),
        "test-v1",
        clock_resolution_ns=15_625_000,
    )
    collector.start()
    assert collector.wait_until_ready(1.0)
    manifest = collector.capture("clock-rejected", max_skew_ns=30_000_000)
    collector.close()
    assert manifest.overall_skew_ns == 0
    assert manifest.conservative_overall_skew_ns == 31_250_000
    assert manifest.authority_status is TemporalAuthorityStatus.REJECTED_SKEW


class TransientWarmAdapter(SyntheticWarmFrameAdapter):
    def __init__(self, timestamps: dict[str, tuple[int, ...]]) -> None:
        super().__init__(timestamps)
        self.calls: dict[str, int] = {}

    def frames(
        self,
        endpoint: LocalRtspEndpoint,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> Iterable[BufferedFrame]:
        call = self.calls.get(endpoint.camera_id, 0) + 1
        self.calls[endpoint.camera_id] = call
        if endpoint.camera_id == CAMERA_IDS[3] and call == 1:
            raise TimeoutError("synthetic transient read failure")
        yield from super().frames(endpoint, policy, cancel)


def test_warm_worker_recovers_transient_camera_failure(tmp_path: Path) -> None:
    timestamps: dict[str, tuple[int, ...]] = {
        camera_id: (1_000_000_000,) for camera_id in CAMERA_IDS
    }
    adapter = TransientWarmAdapter(timestamps)
    collector = WarmTemporalCaptureService(
        endpoints(),
        adapter,
        TemporalBundleRepository(tmp_path),
        CapturePolicy(read_timeout_seconds=0.1, initial_backoff_seconds=0.001),
        "test-v1",
        clock_resolution_ns=1_000_000,
    )
    collector.start()
    assert collector.wait_until_ready(1.0)
    assert adapter.calls[CAMERA_IDS[3]] == 2
    assert collector.status()[CAMERA_IDS[3]]["failure_class"] is None
    collector.close()


def test_invalid_bundle_id_writes_nothing(tmp_path: Path) -> None:
    timestamps: dict[str, tuple[int, ...]] = {
        camera_id: (1_000_000_000,) for camera_id in CAMERA_IDS
    }
    collector = service(tmp_path, timestamps)
    collector.start()
    assert collector.wait_until_ready(1.0)
    with pytest.raises(P03ContractError, match="bundle_id"):
        collector.capture("../unsafe", max_skew_ns=10_000_000)
    collector.close()
    assert not (tmp_path / "captures" / "p03" / "unsafe").exists()
