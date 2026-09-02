"""Reusable construction helpers for the maintained P03 capture runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from spatial_mapping_phase2.p01_observability import load_local_rtsp_endpoints
from spatial_mapping_phase2.p03_capture_service import (
    CapturePolicy,
    CaptureRepository,
    CaptureWorkflowService,
)
from spatial_mapping_phase2.p03_pyav_adapter import PyAvCaptureAdapter
from spatial_mapping_phase2.p03_temporal_capture import (
    TemporalBundleRepository,
    WarmTemporalCaptureService,
)


def build_capture_service(environment: Mapping[str, str]) -> CaptureWorkflowService:
    root = Path(environment["PHASE2_ARTIFACT_ROOT"])
    return CaptureWorkflowService(
        load_local_rtsp_endpoints(environment),
        PyAvCaptureAdapter(),
        CaptureRepository(root),
        "spatial-mapping-phase2-p03-v1",
    )


def build_temporal_capture_service(
    environment: Mapping[str, str],
) -> WarmTemporalCaptureService:
    root = Path(environment["PHASE2_ARTIFACT_ROOT"])
    return WarmTemporalCaptureService(
        load_local_rtsp_endpoints(environment),
        PyAvCaptureAdapter(),
        TemporalBundleRepository(root),
        CapturePolicy(duration_seconds=2.0, read_timeout_seconds=5.0),
        "spatial-mapping-phase2-p03-temporal-v1",
    )
