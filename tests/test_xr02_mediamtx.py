from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS
from spatial_mapping_phase2.xr02_mediamtx import (
    MediaMtxGateway,
    MediaMtxGatewayError,
    MediaMtxGatewayPolicy,
    _child_environment,
    _credential_safe_config,
    _parse_path_metrics,
    _sanitize_log_line,
)


def _endpoints(secret: str = "secret-value") -> tuple[LocalRtspEndpoint, ...]:
    return tuple(
        LocalRtspEndpoint(
            camera_id,
            CAMERA_ENDPOINT_KEYS[camera_id],
            f"rtsp://user:{secret}@192.0.2.{index}/stream",
        )
        for index, camera_id in enumerate(CAMERA_IDS, start=1)
    )


def test_config_is_credential_free_and_disables_filler_and_on_demand(tmp_path: Path) -> None:
    policy = MediaMtxGatewayPolicy(tmp_path / "mediamtx.exe")
    config = _credential_safe_config(policy)
    assert "rtsp://" not in config.lower()
    assert "sourceOnDemand: false" in config
    assert "alwaysAvailable: false" in config
    assert "rtspTransport: tcp" in config
    assert all(f"officecam{index:02d}" in config for index in range(1, 5))


def test_upstream_secrets_exist_only_in_child_environment(tmp_path: Path) -> None:
    endpoints = _endpoints()
    environment = _child_environment(endpoints)
    assert environment["MTX_PATHS_OFFICECAM01_SOURCE"].startswith("rtsp://user:secret-value@")
    gateway = MediaMtxGateway(
        endpoints,
        MediaMtxGatewayPolicy(tmp_path / "mediamtx.exe"),
        tmp_path / "runtime",
    )
    evidence = gateway.evidence()
    assert "secret-value" not in json.dumps(evidence)
    local = gateway.local_endpoints()
    assert tuple(item.camera_id for item in local) == CAMERA_IDS
    assert local[0].for_read_only_adapter() == "rtsp://127.0.0.1:8554/officecam01"


def test_logs_and_metrics_are_reduced_to_credential_safe_health() -> None:
    assert (
        _sanitize_log_line("failed rtsp://user:secret@camera.local/private-path now")
        == "failed <redacted-rtsp-endpoint> now"
    )
    metrics = _parse_path_metrics(
        "\n".join(
            (
                'paths{name="officecam01",state="ready"} 1',
                'paths_inbound_bytes{name="officecam01",state="ready"} 1234',
                'paths_inbound_frames_in_error{name="officecam01",state="ready"} 2',
                'paths_readers{name="officecam01",state="ready",readerType="rtspSession"} 2',
                'paths{name="unrelated",state="ready"} 1',
            )
        )
    )
    assert metrics == {
        "officecam01": {
            "state": "ready",
            "inbound_bytes": "1234",
            "inbound_frame_errors": "2",
            "reader_count": "2",
        }
    }


def test_policy_rejects_ambiguous_ports_and_bad_identity(tmp_path: Path) -> None:
    with pytest.raises(MediaMtxGatewayError, match="distinct"):
        MediaMtxGatewayPolicy(tmp_path / "mediamtx.exe", rtsp_port=8554, api_port=8554)
    with pytest.raises(MediaMtxGatewayError, match="SHA-256"):
        MediaMtxGatewayPolicy(tmp_path / "mediamtx.exe", expected_executable_sha256="bad")
