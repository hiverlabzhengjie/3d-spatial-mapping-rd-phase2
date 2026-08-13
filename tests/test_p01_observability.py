import json
from dataclasses import replace
from pathlib import Path

import pytest

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    CAMERA_IDS,
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    OWNER_INPUT_SCHEMA_VERSION,
    STREAM_PROFILE_SCHEMA_VERSION,
    ApproximatePlanObservation,
    CameraIdentity,
    CameraOwnerInput,
    CapturedDiagnosticArtifact,
    DiagnosticCaptureError,
    DiagnosticCaptureManifest,
    DiagnosticCaptureRequest,
    EndpointConfigurationError,
    LocalRtspEndpoint,
    MountingObservation,
    OwnerInputManifest,
    P01ContractError,
    ReadOnlyObservabilityService,
    StreamPreflightError,
    StreamProbeResult,
    StreamProfile,
    assert_immutable_camera_identities,
    fingerprint_diagnostic_bytes,
    load_local_rtsp_endpoints,
    redact_rtsp_url,
    validate_landmark_inventory,
)

SECRET_ENDPOINT = "rtsp://operator:do-not-disclose@camera.local:554/stream?token=hidden"
UTC_NOW = "2026-08-13T02:03:04Z"


def _endpoint(camera_id: str = "office-cam-01") -> LocalRtspEndpoint:
    return LocalRtspEndpoint(camera_id, CAMERA_ENDPOINT_KEYS[camera_id], SECRET_ENDPOINT)


def _probe() -> StreamProbeResult:
    return StreamProbeResult(
        observed_at=UTC_NOW,
        width_pixels=1920,
        height_pixels=1080,
        codec="h264",
        nominal_fps=20.0,
        observed_fps=19.8,
        time_base="1/90000",
        rotation_degrees=0,
        crop_description="unknown",
        overlay_description="upper-left clock",
        dewarping_indicator="not observed",
        keyframe_behavior="keyframe observed within bounded sample",
        stability_note="no disconnect during bounded sample",
    )


def _owner_manifest(label_suffix: str = "A") -> OwnerInputManifest:
    cameras = tuple(
        CameraOwnerInput(
            identity=CameraIdentity(
                camera_id=camera_id,
                physical_label=f"Camera {index}{label_suffix}",
                endpoint_environment_key=CAMERA_ENDPOINT_KEYS[camera_id],
                stream_profile_version="stream-profile-v1",
            ),
            approximate_plan_position=ApproximatePlanObservation(
                description=f"marked near room {index}",
                uncertainty_note="approximate owner mark; not a world coordinate",
            ),
            mounting=MountingObservation(2.5, 0.05, "owner tape measurement"),
            camera_model=None,
            lens_details=None,
            stream_alteration_confirmation="unknown",
        )
        for index, camera_id in enumerate(CAMERA_IDS, start=1)
    )
    return OwnerInputManifest(OWNER_INPUT_SCHEMA_VERSION, UTC_NOW, cameras)


def _owner_camera_payload(camera: CameraOwnerInput) -> dict[str, object]:
    assert camera.approximate_plan_position is not None
    assert camera.mounting is not None
    return {
        "identity": {
            "camera_id": camera.identity.camera_id,
            "physical_label": camera.identity.physical_label,
            "endpoint_environment_key": camera.identity.endpoint_environment_key,
            "stream_profile_version": camera.identity.stream_profile_version,
        },
        "approximate_plan_position": {
            "description": camera.approximate_plan_position.description,
            "uncertainty_note": camera.approximate_plan_position.uncertainty_note,
        },
        "mounting": {
            "height_metres": camera.mounting.height_metres,
            "uncertainty_metres": camera.mounting.uncertainty_metres,
            "measured_by": camera.mounting.measured_by,
        },
        "camera_model": camera.camera_model,
        "lens_details": camera.lens_details,
        "stream_alteration_confirmation": camera.stream_alteration_confirmation,
    }


def _capture_artifact() -> CapturedDiagnosticArtifact:
    return CapturedDiagnosticArtifact(
        relative_artifact_path="captures/p01/office-cam-01/sample.mp4",
        sha256=fingerprint_diagnostic_bytes(b"sample"),
        source_pts_start_seconds=12.0,
        source_pts_end_seconds=20.0,
        acquisition_started_at=UTC_NOW,
        acquisition_finished_at="2026-08-13T02:03:12Z",
    )


def test_loads_all_four_local_endpoints_without_exposing_values() -> None:
    environment = {
        endpoint_key: f"rtsp://operator:secret{index}@camera{index}.local:554/live"
        for index, endpoint_key in enumerate(CAMERA_ENDPOINT_KEYS.values(), start=1)
    }

    endpoints = load_local_rtsp_endpoints(environment)

    assert tuple(endpoint.camera_id for endpoint in endpoints) == CAMERA_IDS
    assert "secret1" not in repr(endpoints)


def test_missing_or_malformed_endpoint_never_echoes_credential() -> None:
    with pytest.raises(EndpointConfigurationError) as missing_error:
        load_local_rtsp_endpoints({})
    assert SECRET_ENDPOINT not in str(missing_error.value)

    environment = {key: SECRET_ENDPOINT for key in CAMERA_ENDPOINT_KEYS.values()}
    environment["PHASE2_RTSP_CAMERA_2"] = "https://operator:do-not-disclose@wrong.example"
    with pytest.raises(EndpointConfigurationError) as malformed_error:
        load_local_rtsp_endpoints(environment)
    assert "do-not-disclose" not in str(malformed_error.value)


def test_redacts_all_rtsp_credentials_path_and_query() -> None:
    assert redact_rtsp_url(SECRET_ENDPOINT) == "rtsp://camera.local:554"
    assert redact_rtsp_url("not a URL") == "<invalid-rtsp-endpoint>"


def test_owner_manifest_parsing_and_identity_immutability(tmp_path: Path) -> None:
    manifest = _owner_manifest()
    payload = {
        "schema_version": manifest.schema_version,
        "created_at": manifest.created_at,
        "cameras": [_owner_camera_payload(camera) for camera in manifest.cameras],
    }
    path = tmp_path / "p01_owner_input.local.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = OwnerInputManifest.from_json_file(path)
    assert parsed == manifest

    changed_first = replace(
        parsed.cameras[0],
        identity=replace(parsed.cameras[0].identity, physical_label="different physical camera"),
    )
    changed = replace(parsed, cameras=(changed_first, *parsed.cameras[1:]))
    with pytest.raises(P01ContractError, match="immutable"):
        assert_immutable_camera_identities(parsed, changed)


def test_tracked_owner_input_template_validates() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    template = OwnerInputManifest.from_json_file(
        repository_root / "configs" / "p01_owner_input.example.json"
    )

    assert tuple(camera.identity.camera_id for camera in template.cameras) == CAMERA_IDS


def test_owner_manifest_rejects_missing_or_reordered_camera_ids() -> None:
    manifest = _owner_manifest()
    with pytest.raises(P01ContractError, match="fixed ID order"):
        OwnerInputManifest(
            manifest.schema_version,
            manifest.created_at,
            tuple(reversed(manifest.cameras)),
        )


def test_stream_profile_validates_camera_binding_and_observed_values() -> None:
    profile = StreamProfile(
        schema_version=STREAM_PROFILE_SCHEMA_VERSION,
        camera_id="office-cam-01",
        profile_version="stream-profile-v1",
        endpoint_environment_key="PHASE2_RTSP_CAMERA_1",
        observation=_probe(),
    )
    assert profile.observation.width_pixels == 1920
    with pytest.raises(P01ContractError, match="rotation"):
        replace(profile.observation, rotation_degrees=45)
    with pytest.raises(P01ContractError, match="endpoint key"):
        replace(profile, endpoint_environment_key="PHASE2_RTSP_CAMERA_2")


def test_diagnostic_capture_manifest_preserves_timestamp_and_artifact_boundaries() -> None:
    request = DiagnosticCaptureRequest(duration_seconds=10.0, connect_timeout_seconds=3.0)
    manifest = DiagnosticCaptureManifest(
        schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
        capture_id="p01-20260813-0001",
        camera_id="office-cam-01",
        stream_profile_version="stream-profile-v1",
        request=request,
        artifact=_capture_artifact(),
    )

    serialized = manifest.to_sanitized_json()
    assert "do-not-disclose" not in serialized
    assert '"relative_artifact_path": "captures/p01/office-cam-01/sample.mp4"' in serialized
    with pytest.raises(DiagnosticCaptureError, match="at most 15"):
        DiagnosticCaptureRequest(duration_seconds=15.1, connect_timeout_seconds=3.0)
    with pytest.raises(DiagnosticCaptureError, match="below captures"):
        replace(_capture_artifact(), relative_artifact_path="../outside.mp4")


class _FakeAdapter:
    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.calls: list[str] = []

    def probe(self, endpoint_url: str, timeout_seconds: float) -> StreamProbeResult:
        self.calls.append("probe")
        if self.mode == "timeout":
            raise TimeoutError(endpoint_url)
        if self.mode == "disconnect":
            raise ConnectionError(endpoint_url)
        return _probe()

    def capture(
        self, endpoint_url: str, request: DiagnosticCaptureRequest
    ) -> CapturedDiagnosticArtifact:
        self.calls.append("capture")
        if self.mode == "disconnect":
            raise ConnectionError(endpoint_url)
        return _capture_artifact()

    capture_diagnostic = capture


def test_read_only_service_records_successful_probe_and_capture() -> None:
    adapter = _FakeAdapter()
    service = ReadOnlyObservabilityService(adapter)

    assert service.preflight(_endpoint(), 3.0) == _probe()
    assert service.capture(_endpoint(), DiagnosticCaptureRequest(10.0, 3.0)) == _capture_artifact()
    assert adapter.calls == ["probe", "capture"]


@pytest.mark.parametrize(
    "mode, expected_exception, phrase",
    [
        ("timeout", StreamPreflightError, "timed out"),
        ("disconnect", StreamPreflightError, "disconnected"),
    ],
)
def test_read_only_preflight_failures_redact_endpoint(
    mode: str, expected_exception: type[Exception], phrase: str
) -> None:
    service = ReadOnlyObservabilityService(_FakeAdapter(mode))

    with pytest.raises(expected_exception, match=phrase) as error:
        service.preflight(_endpoint(), 3.0)
    assert "do-not-disclose" not in str(error.value)
    assert "PHASE2_RTSP_CAMERA_1" in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_read_only_capture_disconnect_is_safe() -> None:
    service = ReadOnlyObservabilityService(_FakeAdapter("disconnect"))

    with pytest.raises(DiagnosticCaptureError, match="disconnected") as error:
        service.capture(_endpoint(), DiagnosticCaptureRequest(10.0, 3.0))
    assert "do-not-disclose" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_landmark_inventory_requires_distinct_solving_and_held_out_candidates() -> None:
    validate_landmark_inventory(["door corner", "beam edge"], ["window mullion"])
    with pytest.raises(P01ContractError, match="must not overlap"):
        validate_landmark_inventory(["door corner"], ["door corner"])
