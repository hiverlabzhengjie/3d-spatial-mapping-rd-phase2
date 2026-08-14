import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spatial_mapping_phase2.p02_interactive_registration import InteractiveRegistrationError
from spatial_mapping_phase2.p02_registration_service import (
    P02RegistrationService,
    RenderedPlan,
)
from spatial_mapping_phase2.p02_registration_web import create_p02_registration_app


class _FakeRenderer:
    def render_first_page(self, _source_pdf: Path, destination_png: Path) -> RenderedPlan:
        destination_png.write_bytes(b"fake-png")
        return RenderedPlan(1000, 1200)


def _service(tmp_path: Path) -> P02RegistrationService:
    return P02RegistrationService(tmp_path / "workspace", tmp_path / ".env", _FakeRenderer())


def _configured_payload(service: P02RegistrationService) -> dict[str, object]:
    payload = service.load_state().to_dict()
    payload["scale_controls"] = [
        {
            "control_id": "known-width",
            "meaning": "printed dimension between permanent corners",
            "point_a": {"u": 0.0, "v": 0.0},
            "point_b": {"u": 100.0, "v": 0.0},
            "distance_metres": 1.0,
            "distance_uncertainty_metres": 0.02,
            "source_kind": "printed-dimension",
        }
    ]
    payload["frame"] = {
        "origin": {"u": 100.0, "v": 100.0},
        "positive_x_handle": {"u": 0.0, "v": 100.0},
        "origin_feature_meaning": "exact permanent corner",
    }
    cameras = payload["cameras"]
    assert isinstance(cameras, list)
    first = cameras[0]
    assert isinstance(first, dict)
    first.update(
        {
            "physical_label": "camera-one",
            "marker": {"u": 0.0, "v": 200.0},
            "mounting_height_metres": 3.0,
            "height_uncertainty_metres": 0.1,
            "reference_meaning": "bracket centre",
        }
    )
    return payload


def test_service_versions_state_and_exports_credential_free_snapshot(tmp_path: Path) -> None:
    service = _service(tmp_path)
    state = service.upload_plan("office.pdf", b"%PDF-1.4\nsynthetic")
    assert state.revision == 0
    assert service.plan_image_path().read_bytes() == b"fake-png"

    stale_payload = _configured_payload(service)
    saved = service.save_state(stale_payload)
    assert saved.revision == 1
    assert len(list((service.workspace / "history").glob("state-r0-*.json"))) == 1
    with pytest.raises(InteractiveRegistrationError, match="stale revision"):
        service.save_state(stale_payload)

    secret = "rtsp://user:password@192.0.2.1/live"
    service.save_endpoint("office-cam-01", secret)
    second_secret = "rtsp://other:password@192.0.2.2/live"
    service.save_endpoint("office-cam-02", second_secret)
    assert service.load_endpoint("office-cam-01") == secret
    assert service.load_endpoint("office-cam-02") == second_secret
    assert service.load_endpoint("office-cam-03") is None
    path, export = service.export_snapshot()
    assert path.is_file()
    assert export["camera_mounting_priors"][0]["endpoint_configured"] is True
    assert secret not in json.dumps(export)
    assert secret in (tmp_path / ".env").read_text(encoding="utf-8")


def test_upload_rejects_non_pdf_without_creating_state(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(InteractiveRegistrationError, match="PDF signature"):
        service.upload_plan("office.pdf", b"not a pdf")

    assert service.has_state() is False


def test_web_api_serves_ui_and_never_returns_endpoint_secret(tmp_path: Path) -> None:
    app = create_p02_registration_app(tmp_path / "workspace", tmp_path / ".env", _FakeRenderer())
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert "Facility Registration Console" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/status").json() == {"has_state": False}
    upload = client.post(
        "/api/plan",
        content=b"%PDF-1.4\nsynthetic",
        headers={"x-filename": "office.pdf", "content-type": "application/pdf"},
    )
    assert upload.status_code == 200
    service = app.state.registration_service
    saved = client.put("/api/state", json=_configured_payload(service))
    assert saved.status_code == 200

    secret = "rtsp://operator:secret@192.0.2.4/live"
    state_before_endpoint_save = client.get("/api/state").json()
    endpoint = client.put("/api/cameras/office-cam-01/endpoint", json={"rtsp_url": secret})
    assert endpoint.json() == {"configured": True}
    assert client.get("/api/cameras/office-cam-01/endpoint").json() == {
        "configured": True,
        "rtsp_url": secret,
    }
    assert client.get("/api/cameras/office-cam-02/endpoint").json() == {
        "configured": False,
        "rtsp_url": "",
    }
    state_response = client.get("/api/state")
    state_after_endpoint_save = state_response.json()
    for key in ("revision", "scale_controls", "frame", "cameras"):
        assert state_after_endpoint_save[key] == state_before_endpoint_save[key]
    export_response = client.post("/api/export")
    assert secret not in state_response.text
    assert secret not in export_response.text
    assert export_response.json()["export"]["camera_mounting_priors"][0][
        "C_world_mount_prior"
    ] == pytest.approx({"x_metres": 1.0, "y_metres": 1.0, "z_metres": 3.0})
