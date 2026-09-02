import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spatial_mapping_phase2.p04_calibration_domain import (
    FrameReviewStatus,
    P04CalibrationError,
)
from spatial_mapping_phase2.p04_calibration_service import (
    CapturedCandidate,
    ImageDimensions,
    P04CalibrationService,
)
from spatial_mapping_phase2.p04_calibration_web import create_p04_calibration_app


class _FakeInspector:
    def inspect(self, path: Path) -> ImageDimensions:
        if "plan" in path.name:
            return ImageDimensions(1000, 1200)
        return ImageDimensions(1920, 1080)


class _FakeCapturer:
    camera_id = "office-cam-03"

    def capture(self) -> CapturedCandidate:
        return CapturedCandidate(
            b"timed-live-frame",
            "2026-08-14T08:30:00Z",
            90000,
            "1/90000",
        )


def _facility_export() -> dict[str, object]:
    return {
        "schema_version": "p02-interactive-export-v1",
        "source_revision": 3,
        "plan": {
            "source_sha256": "a" * 64,
            "image_width_pixels": 1000,
            "image_height_pixels": 1200,
        },
        "facility_frame": {
            "frame_id": "facility-world-interactive-v1",
            "T_world_from_plan_display_pixel": [
                [0.01, 0.0, 1.0],
                [0.0, 0.01, -2.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }


def _initialized_service(tmp_path: Path) -> tuple[P04CalibrationService, Path]:
    export_path = tmp_path / "facility-export.json"
    export_path.write_text(json.dumps(_facility_export()), encoding="utf-8")
    plan_path = tmp_path / "plan.png"
    plan_path.write_bytes(b"synthetic-plan")
    service = P04CalibrationService(tmp_path / "workspace", _FakeInspector())
    state = service.initialize(export_path, plan_path)
    assert state.revision == 0
    assert state.camera_id == "office-cam-03"
    return service, plan_path


def _add_approved_frame(
    service: P04CalibrationService, tmp_path: Path, frame_id: str = "cam03-quiet-001"
) -> Path:
    frame_path = tmp_path / f"{frame_id}.jpg"
    frame_path.write_bytes(f"synthetic-{frame_id}".encode())
    service.add_frame(frame_path, frame_id, "stream-profile-v1")
    service.review_frame(frame_id, FrameReviewStatus.APPROVED, "quiet and sharp")
    return frame_path


def _landmark_payload(frame_id: str = "cam03-quiet-001") -> dict[str, object]:
    return {
        "landmark_id": "door-north-top",
        "name": "Meeting room door north top",
        "physical_meaning": "exact fixed upper door-jamb corner",
        "frame_id": frame_id,
        "image_point": {"u": 1200.0, "v": 240.0},
        "plan_point": {"u": 100.0, "v": 300.0},
        "z_metres": 2.1,
        "z_source": "tape from finished floor",
        "z_uncertainty_metres": None,
        "role": "solve",
    }


def test_linked_landmark_derives_xy_and_allows_optional_uncertainty(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    _add_approved_frame(service, tmp_path)

    state = service.add_landmark(_landmark_payload())

    landmark = state.landmarks[0]
    assert landmark.image_point.to_dict() == {"u": 1200.0, "v": 240.0}
    assert landmark.world_point.to_dict() == pytest.approx(
        {"x_metres": 2.0, "y_metres": 1.0, "z_metres": 2.1}
    )
    assert landmark.z_uncertainty_metres is None
    path, exported = service.export_snapshot()
    assert path.is_file()
    assert exported["status"] == "ready-for-pose-input-review"
    assert exported["role_counts"] == {"solve": 1, "held-out": 0}


def test_d034_validation_is_exported_separately_from_solve_snapshot(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    _add_approved_frame(service, tmp_path)
    service.add_landmark(_landmark_payload())
    first = _landmark_payload()
    first.update({"landmark_id": "d034-v1", "name": "D034 validation 1"})
    first["role"] = "d034-validation"
    first["image_point"] = {"u": 300.0, "v": 400.0}
    second = _landmark_payload()
    second.update({"landmark_id": "d034-v2", "name": "D034 validation 2"})
    second["role"] = "d034-validation"
    second["image_point"] = {"u": 1500.0, "v": 700.0}
    service.add_landmark(first)
    service.add_landmark(second)

    _, solve_export = service.export_snapshot()
    seal_path, seal = service.export_d034_validation_seal()

    assert [item["landmark_id"] for item in solve_export["landmarks"]] == ["door-north-top"]
    assert solve_export["excluded_d034_validation_landmark_ids"] == ["d034-v1", "d034-v2"]
    assert seal_path.parent.name == "validation_seals"
    assert seal["status"] == "sealed-unconsumed"
    assert seal["solve_data_included"] is False
    assert [item["landmark_id"] for item in seal["validation_landmarks"]] == [
        "d034-v1",
        "d034-v2",
    ]


def test_calibration_readiness_requires_current_four_plus_two_export(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    _add_approved_frame(service, tmp_path)
    for index in range(4):
        item = _landmark_payload()
        item["landmark_id"] = f"solve-{index}"
        item["name"] = f"Solve point {index}"
        item["image_point"] = {"u": 300.0 + index * 250, "v": 250.0 + index * 100}
        service.add_landmark(item)
    for index in range(2):
        item = _landmark_payload()
        item["landmark_id"] = f"validation-{index}"
        item["name"] = f"Validation point {index}"
        item["role"] = "d034-validation"
        item["image_point"] = {"u": 500.0 + index * 600, "v": 700.0}
        service.add_landmark(item)

    before_export = service.calibration_readiness()
    assert before_export["calibrate_ready"] is False
    assert "Export linked points" in before_export["reason"]

    path, _ = service.export_snapshot()
    ready = service.calibration_readiness()
    assert ready["calibrate_ready"] is True
    assert ready["current_export_path"] == str(path.resolve())

    service.remove_landmark("validation-1")
    stale = service.calibration_readiness()
    assert stale["calibrate_ready"] is False
    assert stale["current_export_ready"] is False


@pytest.mark.parametrize("camera_id", ["office-cam-01", "office-cam-02", "office-cam-04"])
def test_workspace_supports_p05_camera_ids(tmp_path: Path, camera_id: str) -> None:
    export_path = tmp_path / "facility-export.json"
    export_path.write_text(json.dumps(_facility_export()), encoding="utf-8")
    plan_path = tmp_path / "plan.png"
    plan_path.write_bytes(b"synthetic-plan")
    service = P04CalibrationService(tmp_path / "workspace", _FakeInspector())

    state = service.initialize(export_path, plan_path, camera_id)

    assert state.camera_id == camera_id
    frame_path = tmp_path / "candidate.jpg"
    frame_path.write_bytes(b"candidate")
    state = service.add_frame(frame_path, "candidate-001", "stream-profile-v1")
    assert state.frames[0].camera_id == camera_id


def test_landmark_requires_approved_frame_and_server_selected_xy(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    frame_path = tmp_path / "candidate.jpg"
    frame_path.write_bytes(b"candidate")
    service.add_frame(frame_path, "cam03-quiet-001", "stream-profile-v1")

    with pytest.raises(P04CalibrationError, match="approve a primary frame"):
        service.add_landmark(_landmark_payload())

    assert service.load_state().landmarks == ()


def test_approving_new_frame_supersedes_prior_without_erasing_history(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    _add_approved_frame(service, tmp_path)
    _add_approved_frame(service, tmp_path, "cam03-quiet-002")

    state = service.load_state()
    statuses = {frame.frame_id: frame.status for frame in state.frames}
    assert statuses == {
        "cam03-quiet-001": FrameReviewStatus.SUPERSEDED,
        "cam03-quiet-002": FrameReviewStatus.APPROVED,
    }
    assert len(list((service.workspace / "history").glob("state-r*-*.json"))) == 4


def test_frame_hash_mismatch_is_rejected_without_workspace_mutation(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    frame_path = tmp_path / "candidate.jpg"
    frame_path.write_bytes(b"candidate")

    with pytest.raises(P04CalibrationError, match="does not match expected"):
        service.add_frame(frame_path, "cam03-quiet-001", "stream-profile-v1", "0" * 64)

    assert service.load_state().revision == 0
    assert tuple((service.workspace / "frames").iterdir()) == ()


def test_duplicate_or_out_of_bounds_landmark_is_rejected(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    _add_approved_frame(service, tmp_path)
    service.add_landmark(_landmark_payload())
    duplicate = _landmark_payload()
    duplicate["landmark_id"] = "door-south-top"

    with pytest.raises(P04CalibrationError, match="landmark names must be unique"):
        service.add_landmark(duplicate)

    outside = _landmark_payload()
    outside["landmark_id"] = "outside-point"
    outside["name"] = "Outside point"
    outside["image_point"] = {"u": 1921.0, "v": 10.0}
    with pytest.raises(P04CalibrationError, match="inside its frame"):
        service.add_landmark(outside)

    assert len(service.load_state().landmarks) == 1


def test_tampered_frame_artifact_is_detected(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    _add_approved_frame(service, tmp_path)
    frame = service.load_state().approved_frame
    assert frame is not None
    stored = service.workspace / frame.relative_path
    stored.write_bytes(b"changed")

    with pytest.raises(P04CalibrationError, match="identity changed"):
        service.frame_image_path(frame.frame_id)


def test_missing_z_sources_can_be_bound_in_one_immutable_revision(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    _add_approved_frame(service, tmp_path)
    landmark = _landmark_payload()
    landmark["z_source"] = None
    service.add_landmark(landmark)

    state = service.set_missing_z_sources("Owner-reported laser measurement")

    assert state.revision == 4
    assert state.landmarks[0].z_source == "Owner-reported laser measurement"
    with pytest.raises(P04CalibrationError, match="no landmarks"):
        service.set_missing_z_sources("different source")


def test_user_timed_capture_waits_then_creates_previewable_candidate(tmp_path: Path) -> None:
    delays: list[float] = []
    export_path = tmp_path / "facility-export.json"
    export_path.write_text(json.dumps(_facility_export()), encoding="utf-8")
    plan_path = tmp_path / "plan.png"
    plan_path.write_bytes(b"synthetic-plan")
    service = P04CalibrationService(
        tmp_path / "workspace", _FakeInspector(), sleeper=delays.append
    )
    service.initialize(export_path, plan_path)

    state = service.capture_candidate(3, _FakeCapturer())

    assert delays == [3.0]
    frame = state.frames[0]
    assert frame.status is FrameReviewStatus.CANDIDATE
    assert frame.capture_kind == "user-timed-live"
    assert frame.observed_at_utc == "2026-08-14T08:30:00Z"
    assert frame.source_pts == 90000
    assert service.frame_image_path(frame.frame_id).read_bytes() == b"timed-live-frame"

    with pytest.raises(P04CalibrationError, match="between 0 and 30"):
        service.capture_candidate(31, _FakeCapturer())


def test_live_capture_rejects_endpoint_workspace_camera_mismatch(tmp_path: Path) -> None:
    service, _ = _initialized_service(tmp_path)
    capturer = _FakeCapturer()
    capturer.camera_id = "office-cam-01"

    with pytest.raises(P04CalibrationError, match="does not match workspace"):
        service.capture_candidate(0, capturer)

    assert service.load_state().frames == ()


def test_web_console_supports_review_link_export_and_rejections(tmp_path: Path) -> None:
    app = create_p04_calibration_app(tmp_path / "workspace", _FakeInspector(), _FakeCapturer())
    service = app.state.calibration_service
    export_path = tmp_path / "facility-export.json"
    export_path.write_text(json.dumps(_facility_export()), encoding="utf-8")
    plan_path = tmp_path / "plan.png"
    plan_path.write_bytes(b"synthetic-plan")
    service.initialize(export_path, plan_path)
    frame_path = tmp_path / "quiet.jpg"
    frame_path.write_bytes(b"quiet-frame")
    client = TestClient(app)

    assert "Calibration correspondence console" in client.get("/").text
    assert "Start timed capture" in client.get("/").text
    assert "Zoom camera frame in" in client.get("/").text
    app_script = client.get("/assets/app.js")
    assert app_script.status_code == 200
    assert "linked ${pointCount" in app_script.text
    assert "Workspace history version ${state.revision}" in app_script.text
    assert client.get("/api/plan-image").content == b"synthetic-plan"
    added = client.post(
        "/api/frames",
        json={
            "source_path": str(frame_path),
            "frame_id": "cam03-quiet-001",
            "profile_version": "stream-profile-v1",
            "expected_sha256": hashlib.sha256(b"quiet-frame").hexdigest(),
        },
    )
    assert added.status_code == 200
    timed = client.post("/api/capture-candidate", json={"delay_seconds": 0})
    assert timed.status_code == 200
    assert timed.json()["frames"][-1]["capture_kind"] == "user-timed-live"
    assert client.post("/api/capture-candidate", json={"delay_seconds": 99}).status_code == 422
    rejected_status = client.put(
        "/api/frames/cam03-quiet-001/review", json={"status": "candidate"}
    )
    assert rejected_status.status_code == 422
    approved = client.put(
        "/api/frames/cam03-quiet-001/review",
        json={"status": "approved", "note": "quiet and sharp"},
    )
    assert approved.status_code == 200
    assert client.get("/api/frames/cam03-quiet-001/image").content == b"quiet-frame"
    bad_role = _landmark_payload()
    bad_role["role"] = "training"
    assert client.post("/api/landmarks", json=bad_role).status_code == 422
    linked = client.post("/api/landmarks", json=_landmark_payload())
    assert linked.status_code == 200
    assert linked.json()["derived"] == {
        "approved_frame_id": "cam03-quiet-001",
        "solve_count": 1,
        "held_out_count": 0,
        "d034_validation_count": 0,
    }
    exported = client.post("/api/export")
    assert exported.status_code == 200
    assert exported.json()["export"]["landmarks"][0]["world_point"] == pytest.approx(
        {"x_metres": 2.0, "y_metres": 1.0, "z_metres": 2.1}
    )
    removed = client.delete("/api/landmarks/door-north-top")
    assert removed.status_code == 200
    assert removed.json()["landmarks"] == []
