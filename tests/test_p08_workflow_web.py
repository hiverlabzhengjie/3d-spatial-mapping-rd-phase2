from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.testclient import TestClient

from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    ArtifactReference,
    BoundedJobManager,
    CameraConfig,
    PhaseRecord,
    PhaseState,
    SafeRerunLauncher,
    SceneWorkspace,
    SceneWorkspaceRepository,
    WorkflowService,
)
from spatial_mapping_phase2.p08_workflow_web import create_p08_workflow_app


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _service(tmp_path: Path) -> tuple[WorkflowService, BoundedJobManager, list[object]]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    recording = artifact_root / "selected.rrd"
    recording.write_bytes(b"rerun")
    past_recording = artifact_root / "past.rrd"
    past_recording.write_bytes(b"past-rerun")
    phases = tuple(
        PhaseRecord(
            phase_id=phase_id,
            state=PhaseState.READY if phase_id == "P08" else PhaseState.PROVISIONAL,
            message=f"{phase_id} status",
            prerequisites=() if index == 0 else (PHASE_ORDER[index - 1],),
            artifact_ids=("selected-rerun",) if phase_id == "P07" else (),
        )
        for index, phase_id in enumerate(PHASE_ORDER)
    )
    scene = SceneWorkspace(
        project_id="web-project",
        scene_id="web-scene",
        display_name="Web scene",
        artifact_root=artifact_root.resolve(),
        cameras=(
            CameraConfig("camera-a", "Camera A", "RTSP_A"),
            CameraConfig("camera-b", "Camera B", "RTSP_B"),
            CameraConfig("camera-c", "Camera C", None, False),
        ),
        phases=phases,
        artifacts=(
            ArtifactReference(
                "selected-rerun",
                "P07",
                "rerun-recording",
                recording.resolve(),
                _sha256(recording),
                "provisional selected inspection",
                True,
            ),
            ArtifactReference(
                "past-rerun",
                "P07",
                "rerun-recording",
                past_recording.resolve(),
                _sha256(past_recording),
                "past inspection",
                False,
            ),
        ),
    )
    repository = SceneWorkspaceRepository(tmp_path / "workspace")
    repository.create(scene)
    viewer = tmp_path / "viewer.exe"
    viewer.write_bytes(b"viewer")
    launches: list[object] = []
    launcher = SafeRerunLauncher(
        viewer,
        (artifact_root,),
        lambda arguments: launches.append(arguments),
    )
    jobs = BoundedJobManager()
    return WorkflowService(repository, jobs, rerun_launcher=launcher), jobs, launches


def test_integrated_pages_status_assets_and_legacy_mount(tmp_path: Path) -> None:
    service, jobs, _launches = _service(tmp_path)
    legacy = FastAPI()

    @legacy.get("/")
    async def legacy_index() -> dict[str, bool]:
        return {"legacy": True}

    @legacy.get("/absolute", response_class=HTMLResponse)
    async def absolute_index() -> str:
        return '<script src="/assets/app.js"></script><img src="/api/image">'

    @legacy.get("/assets/app.js")
    async def absolute_script() -> Response:
        return Response('fetch("/api/status")', media_type="application/javascript")

    try:
        client = TestClient(
            create_p08_workflow_app(
                service,
                {
                    "facility": legacy,
                    "calibration-camera-a": legacy,
                    "calibration-camera-b": legacy,
                },
            )
        )
        assert client.get("/").status_code == 200
        assert client.get("/pages/floor").status_code == 200
        assert client.get("/pages/not-a-page").status_code == 404
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/unknown.js").status_code == 404
        pages = client.get("/api/pages").json()["pages"]
        assert len(pages) == 8
        assert any(page["page_id"] == "artifacts" for page in pages)
        assert next(page for page in pages if page["page_id"] == "facility")["tool_url"] == (
            "/tools/facility/"
        )
        calibration = next(page for page in pages if page["page_id"] == "calibration")
        assert [tool["camera_id"] for tool in calibration["calibration_tools"]] == [
            "camera-a",
            "camera-b",
        ]
        assert client.get("/tools/facility/").json() == {"legacy": True}
        assert client.get("/tools/calibration-camera-b/").json() == {"legacy": True}
        rewritten = client.get("/tools/facility/absolute")
        assert "/tools/facility/assets/app.js" in rewritten.text
        assert "/tools/facility/api/image" in rewritten.text
        assert "/tools/facility/api/status" in client.get("/tools/facility/assets/app.js").text
        status = client.get("/api/status").json()
        assert status["scene_id"] == "web-scene"
        assert len(status["camera_roster"]) == 3
        assert "rtsp://" not in client.get("/api/status").text.lower()
        shell = client.get("/assets/app.js").text.lower()
        assert "operator message" not in shell
        assert "artifacts & authority" not in shell
        assert "phase_id" not in shell
        assert "scene history & storage" in shell
        assert "permanent means permanent" in shell
        assert "type this phrase" not in shell
        assert "5 newest shown" in shell
        assert "delete selected files" in shell
        catalog = client.get("/api/artifacts")
        assert catalog.status_code == 200
        assert catalog.json()["scene_id"] == "web-scene"
        geometry_review = next(
            milestone
            for milestone in catalog.json()["milestones"]
            if milestone["milestone_key"] == "geometry-review"
        )
        assert geometry_review["selected_artifact_id"] == "selected-rerun"
        selected_version = next(
            version
            for version in geometry_review["versions"]
            if version["artifact_id"] == "selected-rerun"
        )
        assert selected_version["retention"]["class"] == "accepted-predecessor"
        assert catalog.json()["storage"]["protected_version_count"] == 1
        protected_impact = client.get("/api/artifacts/selected-rerun/delete-impact")
        assert protected_impact.status_code == 200
        assert protected_impact.json()["allowed"] is False
        assert protected_impact.json()["deletion_token"] is None
        assert protected_impact.json()["protected_retention"]["class"] == ("accepted-predecessor")
        protected_batch = client.post(
            "/api/artifacts/delete-impact",
            json={"artifact_ids": ["selected-rerun"]},
        )
        assert protected_batch.status_code == 200
        assert protected_batch.json()["all_allowed"] is False
        refused_delete = client.post(
            "/api/artifacts/delete",
            json={
                "action_id": "refuse-protected-authority",
                "artifact_id": "selected-rerun",
                "deletion_token": "not-issued",
            },
        )
        assert refused_delete.status_code == 422
        assert Path(selected_version["path"]).is_file()
        verified = client.post(
            "/api/artifacts/verify",
            json={"action_id": "verify-selected", "artifact_id": "selected-rerun"},
        )
        assert verified.status_code == 200
        assert verified.json()["lifecycle"] == "available"
        protected = client.post(
            "/api/artifacts/archive",
            json={
                "action_id": "archive-selected",
                "artifact_id": "selected-rerun",
                "archived": True,
            },
        )
        assert protected.status_code == 422
        assert Path(geometry_review["versions"][0]["path"]).is_file()

        past = next(
            version
            for version in geometry_review["versions"]
            if version["artifact_id"] == "past-rerun"
        )
        deletion_impact = client.get("/api/artifacts/past-rerun/delete-impact")
        assert deletion_impact.status_code == 200
        assert deletion_impact.json()["allowed"] is True
        stale_check = client.post(
            "/api/artifacts/delete",
            json={
                "action_id": "delete-past-wrong",
                "artifact_id": "past-rerun",
                "deletion_token": "wrong-token",
            },
        )
        assert stale_check.status_code == 422
        assert Path(past["path"]).is_file()
        deleted = client.post(
            "/api/artifacts/delete",
            json={
                "action_id": "delete-past-exact",
                "artifact_id": "past-rerun",
                "deletion_token": deletion_impact.json()["deletion_token"],
            },
        )
        assert deleted.status_code == 200
        assert deleted.json()["recoverable_from_console"] is False
        assert not Path(past["path"]).exists()
        after_delete = client.get("/api/artifacts").json()
        assert after_delete["storage"]["deleted_version_count"] == 1
        assert all(
            version["artifact_id"] != "past-rerun"
            for section in after_delete["workflow_sections"]
            for version in section["past_items"]
        )
    finally:
        jobs.close()


def test_batch_deletion_routes_preview_then_remove_selected_past_file(tmp_path: Path) -> None:
    service, jobs, _launches = _service(tmp_path)
    try:
        client = TestClient(create_p08_workflow_app(service))
        preview = client.post(
            "/api/artifacts/delete-impact",
            json={"artifact_ids": ["past-rerun"]},
        )
        assert preview.status_code == 200
        assert preview.json()["all_allowed"] is True
        item = preview.json()["items"][0]
        deleted = client.post(
            "/api/artifacts/delete-batch",
            json={
                "action_id": "delete-past-batch",
                "items": [
                    {
                        "artifact_id": item["artifact_id"],
                        "deletion_token": item["deletion_token"],
                    }
                ],
            },
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted_artifact_count"] == 1
        assert not (tmp_path / "artifacts" / "past.rrd").exists()
    finally:
        jobs.close()


def test_web_actions_share_service_and_reject_unconfigured_floor_and_bad_json(
    tmp_path: Path,
) -> None:
    service, jobs, launches = _service(tmp_path)
    try:
        client = TestClient(create_p08_workflow_app(service))
        floor = client.post("/api/jobs/floor", json={"job_id": "floor-one"})
        assert floor.status_code == 422
        assert "not configured" in floor.json()["detail"]
        malformed = client.post(
            "/api/rerun/launch",
            content="not-json",
            headers={"content-type": "application/json"},
        )
        assert malformed.status_code == 422
        launched = client.post(
            "/api/rerun/launch",
            json={"action_id": "launch-web", "artifact_id": "selected-rerun"},
        )
        assert launched.status_code == 200
        assert launched.json()["status"] == "launched"
        assert len(launches) == 1
        duplicate = client.post(
            "/api/rerun/launch",
            json={"action_id": "launch-web", "artifact_id": "selected-rerun"},
        )
        assert duplicate.status_code == 422
        assert len(launches) == 1
    finally:
        jobs.close()


def test_operator_actions_gate_floor_and_support_reconstruction_review(tmp_path: Path) -> None:
    service, jobs, launches = _service(tmp_path)

    class Cameras:
        def summaries(self, camera_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
            return tuple(
                {"camera_id": camera_id, "ready": True, "status": "Ready"}
                for camera_id in camera_ids
            )

    class Reconstruction:
        def readiness_errors(self) -> tuple[str, ...]:
            return ()

        def run(self, job_id: str, _cancel_event: object) -> dict[str, object]:
            combined = tmp_path / f"{job_id}.npz"
            rerun = tmp_path / "artifacts" / f"{job_id}.rrd"
            combined.write_bytes(b"combined")
            rerun.write_bytes(b"rerun")
            return {
                "output_directory": str(tmp_path / job_id),
                "combined_geometry": {
                    "path": str(combined.resolve()),
                    "sha256": _sha256(combined),
                    "point_count": 2,
                },
                "rerun": {
                    "path": str(rerun.resolve()),
                    "sha256": _sha256(rerun),
                    "byte_count": rerun.stat().st_size,
                },
            }

        def ensure_geometry_review(
            self, _run_directory: Path, _cancel_event: object
        ) -> dict[str, object]:
            review = tmp_path / "artifacts" / "reconstruction-one-review.rrd"
            review.write_bytes(b"camera-rich-review")
            return {
                "rerun": {
                    "path": str(review.resolve()),
                    "sha256": _sha256(review),
                    "byte_count": review.stat().st_size,
                }
            }

    class Floor:
        def run(self, job_id: str, _cancel_event: object) -> dict[str, str]:
            return {"output_directory": str(tmp_path / job_id)}

    service.operator_config = SimpleNamespace(
        geometry_rerun_artifact_id="selected-rerun",
        floor_rerun_artifact_id="missing-floor",
        geometry_approved=False,
        floor_approved=False,
    )
    service.camera_summary_adapter = Cameras()
    service.reconstruction_adapter = Reconstruction()
    service.floor_adapter = Floor()  # type: ignore[assignment]
    try:
        client = TestClient(create_p08_workflow_app(service))
        blocked = client.post("/api/jobs/floor", json={"job_id": "blocked-floor"})
        assert blocked.status_code == 422
        assert "Approve" in blocked.json()["detail"]

        reconstruction = client.post(
            "/api/jobs/reconstruction", json={"job_id": "reconstruction-one"}
        )
        assert reconstruction.status_code == 200
        for _ in range(100):
            job = client.get("/api/jobs/reconstruction-one").json()
            if job["state"] == "complete":
                break
            time.sleep(0.01)
        assert job["state"] == "complete"
        reconstructed_status = client.get("/api/status").json()["operator"]
        assert reconstructed_status["geometry"]["artifact_id"] == "reconstruction-one"
        assert reconstructed_status["geometry"]["available"] is True
        assert reconstructed_status["geometry"]["approved"] is False

        opened = client.post(
            "/api/rerun/launch",
            json={"action_id": "open-geometry-one", "artifact_id": "reconstruction-one"},
        )
        assert opened.status_code == 200
        assert launches
        assert "reconstruction-one-review.rrd" in str(launches[-1])
        review_artifact_id = client.get("/api/status").json()["operator"]["geometry"][
            "artifact_id"
        ]
        assert review_artifact_id != "reconstruction-one"
        approved = client.post(
            "/api/approve",
            json={"action_id": "approve-geometry-one", "target": "geometry"},
        )
        assert approved.status_code == 200
        assert client.get("/api/status").json()["operator"]["floor"]["can_generate"]

        restarted_jobs = BoundedJobManager()
        try:
            restarted = WorkflowService(
                service.repository,
                restarted_jobs,
                rerun_launcher=service.rerun_launcher,
                operator_config=service.operator_config,
                camera_summary_adapter=service.camera_summary_adapter,
                reconstruction_adapter=service.reconstruction_adapter,
                floor_adapter=service.floor_adapter,
            )
            restarted_client = TestClient(create_p08_workflow_app(restarted))
            restored = restarted_client.get("/api/status").json()["operator"]
            assert restored["geometry"]["artifact_id"] == review_artifact_id
            assert restored["geometry"]["approved"] is True
            fresh = restarted_client.post(
                "/api/session/fresh",
                json={"action_id": "archive-current", "session_id": "demo-two"},
            )
            assert fresh.status_code == 200
            reset = restarted_client.get("/api/status").json()
            assert reset["operator"]["geometry"]["available"] is False
            assert reset["operator"]["geometry"]["approved"] is False
            assert reset["jobs"] == []
            assert (service.repository.operator_state_archive / "archive-current.json").is_file()
        finally:
            restarted_jobs.close()
    finally:
        jobs.close()


def test_operator_pipeline_rejects_duplicate_action_while_active(tmp_path: Path) -> None:
    service, jobs, _launches = _service(tmp_path)
    started = threading.Event()
    release = threading.Event()

    class Cameras:
        def summaries(self, camera_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
            return tuple({"camera_id": item, "ready": True} for item in camera_ids)

    class Reconstruction:
        def readiness_errors(self) -> tuple[str, ...]:
            return ()

        def run(self, job_id: str, _cancel_event: object) -> dict[str, object]:
            started.set()
            release.wait(2.0)
            combined = tmp_path / f"{job_id}.npz"
            rerun = tmp_path / "artifacts" / f"{job_id}.rrd"
            combined.write_bytes(b"combined")
            rerun.write_bytes(b"rerun")
            return {
                "combined_geometry": {
                    "path": str(combined.resolve()),
                    "sha256": _sha256(combined),
                },
                "rerun": {
                    "path": str(rerun.resolve()),
                    "sha256": _sha256(rerun),
                    "byte_count": rerun.stat().st_size,
                },
            }

    service.operator_config = SimpleNamespace(
        geometry_rerun_artifact_id="selected-rerun",
        floor_rerun_artifact_id="missing-floor",
        geometry_approved=False,
        floor_approved=False,
    )
    service.camera_summary_adapter = Cameras()
    service.reconstruction_adapter = Reconstruction()
    try:
        client = TestClient(create_p08_workflow_app(service))
        first = client.post("/api/jobs/reconstruction", json={"job_id": "reconstruction-active"})
        assert first.status_code == 200
        assert started.wait(1.0)
        duplicate = client.post(
            "/api/jobs/reconstruction", json={"job_id": "reconstruction-duplicate"}
        )
        assert duplicate.status_code == 422
        assert "active workflow action" in duplicate.json()["detail"]
    finally:
        release.set()
        jobs.close()
