from __future__ import annotations

import hashlib
from pathlib import Path

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
                service, {"p02": legacy, "p04": legacy, "p05": legacy}
            )
        )
        assert client.get("/").status_code == 200
        assert client.get("/pages/geometry").status_code == 200
        assert client.get("/pages/not-a-page").status_code == 404
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/unknown.js").status_code == 404
        pages = client.get("/api/pages").json()["pages"]
        assert len(pages) == 7
        assert next(page for page in pages if page["page_id"] == "facility")["tool_url"] == (
            "/tools/p02/"
        )
        calibration = next(page for page in pages if page["page_id"] == "calibration")
        assert [tool["tool_id"] for tool in calibration["tools"]] == ["p04", "p05"]
        assert client.get("/tools/p02/").json() == {"legacy": True}
        assert client.get("/tools/p05/").json() == {"legacy": True}
        rewritten = client.get("/tools/p02/absolute")
        assert '/tools/p02/assets/app.js' in rewritten.text
        assert '/tools/p02/api/image' in rewritten.text
        assert '/tools/p02/api/status' in client.get(
            "/tools/p02/assets/app.js"
        ).text
        status = client.get("/api/status").json()
        assert status["scene_id"] == "web-scene"
        assert len(status["camera_roster"]) == 3
        assert "rtsp://" not in client.get("/api/status").text.lower()
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
