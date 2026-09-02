from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from spatial_mapping_phase2.managed_scene_geocalib import ManagedSceneGeoCalibConfig
from spatial_mapping_phase2.managed_scene_reconstruction import (
    ManagedSceneReconstructionConfig,
)
from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    BoundedJobManager,
    CameraConfig,
    PhaseRecord,
    PhaseState,
    SafeRerunLauncher,
    SceneWorkspace,
    SceneWorkspaceRepository,
    WorkflowService,
)
from spatial_mapping_phase2.xr03_scene_console import (
    SceneRuntime,
    SceneRuntimeManager,
    create_scene_console_app,
)
from spatial_mapping_phase2.xr03_scene_management import SceneRegistry


def scene_workspace(root: Path) -> None:
    scene = SceneWorkspace(
        project_id="phase2",
        scene_id="office",
        display_name="Office spatial mapping",
        artifact_root=(root.parent / "artifacts").resolve(),
        cameras=(
            CameraConfig("office-cam-01", "Office camera 1"),
            CameraConfig("office-cam-02", "Office camera 2"),
        ),
        phases=tuple(PhaseRecord(phase_id, PhaseState.READY, "Ready") for phase_id in PHASE_ORDER),
    )
    SceneWorkspaceRepository(root).create(scene)


def console(tmp_path: Path) -> tuple[SceneRuntimeManager, TestClient, str]:
    workspace = tmp_path / "office-workspace"
    scene_workspace(workspace)
    registry = SceneRegistry(tmp_path / "registry.sqlite3", tmp_path / "managed-scenes")
    existing = registry.register_existing(workspace)
    manager = SceneRuntimeManager(registry, existing.scene_uuid)
    service = WorkflowService(
        SceneWorkspaceRepository(workspace),
        BoundedJobManager(maximum_workers=1, maximum_outstanding_jobs=2),
    )
    manager.register(existing.scene_uuid, SceneRuntime(service, {}))
    return manager, TestClient(create_scene_console_app(manager)), existing.scene_uuid


def test_scene_home_lists_existing_scene_and_scoped_workflow(tmp_path: Path) -> None:
    manager, client, scene_uuid = console(tmp_path)
    try:
        home = client.get("/")
        listing = client.get("/api/scenes")
        workflow = client.get(f"/scenes/{scene_uuid}/api/status")
        scoped_api = client.get(f"/api/scenes/{scene_uuid}/status")
        legacy_api = client.get("/api/status")

        assert home.status_code == 200
        assert "Choose an environment" in home.text
        assert "Source &amp; AGPL-3.0 licence" in home.text
        assert "Internal R&amp;D · no warranty" in home.text
        assert listing.json()["scenes"][0]["display_name"] == "Office spatial mapping"
        assert workflow.json()["scene_id"] == "office"
        assert scoped_api.json()["scene_id"] == "office"
        assert legacy_api.json()["scene_id"] == "office"
    finally:
        manager.close()


def test_create_scene_opens_isolated_draft_with_variable_camera_roster(tmp_path: Path) -> None:
    manager, client, _scene_uuid = console(tmp_path)
    try:
        response = client.post(
            "/api/scenes",
            json={"display_name": "Warehouse", "camera_names": ["Gate", "Packing", "Dock"]},
        )
        created = response.json()["scene"]
        scene_uuid = created["scene_uuid"]

        status = client.get(f"/api/scenes/{scene_uuid}/status")
        facility = client.get(f"/scenes/{scene_uuid}/tools/facility/api/status")
        capture = client.get(f"/scenes/{scene_uuid}/tools/capture/api/cameras")
        capture_page = client.get(f"/scenes/{scene_uuid}/tools/capture/")
        pages = client.get(f"/api/scenes/{scene_uuid}/pages")
        workflow_asset = client.get(f"/scenes/{scene_uuid}/assets/app.js")

        assert response.status_code == 200
        assert created["readiness"] == "draft"
        assert [item["display_name"] for item in status.json()["camera_roster"]] == [
            "Gate",
            "Packing",
            "Dock",
        ]
        assert len(facility.json()["camera_ids"]) == 3
        assert pages.json()["pages"][2]["tool_url"].startswith(f"/scenes/{scene_uuid}/")
        assert pages.json()["pages"][3]["tool_url"].startswith(f"/scenes/{scene_uuid}/")
        assert capture.json() == {
            "cameras": [
                f"scene-{scene_uuid.split('-')[0]}-cam-01",
                f"scene-{scene_uuid.split('-')[0]}-cam-02",
                f"scene-{scene_uuid.split('-')[0]}-cam-03",
            ]
        }
        assert capture_page.status_code == 200
        assert "Check the live camera views, then capture a short session." in capture_page.text
        assert "Office calibration" not in capture_page.text
        assert f"/scenes/{scene_uuid}/tools/capture/api/cameras" in capture_page.text
        assert f"/scenes/{scene_uuid}/tools/capture/api/capture" in capture_page.text
        assert workflow_asset.status_code == 200
        assert "Camera results are unavailable." not in workflow_asset.text

        status_payload = status.json()
        capabilities = status_payload["operator"]["workflow_capabilities"]
        assert {
            capability_id: capability["state"]
            for capability_id, capability in capabilities.items()
        } == {
            "facility": "available",
            "capture": "available",
            "calibration": "not_provisioned",
            "reconstruction": "not_provisioned",
            "floor": "not_provisioned",
            "final_review": "not_provisioned",
            "live_operations": "not_provisioned",
            "scene_updates": "not_provisioned",
        }
        assert capabilities["calibration"] == {
            "state": "not_provisioned",
            "reason_code": "scene-calibration-not-provisioned",
            "message": "Set up calibration inputs for this scene before pose work can begin.",
        }
        assert {
            step["step_id"]: step["state"] for step in status_payload["operator"]["steps"]
        } == {
            "setup": "attention",
            "capture": "attention",
            "calibration": "blocked",
            "reconstruction": "blocked",
            "floor": "blocked",
            "results": "blocked",
        }
        assert status_payload["live_operations"]["blockers"][-1] == (
            "Live operations are not set up for this scene"
        )
    finally:
        manager.close()


def test_managed_scene_mounts_scene_local_calibration_tools_when_configured(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "office-workspace"
    scene_workspace(workspace)
    registry = SceneRegistry(tmp_path / "registry.sqlite3", tmp_path / "managed-scenes")
    existing = registry.register_existing(workspace)
    python = tmp_path / "python.exe"
    python.write_bytes(b"placeholder")
    source = tmp_path / "geocalib"
    torch_home = tmp_path / "torch-home"
    da3_source = tmp_path / "da3"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    torch_home.mkdir()
    da3_source.mkdir()
    checkpoint.mkdir()
    viewer = tmp_path / "rerun-viewer.exe"
    viewer.write_bytes(b"placeholder")
    repository_root = Path(__file__).parents[1]
    manager = SceneRuntimeManager(
        registry,
        existing.scene_uuid,
        ManagedSceneGeoCalibConfig(python, source, torch_home, repository_root),
        ManagedSceneReconstructionConfig(python, repository_root, da3_source, checkpoint),
        SafeRerunLauncher(viewer, (tmp_path,), launch=lambda _arguments: object()),
    )
    service = WorkflowService(
        SceneWorkspaceRepository(workspace),
        BoundedJobManager(maximum_workers=1, maximum_outstanding_jobs=2),
    )
    manager.register(existing.scene_uuid, SceneRuntime(service, {}))
    client = TestClient(create_scene_console_app(manager))
    try:
        created = client.post(
            "/api/scenes",
            json={"display_name": "Calibrate me", "camera_names": ["North", "South"]},
        ).json()["scene"]
        scene_uuid = created["scene_uuid"]

        pages = client.get(f"/api/scenes/{scene_uuid}/pages").json()["pages"]
        calibration = next(item for item in pages if item["page_id"] == "calibration")
        status = client.get(f"/api/scenes/{scene_uuid}/status").json()

        assert len(calibration["calibration_tools"]) == 2
        assert all(
            tool["tool_url"].startswith(f"/scenes/{scene_uuid}/tools/calibration-")
            for tool in calibration["calibration_tools"]
        )
        assert status["operator"]["workflow_capabilities"]["calibration"]["state"] == "available"
        assert status["operator"]["workflow_capabilities"]["reconstruction"]["state"] == (
            "available"
        )
        assert status["operator"]["workflow_capabilities"]["floor"]["state"] == "available"
        assert status["operator"]["workflow_capabilities"]["final_review"]["state"] == (
            "available"
        )
        assert status["operator"]["workflow_capabilities"]["scene_updates"]["state"] == (
            "available"
        )
        assert status["scene_updates"]["available"] is True
        runtime_service = manager.runtime(scene_uuid).service
        reconstruction_adapter = runtime_service.reconstruction_adapter
        assert reconstruction_adapter is not None
        assert reconstruction_adapter.process_resolution == 448
        assert runtime_service.floor_adapter is not None
        assert runtime_service.floor_adapter.contract is None
        assert runtime_service.floor_preview_adapter is not None
        assert runtime_service.rerun_launcher is not None
        assert str(runtime_service.repository.load().artifact_root).lower() in {
            str(root).lower() for root in runtime_service.rerun_launcher.allowed_artifact_roots
        }
    finally:
        manager.close()


def test_archive_hides_scene_without_deleting_workspace(tmp_path: Path) -> None:
    manager, client, _scene_uuid = console(tmp_path)
    try:
        created = client.post(
            "/api/scenes",
            json={"display_name": "Archive me", "camera_names": ["Camera 1"]},
        ).json()["scene"]
        workspace = Path(created["workspace_root"])

        archived = client.patch(
            f"/api/scenes/{created['scene_uuid']}",
            json={"archived": True, "expected_revision": created["revision"]},
        )

        assert archived.json()["scene"]["lifecycle"] == "archived"
        assert workspace.joinpath("scene.json").is_file()
        assert client.get("/api/scenes").json()["scenes"] == [
            item
            for item in client.get("/api/scenes").json()["scenes"]
            if item["scene_uuid"] != created["scene_uuid"]
        ]
    finally:
        manager.close()


def test_old_page_url_redirects_to_default_scene(tmp_path: Path) -> None:
    manager, client, scene_uuid = console(tmp_path)
    try:
        response = client.get("/pages/live", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == f"/scenes/{scene_uuid}/pages/live"
    finally:
        manager.close()


def test_only_scene_entry_records_an_open_event(tmp_path: Path) -> None:
    manager, client, scene_uuid = console(tmp_path)

    def opened_events() -> int:
        with sqlite3.connect(manager.registry.database_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM scene_events WHERE scene_uuid = ? AND action = 'opened'",
                (scene_uuid,),
            ).fetchone()
        assert row is not None
        return int(row[0])

    try:
        initial_count = opened_events()
        assert client.get(f"/scenes/{scene_uuid}/").status_code == 200
        assert opened_events() == initial_count + 1

        assert client.get(f"/scenes/{scene_uuid}/pages/setup").status_code == 200
        assert client.get(f"/scenes/{scene_uuid}/pages/floor").status_code == 200
        assert opened_events() == initial_count + 1
    finally:
        manager.close()
