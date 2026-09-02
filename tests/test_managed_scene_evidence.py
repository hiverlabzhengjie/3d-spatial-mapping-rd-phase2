from __future__ import annotations

import hashlib
import json
from pathlib import Path

from spatial_mapping_phase2.managed_scene_capture import (
    SCENE_CAPTURE_PROFILE_SCHEMA,
    SCENE_CAPTURE_SESSION_SCHEMA,
    SceneCameraBinding,
    SceneCameraCaptureResult,
    SceneCaptureArtifact,
    SceneCaptureRepository,
    SceneCaptureSessionManifest,
    SceneCaptureStatus,
    SceneRationalTimeBase,
    SceneSourceFrame,
    SceneStorageMode,
    SceneStreamProfile,
    select_scene_capture_bundle,
)
from spatial_mapping_phase2.managed_scene_evidence import ManagedSceneEvidenceCoordinator
from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    BoundedJobManager,
    CameraConfig,
    PhaseRecord,
    PhaseState,
    SceneWorkspace,
    SceneWorkspaceRepository,
    WorkflowService,
)


def _write_facility(workspace: Path, camera_ids: tuple[str, ...]) -> Path:
    source_sha256 = "a" * 64
    (workspace / "exports").mkdir(parents=True)
    (workspace / "display").mkdir()
    (workspace / "sources").mkdir()
    (workspace / "display" / f"{source_sha256}-page-01.png").write_bytes(b"plan-image")
    (workspace / "sources" / f"{source_sha256}.pdf").write_bytes(b"floor-plan")
    path = workspace / "exports" / "facility-r1.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "p02-interactive-export-v1",
                "source_revision": 1,
                "plan": {
                    "source_sha256": source_sha256,
                    "original_filename": "floor-plan.pdf",
                    "image_width_pixels": 100,
                    "image_height_pixels": 80,
                },
                "facility_frame": {
                    "T_world_from_plan_display_pixel": [
                        [0.01, 0.0, 0.0],
                        [0.0, 0.01, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                },
                "camera_mounting_priors": [
                    {"camera_id": camera_id, "C_world_mount_prior": [0.0, 0.0, 3.0]}
                    for camera_id in camera_ids
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_bundle(
    repository: SceneCaptureRepository,
    scene_id: str,
    camera_ids: tuple[str, ...],
    captured_ids: tuple[str, ...],
    *,
    session_id: str = "session-1",
    bundle_id: str = "bundle-1",
) -> None:
    repository.create_session_directory(session_id)
    endpoint_keys = {
        camera_id: f"SCENE_TEST_CAMERA_{index + 1:02d}_RTSP"
        for index, camera_id in enumerate(camera_ids)
    }
    bindings = tuple(
        SceneCameraBinding(camera_id, endpoint_keys[camera_id]) for camera_id in camera_ids
    )
    results: list[SceneCameraCaptureResult] = []
    for index, camera_id in enumerate(camera_ids):
        profile = SceneStreamProfile(
            SCENE_CAPTURE_PROFILE_SCHEMA,
            camera_id,
            "scene-stream-profile-v1",
            endpoint_keys[camera_id],
            f"local-{camera_id}",
            640,
            360,
            "h264",
            SceneRationalTimeBase(1, 90_000),
            None,
            None,
            "2026-09-01T00:00:00Z",
        )
        if camera_id not in captured_ids:
            results.append(
                SceneCameraCaptureResult(
                    camera_id,
                    profile,
                    SceneCaptureStatus.FAILED,
                    (),
                    None,
                    (),
                    "ConnectTimeoutError",
                    "bounded capture failed",
                )
            )
            continue
        content = f"capture-{camera_id}".encode()
        relative = f"captures/managed-scene/sessions/{session_id}/{camera_id}.mp4"
        artifact_path = repository.root / relative
        artifact_path.write_bytes(content)
        frame = SceneSourceFrame(
            f"{camera_id}-frame-1",
            camera_id,
            profile.profile_version,
            index,
            profile.time_base,
            1_000_000 + index * 20_000,
            "2026-09-01T00:00:00Z",
        )
        results.append(
            SceneCameraCaptureResult(
                camera_id,
                profile,
                SceneCaptureStatus.CAPTURED,
                (frame,),
                SceneCaptureArtifact(
                    relative,
                    hashlib.sha256(content).hexdigest(),
                    len(content),
                    SceneStorageMode.PACKET_PRESERVING_MP4,
                    False,
                    None,
                ),
                (),
                None,
                None,
            )
        )
    session = SceneCaptureSessionManifest(
        SCENE_CAPTURE_SESSION_SCHEMA,
        scene_id,
        session_id,
        "2026-09-01T00:00:00Z",
        1,
        2,
        "test-capture-v1",
        "b" * 64,
        bindings,
        tuple(results),
    )
    repository.write_session(session)
    repository.write_bundle(select_scene_capture_bundle(session, bundle_id))


def _scene(workspace: Path, artifact_root: Path, camera_ids: tuple[str, ...]) -> None:
    SceneWorkspaceRepository(workspace).create(
        SceneWorkspace(
            "spatial-mapping",
            "scene-test",
            "Test scene",
            artifact_root.resolve(),
            tuple(
                CameraConfig(camera_id, f"Camera {index + 1}")
                for index, camera_id in enumerate(camera_ids)
            ),
            tuple(
                PhaseRecord(
                    phase_id,
                    PhaseState.READY if phase_id == "P02" else PhaseState.UNAVAILABLE,
                    "Ready" if phase_id == "P02" else "Complete the previous step",
                    () if index == 0 else (PHASE_ORDER[index - 1],),
                )
                for index, phase_id in enumerate(PHASE_ORDER)
            ),
        )
    )


def test_complete_managed_inputs_update_phase_status_and_catalog(tmp_path: Path) -> None:
    camera_ids = ("scene-test-cam-01", "scene-test-cam-02")
    facility = tmp_path / "facility-registration"
    artifacts = tmp_path / "artifacts"
    repository = SceneCaptureRepository(artifacts)
    _write_facility(facility, camera_ids)
    _write_bundle(repository, "scene-test", camera_ids, camera_ids)
    coordinator = ManagedSceneEvidenceCoordinator(
        facility,
        repository,
        "scene-test",
        camera_ids,
        {camera_id: f"Camera {index + 1}" for index, camera_id in enumerate(camera_ids)},
    )

    status = coordinator.status()

    assert status["facility"]["ready"] is True
    assert status["capture"]["ready"] is True
    assert len(coordinator.artifacts()) == 2

    workspace = tmp_path / "workspace"
    _scene(workspace, artifacts, camera_ids)
    service = WorkflowService(
        SceneWorkspaceRepository(workspace),
        BoundedJobManager(),
        adapters={
            "P02": coordinator.phase_adapter("P02"),
            "P03": coordinator.phase_adapter("P03"),
        },
        operator_surface_ids=frozenset({"facility", "capture"}),
        scene_evidence_adapter=coordinator,
    )
    try:
        workflow = service.status()
        assert [item["state"] for item in workflow["phases"][:2]] == ["ready", "ready"]
        assert [item["state"] for item in workflow["operator"]["steps"][:2]] == [
            "complete",
            "complete",
        ]
        catalog = service.artifact_catalog_status()
        selected = {
            item["milestone_key"]: item["selected_artifact_id"] for item in catalog["milestones"]
        }
        assert selected["facility-registration"] is not None
        assert selected["capture-bundle"] is not None
    finally:
        service.close()


def test_partial_current_bundle_fails_closed_with_named_missing_camera(tmp_path: Path) -> None:
    camera_ids = ("scene-test-cam-01", "scene-test-cam-02")
    repository = SceneCaptureRepository(tmp_path / "artifacts")
    _write_facility(tmp_path / "facility-registration", camera_ids)
    _write_bundle(repository, "scene-test", camera_ids, camera_ids[:1])
    coordinator = ManagedSceneEvidenceCoordinator(
        tmp_path / "facility-registration",
        repository,
        "scene-test",
        camera_ids,
        {camera_ids[1]: "Camera 2"},
    )

    capture = coordinator.capture_status()

    assert capture["ready"] is False
    assert capture["missing_camera_ids"] == [camera_ids[1]]
    assert "missing Camera 2" in capture["issues"][-1]
    assert (
        coordinator.artifact_metadata(coordinator.artifacts()[-1])["valid_for_downstream"] is False
    )


def test_catalog_selection_updates_capture_pointer_and_readiness(tmp_path: Path) -> None:
    camera_ids = ("scene-test-cam-01", "scene-test-cam-02")
    facility = tmp_path / "facility-registration"
    artifacts = tmp_path / "artifacts"
    repository = SceneCaptureRepository(artifacts)
    _write_facility(facility, camera_ids)
    _write_bundle(repository, "scene-test", camera_ids, camera_ids)
    _write_bundle(
        repository,
        "scene-test",
        camera_ids,
        camera_ids[:1],
        session_id="session-2",
        bundle_id="bundle-2",
    )
    coordinator = ManagedSceneEvidenceCoordinator(facility, repository, "scene-test", camera_ids)
    workspace = tmp_path / "workspace"
    _scene(workspace, artifacts, camera_ids)
    service = WorkflowService(
        SceneWorkspaceRepository(workspace),
        BoundedJobManager(),
        adapters={
            "P02": coordinator.phase_adapter("P02"),
            "P03": coordinator.phase_adapter("P03"),
        },
        operator_surface_ids=frozenset({"facility", "capture"}),
        scene_evidence_adapter=coordinator,
    )
    try:
        assert coordinator.capture_status()["ready"] is False
        catalog = service.artifact_catalog_status()
        complete = next(
            version
            for milestone in catalog["milestones"]
            if milestone["milestone_key"] == "capture-bundle"
            for version in milestone["versions"]
            if version["metadata"]["valid_for_downstream"]
        )

        service.select_artifact_version(
            "select-complete-capture", complete["artifact_id"], confirm_impacts=True
        )

        current = repository.current_bundle()
        assert current is not None
        assert current[1].bundle_id == "bundle-1"
        assert current[2] == "explicit-pointer"
        assert coordinator.capture_status()["ready"] is True
    finally:
        service.close()
