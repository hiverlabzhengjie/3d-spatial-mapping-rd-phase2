from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.managed_scene_downstream import (
    ManagedSceneLiveConfig,
    ManagedSceneXR02Worker,
)
from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    CameraConfig,
    PhaseRecord,
    PhaseState,
    SceneWorkspace,
    SceneWorkspaceRepository,
)
from spatial_mapping_phase2.xr02_deployment import load_xr02_deployment
from spatial_mapping_phase2.xr03_camera_policy import (
    CameraPolicyRepository,
    SceneCameraPolicy,
)


class FakeWorker:
    created: list[tuple[object, ...]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.created.append((*args, kwargs))

    def status(self) -> dict[str, Any]:
        return {
            "schema": "xr02.wp4.operator_status.v5",
            "active": False,
            "operator_state": "ready",
        }

    def start_live(
        self,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, Any]:
        return {"active": True, "active_mode": "live"}

    def close(self) -> None:
        return None


def test_managed_live_materializes_scene_local_compatibility_package(tmp_path: Path) -> None:
    camera_ids = tuple(f"scene-abcd1234-cam-0{index}" for index in range(1, 5))
    workspace = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    repository = SceneWorkspaceRepository(workspace)
    repository.create(
        SceneWorkspace(
            project_id="phase2",
            scene_id="scene-abcd1234",
            display_name="Managed",
            artifact_root=artifact_root,
            cameras=tuple(CameraConfig(camera_id, camera_id) for camera_id in camera_ids),
            phases=tuple(
                PhaseRecord(phase_id, PhaseState.READY, "Ready") for phase_id in PHASE_ORDER
            ),
        )
    )
    geometry_id = "reconstruction-one"
    p06_directory = workspace / "calibrated-reconstruction-inputs" / geometry_id
    p06_directory.mkdir(parents=True)
    cameras = []
    for index, camera_id in enumerate(camera_ids, start=1):
        cameras.append(
            {
                "camera_id": camera_id,
                "intrinsics": {
                    "K_pinhole": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0, 0, 1]],
                    "distortion": [-0.2],
                    "width_pixels": 1920,
                    "height_pixels": 1080,
                },
                "seed_transform": {
                    "T_world_from_camera": [
                        [1.0, 0.0, 0.0, float(index)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
            }
        )
    (p06_directory / "input-manifest.json").write_text(
        json.dumps({"cameras": cameras}), encoding="utf-8"
    )
    floor = artifact_root / "floor" / "floor-one"
    floor.mkdir(parents=True)
    (floor / "floor-completion-manifest.json").write_text("{}", encoding="utf-8")
    (floor / "authoritative_floor_plane.npz").write_bytes(b"floor")
    repository.write_operator_state(
        {
            "geometry_approved": True,
            "floor_approved": True,
            "active_geometry_artifact_id": geometry_id,
            "current_floor_output_directory": str(floor),
        }
    )
    policy_repository = CameraPolicyRepository(
        repository.camera_policy_path, "phase2", "scene-abcd1234", camera_ids
    )
    policy = SceneCameraPolicy.build(
        "phase2",
        "scene-abcd1234",
        camera_ids,
        [{"group_id": "same", "lens_model": "fixed", "camera_ids": list(camera_ids)}],
        [
            {
                "camera_id_a": camera_ids[left],
                "camera_id_b": camera_ids[right],
                "verdict": "overlap",
            }
            for left in range(4)
            for right in range(left + 1, 4)
        ],
    )
    policy_repository.apply(
        "initial-policy", policy, expected_revision=None, confirm_impacts=False
    )
    secret_file = tmp_path / "secrets.env"
    endpoint_keys = {camera_id: f"CAMERA_{index}" for index, camera_id in enumerate(camera_ids, 1)}
    secret_file.write_text(
        "".join(
            f"CAMERA_{index}=rtsp://user:pass@192.0.2.{index}:554/live\n"
            for index in range(1, 5)
        ),
        encoding="utf-8",
    )
    model = tmp_path / "model.pt"
    reid = tmp_path / "reid.pt"
    model.write_bytes(b"model")
    reid.write_bytes(b"reid")
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "schema_version": "xr02-worker-deployment-v1",
                "operator_state": "unused.json",
                "p06_calibration_manifest": "unused-p06.json",
                "p07_geometry_manifest": "unused-p07.json",
                "p08_floor_manifest": "unused-floor.json",
                "p08_floor": "unused-floor.npz",
                "detector_model": str(model),
                "reid_model": str(reid),
                "environment_file": "unused.env",
                "camera_policy": "unused.sqlite3",
                "ultralytics_config": str(tmp_path / "ultralytics"),
                "hashes": {
                    "p06": "0" * 64,
                    "p07": "1" * 64,
                    "p08_floor_manifest": "2" * 64,
                    "p08_floor": "3" * 64,
                    "detector": "4" * 64,
                    "reid": "5" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    FakeWorker.created.clear()
    adapter = ManagedSceneXR02Worker(
        ManagedSceneLiveConfig(
            python,
            None,
            "spatial_mapping_phase2.xr02_worker_cli",
            base,
            tmp_path / "mediamtx.exe",
            tmp_path / "ffmpeg.exe",
            5.0,
            8095,
        ),
        repository,
        endpoint_keys,
        secret_file,
        artifact_root / "live",
        worker_factory=FakeWorker,
    )

    status = adapter.status()
    assert status["operator_state"] == "ready"
    assert FakeWorker.created == []
    deployment_path = next((workspace / "managed-downstream" / "xr02").glob("*/deployment.json"))
    deployment = load_xr02_deployment(deployment_path)
    p06 = json.loads(deployment.p06_calibration_manifest.read_text(encoding="utf-8"))
    p07 = json.loads(deployment.p07_geometry_manifest.read_text(encoding="utf-8"))
    assert [item["camera_id"] for item in p06["cameras"]] == [
        f"office-cam-0{index}" for index in range(1, 5)
    ]
    assert p07["cameras"][0]["processed_intrinsics"][0][0] == 262.5
    assert CameraPolicyRepository.open(deployment.camera_policy).active().overlap_edges
    assert "scene-abcd1234" not in deployment.environment_file.read_text(encoding="utf-8")

    assert adapter.start_live()["active"] is True
    assert len(FakeWorker.created) == 1
