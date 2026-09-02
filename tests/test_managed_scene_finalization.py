from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spatial_mapping_phase2.p08_floor import FloorProcessingConfig
from spatial_mapping_phase2.p08_workflow import (
    PHASE_ORDER,
    BoundedJobManager,
    CameraConfig,
    FloorWorkflowAdapter,
    JobState,
    P08WorkflowError,
    PhaseRecord,
    PhaseState,
    SafeRerunLauncher,
    SceneWorkspace,
    SceneWorkspaceRepository,
    WorkflowService,
)
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wait_for_job(service: WorkflowService, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        status = service.jobs.status(job_id)
        if status["state"] in {
            JobState.COMPLETE.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        }:
            return status
        time.sleep(0.01)
    raise AssertionError(f"job did not finish: {job_id}")


class _Evidence:
    def __init__(self, root: Path) -> None:
        self.allowed_artifact_roots = (root,)
        self.facility_sha256 = "1" * 64
        self.capture_sha256 = "2" * 64

    def status(self) -> dict[str, Any]:
        return {
            "facility": {
                "ready": True,
                "current_export": {"sha256": self.facility_sha256},
            },
            "capture": {
                "ready": True,
                "current_bundle": {"sha256": self.capture_sha256},
            },
        }

    def artifacts(self) -> tuple[Any, ...]:
        return ()

    def artifact_metadata(self, _artifact: object) -> dict[str, Any]:
        return {}


class _Calibration:
    def __init__(self, camera_ids: tuple[str, ...]) -> None:
        self.camera_ids = camera_ids

    def status(self, _policy: object) -> dict[str, Any]:
        return {
            "all_cameras_ready": True,
            "intrinsic_batch": {"payload_sha256": "3" * 64},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "ready": True,
                    "readiness": "strict-ready",
                    "attempt": {"payload_sha256": f"{index + 4:x}" * 64},
                }
                for index, camera_id in enumerate(self.camera_ids)
            ],
        }

    def prepare_reconstruction_inputs(
        self, policy: SceneCameraPolicy, _baseline: Path | None, output: Path
    ) -> Path:
        output.mkdir(parents=True)
        (output / "input-manifest.json").write_text(
            json.dumps(
                {
                    "camera_policy_sha256": policy.sha256,
                    "intrinsic_policy_sha256": policy.intrinsic_policy_sha256,
                }
            ),
            encoding="utf-8",
        )
        (output / "run-manifest.json").write_text("{}", encoding="utf-8")
        return output


class _Reconstruction:
    supports_scene_camera_policy = True
    p06_run_directory: Path | None = None

    def __init__(self, root: Path) -> None:
        self.root = root

    def readiness_errors(self) -> tuple[str, ...]:
        return ()

    def run(
        self,
        job_id: str,
        _cancel_event: threading.Event,
        *,
        input_run_directory: Path | None = None,
        camera_ids: tuple[str, ...] | None = None,
        camera_policy_sha256: str | None = None,
    ) -> dict[str, Any]:
        assert input_run_directory is not None
        assert camera_ids == ("north", "south")
        assert camera_policy_sha256 is not None
        run = self.root / job_id
        geometry = run / "geometry"
        geometry.mkdir(parents=True)
        combined = geometry / "scene-joint-da3-combined.npz"
        np.savez_compressed(
            combined,
            points=np.array([[-2.0, 1.0, -0.5], [4.0, 7.0, 2.0]], dtype=np.float64),
            colors_rgb=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
            confidence=np.array([2.0, 3.0], dtype=np.float64),
            source_pixel_count=np.array([2, 3], dtype=np.int32),
            source_camera_index=np.array([0, 1], dtype=np.int16),
            camera_ids=np.asarray(camera_ids),
        )
        rerun = run / "scene-joint-da3-combined.rrd"
        rerun.write_bytes(b"geometry-rerun")
        geometry_manifest = run / "geometry-manifest.json"
        geometry_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "xr03-scene-joint-da3-combined-geometry-v1",
                    "success": True,
                    "camera_order": list(camera_ids),
                    "combined": {
                        "path": str(combined.resolve()),
                        "sha256": _sha256(combined),
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
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


class _FloorPreview:
    def run(self, _run: Path, _cancel: threading.Event) -> dict[str, Any]:
        raise AssertionError("managed scenes must use the geometry-bound preview route")

    def run_for_geometry(
        self, run: Path, geometry_manifest: Path, _cancel: threading.Event
    ) -> dict[str, Any]:
        payload = json.loads(geometry_manifest.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "xr03-scene-joint-da3-combined-geometry-v1"
        rerun = run / "managed-scene-final.rrd"
        rerun.write_bytes(b"final-rerun")
        return {
            "rerun": {
                "path": str(rerun.resolve()),
                "sha256": _sha256(rerun),
                "byte_count": rerun.stat().st_size,
            },
            "reused_existing_preview": False,
        }


class _Process:
    pid = 1234


def _service(
    repository: SceneWorkspaceRepository,
    evidence: _Evidence,
    calibration: _Calibration,
    reconstruction: _Reconstruction,
    viewer: Path,
) -> WorkflowService:
    return WorkflowService(
        repository,
        BoundedJobManager(maximum_workers=1, maximum_outstanding_jobs=4),
        floor_adapter=FloorWorkflowAdapter(
            None,
            FloorProcessingConfig(),
            repository.load().artifact_root / "floor",
        ),
        floor_preview_adapter=_FloorPreview(),
        rerun_launcher=SafeRerunLauncher(
            viewer,
            (repository.load().artifact_root,),
            launch=lambda _arguments: _Process(),
        ),
        reconstruction_adapter=reconstruction,
        calibration_adapter=calibration,
        scene_evidence_adapter=evidence,
        operator_surface_ids=frozenset({"facility", "capture"}),
    )


def test_generic_scene_completes_floor_and_final_review_and_survives_restart(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    scene = SceneWorkspace(
        project_id="project",
        scene_id="managed-scene",
        display_name="Managed scene",
        artifact_root=artifacts,
        cameras=(
            CameraConfig("north", "North", "RTSP_NORTH"),
            CameraConfig("south", "South", "RTSP_SOUTH"),
        ),
        phases=tuple(
            PhaseRecord(phase_id, PhaseState.READY, "Ready") for phase_id in PHASE_ORDER
        ),
    )
    repository = SceneWorkspaceRepository(workspace)
    repository.create(scene)
    viewer = tmp_path / "rerun.exe"
    viewer.write_bytes(b"viewer")
    evidence = _Evidence(artifacts)
    calibration = _Calibration(("north", "south"))
    reconstruction = _Reconstruction(artifacts / "reconstruction")
    service = _service(repository, evidence, calibration, reconstruction, viewer)
    try:
        first_policy = service.apply_camera_policy(
            "policy-1",
            {
                "intrinsic_groups": [
                    {
                        "group_id": "fixed-lens",
                        "lens_model": "simple-radial",
                        "camera_ids": ["north", "south"],
                    }
                ],
                "overlap_pair_reviews": [
                    {
                        "camera_id_a": "north",
                        "camera_id_b": "south",
                        "verdict": "no_overlap",
                    }
                ],
            },
            expected_revision=None,
            confirm_impacts=False,
        )
        assert service.operator_status()["inputs_ready"] is True

        service.start_reconstruction_job("reconstruction-1")
        assert _wait_for_job(service, "reconstruction-1")["state"] == "complete"
        service.launch_rerun("open-geometry", "reconstruction-1")
        service.approve_result("approve-geometry", "geometry")

        service.start_floor_job("floor-1")
        assert _wait_for_job(service, "floor-1")["state"] == "complete"
        floor_run = artifacts / "floor" / "floor-1"
        assert (floor_run / "verification.json").is_file()
        with np.load(floor_run / "authoritative_floor_plane.npz", allow_pickle=False) as plane:
            assert np.all(plane["vertices_xyz_metres"][:, 2] == 0.0)

        service.start_floor_preview_job("final-preview-1", "floor-1")
        assert _wait_for_job(service, "final-preview-1")["state"] == "complete"
        service.approve_result("approve-final", "floor")
        status = service.operator_status()
        assert status["floor"]["approved"] is True
        assert {item["step_id"]: item["state"] for item in status["steps"]}["results"] == (
            "complete"
        )
        changed = service.apply_camera_policy(
            "policy-overlap-only",
            {
                "intrinsic_groups": [
                    {
                        "group_id": "fixed-lens",
                        "lens_model": "simple-radial",
                        "camera_ids": ["north", "south"],
                    }
                ],
                "overlap_pair_reviews": [
                    {
                        "camera_id_a": "north",
                        "camera_id_b": "south",
                        "verdict": "overlap",
                    }
                ],
            },
            expected_revision=1,
            confirm_impacts=True,
        )
        assert changed["intrinsic_reprocessing_required"] is False
        assert changed["static_reconstruction_cohort_changed"] is False
        after_overlap = service.operator_status()
        assert after_overlap["inputs_ready"] is True
        assert after_overlap["geometry"]["approved"] is True
        assert after_overlap["floor"]["approved"] is True
        assert {item["step_id"]: item["state"] for item in after_overlap["steps"]}[
            "results"
        ] == "complete"
        source_policy = service._require_camera_policy().by_sha256(  # noqa: SLF001
            first_policy["policy"]["policy_sha256"]
        )
        assert source_policy is not None
        legacy_lineage = service._managed_input_lineage_sha256(  # noqa: SLF001
            source_policy, legacy_full_policy=True
        )
        assert legacy_lineage is not None
        service._geometry_input_lineage_sha256 = legacy_lineage  # noqa: SLF001
        service._persist_operator_state()  # noqa: SLF001
    finally:
        service.close()

    restored = _service(repository, evidence, calibration, reconstruction, viewer)
    try:
        restored_status = restored.operator_status()
        assert restored_status["floor"]["approved"] is True
        assert {item["step_id"]: item["state"] for item in restored_status["steps"]}[
            "results"
        ] == "complete"

        evidence.capture_sha256 = "9" * 64
        stale = restored.operator_status()
        assert stale["inputs_ready"] is False
        assert stale["floor"]["approved"] is False
        assert any(
            "differs from the reconstructed geometry" in issue
            for issue in stale["input_issues"]
        )
        with pytest.raises(P08WorkflowError, match="Resolve the current scene inputs"):
            restored.approve_result("stale-approval", "floor")
    finally:
        restored.close()


def test_dynamic_floor_adapter_cancels_before_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source,
        points=np.array([[0.0, 0.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        colors_rgb=np.zeros((2, 3), dtype=np.uint8),
        confidence=np.ones(2, dtype=np.float64),
        source_pixel_count=np.ones(2, dtype=np.int32),
        source_camera_index=np.zeros(2, dtype=np.int16),
        camera_ids=np.asarray(["camera-a"]),
    )
    cancel = threading.Event()
    cancel.set()
    adapter = FloorWorkflowAdapter(None, FloorProcessingConfig(), tmp_path / "runs")

    with pytest.raises(P08WorkflowError, match="cancelled before"):
        adapter.run_for_source("cancelled", source, _sha256(source), cancel)

    assert not (tmp_path / "runs" / "cancelled").exists()
