from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import pytest

from spatial_mapping_phase2.p08_operator_workflow import (
    CameraInputSummaryAdapter,
    FloorPreviewWorkflowAdapter,
    OperatorWorkflowConfig,
    ReconstructionWorkflowAdapter,
    _run_process,
)
from spatial_mapping_phase2.p08_workflow import P08WorkflowError


def _write_manifest(path: Path) -> None:
    cameras = []
    for index in range(2):
        cameras.append(
            {
                "camera_id": f"camera-{index + 1}",
                "intrinsics": {
                    "model": "simple_radial",
                    "width_pixels": 1920,
                    "height_pixels": 1080,
                    "fx_pixels": 1400.0 + index,
                    "fy_pixels": 1401.0 + index,
                    "cx_pixels": 960.0,
                    "cy_pixels": 540.0,
                    "distortion": [-0.27],
                },
                "seed_transform": {
                    "T_world_from_camera": [
                        [1.0, 0.0, 0.0, float(index)],
                        [0.0, 1.0, 0.0, float(index + 2)],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
            }
        )
    path.write_text(json.dumps({"cameras": cameras}), encoding="utf-8")


def _config(path: Path) -> OperatorWorkflowConfig:
    return OperatorWorkflowConfig(
        camera_input_manifest_path=path,
        camera_input_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        geometry_rerun_artifact_id="geometry-preview",
        floor_rerun_artifact_id="floor-preview",
    )


def test_camera_summary_exposes_exact_intrinsics_and_world_pose(tmp_path: Path) -> None:
    manifest = tmp_path / "inputs.json"
    _write_manifest(manifest)
    summaries = CameraInputSummaryAdapter(_config(manifest)).summaries(("camera-1", "camera-2"))

    assert [camera["camera_id"] for camera in summaries] == ["camera-1", "camera-2"]
    assert summaries[0]["intrinsics"]["fx_pixels"] == 1400.0
    assert summaries[1]["pose"]["position_metres"] == [1.0, 3.0, 3.0]
    assert summaries[0]["pose"]["transform"] == "T_world_from_camera"
    assert summaries[0]["pose"]["orientation_zyx_degrees"] == {
        "yaw": 0.0,
        "pitch": -0.0,
        "roll": 0.0,
    }


def test_camera_summary_rejects_changed_identity_and_roster(tmp_path: Path) -> None:
    manifest = tmp_path / "inputs.json"
    _write_manifest(manifest)
    config = _config(manifest)
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(P08WorkflowError, match="changed"):
        CameraInputSummaryAdapter(config).summaries(("camera-1", "camera-2"))

    _write_manifest(manifest)
    with pytest.raises(P08WorkflowError, match="does not match"):
        CameraInputSummaryAdapter(_config(manifest)).summaries(("camera-1",))


def test_long_job_log_capture_does_not_deadlock_on_large_output(tmp_path: Path) -> None:
    _run_process(
        (sys.executable, "-c", "print('x' * 1000000)"),
        tmp_path,
        threading.Event(),
    )


def test_external_workflow_process_imports_repository_package_and_preserves_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "src" / "spatial_mapping_phase2"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("MARKER = 'repository-source'\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    output = tmp_path / "probe.txt"
    script = scripts / "probe.py"
    script.write_text(
        "import os, sys\n"
        "from spatial_mapping_phase2 import MARKER\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(MARKER + '\\n' + os.environ['PYTHONPATH'])\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", "existing-runtime-path")

    _run_process((sys.executable, str(script), str(output)), tmp_path, threading.Event())

    marker, child_pythonpath = output.read_text(encoding="utf-8").splitlines()
    assert marker == "repository-source"
    assert child_pythonpath.split(os.pathsep) == [
        str((tmp_path / "src").resolve()),
        "existing-runtime-path",
    ]


def test_floor_preview_reuses_complete_immutable_recording(tmp_path: Path) -> None:
    run_directory = tmp_path / "floor-run"
    run_directory.mkdir()
    rerun = run_directory / "candidate-working-facility-geometry-v3-floor-context-v4.rrd"
    rerun.write_bytes(b"complete-rerun")
    rerun_sha = hashlib.sha256(rerun.read_bytes()).hexdigest()
    manifest = run_directory / "floor-rerun-manifest-v4.json"
    manifest.write_text(
        json.dumps(
            {
                "rerun": {
                    "path": str(rerun.resolve()),
                    "sha256": rerun_sha,
                    "byte_count": rerun.stat().st_size,
                }
            }
        ),
        encoding="utf-8",
    )
    adapter = FloorPreviewWorkflowAdapter(
        python_executable=tmp_path / "missing-python",
        repository_root=tmp_path,
        floor_contract_path=tmp_path / "missing-contract",
    )

    result = adapter.run(run_directory, threading.Event())

    assert result["reused_existing_preview"] is True
    assert result["rerun"]["sha256"] == rerun_sha


def test_floor_preview_rejects_partial_existing_output(tmp_path: Path) -> None:
    run_directory = tmp_path / "partial-floor-run"
    run_directory.mkdir()
    (run_directory / "candidate-working-facility-geometry-v3-floor-context-v4.rrd").write_bytes(
        b"partial"
    )
    adapter = FloorPreviewWorkflowAdapter(
        python_executable=tmp_path / "missing-python",
        repository_root=tmp_path,
        floor_contract_path=tmp_path / "missing-contract",
    )

    with pytest.raises(P08WorkflowError, match="incomplete"):
        adapter.run(run_directory, threading.Event())


def _reconstruction_adapter(tmp_path: Path) -> ReconstructionWorkflowAdapter:
    return ReconstructionWorkflowAdapter(
        python_executable=tmp_path / "python.exe",
        repository_root=tmp_path,
        p06_run_directory=tmp_path,
        source_directory=tmp_path,
        checkpoint_directory=tmp_path,
        d041_manifest_path=tmp_path / "manifest.json",
        output_root=tmp_path,
    )


def test_reconstruction_readiness_includes_scene_source_issues(tmp_path: Path) -> None:
    adapter = ReconstructionWorkflowAdapter(
        python_executable=tmp_path / "python.exe",
        repository_root=tmp_path,
        p06_run_directory=None,
        source_directory=tmp_path,
        checkpoint_directory=tmp_path,
        d041_manifest_path=None,
        output_root=tmp_path,
        input_readiness=lambda: ("Select one complete capture bundle",),
    )

    assert "Select one complete capture bundle" in adapter.readiness_errors()


def test_geometry_review_reuses_camera_rich_reconstruction_recording(tmp_path: Path) -> None:
    rerun = tmp_path / "reconstruction.rrd"
    rerun.write_bytes(b"camera-rich-rerun")
    rerun_record = {
        "path": str(rerun.resolve()),
        "sha256": hashlib.sha256(rerun.read_bytes()).hexdigest(),
        "byte_count": rerun.stat().st_size,
        "camera_frustum_count": 2,
        "camera_rgb_image_plane_count": 2,
        "camera_orientation_axis_count": 2,
        "fixed_camera_centre_count": 2,
        "world_space_camera_label_count": 2,
    }
    (tmp_path / "geometry-manifest.json").write_text(
        json.dumps({"camera_order": ["camera-1", "camera-2"], "rerun": rerun_record}),
        encoding="utf-8",
    )

    result = _reconstruction_adapter(tmp_path).ensure_geometry_review(tmp_path, threading.Event())

    assert result["reused_existing_preview"] is True
    assert result["derived_without_da3_rerun"] is False
    assert result["rerun"]["sha256"] == rerun_record["sha256"]


def test_geometry_review_reuses_verified_visual_derivative(tmp_path: Path) -> None:
    source = tmp_path / "source.rrd"
    source.write_bytes(b"old-rerun")
    (tmp_path / "geometry-manifest.json").write_text(
        json.dumps(
            {
                "camera_order": ["camera-1", "camera-2"],
                "rerun": {
                    "path": str(source.resolve()),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "byte_count": source.stat().st_size,
                },
            }
        ),
        encoding="utf-8",
    )
    rerun = tmp_path / "all-four-da3-combined-geometry-review-v2.rrd"
    rerun.write_bytes(b"derived-rerun")
    rerun_sha256 = hashlib.sha256(rerun.read_bytes()).hexdigest()
    manifest = {
        "success": True,
        "camera_order": ["camera-1", "camera-2"],
        "camera_frustum_count": 2,
        "camera_rgb_image_plane_count": 2,
        "camera_orientation_axis_count": 2,
        "fixed_camera_centre_count": 2,
        "world_space_camera_label_count": 2,
        "rerun": {
            "path": str(rerun.resolve()),
            "sha256": rerun_sha256,
            "byte_count": rerun.stat().st_size,
        },
    }
    (tmp_path / "geometry-review-manifest-v2.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    result = _reconstruction_adapter(tmp_path).ensure_geometry_review(tmp_path, threading.Event())

    assert result["reused_existing_preview"] is True
    assert result["derived_without_da3_rerun"] is True
    assert result["rerun"]["sha256"] == rerun_sha256
