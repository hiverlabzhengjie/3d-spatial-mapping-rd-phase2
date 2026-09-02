from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from spatial_mapping_phase2 import managed_scene_reconstruction, p08_operator_workflow
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
from spatial_mapping_phase2.managed_scene_reconstruction import (
    ManagedSceneReconstructionError,
    ManagedSceneReconstructionInputBuilder,
)
from spatial_mapping_phase2.p08_operator_workflow import ReconstructionWorkflowAdapter
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy


def _policy(camera_ids: tuple[str, ...]) -> SceneCameraPolicy:
    return SceneCameraPolicy.build(
        "project-test",
        "scene-test",
        camera_ids,
        [
            {
                "group_id": "fixed-cctv",
                "lens_model": "simple-radial",
                "camera_ids": list(camera_ids),
            }
        ],
        [
            {
                "camera_id_a": camera_ids[left],
                "camera_id_b": camera_ids[right],
                "verdict": "no_overlap",
            }
            for left in range(len(camera_ids))
            for right in range(left + 1, len(camera_ids))
        ],
    )


def _capture_repository(
    root: Path,
    camera_ids: tuple[str, ...],
    *,
    packet_camera: str | None = None,
    captured_ids: tuple[str, ...] | None = None,
) -> SceneCaptureRepository:
    repository = SceneCaptureRepository(root)
    session_id = "session-1"
    repository.create_session_directory(session_id)
    captured = camera_ids if captured_ids is None else captured_ids
    bindings: list[SceneCameraBinding] = []
    results: list[SceneCameraCaptureResult] = []
    for index, camera_id in enumerate(camera_ids):
        key = f"SCENE_TEST_CAMERA_{index + 1:02d}_RTSP"
        bindings.append(SceneCameraBinding(camera_id, key))
        profile = SceneStreamProfile(
            SCENE_CAPTURE_PROFILE_SCHEMA,
            camera_id,
            "profile-v1",
            key,
            f"local-{camera_id}",
            8,
            6,
            "h264",
            SceneRationalTimeBase(1, 100),
            None,
            None,
            "2026-09-01T00:00:00Z",
        )
        if camera_id not in captured:
            results.append(
                SceneCameraCaptureResult(
                    camera_id,
                    profile,
                    SceneCaptureStatus.FAILED,
                    (),
                    None,
                    (),
                    "ConnectTimeout",
                    "failed",
                )
            )
            continue
        packet = camera_id == packet_camera
        suffix = ".mp4" if packet else ".jpg"
        relative = f"captures/managed-scene/sessions/{session_id}/{camera_id}{suffix}"
        artifact_path = repository.root / relative
        if packet:
            artifact_path.write_bytes(b"synthetic packet capture")
            source_pts = 10
            time_base = profile.time_base
            mode = SceneStorageMode.PACKET_PRESERVING_MP4
            reason = None
        else:
            assert cv2.imwrite(str(artifact_path), np.full((6, 8, 3), 20 + index, dtype=np.uint8))
            source_pts = None
            time_base = None
            mode = SceneStorageMode.DECODED_FRAME_FALLBACK
            reason = "test-fallback"
        frame = SceneSourceFrame(
            f"{camera_id}-frame-1",
            camera_id,
            profile.profile_version,
            source_pts,
            time_base,
            1_000 + index,
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
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    artifact_path.stat().st_size,
                    mode,
                    False,
                    reason,
                ),
                (),
                None,
                None,
            )
        )
    session = SceneCaptureSessionManifest(
        SCENE_CAPTURE_SESSION_SCHEMA,
        "scene-test",
        session_id,
        "2026-09-01T00:00:00Z",
        1,
        2,
        "test-capture",
        "a" * 64,
        tuple(bindings),
        tuple(results),
    )
    repository.write_session(session)
    repository.write_bundle(select_scene_capture_bundle(session, "bundle-1"))
    return repository


def test_build_materializes_complete_bundle_and_records_nearest_decodable_pts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera_ids = ("camera-1", "camera-2")
    repository = _capture_repository(tmp_path / "artifacts", camera_ids, packet_camera="camera-2")
    monkeypatch.setattr(
        managed_scene_reconstruction,
        "_decode_nearest_video_frame",
        lambda _path, _pts: (np.full((6, 8, 3), 77, dtype=np.uint8), 14),
    )

    output = ManagedSceneReconstructionInputBuilder(repository, "scene-test", camera_ids).build(
        _policy(camera_ids), tmp_path / "source-input"
    )
    manifest = json.loads((output / "input-manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "managed-scene-da3-source-input-v1"
    assert manifest["camera_order"] == list(camera_ids)
    assert manifest["camera_policy_sha256"] == _policy(camera_ids).sha256
    assert [camera["camera_id"] for camera in manifest["cameras"]] == list(camera_ids)
    assert all(Path(camera["source"]["path"]).is_file() for camera in manifest["cameras"])
    packet = manifest["cameras"][1]
    assert packet["materialization"]["selection_mode"] == "nearest-decodable-pts"
    assert packet["materialization"]["source_pts_delta"] == 4
    assert packet["materialization"]["source_time_delta_seconds"] == pytest.approx(0.04)
    assert (
        json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))["success"] is True
    )


def test_build_rejects_incomplete_bundle_before_creating_output(tmp_path: Path) -> None:
    camera_ids = ("camera-1", "camera-2")
    repository = _capture_repository(
        tmp_path / "artifacts", camera_ids, captured_ids=("camera-1",)
    )
    output = tmp_path / "source-input"

    with pytest.raises(ManagedSceneReconstructionError, match="incomplete"):
        ManagedSceneReconstructionInputBuilder(repository, "scene-test", camera_ids).build(
            _policy(camera_ids), output
        )

    assert not output.exists()


def test_build_never_mutates_an_existing_immutable_input_directory(tmp_path: Path) -> None:
    camera_ids = ("camera-1",)
    repository = _capture_repository(tmp_path / "artifacts", camera_ids)
    output = tmp_path / "source-input"
    output.mkdir()
    marker = output / "retained.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ManagedSceneReconstructionInputBuilder(repository, "scene-test", camera_ids).build(
            _policy(camera_ids), output
        )

    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert tuple(output.iterdir()) == (marker,)


def test_scene_reconstruction_uses_448_and_has_no_office_rollback_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (repository / "src" / "spatial_mapping_phase2").mkdir(parents=True)
    for name in (
        "run_p07_all4_da3_diagnostic.py",
        "export_p07_all4_da3_cloud.py",
        "verify_p07_all4_da3_diagnostic.py",
    ):
        (scripts / name).write_text("# test\n", encoding="utf-8")
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "input-manifest.json").write_text("{}", encoding="utf-8")
    (inputs / "run-manifest.json").write_text("{}", encoding="utf-8")
    output_root = tmp_path / "outputs"
    commands: list[tuple[str, ...]] = []

    def fake_process(command: tuple[str, ...], _cwd: Path, _cancel: threading.Event) -> None:
        commands.append(command)
        if len(commands) != 1:
            return
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        combined = output / "combined.npz"
        rerun = output / "review.rrd"
        combined.write_bytes(b"geometry")
        rerun.write_bytes(b"rerun")
        (output / "inference-manifest.json").write_text("{}", encoding="utf-8")
        (output / "geometry-manifest.json").write_text(
            json.dumps(
                {
                    "combined": {
                        "path": str(combined),
                        "sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
                        "point_count": 3,
                    },
                    "rerun": {
                        "path": str(rerun),
                        "sha256": hashlib.sha256(rerun.read_bytes()).hexdigest(),
                        "byte_count": rerun.stat().st_size,
                    },
                }
            ),
            encoding="utf-8",
        )
        (output / "verification.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(p08_operator_workflow, "_run_process", fake_process)
    adapter = ReconstructionWorkflowAdapter(
        python,
        repository,
        None,
        source,
        checkpoint,
        None,
        output_root,
        process_resolution=448,
    )

    result = adapter.run(
        "job-1",
        threading.Event(),
        input_run_directory=inputs,
        camera_ids=("camera-1", "camera-2"),
        camera_policy_sha256="b" * 64,
    )

    assert result["process_resolution"] == 448
    assert "--process-resolution" in commands[0]
    assert commands[0][commands[0].index("--process-resolution") + 1] == "448"
    assert "--d041-manifest" not in commands[2]
    assert result["da3_cohort_policy"] == "all-enabled-cameras-per-scene-joint"
