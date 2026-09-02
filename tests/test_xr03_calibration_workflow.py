from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from spatial_mapping_phase2.p04_pose_domain import CameraIntrinsics, project_world_points
from spatial_mapping_phase2.xr03_calibration_workflow import (
    CalibrationWorkflowError,
    IntegratedCalibrationWorkflowAdapter,
    SceneCalibrationRepository,
    _intrinsic_batch_matches_policy,
)
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy


class _CalibrationInput:
    def __init__(
        self,
        export_path: Path,
        validation_items: list[dict[str, Any]],
    ) -> None:
        self.export_path = export_path
        self.validation_items = validation_items
        self.revision = 7

    def calibration_readiness(self) -> dict[str, Any]:
        import hashlib

        return {
            "camera_id": "camera-1",
            "source_revision": self.revision,
            "approved_frame_id": "frame-1",
            "solve_count": 4,
            "d034_validation_count": 2,
            "current_export_ready": True,
            "current_export_path": str(self.export_path),
            "current_export_sha256": hashlib.sha256(self.export_path.read_bytes()).hexdigest(),
            "calibrate_ready": True,
            "reason": None,
        }

    def export_d034_validation_seal(self) -> tuple[Path, dict[str, Any]]:
        return Path("validation-seal.json"), {"validation_landmarks": self.validation_items}


def _policy() -> SceneCameraPolicy:
    return SceneCameraPolicy.build(
        "project-a",
        "scene-a",
        ("camera-1",),
        (
            {
                "group_id": "lens-a",
                "lens_model": "Synthetic fixed lens",
                "camera_ids": ["camera-1"],
            },
        ),
        (),
    )


def _landmark(landmark_id: str, world: Any, pixel: Any, role: str) -> dict[str, Any]:
    return {
        "landmark_id": landmark_id,
        "role": role,
        "world_point": {
            "x_metres": float(world[0]),
            "y_metres": float(world[1]),
            "z_metres": float(world[2]),
        },
        "image_point": {"u": float(pixel[0]), "v": float(pixel[1])},
    }


def _adapter(
    tmp_path: Path, *, validation_offset_pixels: float = 0.0
) -> tuple[IntegratedCalibrationWorkflowAdapter, _CalibrationInput]:
    intrinsics = CameraIntrinsics("pinhole", 1920, 1080, 1200, 1200, 960, 540)
    center = np.array([8.7, 5.4, 3.1])
    rvec = np.array([0.35, -0.6, 1.1])
    rotation_camera_from_world, _ = cv2.Rodrigues(rvec)
    translation = -rotation_camera_from_world @ center
    camera_points = np.array(
        [
            [-2.5, -1.5, 7],
            [2.0, -1.2, 8],
            [2.2, 1.8, 9],
            [-2.0, 1.5, 6],
            [0.2, 0.5, 10],
            [2.0, -2.0, 12],
        ]
    )
    world = (rotation_camera_from_world.T @ (camera_points - translation).T).T
    pixels, _ = project_world_points(intrinsics, world, rvec, translation)
    solve = [
        _landmark(f"solve-{index}", world[index], pixels[index], "solve") for index in range(4)
    ]
    validation_pixels = pixels[4:].copy()
    validation_pixels[1, 0] += validation_offset_pixels
    validation = [
        _landmark(
            f"validation-{index}",
            world[index + 4],
            validation_pixels[index],
            "d034-validation",
        )
        for index in range(2)
    ]
    export_path = tmp_path / "correspondences.json"
    export_path.write_text(json.dumps({"landmarks": solve}), encoding="utf-8")
    evidence_path = tmp_path / "intrinsics.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "xr03-independent-intrinsic-estimates-v1",
                "authority": "synthetic-test",
                "estimates": [
                    {
                        "camera_id": "camera-1",
                        "profile_version": "profile-v1",
                        **intrinsics.to_dict(),
                        "within_camera_focal_cv": 0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    facility_path = tmp_path / "facility.json"
    facility_path.write_text(
        json.dumps(
            {
                "camera_mounting_priors": [
                    {
                        "camera_id": "camera-1",
                        "C_world_mount_prior": {
                            "x_metres": float(center[0]),
                            "y_metres": float(center[1]),
                            "z_metres": float(center[2]),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    inputs = _CalibrationInput(export_path, validation)
    repository = SceneCalibrationRepository(
        tmp_path / "calibration.sqlite3", "project-a", "scene-a", ("camera-1",)
    )
    return (
        IntegratedCalibrationWorkflowAdapter(
            repository,
            {"camera-1": inputs},
            evidence_path,
            facility_path,
            tmp_path / "outputs",
        ),
        inputs,
    )


def test_intrinsic_first_calibration_and_strict_review(tmp_path: Path) -> None:
    adapter, inputs = _adapter(tmp_path)
    policy = _policy()

    assert adapter.status(policy)["intrinsics_ready"] is False
    batch = adapter.determine_intrinsics(policy)
    assert batch["assignments"][0]["initial_assignment_label"] == "independent:camera-1"
    attempt = adapter.calibrate_camera("camera-1", policy)
    assert attempt["automated_status"] == "accepted"
    assert attempt["pose"]["fixed_camera_center_world_metres"] == pytest.approx([8.7, 5.4, 3.1])
    assert adapter.status(policy)["all_cameras_ready"] is False

    adapter.review_camera("camera-1", attempt["payload_sha256"], policy)
    assert adapter.status(policy)["cameras"][0]["readiness"] == "strict-ready"

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    source_path = baseline / "camera-1.jpg"
    assert cv2.imwrite(str(source_path), np.zeros((1080, 1920, 3), dtype=np.uint8))
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (baseline / "input-manifest.json").write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "camera_id": "camera-1",
                        "source": {"path": str(source_path), "sha256": source_sha256},
                        "evaluation_mask": {"rectangles_xyxy_derivative_pixels": []},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (baseline / "run-manifest.json").write_text("{}", encoding="utf-8")
    prepared = adapter.prepare_reconstruction_inputs(policy, baseline, tmp_path / "prepared")
    prepared_manifest = json.loads((prepared / "input-manifest.json").read_text(encoding="utf-8"))
    prepared_camera = prepared_manifest["cameras"][0]
    assert prepared_camera["calibration_attempt_sha256"] == attempt["payload_sha256"]
    assert prepared_camera["seed_transform"]["status"] == "strict-ready"
    assert Path(prepared_camera["pinhole_derivative"]["path"]).is_file()

    inputs.revision += 1
    stale = adapter.status(policy)
    assert stale["all_cameras_ready"] is False
    assert stale["cameras"][0]["attempt"] is None


def test_validation_failure_can_only_be_warning_qualified(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, validation_offset_pixels=45.0)
    policy = _policy()
    adapter.determine_intrinsics(policy)
    attempt = adapter.calibrate_camera("camera-1", policy)

    assert attempt["automated_status"] == "rejected"
    assert attempt["can_override"] is True
    with pytest.raises(CalibrationWorkflowError, match="acknowledgement"):
        adapter.override_camera("camera-1", attempt["payload_sha256"], "reviewed", False, policy)
    adapter.override_camera(
        "camera-1",
        attempt["payload_sha256"],
        "Physical overlay is acceptable for this internal R&D run",
        True,
        policy,
    )
    camera = adapter.status(policy)["cameras"][0]
    assert camera["readiness"] == "operator-accepted-with-warning"
    assert camera["attempt"]["automated_status"] == "rejected"


def test_hard_solve_failure_cannot_be_overridden(tmp_path: Path) -> None:
    adapter, inputs = _adapter(tmp_path)
    correspondence = json.loads(inputs.export_path.read_text(encoding="utf-8"))
    for item in correspondence["landmarks"]:
        item["image_point"] = {"u": 960.0, "v": 540.0}
    inputs.export_path.write_text(json.dumps(correspondence), encoding="utf-8")
    policy = _policy()
    adapter.determine_intrinsics(policy)

    attempt = adapter.calibrate_camera("camera-1", policy)
    assert attempt["automated_status"] == "rejected"
    assert attempt["pose"] is None
    assert attempt["can_override"] is False
    with pytest.raises(CalibrationWorkflowError, match="no usable pose"):
        adapter.override_camera(
            "camera-1",
            attempt["payload_sha256"],
            "The operator wants to force this",
            True,
            policy,
        )


def test_legacy_intrinsic_batch_remains_current_after_overlap_only_change() -> None:
    cameras = ("camera-1", "camera-2")
    groups = (
        {
            "group_id": "fixed-lens",
            "lens_model": "simple-radial",
            "camera_ids": list(cameras),
        },
    )
    first = SceneCameraPolicy.build(
        "project-a",
        "scene-a",
        cameras,
        groups,
        (
            {
                "camera_id_a": "camera-1",
                "camera_id_b": "camera-2",
                "verdict": "no_overlap",
            },
        ),
    )
    changed = SceneCameraPolicy.build(
        "project-a",
        "scene-a",
        cameras,
        groups,
        (
            {
                "camera_id_a": "camera-1",
                "camera_id_b": "camera-2",
                "verdict": "overlap",
            },
        ),
    )
    legacy_batch = {
        "camera_policy_sha256": first.sha256,
        "assignments": [
            {
                "camera_id": camera_id,
                "group_id": "fixed-lens",
                "lens_model": "simple-radial",
            }
            for camera_id in cameras
        ],
    }

    assert first.sha256 != changed.sha256
    assert _intrinsic_batch_matches_policy(legacy_batch, changed) is True
