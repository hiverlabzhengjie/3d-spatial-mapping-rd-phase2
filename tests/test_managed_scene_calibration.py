from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from spatial_mapping_phase2.managed_scene_calibration import (
    ManagedSceneCalibrationCoordinator,
    calibration_generation_manifest,
)
from spatial_mapping_phase2.managed_scene_geocalib import (
    ManagedSceneGeoCalibConfig,
    ManagedSceneGeoCalibError,
    ManagedSceneGeoCalibRunner,
)
from spatial_mapping_phase2.p04_calibration_domain import FrameReviewStatus


class _Evidence:
    def __init__(self, export: Path, plan: Path) -> None:
        self.export = export
        self.plan = plan

    def facility_status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "issues": [],
            "current_export": _identity(self.export),
            "plan_image": _identity(self.plan),
        }


class _Capture:
    pass


class _FrameService:
    def __init__(self, frames: list[Any], root: Path, *, approved: bool = True) -> None:
        self.frames = frames
        self.root = root
        self.approved = approved

    def load_state(self) -> Any:
        return SimpleNamespace(
            frames=tuple(self.frames),
            approved_frame=self.frames[0] if self.approved else None,
        )

    def frame_image_path(self, frame_id: str) -> Path:
        return self.root / f"{frame_id}.jpg"


def _identity(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _facility(path: Path, plan: Path, revision: int) -> None:
    plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema_version": "p02-interactive-export-v1",
                "source_revision": revision,
                "plan": {
                    "source_sha256": plan_sha256,
                    "image_width_pixels": 80,
                    "image_height_pixels": 60,
                },
                "facility_frame": {
                    "frame_id": "facility-world",
                    "T_world_from_plan_display_pixel": [
                        [0.1, 0.0, 0.0],
                        [0.0, 0.1, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
                "camera_mounting_priors": [
                    {
                        "camera_id": "camera-1",
                        "C_world_mount_prior": {
                            "x_metres": 1.0,
                            "y_metres": 2.0,
                            "z_metres": 3.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _geocalib_config(tmp_path: Path) -> ManagedSceneGeoCalibConfig:
    source = tmp_path / "geocalib"
    torch_home = tmp_path / "torch-home"
    repository = tmp_path / "repository"
    source.mkdir()
    torch_home.mkdir()
    (repository / "src").mkdir(parents=True)
    return ManagedSceneGeoCalibConfig(Path(sys.executable), source, torch_home, repository)


def test_facility_revision_creates_new_calibration_generation(tmp_path: Path) -> None:
    plan = tmp_path / "plan.png"
    assert cv2.imwrite(str(plan), np.zeros((60, 80, 3), dtype=np.uint8))
    first = tmp_path / "facility-1.json"
    second = tmp_path / "facility-2.json"
    _facility(first, plan, 1)
    _facility(second, plan, 2)
    evidence = _Evidence(first, plan)
    root = tmp_path / "calibration"
    coordinator = ManagedSceneCalibrationCoordinator(
        evidence,  # type: ignore[arg-type]
        _Capture(),
        "project-a",
        "scene-a",
        ("camera-1",),
        root,
        _geocalib_config(tmp_path),
    )

    original = coordinator.current_service("camera-1")
    evidence.export = second
    current = coordinator.current_service("camera-1")

    assert original.workspace != current.workspace
    assert original.workspace.is_dir()
    assert current.workspace.is_dir()
    assert calibration_generation_manifest(root)["generation_count"] == 2


def test_geocalib_runner_reuses_exact_frame_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _geocalib_config(tmp_path)
    frames: list[Any] = []
    for index in range(3):
        frame_id = f"frame-{index}"
        content = f"jpeg-{index}".encode()
        (tmp_path / f"{frame_id}.jpg").write_bytes(content)
        frames.append(
            SimpleNamespace(
                frame_id=frame_id,
                profile_version="profile-v1",
                sha256=hashlib.sha256(content).hexdigest(),
                image_width_pixels=1920,
                image_height_pixels=1080,
                status=(FrameReviewStatus.APPROVED if index == 0 else FrameReviewStatus.CANDIDATE),
            )
        )
    service = _FrameService(frames, tmp_path)
    runner = ManagedSceneGeoCalibRunner(config, tmp_path / "intrinsics")
    calls: list[Path] = []
    monkeypatch.setattr(ManagedSceneGeoCalibConfig, "validate", lambda _config: None)

    def fake_worker(_runner: ManagedSceneGeoCalibRunner, request_path: Path) -> None:
        calls.append(request_path)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        Path(request["output_path"]).write_text(
            json.dumps(
                {
                    "schema_version": "xr03-independent-intrinsic-estimates-v1",
                    "request_sha256": request["request_sha256"],
                    "authority": "test",
                    "estimates": [{"camera_id": "camera-1"}],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(ManagedSceneGeoCalibRunner, "_run_worker", fake_worker)

    first = runner.prepare(("camera-1",), {"camera-1": service})
    second = runner.prepare(("camera-1",), {"camera-1": service})

    assert first == second == runner.current_evidence_path
    assert len(calls) == 1


def test_geocalib_requires_approved_primary_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _geocalib_config(tmp_path)
    monkeypatch.setattr(ManagedSceneGeoCalibConfig, "validate", lambda _config: None)
    frame = SimpleNamespace(
        frame_id="frame-1",
        profile_version="profile-v1",
        sha256="0" * 64,
        image_width_pixels=1920,
        image_height_pixels=1080,
        status=FrameReviewStatus.CANDIDATE,
    )
    service = _FrameService([frame, frame, frame], tmp_path, approved=False)

    with pytest.raises(ManagedSceneGeoCalibError, match="approve one primary"):
        ManagedSceneGeoCalibRunner(config, tmp_path / "intrinsics").prepare(
            ("camera-1",), {"camera-1": service}
        )
