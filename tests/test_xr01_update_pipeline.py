from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from spatial_mapping_phase2.p08_scene_updates import SceneUpdateSchedule, UpdateMode
from spatial_mapping_phase2.xr01_update_pipeline import (
    CapturedJpeg,
    median_rgb_stack,
    prepare_scene_update_inputs,
)


class _Frames:
    def capture_jpeg(self, camera_id: str) -> CapturedJpeg:
        value = 30 if camera_id == "camera-a" else 90
        image = np.full((4, 6, 3), value, dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        assert success
        return CapturedJpeg(camera_id, encoded.tobytes(), datetime.now(UTC).isoformat())


def test_median_stack_removes_one_transient_foreground() -> None:
    background = np.full((2, 3, 3), 20, dtype=np.uint8)
    transient = background.copy()
    transient[0, 0] = 240
    result = median_rgb_stack((background, transient, background, transient, background))
    assert np.array_equal(result, background)


def test_manual_input_preparation_preserves_camera_conditions_and_omits_secrets(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    cameras = []
    for camera_id in ("camera-a", "camera-b"):
        cameras.append(
            {
                "camera_id": camera_id,
                "pinhole_derivative": {"source_intrinsic_label": "working"},
                "intrinsics": {
                    "K_pinhole": [[4.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                    "distortion": [0.0],
                },
                "seed_transform": {"T_camera_from_world_for_DA3": np.eye(4).tolist()},
                "evaluation_mask": {"path": "unchanged"},
            }
        )
    (baseline / "input-manifest.json").write_text(
        json.dumps({"schema_version": "p06-da3-input-manifest-v1", "cameras": cameras}),
        encoding="utf-8",
    )
    (baseline / "run-manifest.json").write_text('{"selected": true}\n', encoding="utf-8")
    prepared = prepare_scene_update_inputs(
        "manual-1",
        SceneUpdateSchedule(False, UpdateMode.MANUAL, "Asia/Singapore"),
        _Frames(),
        ("camera-a", "camera-b"),
        baseline,
        tmp_path / "updates",
        threading.Event(),
    )

    payload = (prepared.directory / "input-manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(payload)
    assert manifest["capture_policy"]["frame_count_per_camera"] == 1
    assert manifest["cameras"][0]["seed_transform"] == cameras[0]["seed_transform"]
    assert manifest["cameras"][0]["intrinsics"] == cameras[0]["intrinsics"]
    assert "rtsp://" not in payload.lower()
    assert "password" not in payload.lower()
    provenance = prepared.directory / "capture-provenance.json"
    assert hashlib.sha256(provenance.read_bytes()).hexdigest()
