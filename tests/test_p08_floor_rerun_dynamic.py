from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_exporter() -> ModuleType:
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "wp3_test_export_p08_floor_rerun", scripts / "export_p08_floor_rerun.py"
        )
        if spec is None or spec.loader is None:
            raise AssertionError("floor Rerun exporter could not be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_dynamic_floor_rerun_camera_context_supports_variable_roster(tmp_path: Path) -> None:
    exporter = _load_exporter()
    camera_ids = ("north", "south", "loading-bay")
    images = np.arange(3 * 2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3, 3)
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 3, axis=0)
    intrinsics[:, 0, 0] = 500.0
    intrinsics[:, 1, 1] = 500.0
    raw = tmp_path / "selected-raw.npz"
    np.savez_compressed(raw, processed_images=images, intrinsics=intrinsics)
    records: list[dict[str, Any]] = []
    for index, camera_id in enumerate(camera_ids):
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = float(index)
        records.append(
            {
                "camera_id": camera_id,
                "T_world_from_camera": transform.tolist(),
                "processed_frame_rgb_array_sha256": _array_sha256(images[index]),
                "processed_intrinsics_array_sha256": _array_sha256(intrinsics[index]),
                "T_world_from_camera_array_sha256": _array_sha256(transform),
            }
        )
    manifest = {
        "camera_order": list(camera_ids),
        "inputs": {
            "selected_raw": {
                "path": str(raw.resolve()),
                "sha256": _sha256(raw),
            }
        },
        "camera_records": records,
    }

    cameras, identity = exporter._load_dynamic_cameras(manifest)

    assert [camera.camera_id for camera in cameras] == list(camera_ids)
    assert identity["sha256"] == _sha256(raw)
    manifest["camera_records"] = list(reversed(records))
    with pytest.raises(ValueError, match="declared roster"):
        exporter._load_dynamic_cameras(manifest)
