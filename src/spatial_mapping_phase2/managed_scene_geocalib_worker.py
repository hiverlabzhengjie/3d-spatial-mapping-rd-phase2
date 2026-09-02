"""Pinned GeoCalib model worker for one immutable managed-scene request."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_SOURCE_COMMIT = "97b8968e7798a66bf04fcf791fb535624241bda7"
EXPECTED_DISTORTED_WEIGHT_SHA256 = (
    "13cc505928e3ff4eb26c00bff73861ab2b11b804a546323456cf5462e1f8f447"
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise RuntimeError("provide exactly one managed-scene GeoCalib request path")
    run(Path(arguments[0]))
    return 0


def run(request_path: Path) -> dict[str, Any]:
    request = _read_json(request_path)
    if request.get("schema_version") != "managed-scene-geocalib-request-v1":
        raise RuntimeError("unsupported managed-scene GeoCalib request schema")
    source_directory = Path(_string(request, "geocalib_source_directory")).resolve()
    if _source_commit(source_directory) != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("GeoCalib source differs from the pinned Phase 2 revision")
    torch_home = Path(_string(request, "torch_home")).resolve()
    os.environ["TORCH_HOME"] = str(torch_home)
    sys.path.insert(0, str(source_directory))

    import numpy as np
    import torch
    from geocalib import GeoCalib  # type: ignore[import-not-found]

    from spatial_mapping_phase2.p04_intrinsic_domain import (
        looks_like_geocalib_initialization,
        summarize_intrinsic_candidate,
    )

    weights = _string(request, "weights")
    camera_model = _string(request, "camera_model")
    if weights != "distorted" or camera_model != "simple_radial":
        raise RuntimeError("managed-scene GeoCalib supports the pinned simple-radial candidate")
    weight_path = torch_home / "hub" / "geocalib" / "distorted.tar"
    if _sha256(weight_path) != EXPECTED_DISTORTED_WEIGHT_SHA256:
        raise RuntimeError("GeoCalib distorted checkpoint differs from the pinned identity")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCalib(weights=weights).to(device)
    estimates: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    cameras = request.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise RuntimeError("GeoCalib request camera roster is missing")
    for camera in cameras:
        if not isinstance(camera, dict):
            raise RuntimeError("GeoCalib camera request is malformed")
        camera_id = _string(camera, "camera_id")
        profile_version = _string(camera, "profile_version")
        frames = camera.get("frames")
        if not isinstance(frames, list) or len(frames) < 2:
            raise RuntimeError(f"{camera_id} requires at least two GeoCalib frames")
        tensors = []
        frame_records: list[dict[str, Any]] = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise RuntimeError(f"{camera_id} GeoCalib frame is malformed")
            path = Path(_string(frame, "path")).resolve()
            expected_sha256 = _string(frame, "sha256")
            if _sha256(path) != expected_sha256:
                raise RuntimeError(f"{camera_id} GeoCalib frame identity changed")
            tensor = model.load_image(path)
            expected_shape = (
                3,
                _integer(frame, "height_pixels"),
                _integer(frame, "width_pixels"),
            )
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(f"{camera_id} decoded frame dimensions changed")
            tensors.append(tensor)
            frame_records.append(
                {
                    "frame_id": _string(frame, "frame_id"),
                    "sha256": expected_sha256,
                    "width_pixels": expected_shape[2],
                    "height_pixels": expected_shape[1],
                }
            )
        individual = [
            model.calibrate(
                tensor[None].to(device),
                camera_model=camera_model,
                shared_intrinsics=False,
            )
            for tensor in tensors
        ]
        shared = model.calibrate(
            torch.stack(tensors).to(device),
            camera_model=camera_model,
            shared_intrinsics=True,
        )
        individual_camera = np.concatenate(
            [result["camera"].numpy() for result in individual], axis=0
        ).tolist()
        gravity = np.concatenate(
            [result["gravity"].vec3d.detach().cpu().numpy() for result in individual],
            axis=0,
        ).tolist()
        shared_camera = shared["camera"].numpy().tolist()
        flat = [number for row in (*individual_camera, *shared_camera, *gravity) for number in row]
        if not all(math.isfinite(float(value)) for value in flat):
            raise RuntimeError(f"{camera_id} GeoCalib returned non-finite values")
        if looks_like_geocalib_initialization(individual_camera, gravity):
            raise RuntimeError(f"{camera_id} GeoCalib optimizer remained at initialization")
        summary = summarize_intrinsic_candidate(individual_camera, shared_camera, gravity, 1)
        row = shared_camera[0]
        estimates.append(
            {
                "camera_id": camera_id,
                "profile_version": profile_version,
                "model": camera_model,
                "width_pixels": int(round(float(row[0]))),
                "height_pixels": int(round(float(row[1]))),
                "fx_pixels": float(row[2]),
                "fy_pixels": float(row[3]),
                "cx_pixels": float(row[4]),
                "cy_pixels": float(row[5]),
                "distortion": [float(row[6])],
                "within_camera_focal_cv": float(summary.focal_cv),
            }
        )
        input_records.append({"camera_id": camera_id, "frames": frame_records})
    result = {
        "schema_version": "xr03-independent-intrinsic-estimates-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": _string(request, "request_sha256"),
        "authority": "scene-specific-pinned-geocalib",
        "candidate_policy": (
            "independent per-camera multi-frame simple-radial GeoCalib; lens-group profiles "
            "remain challengers and are never forced"
        ),
        "estimates": estimates,
        "inputs": input_records,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "geocalib_source_commit": EXPECTED_SOURCE_COMMIT,
            "weight_sha256": _sha256(weight_path),
        },
    }
    output_path = Path(_string(request, "output_path")).resolve()
    if output_path.exists():
        raise RuntimeError("GeoCalib evidence output already exists")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("GeoCalib request must be an object")
    return value


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise RuntimeError(f"{key} must be a non-blank string")
    return item.strip()


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise RuntimeError(f"{key} must be a positive integer")
    return item


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required GeoCalib input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(path: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={path.as_posix()}",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot read GeoCalib source identity")
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
