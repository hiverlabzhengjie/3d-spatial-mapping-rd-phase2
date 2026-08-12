#!/usr/bin/env python3
"""Run bounded P00 native supporting-stack smoke operations with retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spatial_mapping_phase2.supporting_stack_contract import (  # noqa: E402
    validate_supporting_stack_result,
)

GEOCALIB_SOURCE_COMMIT = "97b8968e7798a66bf04fcf791fb535624241bda7"
GEOCALIB_WEIGHT_URL = "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--geocalib-source-dir", required=True, type=Path)
    parser.add_argument("--torch-home", required=True, type=Path)
    parser.add_argument("--code-revision", required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_run_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing smoke-run directory: {path}")
    path.mkdir(parents=True)


def _record_component(
    result: dict[str, Any], name: str, operation: Callable[[], dict[str, Any]]
) -> None:
    started = time.perf_counter()
    try:
        details = operation()
        result["components"][name] = {
            "success": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            **details,
        }
    except BaseException as error:  # Retain every component failure in the manifest.
        result["components"][name] = {
            "success": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }


def _smoke_scipy() -> dict[str, Any]:
    import numpy as np
    import scipy
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    expected = np.array([1.5, -2.0])
    fit = least_squares(lambda values: values - expected, x0=np.zeros(2))
    rotation = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
    matrix = rotation.as_matrix()
    if (
        not fit.success
        or not np.allclose(fit.x, expected)
        or not np.allclose(matrix.T @ matrix, np.eye(3))
    ):
        raise RuntimeError("SciPy optimization/rotation smoke result failed validation.")
    return {"version": scipy.__version__, "least_squares_solution": fit.x.tolist()}


def _smoke_opencv() -> dict[str, Any]:
    import cv2
    import numpy as np

    object_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.2, 0.3, 1.0],
            [0.8, 0.4, 1.2],
        ],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[420.0, 0.0, 160.0], [0.0, 415.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    known_rvec = np.array([[0.08], [-0.03], [0.04]], dtype=np.float64)
    known_tvec = np.array([[0.1], [-0.15], [4.5]], dtype=np.float64)
    image_points, _ = cv2.projectPoints(object_points, known_rvec, known_tvec, camera_matrix, None)
    solved, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None)
    if not solved:
        raise RuntimeError("OpenCV solvePnP did not solve the synthetic projection.")
    reprojection, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    maximum_error = float(np.max(np.abs(reprojection - image_points)))
    if maximum_error > 1e-6:
        raise RuntimeError(f"OpenCV synthetic reprojection error is too large: {maximum_error}")
    return {"version": cv2.__version__, "max_reprojection_error_pixels": maximum_error}


def _smoke_open3d() -> dict[str, Any]:
    import numpy as np
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    extent = np.asarray(cloud.get_axis_aligned_bounding_box().get_extent())
    if len(cloud.points) != 4 or not np.allclose(extent, np.ones(3)):
        raise RuntimeError("Open3D point-cloud smoke result failed validation.")
    return {
        "version": o3d.__version__,
        "point_count": len(cloud.points),
        "extent": extent.tolist(),
    }


def _smoke_rerun(run_dir: Path) -> dict[str, Any]:
    import rerun as rr

    recording_path = run_dir / "rerun_stack_smoke.rrd"
    rr.init("p00-supporting-stack-smoke", spawn=False)
    rr.save(str(recording_path))
    rr.log("world/smoke_points", rr.Points3D([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    if not recording_path.is_file() or recording_path.stat().st_size == 0:
        raise RuntimeError("Rerun did not write a non-empty recording.")
    return {
        "version": rr.__version__,
        "recording_path": str(recording_path),
        "recording_sha256": _sha256_file(recording_path),
        "recording_bytes": recording_path.stat().st_size,
    }


def _smoke_pyav_ffmpeg(run_dir: Path) -> dict[str, Any]:
    import av

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError(
            "Both ffmpeg and ffprobe must be on PATH for the PyAV/FFmpeg smoke."
        )
    video_path = run_dir / "ffmpeg_synthetic_smoke.mp4"
    encode = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x24:r=3",
            "-t",
            "1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if encode.returncode != 0:
        raise RuntimeError(f"FFmpeg synthetic encode failed: {encode.stderr.strip()}")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"FFprobe synthetic inspection failed: {probe.stderr.strip()}")
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1 or streams[0].get("width") != 32 or streams[0].get("height") != 24:
        raise RuntimeError("FFprobe did not report the expected synthetic video stream.")
    with av.open(str(video_path)) as container:
        decoded_frames = sum(1 for frame in container.decode(video=0) if frame is not None)
    if decoded_frames < 3:
        raise RuntimeError(f"PyAV decoded too few frames: {decoded_frames}")
    return {
        "pyav_version": av.__version__,
        "ffmpeg_path": ffmpeg,
        "ffprobe_path": ffprobe,
        "stream": streams[0],
        "decoded_frame_count": decoded_frames,
        "video_path": str(video_path),
        "video_sha256": _sha256_file(video_path),
    }


def _smoke_pillow_heif(run_dir: Path) -> dict[str, Any]:
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    image_path = run_dir / "pillow_heif_smoke.heic"
    Image.new("RGB", (8, 6), color=(12, 34, 56)).save(image_path, format="HEIF", quality=100)
    with Image.open(image_path) as decoded:
        if decoded.size != (8, 6) or decoded.mode != "RGB":
            raise RuntimeError("Pillow-Heif did not round-trip the expected image shape/mode.")
    return {
        "version": pillow_heif.__version__,
        "image_path": str(image_path),
        "image_sha256": _sha256_file(image_path),
    }


def _smoke_pre_commit() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pre_commit", "--version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pre-commit version command failed: {completed.stderr.strip()}")
    return {
        "version": importlib.metadata.version("pre-commit"),
        "command_output": completed.stdout.strip(),
    }


def _smoke_web() -> dict[str, Any]:
    import fastapi
    import uvicorn
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    application = FastAPI()

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    response = TestClient(application).get("/healthz")
    if response.status_code != 200 or response.json() != {"status": "ok"}:
        raise RuntimeError("FastAPI health endpoint smoke did not return the expected response.")
    configuration = uvicorn.Config(application, host="127.0.0.1", port=0, log_level="warning")
    configuration.load()
    if configuration.loaded_app is None:
        raise RuntimeError("Uvicorn did not load the FastAPI application.")
    return {"fastapi_version": fastapi.__version__, "uvicorn_version": uvicorn.__version__}


def _smoke_geocalib(torch_home: Path, source_dir: Path) -> dict[str, Any]:
    os.environ["TORCH_HOME"] = str(torch_home)
    import torch
    from geocalib import GeoCalib

    source_commit = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if source_commit.returncode != 0:
        raise RuntimeError(f"Cannot read GeoCalib source identity: {source_commit.stderr.strip()}")
    if source_commit.stdout.strip() != GEOCALIB_SOURCE_COMMIT:
        raise RuntimeError("GeoCalib source commit differs from the WP8 pinned revision.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCalib(weights="pinhole").to(device)
    image = torch.full((3, 96, 128), 0.5, dtype=torch.float32, device=device)
    calibration = model.calibrate(image)
    required = ("camera", "gravity", "covariance")
    if any(key not in calibration for key in required):
        raise RuntimeError("GeoCalib result omits a required calibration field.")
    tensor_fields = {
        name: value
        for name, value in calibration.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensor_fields or any(
        not torch.isfinite(value).all() for value in tensor_fields.values()
    ):
        raise RuntimeError("GeoCalib returned missing or non-finite tensor calibration data.")
    weight_path = torch_home / "hub" / "geocalib" / "pinhole.tar"
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"GeoCalib did not retain the expected pinhole weight file: {weight_path}"
        )
    return {
        "source_commit": source_commit.stdout.strip(),
        "weight_url": GEOCALIB_WEIGHT_URL,
        "weight_path": str(weight_path),
        "weight_sha256": _sha256_file(weight_path),
        "device": str(device),
        "result_keys": sorted(calibration),
        "tensor_shapes": {name: list(value.shape) for name, value in tensor_fields.items()},
    }


def main() -> int:
    args = _parse_arguments()
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P00 WP8 native supporting-stack smoke",
        "success": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "code_revision": args.code_revision,
        "geocalib_source_dir": str(args.geocalib_source_dir),
        "geocalib_expected_source_commit": GEOCALIB_SOURCE_COMMIT,
        "torch_home": str(args.torch_home),
        "host": {"platform": platform.platform(), "python": sys.version},
        "components": {},
    }
    try:
        _prepare_run_directory(args.run_dir)
        args.torch_home.mkdir(parents=True, exist_ok=True)
        _record_component(result, "scipy", _smoke_scipy)
        _record_component(result, "opencv", _smoke_opencv)
        _record_component(result, "open3d", _smoke_open3d)
        _record_component(result, "rerun", lambda: _smoke_rerun(args.run_dir))
        _record_component(result, "pyav_ffmpeg", lambda: _smoke_pyav_ffmpeg(args.run_dir))
        _record_component(result, "pillow_heif", lambda: _smoke_pillow_heif(args.run_dir))
        _record_component(result, "pre_commit", _smoke_pre_commit)
        _record_component(result, "web", _smoke_web)
        _record_component(
            result,
            "geocalib",
            lambda: _smoke_geocalib(args.torch_home, args.geocalib_source_dir),
        )
        result["success"] = all(
            component.get("success") is True for component in result["components"].values()
        )
        if result["success"]:
            validate_supporting_stack_result(result)
    except BaseException as error:
        result["orchestration_error_type"] = type(error).__name__
        result["orchestration_error_message"] = str(error)
        result["orchestration_traceback"] = traceback.format_exc()
    result["completed_at_utc"] = datetime.now(UTC).isoformat()
    output_path = args.run_dir / "supporting_stack_smoke.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
