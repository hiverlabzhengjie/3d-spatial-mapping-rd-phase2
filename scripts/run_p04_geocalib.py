"""Run immutable multi-frame GeoCalib candidates for the P04 Camera 3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p04_intrinsic_domain import (
    looks_like_geocalib_initialization,
    summarize_intrinsic_candidate,
)

EXPECTED_SOURCE_COMMIT = "97b8968e7798a66bf04fcf791fb535624241bda7"
CAMERA_MODELS = ("pinhole", "simple_radial", "simple_divisional", "radial")
DISTORTION_PARAMETER_COUNTS = {
    "pinhole": 0,
    "simple_radial": 1,
    "simple_divisional": 1,
    "radial": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--geocalib-source-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("pinhole", "distorted"), required=True)
    parser.add_argument("--camera-model", choices=CAMERA_MODELS, required=True)
    parser.add_argument("--camera-id", default="office-cam-03")
    parser.add_argument("--profile-version", default="stream-profile-v1")
    parser.add_argument(
        "--frame",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="Immutable Camera 3 frame identity and local image path; repeat for each frame",
    )
    arguments = parser.parse_args()
    run(arguments)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.run_dir.exists():
        raise RuntimeError("run directory already exists")
    source_commit = _source_commit(arguments.geocalib_source_dir)
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("GeoCalib source commit differs from the P04 pinned revision")
    frames = [_parse_frame(value) for value in arguments.frame]
    if len(frames) < 2 or len({item[0] for item in frames}) != len(frames):
        raise RuntimeError("provide at least two uniquely identified frames")
    camera_id = str(arguments.camera_id).strip()
    profile_version = str(arguments.profile_version).strip()
    if not camera_id or not profile_version:
        raise RuntimeError("camera and profile identities must be non-blank")
    arguments.run_dir.mkdir(parents=True)
    raw_dir = arguments.run_dir / "raw"
    raw_dir.mkdir()
    os.environ["TORCH_HOME"] = str(arguments.torch_home.resolve())

    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from geocalib import GeoCalib  # type: ignore[import-not-found]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCalib(weights=arguments.weights).to(device)
    tensors = []
    frame_records: list[dict[str, Any]] = []
    for frame_id, path in frames:
        content = path.read_bytes()
        tensor = model.load_image(path)
        if tuple(tensor.shape) != (3, 1080, 1920):
            raise RuntimeError(f"{frame_id} is not a native 1920x1080 RGB frame")
        tensors.append(tensor)
        frame_records.append(
            {
                "frame_id": frame_id,
                "camera_id": camera_id,
                "profile_version": profile_version,
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
                "source_path": str(path.resolve()),
            }
        )
    batch = torch.stack(tensors).to(device)
    individual_results = [
        model.calibrate(
            tensor[None].to(device),
            camera_model=arguments.camera_model,
            shared_intrinsics=False,
        )
        for tensor in tensors
    ]
    shared = model.calibrate(batch, camera_model=arguments.camera_model, shared_intrinsics=True)
    individual_camera_array = np.concatenate(
        [result["camera"].numpy() for result in individual_results], axis=0
    )
    individual_gravity_array = np.concatenate(
        [result["gravity"].vec3d.detach().cpu().numpy() for result in individual_results],
        axis=0,
    )
    individual_camera = individual_camera_array.tolist()
    shared_camera = shared["camera"].numpy().tolist()
    gravity = individual_gravity_array.tolist()

    raw_values: dict[str, Any] = {}
    raw_values["individual_camera"] = individual_camera_array
    raw_values["individual_gravity"] = individual_gravity_array
    for frame_index, result in enumerate(individual_results):
        for key, value in result.items():
            if key in {"camera", "gravity"}:
                continue
            if isinstance(value, torch.Tensor):
                raw_values[f"individual_{frame_index}_{key}"] = value.detach().cpu().numpy()
    raw_values["shared_camera"] = shared["camera"].numpy()
    raw_values["shared_gravity"] = shared["gravity"].vec3d.detach().cpu().numpy()
    for key, value in shared.items():
        if key in {"camera", "gravity"}:
            continue
        if isinstance(value, torch.Tensor):
            raw_values[f"shared_{key}"] = value.detach().cpu().numpy()
    finite_output = all(np.isfinite(value).all() for value in raw_values.values())
    stalled = looks_like_geocalib_initialization(individual_camera, gravity)
    summary = (
        summarize_intrinsic_candidate(
            individual_camera,
            shared_camera,
            gravity,
            DISTORTION_PARAMETER_COUNTS[arguments.camera_model],
        )
        if finite_output and not stalled
        else None
    )
    raw_path = raw_dir / f"{arguments.weights}-{arguments.camera_model}.npz"
    np.savez_compressed(raw_path, **raw_values)
    weight_path = arguments.torch_home / "hub" / "geocalib" / f"{arguments.weights}.tar"
    manifest: dict[str, Any] = {
        "schema_version": "p04-geocalib-candidate-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "camera_id": camera_id,
        "profile_version": profile_version,
        "frames": frame_records,
        "candidate": {
            "weights": arguments.weights,
            "camera_model": arguments.camera_model,
            "shared_intrinsics": True,
            "individual_camera_parameters": individual_camera,
            "shared_camera_parameters": shared_camera,
            "individual_gravity_camera": gravity,
            "stability": None if summary is None else summary.to_dict(),
        },
        "identities": {
            "geocalib_source_commit": source_commit,
            "weight_sha256": _sha256(weight_path),
            "raw_npz_relative_path": str(raw_path.relative_to(arguments.run_dir)),
            "raw_npz_sha256": _sha256(raw_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "status": (
            "provisional-candidate"
            if summary is not None
            else "rejected-nonfinite-or-stalled-optimizer"
        ),
        "authority_note": (
            "GeoCalib intrinsic/gravity evidence only; model selection and pose remain pending"
        ),
    }
    manifest_path = arguments.run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _parse_frame(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise RuntimeError("frame must use ID=PATH")
    frame_id, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not frame_id.strip() or not path.is_file():
        raise RuntimeError("frame identity must be non-blank and its path must exist")
    return frame_id.strip(), path


def _source_commit(source_dir: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={source_dir.resolve().as_posix()}",
            "-C",
            str(source_dir),
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


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required immutable artifact is missing: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
