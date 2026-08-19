"""Run the frozen P06 matrix with the exact local DA3 Nested Giant-Large 1.1 runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p06_da3_evaluation import (
    CHECKPOINT_NAME,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    PROCESS_RESOLUTION,
    REPETITIONS,
    SOURCE_COMMIT,
    array_sha256,
    file_sha256,
    validate_complete_case_matrix,
    validate_prediction_arrays,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.input_manifest.parent
    raw_dir = run_dir / "raw"
    output_manifest_path = run_dir / "run-manifest.json"
    if raw_dir.exists() or output_manifest_path.exists():
        parser.error("raw output or run manifest already exists; P06 execution never overwrites")
    input_manifest = _read_json(args.input_manifest)
    cases_value = input_manifest.get("case_matrix")
    if not isinstance(cases_value, list) or not all(
        isinstance(item, dict) for item in cases_value
    ):
        parser.error("input manifest case matrix is malformed")
    cases = validate_complete_case_matrix(cases_value)
    model_requirement = input_manifest.get("model_requirement")
    if not isinstance(model_requirement, dict):
        parser.error("input manifest model requirement is missing")
    expected_requirement = {
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_commit": SOURCE_COMMIT,
        "process_resolution": PROCESS_RESOLUTION,
        "process_resolution_method": "upper_bound_resize",
        "repetitions": REPETITIONS,
    }
    if model_requirement != expected_requirement:
        parser.error("input manifest differs from the exact frozen model/runtime requirement")
    cameras_value = input_manifest.get("cameras")
    cameras = _camera_records(cameras_value)
    for camera_id, camera in cameras.items():
        derivative = camera.get("pinhole_derivative")
        if not isinstance(derivative, dict):
            parser.error(f"pinhole derivative record is malformed for {camera_id}")
        path = Path(str(derivative.get("path")))
        if file_sha256(path) != derivative.get("sha256"):
            parser.error(f"pinhole derivative identity mismatch for {camera_id}")

    source_head = _git(args.source_dir, "rev-parse", "HEAD")
    source_status = _git(args.source_dir, "status", "--porcelain")
    if source_head != SOURCE_COMMIT:
        parser.error("local DA3 source is not at the frozen commit")
    if source_status:
        parser.error("local DA3 source cache has uncommitted changes")
    weights_path = args.checkpoint_dir / "model.safetensors"
    if not weights_path.is_file() or file_sha256(weights_path) != CHECKPOINT_SHA256:
        parser.error("mandatory DA3 1.1 checkpoint weight identity mismatch")
    config_path = args.checkpoint_dir / "config.json"
    config = _read_json(config_path)
    if config.get("model_name") != "da3nested-giant-large":
        parser.error("checkpoint config is not the nested giant-large architecture")
    if importlib.util.find_spec("xformers") is not None:
        parser.error("xformers is installed but prohibited by D015")

    # D015 requires this environment control before Torch import.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import psutil  # type: ignore[import-untyped]
    import torch
    from depth_anything_3.api import DepthAnything3  # type: ignore[import-untyped]

    torch.use_deterministic_algorithms(True)  # type: ignore[no-untyped-call]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw_dir.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": "p06-da3-run-manifest-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "success": False,
        "input_manifest_path": str(args.input_manifest.resolve()),
        "input_manifest_sha256": file_sha256(args.input_manifest),
        "authority": (
            "diagnostic evaluation only; no accepted geometry, pose, connectivity, scale, "
            "facility-frame or fusion authority"
        ),
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "numpy_version": np.__version__,
            "opencv_version": _version("opencv-python"),
            "psutil_version": psutil.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_total_vram_gib": (
                round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,  # type: ignore[attr-defined]
                    6,
                )
                if torch.cuda.is_available()
                else None
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),  # type: ignore[no-untyped-call]
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "xformers_installed": False,
            "gsplat_installed": importlib.util.find_spec("gsplat") is not None,
            "infer_gs": False,
            "manual_precision_change": False,
            "torch_compile": False,
        },
        "model": {
            "checkpoint_name": CHECKPOINT_NAME,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_path": str(args.checkpoint_dir.resolve()),
            "checkpoint_weight_sha256": CHECKPOINT_SHA256,
            "checkpoint_weight_byte_count": weights_path.stat().st_size,
            "checkpoint_config_sha256": file_sha256(config_path),
            "source_path": str(args.source_dir.resolve()),
            "source_commit": source_head,
            "source_status_porcelain": source_status,
        },
        "execution": {
            "process_resolution": PROCESS_RESOLUTION,
            "process_resolution_method": "upper_bound_resize",
            "repetitions": REPETITIONS,
            "persistent_model": True,
            "rng_seed_reset_before_each_invocation": 42,
            "posed_output_boundary": (
                "exact vendor prediction before public-API Umeyama alignment; required for a "
                "common two/three-view diagnostic boundary after retained two-view rank failure"
            ),
            "posed_post_prediction_alignment": False,
            "export_dir": None,
            "export_format": None,
        },
        "cases": [],
    }
    process = psutil.Process()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the selected exact P06 runtime")
        load_cpu_start = time.perf_counter()
        model = DepthAnything3.from_pretrained(str(args.checkpoint_dir))
        load_cpu_seconds = time.perf_counter() - load_cpu_start
        transfer_start = time.perf_counter()
        model = model.to("cuda")
        torch.cuda.synchronize()
        transfer_seconds = time.perf_counter() - transfer_start
        manifest["model_load"] = {
            "cpu_load_seconds": load_cpu_seconds,
            "gpu_transfer_seconds": transfer_seconds,
            "rss_gib_after_load": process.memory_info().rss / 1024**3,
            "vram_allocated_gib_after_load": torch.cuda.memory_allocated() / 1024**3,
            "vram_reserved_gib_after_load": torch.cuda.memory_reserved() / 1024**3,
        }
        all_success = True
        for case in cases:
            case_record: dict[str, Any] = {**case.to_dict(), "success": False, "repetitions": []}
            paths = [
                str(Path(str(cameras[camera_id]["pinhole_derivative"]["path"])))
                for camera_id in case.camera_ids
            ]
            extrinsics: NDArray[Any] | None = None
            intrinsics: NDArray[Any] | None = None
            if case.uses_input_camera_parameters:
                extrinsics = np.asarray(
                    [
                        cameras[camera_id]["seed_transform"]["T_camera_from_world_for_DA3"]
                        for camera_id in case.camera_ids
                    ],
                    dtype=np.float32,
                )
                intrinsics = np.asarray(
                    [
                        cameras[camera_id]["intrinsics"]["K_pinhole"]
                        for camera_id in case.camera_ids
                    ],
                    dtype=np.float32,
                )
            try:
                for repetition in range(1, REPETITIONS + 1):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    _reset_rng(torch, 42)
                    started = time.perf_counter()
                    if case.uses_input_camera_parameters:
                        if extrinsics is None or intrinsics is None:
                            raise RuntimeError("posed case omitted frozen camera conditions")
                        prediction = _posed_prediction_before_umeyama(
                            model,
                            paths,
                            extrinsics,
                            intrinsics,
                            case.reference_view_strategy,
                        )
                    else:
                        prediction = model.inference(
                            paths,
                            extrinsics=None,
                            intrinsics=None,
                            align_to_input_ext_scale=True,
                            infer_gs=False,
                            use_ray_pose=False,
                            ref_view_strategy=case.reference_view_strategy,
                            process_res=PROCESS_RESOLUTION,
                            process_res_method="upper_bound_resize",
                            export_dir=None,
                        )
                    torch.cuda.synchronize()
                    confidence = prediction.conf
                    predicted_extrinsics = prediction.extrinsics
                    predicted_intrinsics = prediction.intrinsics
                    processed_images = prediction.processed_images
                    if (
                        confidence is None
                        or predicted_extrinsics is None
                        or predicted_intrinsics is None
                        or processed_images is None
                    ):
                        raise RuntimeError("DA3 prediction omitted a mandatory P06 raw field")
                    arrays = {
                        "depth": np.asarray(prediction.depth),
                        "confidence": np.asarray(confidence),
                        "extrinsics": np.asarray(predicted_extrinsics),
                        "intrinsics": np.asarray(predicted_intrinsics),
                        "processed_images": np.asarray(processed_images),
                    }
                    shapes = validate_prediction_arrays(arrays, len(case.camera_ids))
                    raw_name = f"{case.case_id}-repeat-{repetition}.npz"
                    raw_path = raw_dir / raw_name
                    optional: dict[str, NDArray[Any]] = {
                        "is_metric": np.asarray(prediction.is_metric),
                    }
                    if prediction.sky is not None:
                        optional["sky"] = np.asarray(prediction.sky)
                    if prediction.scale_factor is not None:
                        optional["scale_factor"] = np.asarray(prediction.scale_factor)
                    np.savez_compressed(raw_path, **arrays, **optional)
                    repetition_record = {
                        "repetition": repetition,
                        "raw_path": str(raw_path.resolve()),
                        "raw_sha256": file_sha256(raw_path),
                        "raw_byte_count": raw_path.stat().st_size,
                        "output_shapes": shapes,
                        "array_sha256": {
                            field: array_sha256(value) for field, value in arrays.items()
                        },
                        "is_metric": int(prediction.is_metric),
                        "sky_preserved": prediction.sky is not None,
                        "scale_factor": (
                            None
                            if prediction.scale_factor is None
                            else float(prediction.scale_factor)
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                        "rss_gib_after_case": process.memory_info().rss / 1024**3,
                        "vram_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "vram_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    }
                    case_record["repetitions"].append(repetition_record)
                case_record["success"] = True
            except BaseException as error:  # raw diagnostic failures must be retained
                all_success = False
                case_record["error_type"] = type(error).__name__
                case_record["error_message"] = str(error)
                case_record["traceback"] = traceback.format_exc()
            manifest["cases"].append(case_record)
        manifest["success"] = all_success
    except BaseException as error:  # model/runtime failure still produces a manifest
        manifest["error_type"] = type(error).__name__
        manifest["error_message"] = str(error)
        manifest["traceback"] = traceback.format_exc()
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_manifest": str(output_manifest_path),
                "run_manifest_sha256": file_sha256(output_manifest_path),
                "success": manifest["success"],
                "case_count": len(manifest["cases"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["success"] else 1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is malformed: {path}")
    return value


def _camera_records(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("input manifest cameras must be a list of records")
    records = {str(item.get("camera_id")): item for item in value}
    if len(records) != 4:
        raise ValueError("input manifest must contain four unique camera records")
    return records


def _git(source_dir: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _posed_prediction_before_umeyama(
    model: Any,
    paths: list[str],
    extrinsics: NDArray[Any],
    intrinsics: NDArray[Any],
    reference_view_strategy: str,
) -> Any:
    """Run exact vendor learned computation but omit rank-invalid public Sim(3) alignment.

    Nested-model metric scaling occurs inside the learned model before this boundary. This helper
    does not edit vendor source, change inputs, estimate a replacement alignment or copy supplied
    camera parameters into the output.
    """

    images_cpu, processed_extrinsics, processed_intrinsics = model._preprocess_inputs(
        paths,
        extrinsics,
        intrinsics,
        PROCESS_RESOLUTION,
        "upper_bound_resize",
    )
    images, extrinsics_tensor, intrinsics_tensor = model._prepare_model_inputs(
        images_cpu,
        processed_extrinsics,
        processed_intrinsics,
    )
    normalized_extrinsics = model._normalize_extrinsics(extrinsics_tensor.clone())
    raw_output = model._run_model_forward(
        images,
        normalized_extrinsics,
        intrinsics_tensor,
        [],
        False,
        False,
        reference_view_strategy,
    )
    prediction = model._convert_to_prediction(raw_output)
    return model._add_processed_images(prediction, images_cpu)


def _reset_rng(torch_module: Any, seed: int) -> None:
    """Reset every used RNG before an invocation to control vendor quantile subsampling."""

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
