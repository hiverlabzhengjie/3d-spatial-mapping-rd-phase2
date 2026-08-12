#!/usr/bin/env python3
"""Execute one provenance-bound P00 DA3 measurement run in a clean process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
CHECKPOINT_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
FIXTURE_PATHS = {
    "one_view": ("assets/examples/SOH/000.png",),
    "two_view": ("assets/examples/SOH/000.png", "assets/examples/SOH/010.png"),
    "three_view": (
        "assets/examples/SOH/000.png",
        "assets/examples/SOH/010.png",
        "assets/images/da3_radar.png",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _sha256_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--process-res", required=True, type=int)
    parser.add_argument("--deterministic-runtime", action="store_true")
    parser.add_argument("--cublas-workspace-config")
    parser.add_argument("--single-input-path", type=Path)
    return parser.parse_args()


def _build_inputs(args: argparse.Namespace) -> dict[str, list[Path]]:
    if args.single_input_path is not None:
        return {
            case_name: [args.single_input_path] * len(relative_paths)
            for case_name, relative_paths in FIXTURE_PATHS.items()
        }
    return {
        case_name: [args.source_dir / relative_path for relative_path in relative_paths]
        for case_name, relative_paths in FIXTURE_PATHS.items()
    }


def _instrument_phase(
    phase_seconds: dict[str, float], name: str, function: Callable[..., Any]
) -> Callable[..., Any]:
    def timed(*args: object, **kwargs: object) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            phase_seconds[name] = time.perf_counter() - started

    return timed


def main() -> int:
    args = _parse_arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.cublas_workspace_config is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = args.cublas_workspace_config

    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P00 WP5 native DA3 measurement worker",
        "success": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "runtime_id": args.runtime_id,
        "code_revision": args.code_revision,
        "source_path": str(args.source_dir),
        "source_commit": SOURCE_COMMIT,
        "checkpoint_path": str(args.checkpoint_dir),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "process_res": args.process_res,
        "process_res_method": "upper_bound_resize",
        "batching": (
            "One joint public-API inference call per view-count case; DA3 exposes no "
            "batch_size argument."
        ),
        "runtime_controls": {
            "deterministic_runtime": args.deterministic_runtime,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "cases": [],
    }
    try:
        if args.deterministic_runtime and args.cublas_workspace_config is None:
            raise ValueError("Deterministic runtime requires a cuBLAS workspace configuration.")
        if not args.dependency_lock.is_file():
            raise FileNotFoundError(f"Dependency lock is missing: {args.dependency_lock}")
        result["dependency_lock_path"] = str(args.dependency_lock)
        result["dependency_lock_sha256"] = _sha256_file(args.dependency_lock)

        inputs = _build_inputs(args)
        missing_inputs = [
            str(path) for paths in inputs.values() for path in paths if not path.is_file()
        ]
        if missing_inputs:
            raise FileNotFoundError(f"Input file(s) missing: {missing_inputs}")
        safetensors_path = args.checkpoint_dir / "model.safetensors"
        if not safetensors_path.is_file():
            raise FileNotFoundError(f"Mandatory checkpoint weights missing: {safetensors_path}")
        result["checkpoint_safetensors_sha256"] = _sha256_file(safetensors_path)
        result["inputs"] = {
            case_name: [{"path": str(path), "sha256": _sha256_file(path)} for path in paths]
            for case_name, paths in inputs.items()
        }
        result["input_manifest_sha256"] = _sha256_json(
            {
                case_name: [item["sha256"] for item in input_items]
                for case_name, input_items in result["inputs"].items()
            }
        )

        import numpy as np
        import psutil
        import torch
        from depth_anything_3.api import DepthAnything3

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the P00 DA3 measurement worker.")
        if args.deterministic_runtime:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        result.update(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "device_total_vram_gib": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 3
                ),
                "system_memory_before_load": {
                    "total_gib": round(psutil.virtual_memory().total / (1024**3), 3),
                    "available_gib": round(psutil.virtual_memory().available / (1024**3), 3),
                    "used_gib": round(psutil.virtual_memory().used / (1024**3), 3),
                },
            }
        )
        load_started = time.perf_counter()
        model = DepthAnything3.from_pretrained(str(args.checkpoint_dir))
        cpu_load_seconds = time.perf_counter() - load_started
        gpu_transfer_started = time.perf_counter()
        model = model.to("cuda")
        torch.cuda.synchronize()
        gpu_transfer_seconds = time.perf_counter() - gpu_transfer_started
        result["load"] = {
            "cpu_load_seconds": round(cpu_load_seconds, 3),
            "gpu_transfer_seconds": round(gpu_transfer_seconds, 3),
            "rss_gib_after_load": round(psutil.Process().memory_info().rss / (1024**3), 3),
            "vram_allocated_gib_after_load": round(torch.cuda.memory_allocated() / (1024**3), 3),
            "vram_reserved_gib_after_load": round(torch.cuda.memory_reserved() / (1024**3), 3),
        }
        original_preprocess_inputs = model._preprocess_inputs
        original_run_model_forward = model._run_model_forward
        original_convert_to_prediction = model._convert_to_prediction
        for case_name, paths in inputs.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            phase_seconds: dict[str, float] = {}
            model._preprocess_inputs = _instrument_phase(  # type: ignore[method-assign]
                phase_seconds, "input_preprocess", original_preprocess_inputs
            )
            model._run_model_forward = _instrument_phase(  # type: ignore[method-assign]
                phase_seconds, "model_forward", original_run_model_forward
            )
            model._convert_to_prediction = _instrument_phase(  # type: ignore[method-assign]
                phase_seconds, "output_conversion", original_convert_to_prediction
            )
            started = time.perf_counter()
            prediction = model.inference(
                [str(path) for path in paths],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
            torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - started
            arrays = {
                "depth": np.asarray(prediction.depth),
                "confidence": np.asarray(prediction.conf),
                "extrinsics": np.asarray(prediction.extrinsics),
                "intrinsics": np.asarray(prediction.intrinsics),
            }
            raw_path = args.run_dir / f"{args.result_name}_{case_name}_raw_prediction.npz"
            np.savez_compressed(raw_path, **arrays)
            phase_total = sum(phase_seconds.values())
            result["cases"].append(
                {
                    "case_id": case_name,
                    "requested_view_count": len(paths),
                    "output_shapes": {
                        field: list(array.shape) for field, array in arrays.items()
                    },
                    "all_finite": {
                        field: bool(np.isfinite(array).all()) for field, array in arrays.items()
                    },
                    "sha256": {field: _sha256_array(array) for field, array in arrays.items()},
                    "raw_prediction_path": str(raw_path),
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "phase_seconds": {
                        **{name: round(seconds, 6) for name, seconds in phase_seconds.items()},
                        "other_inference_seconds": round(
                            max(elapsed_seconds - phase_total, 0.0), 6
                        ),
                    },
                    "vram_peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / (1024**3), 3
                    ),
                    "vram_peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / (1024**3), 3
                    ),
                    "rss_gib_after_case": round(
                        psutil.Process().memory_info().rss / (1024**3), 3
                    ),
                    "system_memory_after_case": {
                        "available_gib": round(psutil.virtual_memory().available / (1024**3), 3),
                        "used_gib": round(psutil.virtual_memory().used / (1024**3), 3),
                    },
                }
            )
        result["success"] = True
    except BaseException as error:  # keep all expected and unexpected failures as evidence
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        result["traceback"] = traceback.format_exc()

    output_path = args.run_dir / f"{args.result_name}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
