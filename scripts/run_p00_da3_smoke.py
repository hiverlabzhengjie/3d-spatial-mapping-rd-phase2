#!/usr/bin/env python3
"""Run the P00 controlled DA3 checkpoint smoke against pinned upstream fixtures.

This is a P00 feasibility harness, not a geometry-acceptance pipeline.  It loads
the mandatory checkpoint only from a caller-supplied local directory and invokes
the public DA3 API for one, two, and three input views.  It records provenance,
resource measurements, finite-value checks, and optional raw model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
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


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--write-raw", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        name: [args.source_dir / relative_path for relative_path in relative_paths]
        for name, relative_paths in FIXTURE_PATHS.items()
    }
    missing_inputs = [
        str(path) for paths in inputs.values() for path in paths if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"Pinned DA3 fixture(s) missing: {missing_inputs}")
    safetensors_path = args.checkpoint_dir / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Mandatory checkpoint weights missing: {safetensors_path}")

    import numpy as np
    import psutil
    import torch
    from depth_anything_3.api import DepthAnything3

    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P00 WP4 controlled DA3 checkpoint smoke",
        "success": False,
        "platform_label": args.platform_label,
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_total_vram_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3)
            if torch.cuda.is_available()
            else None
        ),
        "source_path": str(args.source_dir),
        "source_commit": SOURCE_COMMIT,
        "checkpoint_path": str(args.checkpoint_dir),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_safetensors_sha256": _sha256_file(safetensors_path),
        "process_res": args.process_res,
        "process_res_method": "upper_bound_resize",
        "batching": (
            "Each case is one joint DA3 public-API inference call; that API exposes no "
            "batch_size argument."
        ),
        "inputs": {
            name: [{"path": str(path), "sha256": _sha256_file(path)} for path in paths]
            for name, paths in inputs.items()
        },
        "cases": [],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this P00 DA3 smoke.")
        load_started = time.perf_counter()
        model = DepthAnything3.from_pretrained(str(args.checkpoint_dir))
        cpu_load_seconds = time.perf_counter() - load_started
        gpu_started = time.perf_counter()
        model = model.to("cuda")
        torch.cuda.synchronize()
        gpu_transfer_seconds = time.perf_counter() - gpu_started
        result["load"] = {
            "cpu_load_seconds": round(cpu_load_seconds, 3),
            "gpu_transfer_seconds": round(gpu_transfer_seconds, 3),
            "rss_gib_after_load": round(psutil.Process().memory_info().rss / (1024**3), 3),
            "vram_allocated_gib_after_load": round(torch.cuda.memory_allocated() / (1024**3), 3),
            "vram_reserved_gib_after_load": round(torch.cuda.memory_reserved() / (1024**3), 3),
        }
        for name, paths in inputs.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            prediction = model.inference(
                [str(path) for path in paths],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
            torch.cuda.synchronize()
            arrays = {
                "depth": np.asarray(prediction.depth),
                "confidence": np.asarray(prediction.conf),
                "extrinsics": np.asarray(prediction.extrinsics),
                "intrinsics": np.asarray(prediction.intrinsics),
            }
            if args.write_raw:
                raw_file_name = f"{args.result_name}_{name}_raw_prediction.npz"
                raw_prediction_path = args.run_dir / raw_file_name
                np.savez_compressed(raw_prediction_path, **arrays)
            result["cases"].append(
                {
                    "name": name,
                    "requested_view_count": len(paths),
                    "output_shapes": {
                        field: list(array.shape) for field, array in arrays.items()
                    },
                    "all_finite": {
                        field: bool(np.isfinite(array).all()) for field, array in arrays.items()
                    },
                    "sha256": {field: _sha256_array(array) for field, array in arrays.items()},
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "vram_peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / (1024**3), 3
                    ),
                    "vram_peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / (1024**3), 3
                    ),
                    "rss_gib_after_case": round(psutil.Process().memory_info().rss / (1024**3), 3),
                }
            )
        result["success"] = True
    except BaseException as error:  # evidence must retain the original failure
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        result["traceback"] = traceback.format_exc()
    output_path = args.run_dir / f"{args.result_name}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
