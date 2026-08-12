#!/usr/bin/env python3
"""Measure source-preserving persistent-worker reuse for the native P00 DA3 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

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
OUTPUT_FIELDS = ("depth", "confidence", "extrinsics", "intrinsics")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--process-res", default=252, choices=(252, 504), type=int)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def main() -> int:
    args = _parse_arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    from spatial_mapping_phase2.measurement_analysis import summarize_array_difference
    from spatial_mapping_phase2.native_inference_policy import NativeInferencePolicy

    policy = NativeInferencePolicy(process_res=args.process_res)
    policy.validate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = policy.cublas_workspace_config
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P00 WP6 native DA3 persistent-worker optimization probe",
        "success": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "code_revision": args.code_revision,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "process_res": args.process_res,
        "policy": {
            "persistent_model_worker": policy.persistent_model_worker,
            "CUBLAS_WORKSPACE_CONFIG": policy.cublas_workspace_config,
            "deterministic_algorithms": policy.deterministic_algorithms,
            "cudnn_deterministic": policy.cudnn_deterministic,
            "cudnn_benchmark": policy.cudnn_benchmark,
            "use_upstream_autocast": policy.use_upstream_autocast,
            "use_upstream_swiglu_fallback": policy.use_upstream_swiglu_fallback,
            "enable_xformers": policy.enable_xformers,
            "enable_torch_compile": policy.enable_torch_compile,
        },
        "passes": [],
    }
    try:
        checkpoint_path = args.checkpoint_dir / "model.safetensors"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Mandatory checkpoint weights missing: {checkpoint_path}")
        inputs = {
            name: [args.source_dir / relative_path for relative_path in relative_paths]
            for name, relative_paths in FIXTURE_PATHS.items()
        }
        missing_inputs = [
            str(path) for paths in inputs.values() for path in paths if not path.is_file()
        ]
        if missing_inputs:
            raise FileNotFoundError(f"Pinned fixture(s) missing: {missing_inputs}")

        import numpy as np
        import torch
        from depth_anything_3.api import DepthAnything3

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the WP6 native optimization probe.")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        result["cuda"] = {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "upstream_selected_autocast_dtype": "bfloat16"
            if torch.cuda.is_bf16_supported()
            else "float16",
        }
        load_started = time.perf_counter()
        model = DepthAnything3.from_pretrained(str(args.checkpoint_dir)).to("cuda")
        torch.cuda.synchronize()
        result["load_seconds"] = round(time.perf_counter() - load_started, 3)
        result["checkpoint_safetensors_sha256"] = _sha256_file(checkpoint_path)
        raw_arrays_by_pass: dict[str, dict[str, dict[str, Any]]] = {}
        for pass_name in ("first_pass", "warm_pass"):
            pass_cases: list[dict[str, Any]] = []
            raw_arrays_by_pass[pass_name] = {}
            for case_name, paths in inputs.items():
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
                raw_path = args.run_dir / f"{pass_name}_{case_name}_raw_prediction.npz"
                np.savez_compressed(raw_path, **arrays)
                raw_arrays_by_pass[pass_name][case_name] = arrays
                pass_cases.append(
                    {
                        "case_id": case_name,
                        "requested_view_count": len(paths),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "sha256": {field: _sha256_array(array) for field, array in arrays.items()},
                        "raw_prediction_path": str(raw_path),
                        "vram_peak_allocated_gib": round(
                            torch.cuda.max_memory_allocated() / (1024**3), 3
                        ),
                    }
                )
            result["passes"].append({"name": pass_name, "cases": pass_cases})
        comparisons: list[dict[str, Any]] = []
        for case_name in FIXTURE_PATHS:
            fields = {
                field: summarize_array_difference(
                    raw_arrays_by_pass["first_pass"][case_name][field],
                    raw_arrays_by_pass["warm_pass"][case_name][field],
                )
                for field in OUTPUT_FIELDS
            }
            comparisons.append(
                {
                    "case_id": case_name,
                    "fields": fields,
                    "all_fields_exact_equal": all(item["exact_equal"] for item in fields.values()),
                }
            )
        result["first_to_warm_comparison"] = comparisons
        result["success"] = all(item["all_fields_exact_equal"] for item in comparisons)
    except BaseException as error:  # preserve rejected or failed optimization evidence
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        result["traceback"] = traceback.format_exc()

    output_path = args.run_dir / "wp6_persistent_worker_probe.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
