#!/usr/bin/env python3
"""Orchestrate the native P00 DA3 repeatability and failure measurement matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


WORKER_SCRIPT = Path(__file__).with_name("run_p00_da3_measurement_worker.py")
OUTPUT_FIELDS = ("depth", "confidence", "extrinsics", "intrinsics")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--runtime-id", default="native-windows-py311")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--resolutions", nargs="+", default=[252, 504], type=int)
    parser.add_argument("--repeat-count", default=2, type=int)
    parser.add_argument("--cublas-workspace-config", default=":4096:8")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _nvidia_smi_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _read_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_worker(
    args: argparse.Namespace,
    *,
    result_name: str,
    process_res: int,
    checkpoint_dir: Path | None = None,
    single_input_path: Path | None = None,
) -> dict[str, object]:
    result_path = args.run_dir / f"{result_name}.json"
    existing_result = _read_result(result_path)
    if args.resume and existing_result is not None:
        return {
            "result_name": result_name,
            "resumed": True,
            "return_code": 0 if existing_result.get("success") else 1,
            "result": existing_result,
        }
    command = [
        str(args.python),
        str(WORKER_SCRIPT),
        "--source-dir",
        str(args.source_dir),
        "--checkpoint-dir",
        str(checkpoint_dir or args.checkpoint_dir),
        "--run-dir",
        str(args.run_dir),
        "--result-name",
        result_name,
        "--runtime-id",
        args.runtime_id,
        "--code-revision",
        args.code_revision,
        "--dependency-lock",
        str(args.dependency_lock),
        "--process-res",
        str(process_res),
        "--deterministic-runtime",
        "--cublas-workspace-config",
        args.cublas_workspace_config,
    ]
    if single_input_path is not None:
        command.extend(("--single-input-path", str(single_input_path)))
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = args.cublas_workspace_config
    before_gpu = _nvidia_smi_snapshot()
    started_at_utc = datetime.now(UTC).isoformat()
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment
    )
    finished_at_utc = datetime.now(UTC).isoformat()
    (args.run_dir / f"{result_name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (args.run_dir / f"{result_name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return {
        "result_name": result_name,
        "command": command,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "return_code": completed.returncode,
        "gpu_before": before_gpu,
        "gpu_after": _nvidia_smi_snapshot(),
        "result": _read_result(result_path),
    }


def _case_by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in result["cases"]}


def _compare_repetitions(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    from spatial_mapping_phase2.measurement_analysis import summarize_array_difference

    first_result = first["result"]
    second_result = second["result"]
    if not isinstance(first_result, dict) or not isinstance(second_result, dict):
        return {"comparison_available": False, "reason": "A worker result manifest is missing."}
    if not first_result.get("success") or not second_result.get("success"):
        return {"comparison_available": False, "reason": "A normal worker did not succeed."}

    import numpy as np

    comparisons: list[dict[str, object]] = []
    first_cases = _case_by_id(first_result)
    second_cases = _case_by_id(second_result)
    for case_id, first_case in first_cases.items():
        second_case = second_cases[case_id]
        with np.load(first_case["raw_prediction_path"]) as first_arrays:
            with np.load(second_case["raw_prediction_path"]) as second_arrays:
                fields = {
                    field: summarize_array_difference(first_arrays[field], second_arrays[field])
                    for field in OUTPUT_FIELDS
                }
        comparisons.append(
            {
                "case_id": case_id,
                "fields": fields,
                "all_fields_exact_equal": all(
                    comparison["exact_equal"] for comparison in fields.values()
                ),
            }
        )
    return {"comparison_available": True, "cases": comparisons}


def main() -> int:
    args = _parse_arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.repeat_count < 2:
        raise ValueError("WP5 repeatability requires at least two clean-process repetitions.")
    if any(resolution <= 0 for resolution in args.resolutions):
        raise ValueError("Every requested process resolution must be positive.")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "P00 WP5 native DA3 measurement matrix",
        "success": False,
        "runtime_id": args.runtime_id,
        "code_revision": args.code_revision,
        "resolutions": args.resolutions,
        "repeat_count": args.repeat_count,
        "runtime_controls": {
            "deterministic_runtime": True,
            "CUBLAS_WORKSPACE_CONFIG": args.cublas_workspace_config,
        },
        "normal_runs": [],
        "repeatability": [],
        "failure_probes": [],
    }
    normal_runs_by_resolution: dict[int, list[dict[str, object]]] = {}
    for resolution in args.resolutions:
        resolution_runs: list[dict[str, object]] = []
        for repeat_index in range(args.repeat_count):
            result_name = f"native_r{resolution}_repeat{repeat_index + 1}"
            run = _run_worker(args, result_name=result_name, process_res=resolution)
            resolution_runs.append(run)
            summary["normal_runs"].append(run)
        normal_runs_by_resolution[resolution] = resolution_runs

    for resolution, runs in normal_runs_by_resolution.items():
        comparisons = [
            _compare_repetitions(runs[0], other_run) for other_run in runs[1:]
        ]
        summary["repeatability"].append(
            {
                "process_res": resolution,
                "reference": runs[0]["result_name"],
                "comparisons": comparisons,
            }
        )

    missing_checkpoint = args.run_dir / "expected_missing_checkpoint"
    summary["failure_probes"].append(
        _run_worker(
            args,
            result_name="failure_missing_checkpoint",
            process_res=504,
            checkpoint_dir=missing_checkpoint,
        )
    )
    malformed_input = args.run_dir / "malformed_input.bin"
    malformed_input.write_bytes(b"P00 controlled malformed image payload\n")
    summary["failure_probes"].append(
        _run_worker(
            args,
            result_name="failure_malformed_input",
            process_res=504,
            single_input_path=malformed_input,
        )
    )

    normal_success = all(
        isinstance(run["result"], dict) and run["result"].get("success")
        for run in summary["normal_runs"]
    )
    expected_failures_retained = all(
        isinstance(probe["result"], dict) and not probe["result"].get("success")
        for probe in summary["failure_probes"]
    )
    summary["success"] = normal_success and expected_failures_retained
    summary_path = args.run_dir / "wp5_measurement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
