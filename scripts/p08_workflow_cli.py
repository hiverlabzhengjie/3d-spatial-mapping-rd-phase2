"""Create, inspect, and operate the shared P02-P08 workflow service from the CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p08_floor import FloorProcessingConfig
from spatial_mapping_phase2.p08_workflow import (
    BoundedJobManager,
    FloorWorkflowAdapter,
    JobState,
    P08WorkflowError,
    SafeRerunLauncher,
    SceneWorkspace,
    SceneWorkspaceRepository,
    WorkflowService,
    load_frozen_floor_input,
)


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    repository = SceneWorkspaceRepository(arguments.workspace_dir)
    if arguments.command == "create-scene":
        scene = SceneWorkspace.from_dict(_read_json(arguments.scene_config))
        path = repository.create(scene)
        print(json.dumps({"scene": _identity(path)}, indent=2, sort_keys=True))
        return
    service = build_workflow_service(arguments, repository)
    try:
        if arguments.command == "status":
            result = service.status()
        elif arguments.command == "floor":
            result = _run_floor(service, arguments.job_id, arguments.wait_seconds)
        elif arguments.command == "launch-rerun":
            result = service.launch_rerun(arguments.action_id, arguments.artifact_id)
        else:
            raise P08WorkflowError("unknown workflow CLI command")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        service.jobs.close()


def build_workflow_service(
    arguments: argparse.Namespace,
    repository: SceneWorkspaceRepository | None = None,
) -> WorkflowService:
    repo = repository or SceneWorkspaceRepository(arguments.workspace_dir)
    jobs = BoundedJobManager(
        maximum_workers=arguments.maximum_workers,
        maximum_outstanding_jobs=arguments.maximum_outstanding_jobs,
    )
    floor_adapter = None
    floor_values = (
        arguments.floor_contract,
        arguments.floor_config,
        arguments.floor_output_root,
    )
    if any(value is not None for value in floor_values):
        if not all(value is not None for value in floor_values):
            jobs.close()
            raise P08WorkflowError(
                "floor adapter requires contract, config, and output root together"
            )
        floor_adapter = FloorWorkflowAdapter(
            load_frozen_floor_input(_read_json(arguments.floor_contract)),
            FloorProcessingConfig(**_read_json(arguments.floor_config)),
            arguments.floor_output_root,
        )
    rerun_launcher = None
    if arguments.viewer is not None or arguments.allowed_artifact_root:
        if arguments.viewer is None or not arguments.allowed_artifact_root:
            jobs.close()
            raise P08WorkflowError("Rerun launch requires a viewer and allowed root")
        rerun_launcher = SafeRerunLauncher(
            arguments.viewer,
            tuple(arguments.allowed_artifact_root),
        )
    return WorkflowService(
        repository=repo,
        jobs=jobs,
        floor_adapter=floor_adapter,
        rerun_launcher=rerun_launcher,
    )


def _run_floor(service: WorkflowService, job_id: str, wait_seconds: float) -> dict[str, Any]:
    service.start_floor_job(job_id)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        status = service.jobs.status(job_id)
        if status["state"] in {
            JobState.COMPLETE.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        }:
            return status
        time.sleep(0.05)
    service.jobs.cancel(job_id)
    raise P08WorkflowError("floor CLI wait timed out; cancellation requested")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--floor-contract", type=Path)
    parser.add_argument("--floor-config", type=Path)
    parser.add_argument("--floor-output-root", type=Path)
    parser.add_argument("--viewer", type=Path)
    parser.add_argument("--allowed-artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--maximum-workers", type=int, default=1)
    parser.add_argument("--maximum-outstanding-jobs", type=int, default=4)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-scene")
    create.add_argument("--scene-config", type=Path, required=True)
    subparsers.add_parser("status")
    floor = subparsers.add_parser("floor")
    floor.add_argument("--job-id", required=True)
    floor.add_argument("--wait-seconds", type=float, default=120.0)
    launch = subparsers.add_parser("launch-rerun")
    launch.add_argument("--action-id", required=True)
    launch.add_argument("--artifact-id", required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P08WorkflowError(f"JSON root must be an object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    import hashlib

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


if __name__ == "__main__":
    main()
