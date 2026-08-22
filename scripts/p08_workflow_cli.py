"""Create, inspect, and operate the shared P02-P08 workflow service from the CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p08_floor import FloorProcessingConfig
from spatial_mapping_phase2.p08_operator_workflow import (
    CameraInputSummaryAdapter,
    FloorPreviewWorkflowAdapter,
    OperatorWorkflowConfig,
    ReconstructionWorkflowAdapter,
)
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
            result = _run_job(service, "floor", arguments.job_id, arguments.wait_seconds)
        elif arguments.command == "reconstruct":
            result = _run_job(service, "reconstruct", arguments.job_id, arguments.wait_seconds)
        elif arguments.command == "floor-preview":
            service.start_floor_preview_job(arguments.job_id, arguments.floor_job_id)
            result = _wait_for_job(service, arguments.job_id, arguments.wait_seconds)
        elif arguments.command == "launch-rerun":
            result = service.launch_rerun(arguments.action_id, arguments.artifact_id)
        elif arguments.command == "approve":
            result = service.approve_result(arguments.action_id, arguments.target)
        elif arguments.command == "artifacts":
            result = service.artifact_catalog_status()
        elif arguments.command == "artifact-impact":
            result = service.artifact_selection_impact(arguments.artifact_id)
        elif arguments.command == "artifact-delete-impact":
            result = service.artifact_deletion_impact(arguments.artifact_id)
        elif arguments.command == "artifact-delete-batch-impact":
            result = service.artifact_batch_deletion_impact(arguments.artifact_ids)
        elif arguments.command == "artifact-select":
            result = service.select_artifact_version(
                arguments.action_id,
                arguments.artifact_id,
                confirm_impacts=arguments.confirm_impacts,
            )
        elif arguments.command == "artifact-verify":
            result = service.verify_artifact_version(arguments.action_id, arguments.artifact_id)
        elif arguments.command in {"artifact-archive", "artifact-restore"}:
            result = service.archive_artifact_version(
                arguments.action_id,
                arguments.artifact_id,
                archived=arguments.command == "artifact-archive",
            )
        elif arguments.command == "artifact-delete":
            result = service.delete_artifact_version(
                arguments.action_id,
                arguments.artifact_id,
                deletion_token=arguments.deletion_token,
            )
        elif arguments.command == "artifact-delete-batch":
            result = service.delete_artifact_versions(
                arguments.action_id,
                _deletion_items(arguments.items),
            )
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
    operator_config = None
    camera_summary_adapter = None
    operator_profile = getattr(arguments, "operator_profile", None)
    if operator_profile is not None:
        operator_config = OperatorWorkflowConfig.from_dict(_read_json(operator_profile))
        camera_summary_adapter = CameraInputSummaryAdapter(operator_config)
    reconstruction_adapter = None
    reconstruction_values = (
        getattr(arguments, "workflow_python", None),
        getattr(arguments, "repository_root", None),
        getattr(arguments, "p06_run_directory", None),
        getattr(arguments, "da3_source_directory", None),
        getattr(arguments, "da3_checkpoint_directory", None),
        getattr(arguments, "d041_manifest", None),
        getattr(arguments, "reconstruction_output_root", None),
    )
    reconstruction_requested = any(value is not None for value in reconstruction_values[2:])
    if reconstruction_requested:
        if not all(value is not None for value in reconstruction_values):
            jobs.close()
            raise P08WorkflowError("static reconstruction requires all configured paths")
        reconstruction_adapter = ReconstructionWorkflowAdapter(
            python_executable=_required_path(reconstruction_values[0], "workflow Python"),
            repository_root=_required_path(reconstruction_values[1], "repository root"),
            p06_run_directory=_required_path(reconstruction_values[2], "reconstruction inputs"),
            source_directory=_required_path(reconstruction_values[3], "DA3 source"),
            checkpoint_directory=_required_path(reconstruction_values[4], "DA3 checkpoint"),
            d041_manifest_path=_required_path(reconstruction_values[5], "rollback manifest"),
            output_root=_required_path(reconstruction_values[6], "reconstruction output root"),
            expected_geometry_sha256=getattr(arguments, "expected_geometry_sha256", None),
        )
    floor_preview_adapter = None
    if (
        getattr(arguments, "workflow_python", None) is not None
        and getattr(arguments, "repository_root", None) is not None
        and arguments.floor_contract is not None
    ):
        floor_preview_adapter = FloorPreviewWorkflowAdapter(
            python_executable=arguments.workflow_python,
            repository_root=arguments.repository_root,
            floor_contract_path=arguments.floor_contract,
        )
    return WorkflowService(
        repository=repo,
        jobs=jobs,
        floor_adapter=floor_adapter,
        rerun_launcher=rerun_launcher,
        operator_config=operator_config,
        camera_summary_adapter=camera_summary_adapter,
        reconstruction_adapter=reconstruction_adapter,
        floor_preview_adapter=floor_preview_adapter,
    )


def _run_job(
    service: WorkflowService, action: str, job_id: str, wait_seconds: float
) -> dict[str, Any]:
    if action == "floor":
        service.start_floor_job(job_id)
    else:
        service.start_reconstruction_job(job_id)
    return _wait_for_job(service, job_id, wait_seconds)


def _wait_for_job(service: WorkflowService, job_id: str, wait_seconds: float) -> dict[str, Any]:
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
    raise P08WorkflowError("workflow CLI wait timed out; cancellation requested")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--floor-contract", type=Path)
    parser.add_argument("--floor-config", type=Path)
    parser.add_argument("--floor-output-root", type=Path)
    parser.add_argument("--operator-profile", type=Path)
    parser.add_argument("--workflow-python", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--p06-run-directory", type=Path)
    parser.add_argument("--da3-source-directory", type=Path)
    parser.add_argument("--da3-checkpoint-directory", type=Path)
    parser.add_argument("--d041-manifest", type=Path)
    parser.add_argument("--reconstruction-output-root", type=Path)
    parser.add_argument("--expected-geometry-sha256")
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
    reconstruct = subparsers.add_parser("reconstruct")
    reconstruct.add_argument("--job-id", required=True)
    reconstruct.add_argument("--wait-seconds", type=float, default=600.0)
    floor_preview = subparsers.add_parser("floor-preview")
    floor_preview.add_argument("--job-id", required=True)
    floor_preview.add_argument("--floor-job-id", required=True)
    floor_preview.add_argument("--wait-seconds", type=float, default=180.0)
    launch = subparsers.add_parser("launch-rerun")
    launch.add_argument("--action-id", required=True)
    launch.add_argument("--artifact-id", required=True)
    approve = subparsers.add_parser("approve")
    approve.add_argument("--action-id", required=True)
    approve.add_argument("--target", choices=("geometry", "floor"), required=True)
    subparsers.add_parser("artifacts")
    impact = subparsers.add_parser("artifact-impact")
    impact.add_argument("--artifact-id", required=True)
    delete_impact = subparsers.add_parser("artifact-delete-impact")
    delete_impact.add_argument("--artifact-id", required=True)
    batch_delete_impact = subparsers.add_parser("artifact-delete-batch-impact")
    batch_delete_impact.add_argument(
        "--artifact-id", dest="artifact_ids", action="append", required=True
    )
    select = subparsers.add_parser("artifact-select")
    select.add_argument("--action-id", required=True)
    select.add_argument("--artifact-id", required=True)
    select.add_argument("--confirm-impacts", action="store_true")
    verify = subparsers.add_parser("artifact-verify")
    verify.add_argument("--action-id", required=True)
    verify.add_argument("--artifact-id", required=True)
    archive = subparsers.add_parser("artifact-archive")
    archive.add_argument("--action-id", required=True)
    archive.add_argument("--artifact-id", required=True)
    restore = subparsers.add_parser("artifact-restore")
    restore.add_argument("--action-id", required=True)
    restore.add_argument("--artifact-id", required=True)
    delete = subparsers.add_parser("artifact-delete")
    delete.add_argument("--action-id", required=True)
    delete.add_argument("--artifact-id", required=True)
    delete.add_argument("--deletion-token", required=True)
    batch_delete = subparsers.add_parser("artifact-delete-batch")
    batch_delete.add_argument("--action-id", required=True)
    batch_delete.add_argument(
        "--item",
        dest="items",
        action="append",
        required=True,
        metavar="ARTIFACT_ID=DELETION_TOKEN",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P08WorkflowError(f"JSON root must be an object: {path}")
    return value


def _required_path(value: Any, label: str) -> Path:
    if not isinstance(value, Path):
        raise P08WorkflowError(f"{label} must be configured as a path")
    return value


def _deletion_items(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        artifact_id, separator, token = value.partition("=")
        if not separator or not artifact_id.strip() or not token.strip():
            raise P08WorkflowError("each --item must use ARTIFACT_ID=DELETION_TOKEN")
        if artifact_id in result:
            raise P08WorkflowError("artifact IDs in a deletion batch must be unique")
        result[artifact_id] = token
    return result


def _identity(path: Path) -> dict[str, Any]:
    import hashlib

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


if __name__ == "__main__":
    main()
