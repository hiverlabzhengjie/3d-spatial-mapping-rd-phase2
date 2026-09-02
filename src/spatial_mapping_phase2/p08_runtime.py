"""Typed construction of the maintained P02-P08 workflow runtime.

Command-line and web entry points intentionally share this module so an installed package does
not depend on importing another file from ``scripts/``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p04_calibration_service import P04CalibrationService
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
    P08WorkflowError,
    SafeRerunLauncher,
    SceneWorkspaceRepository,
    WorkflowService,
    load_frozen_floor_input,
)
from spatial_mapping_phase2.xr03_calibration_workflow import (
    IntegratedCalibrationWorkflowAdapter,
    SceneCalibrationRepository,
)


@dataclass(frozen=True)
class WorkflowRuntimeConfig:
    """Validated inputs used to assemble one scene workflow service."""

    workspace_dir: Path
    floor_contract: Path | None = None
    floor_config: Path | None = None
    floor_output_root: Path | None = None
    operator_profile: Path | None = None
    workflow_python: Path | None = None
    repository_root: Path | None = None
    p06_run_directory: Path | None = None
    da3_source_directory: Path | None = None
    da3_checkpoint_directory: Path | None = None
    d041_manifest: Path | None = None
    reconstruction_output_root: Path | None = None
    expected_geometry_sha256: str | None = None
    intrinsic_evidence: Path | None = None
    facility_export: Path | None = None
    calibration_output_root: Path | None = None
    calibration_workspaces: Mapping[str, Path] = field(default_factory=dict)
    viewer: Path | None = None
    allowed_artifact_roots: tuple[Path, ...] = ()
    maximum_workers: int = 1
    maximum_outstanding_jobs: int = 4


def workflow_runtime_config_from_arguments(arguments: Any) -> WorkflowRuntimeConfig:
    """Translate the shared CLI argument contract at the package boundary."""

    return WorkflowRuntimeConfig(
        workspace_dir=arguments.workspace_dir,
        floor_contract=arguments.floor_contract,
        floor_config=arguments.floor_config,
        floor_output_root=arguments.floor_output_root,
        operator_profile=arguments.operator_profile,
        workflow_python=arguments.workflow_python,
        repository_root=arguments.repository_root,
        p06_run_directory=arguments.p06_run_directory,
        da3_source_directory=arguments.da3_source_directory,
        da3_checkpoint_directory=arguments.da3_checkpoint_directory,
        d041_manifest=arguments.d041_manifest,
        reconstruction_output_root=arguments.reconstruction_output_root,
        expected_geometry_sha256=arguments.expected_geometry_sha256,
        intrinsic_evidence=arguments.intrinsic_evidence,
        facility_export=arguments.facility_export,
        calibration_output_root=arguments.calibration_output_root,
        calibration_workspaces=parse_calibration_workspaces(arguments.calibration_workspace),
        viewer=arguments.viewer,
        allowed_artifact_roots=tuple(arguments.allowed_artifact_root),
        maximum_workers=arguments.maximum_workers,
        maximum_outstanding_jobs=arguments.maximum_outstanding_jobs,
    )


def parse_calibration_workspaces(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        camera_id, separator, raw_path = value.partition("=")
        if not separator or not camera_id or not raw_path:
            raise P08WorkflowError("calibration workspace must use CAMERA_ID=PATH")
        if camera_id in result:
            raise P08WorkflowError(f"duplicate calibration camera: {camera_id}")
        result[camera_id] = Path(raw_path)
    return result


def build_workflow_service(
    config: WorkflowRuntimeConfig,
    repository: SceneWorkspaceRepository | None = None,
) -> WorkflowService:
    """Build one service and close its job manager if configuration is rejected."""

    repo = repository or SceneWorkspaceRepository(config.workspace_dir)
    jobs = BoundedJobManager(
        maximum_workers=config.maximum_workers,
        maximum_outstanding_jobs=config.maximum_outstanding_jobs,
    )
    try:
        floor_adapter = _floor_adapter(config)
        rerun_launcher = _rerun_launcher(config)
        operator_config, camera_summary_adapter = _operator_inputs(config)
        reconstruction_adapter = _reconstruction_adapter(config)
        floor_preview_adapter = _floor_preview_adapter(config)
        calibration_adapter = _calibration_adapter(config, repo)
    except Exception:
        jobs.close()
        raise
    return WorkflowService(
        repository=repo,
        jobs=jobs,
        floor_adapter=floor_adapter,
        rerun_launcher=rerun_launcher,
        operator_config=operator_config,
        camera_summary_adapter=camera_summary_adapter,
        reconstruction_adapter=reconstruction_adapter,
        calibration_adapter=calibration_adapter,
        floor_preview_adapter=floor_preview_adapter,
    )


def _floor_adapter(config: WorkflowRuntimeConfig) -> FloorWorkflowAdapter | None:
    values = (config.floor_contract, config.floor_config, config.floor_output_root)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise P08WorkflowError("floor adapter requires contract, config, and output root together")
    return FloorWorkflowAdapter(
        load_frozen_floor_input(_read_json(_path(config.floor_contract, "floor contract"))),
        FloorProcessingConfig(**_read_json(_path(config.floor_config, "floor config"))),
        _path(config.floor_output_root, "floor output root"),
    )


def _rerun_launcher(config: WorkflowRuntimeConfig) -> SafeRerunLauncher | None:
    if config.viewer is None and not config.allowed_artifact_roots:
        return None
    if config.viewer is None or not config.allowed_artifact_roots:
        raise P08WorkflowError("Rerun launch requires a viewer and allowed root")
    return SafeRerunLauncher(config.viewer, config.allowed_artifact_roots)


def _operator_inputs(
    config: WorkflowRuntimeConfig,
) -> tuple[OperatorWorkflowConfig | None, CameraInputSummaryAdapter | None]:
    if config.operator_profile is None:
        return None, None
    operator_config = OperatorWorkflowConfig.from_dict(_read_json(config.operator_profile))
    return operator_config, CameraInputSummaryAdapter(operator_config)


def _reconstruction_adapter(
    config: WorkflowRuntimeConfig,
) -> ReconstructionWorkflowAdapter | None:
    values = (
        config.workflow_python,
        config.repository_root,
        config.p06_run_directory,
        config.da3_source_directory,
        config.da3_checkpoint_directory,
        config.d041_manifest,
        config.reconstruction_output_root,
    )
    if not any(value is not None for value in values[2:]):
        return None
    if not all(value is not None for value in values):
        raise P08WorkflowError("static reconstruction requires all configured paths")
    return ReconstructionWorkflowAdapter(
        python_executable=_path(config.workflow_python, "workflow Python"),
        repository_root=_path(config.repository_root, "repository root"),
        p06_run_directory=_path(config.p06_run_directory, "reconstruction inputs"),
        source_directory=_path(config.da3_source_directory, "DA3 source"),
        checkpoint_directory=_path(config.da3_checkpoint_directory, "DA3 checkpoint"),
        d041_manifest_path=_path(config.d041_manifest, "rollback manifest"),
        output_root=_path(config.reconstruction_output_root, "reconstruction output root"),
        expected_geometry_sha256=config.expected_geometry_sha256,
    )


def _floor_preview_adapter(
    config: WorkflowRuntimeConfig,
) -> FloorPreviewWorkflowAdapter | None:
    if (
        config.workflow_python is None
        or config.repository_root is None
        or config.floor_contract is None
    ):
        return None
    return FloorPreviewWorkflowAdapter(
        python_executable=config.workflow_python,
        repository_root=config.repository_root,
        floor_contract_path=config.floor_contract,
    )


def _calibration_adapter(
    config: WorkflowRuntimeConfig,
    repository: SceneWorkspaceRepository,
) -> IntegratedCalibrationWorkflowAdapter | None:
    values = (
        config.intrinsic_evidence,
        config.facility_export,
        config.calibration_output_root,
    )
    requested = bool(config.calibration_workspaces) or any(value is not None for value in values)
    if not requested:
        return None
    if not all(value is not None for value in values) or not config.calibration_workspaces:
        raise P08WorkflowError(
            "integrated calibration requires intrinsic evidence, facility export, output "
            "root, and CAMERA_ID=PATH workspaces"
        )
    scene = repository.load()
    camera_ids = tuple(camera.camera_id for camera in scene.cameras if camera.enabled)
    if set(config.calibration_workspaces) != set(camera_ids):
        raise P08WorkflowError("calibration workspaces must match the enabled scene roster")
    services = {
        camera_id: P04CalibrationService(config.calibration_workspaces[camera_id])
        for camera_id in camera_ids
    }
    return IntegratedCalibrationWorkflowAdapter(
        SceneCalibrationRepository(
            repository.calibration_workflow_path,
            scene.project_id,
            scene.scene_id,
            camera_ids,
        ),
        services,
        _path(config.intrinsic_evidence, "intrinsic evidence"),
        _path(config.facility_export, "facility export"),
        _path(config.calibration_output_root, "calibration output root"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P08WorkflowError(f"JSON root must be an object: {path}")
    return value


def _path(value: Path | None, label: str) -> Path:
    if value is None:
        raise P08WorkflowError(f"{label} must be configured as a path")
    return value
