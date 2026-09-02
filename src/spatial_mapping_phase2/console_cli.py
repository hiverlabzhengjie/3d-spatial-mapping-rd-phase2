"""Run the integrated P02-P08 localhost workflow console."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn

from spatial_mapping_phase2.console_preflight import console_preflight
from spatial_mapping_phase2.console_profile import (
    ConsoleProfileError,
    load_console_profile_arguments,
)
from spatial_mapping_phase2.managed_scene_downstream import ManagedSceneLiveConfig
from spatial_mapping_phase2.managed_scene_geocalib import ManagedSceneGeoCalibConfig
from spatial_mapping_phase2.managed_scene_reconstruction import (
    ManagedSceneReconstructionConfig,
)
from spatial_mapping_phase2.p04_calibration_web import create_p04_calibration_app
from spatial_mapping_phase2.p08_runtime import (
    build_workflow_service,
    parse_calibration_workspaces,
    workflow_runtime_config_from_arguments,
)
from spatial_mapping_phase2.p08_workflow import SceneWorkspaceRepository
from spatial_mapping_phase2.xr01_update_pipeline import CapturedJpeg, SceneUpdatePipelineAdapter
from spatial_mapping_phase2.xr03_live_operations import SupervisedXR02Worker
from spatial_mapping_phase2.xr03_scene_console import (
    SceneRuntime,
    SceneRuntimeManager,
    create_scene_console_app,
)
from spatial_mapping_phase2.xr03_scene_management import SceneRegistry


@dataclass(frozen=True)
class _P03FreshFrameProvider:
    service: Any

    def capture_jpeg(self, camera_id: str) -> CapturedJpeg:
        from spatial_mapping_phase2.p03_capture_service import CapturePolicy

        frame = self.service.preview(camera_id, CapturePolicy(duration_seconds=1.0))
        return CapturedJpeg(camera_id, frame.content, frame.observed_at_utc)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile", type=Path)
    profile_arguments, remaining = profile_parser.parse_known_args(argv)
    expanded: list[str] = []
    if profile_arguments.profile is not None:
        try:
            expanded = load_console_profile_arguments(profile_arguments.profile)
        except ConsoleProfileError as error:
            parser.error(str(error))
    arguments = parser.parse_args([*expanded, *remaining])
    if arguments.preflight:
        result = console_preflight(arguments)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready"] else 2
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("integrated workflow console may bind only to localhost")
    if arguments.enable_live_p03 and importlib.util.find_spec("av") is None:
        parser.error(
            "live camera preview requires PyAV; launch this console with the "
            "configured pinned native Python runtime"
        )
    repository = SceneWorkspaceRepository(arguments.workspace_dir)
    service = build_workflow_service(workflow_runtime_config_from_arguments(arguments), repository)
    legacy_apps: dict[str, Any] = {}
    capture_service: Any = None
    if arguments.p02_workspace is not None:
        from spatial_mapping_phase2.p02_registration_web import create_p02_registration_app

        legacy_apps["facility"] = create_p02_registration_app(
            arguments.p02_workspace, arguments.secret_file
        )
    if arguments.p04_workspace is not None:
        legacy_apps["calibration-default"] = create_p04_calibration_app(arguments.p04_workspace)
    if arguments.p05_workspace is not None:
        legacy_apps["calibration-pose-review"] = create_p04_calibration_app(
            arguments.p05_workspace
        )
    for camera_id, workspace in parse_calibration_workspaces(
        arguments.calibration_workspace
    ).items():
        legacy_apps[f"calibration-{camera_id}"] = create_p04_calibration_app(workspace)
    if arguments.enable_live_p03:
        from spatial_mapping_phase2.p03_capture_web import create_p03_capture_app
        from spatial_mapping_phase2.p03_runtime import (
            build_capture_service,
            build_temporal_capture_service,
        )
        from spatial_mapping_phase2.runtime_environment import load_environment_file

        environment = load_environment_file(arguments.secret_file)
        capture_service = build_capture_service(environment)
        legacy_apps["capture"] = create_p03_capture_app(
            capture_service,
            temporal_factory=lambda: build_temporal_capture_service(environment),
        )
        if (
            service.reconstruction_adapter is not None
            and service.floor_adapter is not None
            and service.floor_preview_adapter is not None
            and arguments.p06_run_directory is not None
            and arguments.reconstruction_output_root is not None
        ):
            service.enable_scene_updates(
                SceneUpdatePipelineAdapter(
                    frame_provider=_P03FreshFrameProvider(capture_service),
                    camera_ids=tuple(
                        camera.camera_id for camera in repository.load().cameras if camera.enabled
                    ),
                    baseline_input_directory=arguments.p06_run_directory,
                    input_output_root=arguments.reconstruction_output_root / "scene-update-inputs",
                    reconstruction=service.reconstruction_adapter,
                    floor=service.floor_adapter,
                    floor_preview=service.floor_preview_adapter,
                    camera_policy_provider=service.camera_policy_status,
                )
            )
    managed_live = None
    if arguments.enable_xr02_live_operations:
        xr02_python = arguments.xr02_python
        worker_script = arguments.xr02_worker_script
        worker_module = arguments.xr02_worker_module
        deployment_config = arguments.xr02_deployment_config
        output_root = arguments.xr02_output_root
        mediamtx_binary = arguments.xr02_mediamtx_binary
        ffmpeg_binary = arguments.xr02_ffmpeg_binary
        missing = [
            option
            for option, value in (
                ("--xr02-python", xr02_python),
                ("--xr02-deployment-config", deployment_config),
                ("--xr02-output-root", output_root),
                ("--xr02-mediamtx-binary", mediamtx_binary),
                ("--xr02-ffmpeg-binary", ffmpeg_binary),
            )
            if value is None
        ]
        if missing:
            parser.error("integrated XR02 requires " + ", ".join(missing))
        if (worker_script is None) == (worker_module is None):
            parser.error("integrated XR02 requires exactly one worker script or module")
        assert xr02_python is not None
        assert deployment_config is not None
        assert output_root is not None
        assert mediamtx_binary is not None
        assert ffmpeg_binary is not None
        service.enable_live_operations(
            SupervisedXR02Worker(
                xr02_python,
                worker_script,
                (
                    "--deployment-config",
                    str(deployment_config),
                    "--output-root",
                    str(output_root),
                    "--mediamtx-binary",
                    str(mediamtx_binary),
                    "--record-trial-video",
                    "--ffmpeg-binary",
                    str(ffmpeg_binary),
                    "--recording-free-space-reserve-gb",
                    str(arguments.xr02_recording_free_space_reserve_gb),
                ),
                worker_module=worker_module,
                port=arguments.xr02_worker_port,
            )
        )
        managed_live = ManagedSceneLiveConfig(
            python_executable=xr02_python,
            worker_script=worker_script,
            worker_module=worker_module,
            base_deployment_config=deployment_config,
            mediamtx_binary=mediamtx_binary,
            ffmpeg_binary=ffmpeg_binary,
            recording_free_space_reserve_gb=arguments.xr02_recording_free_space_reserve_gb,
            port=arguments.xr02_worker_port + 1,
        )
    scene_registry_path = arguments.scene_registry or (
        arguments.workspace_dir.parent / "scene-control" / "scene-registry.sqlite3"
    )
    managed_scene_root = arguments.managed_scene_root or (
        arguments.workspace_dir.parent / "scene-control" / "scenes"
    )
    registry = SceneRegistry(scene_registry_path, managed_scene_root)
    registry.clear_stale_resources()
    existing = registry.register_existing(repository.root)
    geocalib_values = (
        arguments.geocalib_python,
        arguments.geocalib_source_directory,
        arguments.geocalib_torch_home,
        arguments.repository_root,
    )
    if any(value is not None for value in geocalib_values) and not all(
        value is not None for value in geocalib_values
    ):
        parser.error(
            "managed-scene calibration requires --geocalib-python, "
            "--geocalib-source-directory, --geocalib-torch-home and --repository-root"
        )
    managed_geocalib = None
    if all(value is not None for value in geocalib_values):
        assert arguments.geocalib_python is not None
        assert arguments.geocalib_source_directory is not None
        assert arguments.geocalib_torch_home is not None
        assert arguments.repository_root is not None
        managed_geocalib = ManagedSceneGeoCalibConfig(
            arguments.geocalib_python,
            arguments.geocalib_source_directory,
            arguments.geocalib_torch_home,
            arguments.repository_root,
        )
    reconstruction_values = (
        arguments.workflow_python,
        arguments.repository_root,
        arguments.da3_source_directory,
        arguments.da3_checkpoint_directory,
    )
    managed_reconstruction = None
    if all(value is not None for value in reconstruction_values):
        assert arguments.workflow_python is not None
        assert arguments.repository_root is not None
        assert arguments.da3_source_directory is not None
        assert arguments.da3_checkpoint_directory is not None
        managed_reconstruction = ManagedSceneReconstructionConfig(
            arguments.workflow_python,
            arguments.repository_root,
            arguments.da3_source_directory,
            arguments.da3_checkpoint_directory,
        )
    runtime_manager = SceneRuntimeManager(
        registry,
        existing.scene_uuid,
        managed_geocalib,
        managed_reconstruction,
        service.rerun_launcher,
        managed_live,
    )
    runtime_manager.register(
        existing.scene_uuid,
        SceneRuntime(
            service,
            legacy_apps,
            close_callback=capture_service.close if capture_service is not None else None,
        ),
    )
    app = create_scene_console_app(runtime_manager)
    try:
        uvicorn.run(app, host=arguments.host, port=arguments.port, access_log=False)
    finally:
        runtime_manager.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        help="Versioned JSON launch profile; explicit CLI options override scalar profile values.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check the configured runtime without starting services or changing state.",
    )
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path)
    parser.add_argument("--managed-scene-root", type=Path)
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
    parser.add_argument("--intrinsic-evidence", type=Path)
    parser.add_argument("--facility-export", type=Path)
    parser.add_argument("--calibration-output-root", type=Path)
    parser.add_argument("--viewer", type=Path)
    parser.add_argument("--allowed-artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--maximum-workers", type=int, default=1)
    parser.add_argument("--maximum-outstanding-jobs", type=int, default=4)
    parser.add_argument("--p02-workspace", type=Path)
    parser.add_argument("--p04-workspace", type=Path)
    parser.add_argument("--p05-workspace", type=Path)
    parser.add_argument("--geocalib-python", type=Path)
    parser.add_argument("--geocalib-source-directory", type=Path)
    parser.add_argument("--geocalib-torch-home", type=Path)
    parser.add_argument(
        "--calibration-workspace",
        action="append",
        default=[],
        metavar="CAMERA_ID=PATH",
    )
    parser.add_argument("--enable-live-p03", action="store_true")
    parser.add_argument("--enable-xr02-live-operations", action="store_true")
    parser.add_argument(
        "--xr02-python",
        type=Path,
    )
    parser.add_argument(
        "--xr02-worker-script",
        type=Path,
    )
    parser.add_argument("--xr02-worker-module")
    parser.add_argument("--xr02-deployment-config", type=Path)
    parser.add_argument("--xr02-worker-port", type=int, default=8094)
    parser.add_argument("--xr02-output-root", type=Path)
    parser.add_argument("--xr02-mediamtx-binary", type=Path)
    parser.add_argument("--xr02-ffmpeg-binary", type=Path)
    parser.add_argument("--xr02-recording-free-space-reserve-gb", type=float, default=5.0)
    parser.add_argument("--secret-file", type=Path, default=Path(".env"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
