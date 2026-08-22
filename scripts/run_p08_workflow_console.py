"""Run the integrated P02-P08 localhost workflow console."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import uvicorn
from p08_workflow_cli import build_workflow_service

from spatial_mapping_phase2.p02_registration_web import create_p02_registration_app
from spatial_mapping_phase2.p04_calibration_web import create_p04_calibration_app
from spatial_mapping_phase2.p08_workflow import SceneWorkspaceRepository
from spatial_mapping_phase2.p08_workflow_web import create_p08_workflow_app


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("integrated workflow console may bind only to localhost")
    if arguments.enable_live_p03 and importlib.util.find_spec("av") is None:
        parser.error(
            "live camera preview requires PyAV; launch this console with the "
            "configured pinned native Python runtime"
        )
    repository = SceneWorkspaceRepository(arguments.workspace_dir)
    service = build_workflow_service(arguments, repository)
    legacy_apps: dict[str, Any] = {}
    capture_service: Any = None
    if arguments.p02_workspace is not None:
        legacy_apps["facility"] = create_p02_registration_app(
            arguments.p02_workspace, arguments.secret_file
        )
    if arguments.p04_workspace is not None:
        legacy_apps["calibration-default"] = create_p04_calibration_app(arguments.p04_workspace)
    if arguments.p05_workspace is not None:
        legacy_apps["calibration-pose-review"] = create_p04_calibration_app(
            arguments.p05_workspace
        )
    for camera_id, workspace in _calibration_workspaces(arguments.calibration_workspace).items():
        legacy_apps[f"calibration-{camera_id}"] = create_p04_calibration_app(workspace)
    if arguments.enable_live_p03:
        from p03_capture_cli import _environment, build_service, build_temporal_service

        from spatial_mapping_phase2.p03_capture_web import create_p03_capture_app

        environment = _environment(arguments.secret_file)
        capture_service = build_service(environment)
        legacy_apps["capture"] = create_p03_capture_app(
            capture_service,
            temporal_factory=lambda: build_temporal_service(environment),
        )
    app = create_p08_workflow_app(service, legacy_apps)
    try:
        uvicorn.run(
            app,
            host=arguments.host,
            port=arguments.port,
            access_log=False,
        )
    finally:
        if capture_service is not None:
            capture_service.close()
        service.jobs.close()


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
    parser.add_argument("--p02-workspace", type=Path)
    parser.add_argument("--p04-workspace", type=Path)
    parser.add_argument("--p05-workspace", type=Path)
    parser.add_argument(
        "--calibration-workspace",
        action="append",
        default=[],
        metavar="CAMERA_ID=PATH",
    )
    parser.add_argument("--enable-live-p03", action="store_true")
    parser.add_argument("--secret-file", type=Path, default=Path(".env"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    return parser


def _calibration_workspaces(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        camera_id, separator, raw_path = value.partition("=")
        if not separator or not camera_id or not raw_path:
            raise ValueError("calibration workspace must use CAMERA_ID=PATH")
        if camera_id in result:
            raise ValueError(f"duplicate calibration camera: {camera_id}")
        result[camera_id] = Path(raw_path)
    return result


if __name__ == "__main__":
    main()
