"""Strict, secret-free launch profiles for the maintained combined console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConsoleProfileError(ValueError):
    """Raised when a launch profile is ambiguous or malformed."""


SCHEMA_VERSION = "phase2-console-profile-v1"

_PATH_OPTIONS = {
    "workspace_dir": "--workspace-dir",
    "scene_registry": "--scene-registry",
    "managed_scene_root": "--managed-scene-root",
    "floor_contract": "--floor-contract",
    "floor_config": "--floor-config",
    "floor_output_root": "--floor-output-root",
    "operator_profile": "--operator-profile",
    "workflow_python": "--workflow-python",
    "repository_root": "--repository-root",
    "p06_run_directory": "--p06-run-directory",
    "da3_source_directory": "--da3-source-directory",
    "da3_checkpoint_directory": "--da3-checkpoint-directory",
    "d041_manifest": "--d041-manifest",
    "reconstruction_output_root": "--reconstruction-output-root",
    "intrinsic_evidence": "--intrinsic-evidence",
    "facility_export": "--facility-export",
    "calibration_output_root": "--calibration-output-root",
    "viewer": "--viewer",
    "p02_workspace": "--p02-workspace",
    "p04_workspace": "--p04-workspace",
    "p05_workspace": "--p05-workspace",
    "geocalib_python": "--geocalib-python",
    "geocalib_source_directory": "--geocalib-source-directory",
    "geocalib_torch_home": "--geocalib-torch-home",
    "xr02_python": "--xr02-python",
    "xr02_worker_script": "--xr02-worker-script",
    "xr02_deployment_config": "--xr02-deployment-config",
    "xr02_output_root": "--xr02-output-root",
    "xr02_mediamtx_binary": "--xr02-mediamtx-binary",
    "xr02_ffmpeg_binary": "--xr02-ffmpeg-binary",
    "secret_file": "--secret-file",
}
_STRING_OPTIONS = {
    "expected_geometry_sha256": "--expected-geometry-sha256",
    "xr02_worker_module": "--xr02-worker-module",
    "host": "--host",
}
_INTEGER_OPTIONS = {
    "maximum_workers": "--maximum-workers",
    "maximum_outstanding_jobs": "--maximum-outstanding-jobs",
    "xr02_worker_port": "--xr02-worker-port",
    "port": "--port",
}
_FLOAT_OPTIONS = {
    "xr02_recording_free_space_reserve_gb": "--xr02-recording-free-space-reserve-gb",
}
_BOOLEAN_OPTIONS = {
    "enable_live_p03": "--enable-live-p03",
    "enable_xr02_live_operations": "--enable-xr02-live-operations",
}
_LIST_OPTIONS = {"allowed_artifact_roots", "calibration_workspaces"}
_KNOWN_FIELDS = (
    {"schema_version"}
    | set(_PATH_OPTIONS)
    | set(_STRING_OPTIONS)
    | set(_INTEGER_OPTIONS)
    | set(_FLOAT_OPTIONS)
    | set(_BOOLEAN_OPTIONS)
    | _LIST_OPTIONS
)


def load_console_profile_arguments(path: Path) -> list[str]:
    """Translate a versioned JSON profile into the established CLI contract."""

    profile_path = path.resolve()
    try:
        value = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConsoleProfileError(f"cannot read console profile: {profile_path}") from error
    if not isinstance(value, dict):
        raise ConsoleProfileError("console profile root must be an object")
    unknown = sorted(set(value) - _KNOWN_FIELDS)
    if unknown:
        raise ConsoleProfileError("unknown console profile fields: " + ", ".join(unknown))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ConsoleProfileError(f"console profile schema must be {SCHEMA_VERSION}")
    if "workspace_dir" not in value:
        raise ConsoleProfileError("console profile requires workspace_dir")

    arguments: list[str] = []
    base = profile_path.parent
    for field, option in _PATH_OPTIONS.items():
        if field in value:
            arguments.extend((option, str(_profile_path(value[field], field, base))))
    for field, option in _STRING_OPTIONS.items():
        if field in value:
            arguments.extend((option, _string(value[field], field)))
    for field, option in _INTEGER_OPTIONS.items():
        if field in value:
            arguments.extend((option, str(_integer(value[field], field))))
    for field, option in _FLOAT_OPTIONS.items():
        if field in value:
            arguments.extend((option, str(_number(value[field], field))))
    for field, option in _BOOLEAN_OPTIONS.items():
        if field in value and _boolean(value[field], field):
            arguments.append(option)

    roots = value.get("allowed_artifact_roots", [])
    if not isinstance(roots, list):
        raise ConsoleProfileError("allowed_artifact_roots must be a list")
    for index, root in enumerate(roots):
        arguments.extend(
            (
                "--allowed-artifact-root",
                str(_profile_path(root, f"allowed_artifact_roots[{index}]", base)),
            )
        )

    workspaces = value.get("calibration_workspaces", {})
    if not isinstance(workspaces, dict):
        raise ConsoleProfileError("calibration_workspaces must be an object")
    for camera_id, workspace in workspaces.items():
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ConsoleProfileError("calibration workspace camera IDs must be non-empty strings")
        resolved = _profile_path(workspace, f"calibration_workspaces.{camera_id}", base)
        arguments.extend(("--calibration-workspace", f"{camera_id}={resolved}"))
    return arguments


def _profile_path(value: Any, field: str, base: Path) -> Path:
    raw = _string(value, field)
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConsoleProfileError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConsoleProfileError(f"{field} must be an integer")
    return int(value)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConsoleProfileError(f"{field} must be a number")
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConsoleProfileError(f"{field} must be true or false")
    return value
