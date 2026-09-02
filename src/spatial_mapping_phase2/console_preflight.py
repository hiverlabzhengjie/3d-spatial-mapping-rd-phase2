"""Read-only preflight for the combined console's external runtime contract."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.managed_scene_geocalib import EXPECTED_DISTORTED_WEIGHT_SHA256
from spatial_mapping_phase2.p01_observability import load_local_rtsp_endpoints
from spatial_mapping_phase2.runtime_environment import load_environment_file
from spatial_mapping_phase2.xr02_deployment import (
    XR02Deployment,
    XR02DeploymentError,
    load_xr02_deployment,
)


def console_preflight(arguments: Any) -> dict[str, Any]:
    """Return actionable checks without creating databases, jobs, or external processes."""

    checks: list[dict[str, str]] = []
    _check_path(checks, "scene workspace", arguments.workspace_dir / "scene.json", "file")
    _check_optional_paths(
        checks,
        {
            "scene registry parent": (arguments.scene_registry, "parent"),
            "managed scene root parent": (arguments.managed_scene_root, "parent"),
            "floor contract": (arguments.floor_contract, "file"),
            "floor config": (arguments.floor_config, "file"),
            "floor output parent": (arguments.floor_output_root, "parent"),
            "operator profile": (arguments.operator_profile, "file"),
            "workflow Python": (arguments.workflow_python, "file"),
            "repository root": (arguments.repository_root, "directory"),
            "P06 run directory": (arguments.p06_run_directory, "directory"),
            "DA3 source": (arguments.da3_source_directory, "directory"),
            "DA3 checkpoint": (arguments.da3_checkpoint_directory, "directory"),
            "rollback manifest": (arguments.d041_manifest, "file"),
            "reconstruction output parent": (arguments.reconstruction_output_root, "parent"),
            "intrinsic evidence": (arguments.intrinsic_evidence, "file"),
            "facility export": (arguments.facility_export, "file"),
            "calibration output parent": (arguments.calibration_output_root, "parent"),
            "Rerun viewer": (arguments.viewer, "file"),
            "P02 workspace": (arguments.p02_workspace, "directory"),
            "P04 workspace": (arguments.p04_workspace, "directory"),
            "P05 workspace": (arguments.p05_workspace, "directory"),
            "GeoCalib Python": (getattr(arguments, "geocalib_python", None), "file"),
            "GeoCalib source": (
                getattr(arguments, "geocalib_source_directory", None),
                "directory",
            ),
            "GeoCalib model cache": (
                getattr(arguments, "geocalib_torch_home", None),
                "directory",
            ),
        },
    )
    for index, root in enumerate(arguments.allowed_artifact_root):
        _check_path(checks, f"allowed artifact root {index + 1}", root, "directory")
    for value in arguments.calibration_workspace:
        camera_id, separator, raw_path = value.partition("=")
        if not separator or not camera_id or not raw_path:
            _result(checks, "calibration workspaces", False, "use CAMERA_ID=PATH")
        else:
            _check_path(checks, f"calibration workspace {camera_id}", Path(raw_path), "directory")
    geocalib_torch_home = getattr(arguments, "geocalib_torch_home", None)
    if geocalib_torch_home is not None:
        _check_file_hash(
            checks,
            "GeoCalib distorted checkpoint",
            geocalib_torch_home / "hub" / "geocalib" / "distorted.tar",
            EXPECTED_DISTORTED_WEIGHT_SHA256,
        )
    if arguments.repository_root is not None:
        _check_path(
            checks,
            "repository package source",
            arguments.repository_root / "src" / "spatial_mapping_phase2",
            "directory",
        )

    _result(
        checks,
        "localhost binding",
        arguments.host in {"127.0.0.1", "localhost", "::1"},
        "configured for loopback only",
    )
    _result(
        checks,
        "console port",
        1024 <= arguments.port <= 65535,
        "must be within 1024..65535",
    )
    if arguments.p02_workspace is not None:
        _result(
            checks,
            "PyMuPDF import",
            importlib.util.find_spec("fitz") is not None,
            "required for the facility-plan tool",
        )
    if arguments.enable_live_p03:
        _check_path(checks, "local secret file", arguments.secret_file, "file")
        _result(
            checks,
            "PyAV import",
            importlib.util.find_spec("av") is not None,
            "required for live P03 capture",
        )
        if arguments.secret_file.is_file():
            try:
                endpoints = load_local_rtsp_endpoints(load_environment_file(arguments.secret_file))
                _result(
                    checks,
                    "RTSP endpoint roster",
                    bool(endpoints),
                    f"{len(endpoints)} credential-bearing endpoint(s) configured locally",
                )
            except (KeyError, OSError, UnicodeError, ValueError) as error:
                _result(checks, "RTSP endpoint roster", False, str(error))
    if arguments.enable_xr02_live_operations:
        _check_path(checks, "XR02 Python", arguments.xr02_python, "file")
        _check_xr02_worker_entrypoint(
            checks, arguments.xr02_worker_script, arguments.xr02_worker_module
        )
        _check_xr02_deployment(checks, arguments.xr02_deployment_config)
        _check_path(checks, "MediaMTX", arguments.xr02_mediamtx_binary, "file")
        _check_path(checks, "FFmpeg", arguments.xr02_ffmpeg_binary, "file")
        _check_path(checks, "XR02 output parent", arguments.xr02_output_root, "parent")
        _result(
            checks,
            "XR02 worker port",
            1024 <= arguments.xr02_worker_port <= 65535
            and arguments.xr02_worker_port != arguments.port,
            "must be a distinct port within 1024..65535",
        )
    return {
        "schema_version": "phase2-console-preflight-v1",
        "ready": all(check["status"] == "pass" for check in checks),
        "checks": checks,
    }


def _check_optional_paths(
    checks: list[dict[str, str]],
    values: dict[str, tuple[Path | None, str]],
) -> None:
    for label, (path, kind) in values.items():
        if path is not None:
            _check_path(checks, label, path, kind)


def _check_path(checks: list[dict[str, str]], label: str, path: Path | None, kind: str) -> None:
    if path is None:
        _result(checks, label, False, "not configured")
        return
    resolved = path.resolve()
    if kind == "file":
        passed = resolved.is_file()
        detail = str(resolved) if passed else f"unavailable: {resolved}"
    elif kind == "directory":
        passed = resolved.is_dir()
        detail = str(resolved) if passed else f"unavailable: {resolved}"
    elif kind == "parent":
        passed, detail = _check_creatable_location(resolved)
    else:
        raise ValueError(f"unsupported path preflight kind: {kind}")
    _result(checks, label, passed, detail)


def _check_creatable_location(path: Path) -> tuple[bool, str]:
    if path.exists():
        return True, str(path)
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.is_dir() and os.access(ancestor, os.W_OK):
        return True, f"will create below {ancestor}"
    return False, f"no writable parent for: {path}"


def _check_xr02_deployment(checks: list[dict[str, str]], path: Path | None) -> None:
    if path is None:
        _result(checks, "XR02 deployment config", False, "not configured")
        return
    try:
        deployment = load_xr02_deployment(path)
    except XR02DeploymentError as error:
        _result(checks, "XR02 deployment config", False, str(error))
        return
    _result(checks, "XR02 deployment config", True, str(deployment.source_path))
    _check_path(checks, "XR02 operator state", deployment.operator_state, "file")
    _check_path(checks, "XR02 environment file", deployment.environment_file, "file")
    _check_path(checks, "XR02 camera policy", deployment.camera_policy, "file")
    _check_path(checks, "XR02 Ultralytics config", deployment.ultralytics_config, "parent")
    if deployment.wp2_overlay is not None:
        _check_path(checks, "XR02 WP2 overlay", deployment.wp2_overlay, "directory")
    if deployment.environment_file.is_file():
        try:
            endpoints = load_local_rtsp_endpoints(
                load_environment_file(deployment.environment_file)
            )
            _result(
                checks,
                "XR02 RTSP endpoint roster",
                bool(endpoints),
                f"{len(endpoints)} credential-bearing endpoint(s) configured locally",
            )
        except (KeyError, OSError, UnicodeError, ValueError) as error:
            _result(checks, "XR02 RTSP endpoint roster", False, str(error))
    for label, source, expected in _xr02_hashed_inputs(deployment):
        _check_file_hash(checks, label, source, expected)


def _check_xr02_worker_entrypoint(
    checks: list[dict[str, str]], script: Path | None, module: str | None
) -> None:
    if (script is None) == (module is None):
        _result(
            checks,
            "XR02 worker entrypoint",
            False,
            "configure exactly one worker script or installed module",
        )
    elif script is not None:
        _check_path(checks, "XR02 worker entrypoint", script, "file")
    else:
        assert module is not None
        valid = re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) is not None
        _result(
            checks,
            "XR02 worker entrypoint",
            valid,
            module if valid else "invalid Python module name",
        )


def _xr02_hashed_inputs(
    deployment: XR02Deployment,
) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("XR02 P06 calibration", deployment.p06_calibration_manifest, deployment.hashes.p06),
        ("XR02 P07 geometry", deployment.p07_geometry_manifest, deployment.hashes.p07),
        (
            "XR02 P08 floor manifest",
            deployment.p08_floor_manifest,
            deployment.hashes.p08_floor_manifest,
        ),
        ("XR02 P08 floor", deployment.p08_floor, deployment.hashes.p08_floor),
        ("XR02 detector model", deployment.detector_model, deployment.hashes.detector),
        ("XR02 ReID model", deployment.reid_model, deployment.hashes.reid),
    )


def _check_file_hash(checks: list[dict[str, str]], label: str, path: Path, expected: str) -> None:
    if not path.is_file():
        _result(checks, label, False, f"unavailable: {path}")
        return
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _result(checks, label, False, f"cannot read {path}: {error}")
        return
    actual = digest.hexdigest()
    _result(
        checks,
        label,
        actual == expected,
        str(path) if actual == expected else f"SHA-256 mismatch: {path}",
    )


def _result(checks: list[dict[str, str]], label: str, passed: bool, detail: str) -> None:
    checks.append({"check": label, "status": "pass" if passed else "fail", "detail": detail})
