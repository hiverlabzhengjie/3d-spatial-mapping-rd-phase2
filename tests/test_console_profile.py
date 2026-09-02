from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spatial_mapping_phase2.console_preflight import console_preflight
from spatial_mapping_phase2.console_profile import (
    ConsoleProfileError,
    load_console_profile_arguments,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_profile_resolves_paths_and_rejects_unknown_fields(tmp_path: Path) -> None:
    profile = tmp_path / "profiles" / "console.json"
    profile.parent.mkdir()
    _write(
        profile,
        {
            "schema_version": "phase2-console-profile-v1",
            "workspace_dir": "../workspace",
            "enable_live_p03": True,
            "allowed_artifact_roots": ["../artifacts"],
            "calibration_workspaces": {"camera-a": "../calibration-a"},
            "xr02_deployment_config": "../xr02.json",
            "port": 8088,
        },
    )

    arguments = load_console_profile_arguments(profile)

    assert arguments[:2] == ["--workspace-dir", str((tmp_path / "workspace").resolve())]
    assert "--enable-live-p03" in arguments
    assert str((tmp_path / "artifacts").resolve()) in arguments
    assert str((tmp_path / "xr02.json").resolve()) in arguments
    assert f"camera-a={(tmp_path / 'calibration-a').resolve()}" in arguments

    _write(
        profile,
        {
            "schema_version": "phase2-console-profile-v1",
            "workspace_dir": "../workspace",
            "mystery": True,
        },
    )
    with pytest.raises(ConsoleProfileError, match="unknown console profile fields: mystery"):
        load_console_profile_arguments(profile)


def test_preflight_is_read_only_and_reports_missing_runtime(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scene = workspace / "scene.json"
    scene.write_text("{}", encoding="utf-8")
    scene_registry = tmp_path / "nested" / "state" / "registry.sqlite3"
    managed_scene_root = tmp_path / "nested" / "managed-scenes"
    arguments = argparse.Namespace(
        workspace_dir=workspace,
        scene_registry=scene_registry,
        managed_scene_root=managed_scene_root,
        floor_contract=None,
        floor_config=None,
        floor_output_root=None,
        operator_profile=None,
        workflow_python=None,
        repository_root=None,
        p06_run_directory=None,
        da3_source_directory=None,
        da3_checkpoint_directory=None,
        d041_manifest=None,
        reconstruction_output_root=None,
        intrinsic_evidence=None,
        facility_export=None,
        calibration_output_root=None,
        viewer=None,
        p02_workspace=None,
        p04_workspace=None,
        p05_workspace=None,
        allowed_artifact_root=[],
        calibration_workspace=[],
        host="127.0.0.1",
        port=8088,
        enable_live_p03=False,
        secret_file=tmp_path / ".env",
        enable_xr02_live_operations=True,
        xr02_python=tmp_path / "missing-python.exe",
        xr02_worker_script=tmp_path / "missing-worker.py",
        xr02_worker_module=None,
        xr02_deployment_config=None,
        xr02_mediamtx_binary=tmp_path / "missing-mediamtx.exe",
        xr02_ffmpeg_binary=None,
        xr02_output_root=tmp_path / "runs",
        xr02_worker_port=8094,
    )

    result = console_preflight(arguments)

    assert result["ready"] is False
    failures = {item["check"] for item in result["checks"] if item["status"] == "fail"}
    assert {
        "XR02 Python",
        "XR02 worker entrypoint",
        "XR02 deployment config",
        "MediaMTX",
        "FFmpeg",
    } <= failures
    checks = {item["check"]: item for item in result["checks"]}
    assert checks["scene registry parent"]["status"] == "pass"
    assert checks["managed scene root parent"]["status"] == "pass"
    assert "will create below" in checks["scene registry parent"]["detail"]
    assert not scene_registry.parent.exists()
    assert not managed_scene_root.exists()
    assert sorted(workspace.iterdir()) == [scene]


def test_cached_console_import_does_not_require_optional_runtime_extras() -> None:
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in {'av', 'fitz'} or name.startswith(('av.', 'fitz.')):
        raise AssertionError('cached console imported an optional runtime')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import spatial_mapping_phase2.console_cli
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
