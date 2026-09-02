from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient

from spatial_mapping_phase2.p08_workflow import (
    BoundedJobManager,
    SceneWorkspaceRepository,
    WorkflowService,
)
from spatial_mapping_phase2.p08_workflow_web import create_p08_workflow_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI = REPOSITORY_ROOT / "scripts" / "p08_workflow_cli.py"
CONSOLE = REPOSITORY_ROOT / "scripts" / "run_p08_workflow_console.py"


def _scene_config(tmp_path: Path) -> dict[str, Any]:
    phases = []
    for index in range(2, 9):
        phase_id = f"P{index:02d}"
        phases.append(
            {
                "phase_id": phase_id,
                "state": "ready",
                "message": f"{phase_id} test state",
                "prerequisites": [] if index == 2 else [f"P{index - 1:02d}"],
                "workspace_reference_ids": [],
                "artifact_ids": [],
            }
        )
    return {
        "schema_version": "p08-scene-workspace-v1",
        "project_id": "cli-project",
        "scene_id": "three-camera-scene",
        "display_name": "CLI parity scene",
        "artifact_root": str((tmp_path / "artifacts").resolve()),
        "cameras": [
            {
                "camera_id": f"camera-{index}",
                "display_name": f"Camera {index}",
                "endpoint_environment_key": f"RTSP_CAMERA_{index}",
                "enabled": True,
            }
            for index in range(1, 4)
        ],
        "phases": phases,
        "workspace_references": [],
        "artifacts": [],
    }


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_create_and_status_match_integrated_web_status(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scene_config = tmp_path / "scene.json"
    scene_config.write_text(json.dumps(_scene_config(tmp_path)), encoding="utf-8")
    created = _run(
        str(CLI),
        "--workspace-dir",
        str(workspace),
        "create-scene",
        "--scene-config",
        str(scene_config),
    )
    assert created.returncode == 0, created.stderr
    created_payload = cast(dict[str, Any], json.loads(created.stdout))
    assert created_payload["scene"]["sha256"]

    cli_status = _run(str(CLI), "--workspace-dir", str(workspace), "status")
    assert cli_status.returncode == 0, cli_status.stderr
    cli_payload = cast(dict[str, Any], json.loads(cli_status.stdout))

    jobs = BoundedJobManager()
    try:
        service = WorkflowService(SceneWorkspaceRepository(workspace), jobs)
        client = TestClient(create_p08_workflow_app(service))
        assert client.get("/api/status").json() == cli_payload
        assert len(cli_payload["camera_roster"]) == 3
    finally:
        jobs.close()


def test_console_rejects_non_localhost_bind_before_loading_workspace(
    tmp_path: Path,
) -> None:
    result = _run(
        str(CONSOLE),
        "--workspace-dir",
        str(tmp_path / "not-created"),
        "--host",
        "0.0.0.0",
    )
    assert result.returncode == 2
    assert "may bind only to localhost" in result.stderr


def test_console_profile_preflight_does_not_start_or_mutate_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scene = workspace / "scene.json"
    scene.write_text("{}", encoding="utf-8")
    profile = tmp_path / "console-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "phase2-console-profile-v1",
                "workspace_dir": "workspace",
                "host": "127.0.0.1",
                "port": 8088,
            }
        ),
        encoding="utf-8",
    )

    result = _run(str(CONSOLE), "--profile", str(profile), "--preflight")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ready"] is True
    assert sorted(workspace.iterdir()) == [scene]


def test_cli_camera_policy_apply_matches_web_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scene_config = tmp_path / "scene.json"
    scene_config.write_text(json.dumps(_scene_config(tmp_path)), encoding="utf-8")
    assert (
        _run(
            str(CLI),
            "--workspace-dir",
            str(workspace),
            "create-scene",
            "--scene-config",
            str(scene_config),
        ).returncode
        == 0
    )
    policy_file = tmp_path / "camera-policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "intrinsic_groups": [
                    {
                        "group_id": "lens-a",
                        "lens_model": "Model A",
                        "camera_ids": ["camera-1", "camera-2", "camera-3"],
                    }
                ],
                "overlap_pair_reviews": [
                    {
                        "camera_id_a": left,
                        "camera_id_b": right,
                        "verdict": "no_overlap",
                    }
                    for left, right in (
                        ("camera-1", "camera-2"),
                        ("camera-1", "camera-3"),
                        ("camera-2", "camera-3"),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    applied = _run(
        str(CLI),
        "--workspace-dir",
        str(workspace),
        "camera-policy-apply",
        "--action-id",
        "cli-camera-policy-first",
        "--policy-file",
        str(policy_file),
    )
    assert applied.returncode == 0, applied.stderr
    cli_hash = json.loads(applied.stdout)["policy"]["policy_sha256"]

    service = WorkflowService(SceneWorkspaceRepository(workspace), BoundedJobManager())
    try:
        client = TestClient(create_p08_workflow_app(service))
        assert (
            client.get("/api/camera-policy").json()["active_policy"]["policy_sha256"] == cli_hash
        )
    finally:
        service.close()


def test_cli_deletion_impact_enforces_baseline_retention_for_single_and_batch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    recording = artifact_root / "authority.rrd"
    recording.write_bytes(b"authority")
    config = _scene_config(tmp_path)
    config["artifacts"] = [
        {
            "artifact_id": "accepted-authority",
            "phase_id": "P07",
            "kind": "rerun-recording",
            "path": str(recording.resolve()),
            "sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
            "authority": "accepted predecessor test",
            "selected": True,
        }
    ]
    scene_config = tmp_path / "scene.json"
    scene_config.write_text(json.dumps(config), encoding="utf-8")
    created = _run(
        str(CLI),
        "--workspace-dir",
        str(workspace),
        "create-scene",
        "--scene-config",
        str(scene_config),
    )
    assert created.returncode == 0, created.stderr

    single = _run(
        str(CLI),
        "--workspace-dir",
        str(workspace),
        "artifact-delete-impact",
        "--artifact-id",
        "accepted-authority",
    )
    assert single.returncode == 0, single.stderr
    single_payload = cast(dict[str, Any], json.loads(single.stdout))
    assert single_payload["allowed"] is False
    assert single_payload["protected_retention"]["class"] == "accepted-predecessor"
    assert single_payload["deletion_token"] is None

    batch = _run(
        str(CLI),
        "--workspace-dir",
        str(workspace),
        "artifact-delete-batch-impact",
        "--artifact-id",
        "accepted-authority",
    )
    assert batch.returncode == 0, batch.stderr
    batch_payload = cast(dict[str, Any], json.loads(batch.stdout))
    assert batch_payload["all_allowed"] is False
    assert recording.is_file()
