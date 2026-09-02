from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spatial_mapping_phase2.console_preflight import _check_xr02_deployment
from spatial_mapping_phase2.xr02_deployment import (
    XR02DeploymentError,
    load_xr02_deployment,
)


def _deployment_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, Path]]:
    manifest = tmp_path / "config" / "xr02.json"
    manifest.parent.mkdir()
    sources = {
        "operator_state": tmp_path / "inputs" / "operator-state.json",
        "p06_calibration_manifest": tmp_path / "inputs" / "p06.json",
        "p07_geometry_manifest": tmp_path / "inputs" / "p07.json",
        "p08_floor_manifest": tmp_path / "inputs" / "p08.json",
        "p08_floor": tmp_path / "inputs" / "floor.npz",
        "detector_model": tmp_path / "models" / "detector.pt",
        "reid_model": tmp_path / "models" / "reid.pt",
        "environment_file": tmp_path / "secret" / ".env",
        "camera_policy": tmp_path / "inputs" / "camera-policy.sqlite3",
    }
    for index, path in enumerate(sources.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode())
    sources["environment_file"].write_text(
        "\n".join(
            f"PHASE2_RTSP_CAMERA_{index}=rtsp://user:secret@example.test/camera-{index}"
            for index in range(1, 5)
        ),
        encoding="utf-8",
    )
    hash_sources = {
        "p06": sources["p06_calibration_manifest"],
        "p07": sources["p07_geometry_manifest"],
        "p08_floor_manifest": sources["p08_floor_manifest"],
        "p08_floor": sources["p08_floor"],
        "detector": sources["detector_model"],
        "reid": sources["reid_model"],
    }
    value: dict[str, object] = {
        "schema_version": "xr02-worker-deployment-v1",
        **{field: str(Path("..") / path.relative_to(tmp_path)) for field, path in sources.items()},
        "ultralytics_config": "ultralytics-config",
        "hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest().upper()
            for name, path in hash_sources.items()
        },
    }
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest, value, sources


def test_deployment_resolves_paths_and_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest, value, sources = _deployment_fixture(tmp_path)

    deployment = load_xr02_deployment(manifest)

    assert deployment.p06_calibration_manifest == sources["p06_calibration_manifest"].resolve()
    assert (
        deployment.hashes.p06
        == hashlib.sha256(sources["p06_calibration_manifest"].read_bytes()).hexdigest()
    )
    value["mystery"] = True
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(XR02DeploymentError, match="unknown XR02 deployment fields: mystery"):
        load_xr02_deployment(manifest)


def test_deployment_preflight_verifies_hashes_without_creating_paths(tmp_path: Path) -> None:
    manifest, _value, sources = _deployment_fixture(tmp_path)
    ultralytics = manifest.parent / "ultralytics-config"
    checks: list[dict[str, str]] = []

    _check_xr02_deployment(checks, manifest)

    assert all(check["status"] == "pass" for check in checks)
    assert not ultralytics.exists()
    assert all("secret@example.test" not in check["detail"] for check in checks)

    sources["detector_model"].write_bytes(b"changed")
    changed: list[dict[str, str]] = []
    _check_xr02_deployment(changed, manifest)
    detector = next(check for check in changed if check["check"] == "XR02 detector model")
    assert detector["status"] == "fail"
    assert detector["detail"].startswith("SHA-256 mismatch:")
