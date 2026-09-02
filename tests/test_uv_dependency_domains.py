from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "==0.12.7"
PYTHON_VERSION = "3.11.4"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu124"


def _toml(relative_path: str) -> dict[str, Any]:
    with (REPOSITORY_ROOT / relative_path).open("rb") as stream:
        return tomllib.load(stream)


def _dependency_names(project: dict[str, Any]) -> set[str]:
    dependencies = project["project"]["dependencies"]
    return {str(item).split("==", 1)[0].lower() for item in dependencies}


def test_all_uv_domains_pin_the_tool_and_python() -> None:
    projects = [
        "pyproject.toml",
        "environments/da3/pyproject.toml",
        "environments/xr02/pyproject.toml",
    ]
    python_files = [
        ".python-version",
        "environments/da3/.python-version",
        "environments/xr02/.python-version",
    ]

    for project_path in projects:
        assert _toml(project_path)["tool"]["uv"]["required-version"] == UV_VERSION
    for python_path in python_files:
        assert (REPOSITORY_ROOT / python_path).read_text(
            encoding="utf-8"
        ).strip() == PYTHON_VERSION


def test_gpu_projects_are_independent_windows_runtime_substrates() -> None:
    for project_path in ["environments/da3/pyproject.toml", "environments/xr02/pyproject.toml"]:
        project = _toml(project_path)
        uv = project["tool"]["uv"]
        assert uv["package"] is False
        assert "workspace" not in uv
        assert uv["environments"] == ["sys_platform == 'win32'"]
        assert project["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cu124"}
        assert project["tool"]["uv"]["index"][0] == {
            "name": "pytorch-cu124",
            "url": PYTORCH_INDEX,
            "explicit": True,
        }


def test_da3_and_xr02_keep_their_incompatible_numpy_domains() -> None:
    da3 = _toml("environments/da3/pyproject.toml")
    xr02 = _toml("environments/xr02/pyproject.toml")

    assert "numpy==1.26.4" in da3["project"]["dependencies"]
    assert "numpy==2.2.6" in xr02["project"]["dependencies"]
    assert "xformers" not in _dependency_names(da3)
    assert "spatial-mapping-phase2" not in _dependency_names(da3)
    assert "spatial-mapping-phase2" not in _dependency_names(xr02)


def test_xr02_lock_input_consolidates_the_accepted_overlay_and_source() -> None:
    xr02 = _toml("environments/xr02/pyproject.toml")
    dependencies = xr02["project"]["dependencies"]
    boxmot = xr02["tool"]["uv"]["sources"]["boxmot"]

    assert {
        "av==16.0.1",
        "supervision==0.30.0",
        "defusedxml==0.7.1",
        "pydeprecate==0.11.0",
    } <= set(dependencies)
    assert boxmot == {
        "git": "https://github.com/mikel-brostrom/boxmot.git",
        "rev": "8f8babc5302024b13db7e7faeb50b3da55d1e815",
    }


def test_each_uv_project_has_its_own_lockfile() -> None:
    lockfiles = ["uv.lock", "environments/da3/uv.lock", "environments/xr02/uv.lock"]
    for lockfile in lockfiles:
        lock = _toml(lockfile)
        assert lock["version"] >= 1
        assert lock["requires-python"] == "==3.11.*"
