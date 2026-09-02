from __future__ import annotations

from pathlib import Path

import pytest

from spatial_mapping_phase2.xr03_live_operations import (
    LiveOperationsError,
    _worker_launch_command,
)


def test_worker_launch_command_supports_source_script_and_installed_module(
    tmp_path: Path,
) -> None:
    python = tmp_path / "runtime" / "python.exe"
    script = tmp_path / "repository" / "scripts" / "worker.py"
    python.parent.mkdir()
    script.parent.mkdir(parents=True)
    python.touch()
    script.touch()

    script_command, working_directory = _worker_launch_command(
        python, script, None, 8094, ("--deployment-config", "config.json")
    )
    module_command, module_working_directory = _worker_launch_command(
        python,
        None,
        "spatial_mapping_phase2.xr02_worker_cli",
        8094,
        ("--deployment-config", "config.json"),
    )

    assert script_command[:2] == [str(python.resolve()), str(script.resolve())]
    assert working_directory == str(script.resolve().parents[1])
    assert module_command[:3] == [
        str(python.resolve()),
        "-m",
        "spatial_mapping_phase2.xr02_worker_cli",
    ]
    assert module_working_directory is None
    assert module_command[-2:] == ["--deployment-config", "config.json"]


@pytest.mark.parametrize(
    ("script", "module"),
    ((None, None), (Path("worker.py"), "worker.module"), (None, "bad-module!")),
)
def test_worker_launch_command_rejects_ambiguous_or_invalid_entrypoint(
    script: Path | None, module: str | None
) -> None:
    with pytest.raises(LiveOperationsError):
        _worker_launch_command(Path("python.exe"), script, module, 8094, ())
