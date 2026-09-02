"""Credential-safe loading of local runtime key/value files."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def load_environment_file(path: Path) -> dict[str, str]:
    """Load local key/value data without logging credential-bearing values."""

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def repository_source_environment(
    repository_root: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return a child-process environment with this repository's package source first."""

    environment = dict(os.environ if base is None else base)
    source_root = str((repository_root / "src").resolve())
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else os.pathsep.join((source_root, existing))
    )
    return environment
