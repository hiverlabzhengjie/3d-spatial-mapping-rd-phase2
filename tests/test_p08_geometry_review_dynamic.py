from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_exporter() -> ModuleType:
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    dependency_name = "export_p07_all4_da3_cloud"
    previous_dependency = sys.modules.get(dependency_name)
    dependency = ModuleType(dependency_name)
    dependency._array_sha256 = lambda _value: "unused"  # type: ignore[attr-defined]
    dependency.write_geometry_review_rerun = lambda **_values: ()  # type: ignore[attr-defined]
    sys.modules[dependency_name] = dependency
    try:
        spec = importlib.util.spec_from_file_location(
            "test_export_p08_geometry_review_rerun",
            scripts / "export_p08_geometry_review_rerun.py",
        )
        if spec is None or spec.loader is None:
            raise AssertionError("geometry-review exporter could not be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))
        if previous_dependency is None:
            del sys.modules[dependency_name]
        else:
            sys.modules[dependency_name] = previous_dependency


def test_geometry_review_accepts_generic_scene_camera_roster() -> None:
    exporter = _load_exporter()

    camera_order = exporter._camera_order(
        {
            "schema_version": "xr03-scene-joint-da3-combined-geometry-v1",
            "camera_order": ["north", "south", "loading-bay"],
        }
    )

    assert camera_order == ("north", "south", "loading-bay")


def test_geometry_review_preserves_strict_historical_camera_roster() -> None:
    exporter = _load_exporter()

    with pytest.raises(ValueError, match="exact Camera"):
        exporter._camera_order(
            {
                "schema_version": "p07-all4-da3-combined-geometry-v1",
                "camera_order": ["north", "south"],
            }
        )
