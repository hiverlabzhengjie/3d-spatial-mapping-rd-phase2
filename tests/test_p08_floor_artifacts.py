from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spatial_mapping_phase2.p08_floor import FloorProcessingConfig, P08FloorError
from spatial_mapping_phase2.p08_floor_artifacts import (
    FrozenP07FloorInput,
    create_floor_artifact_run,
    create_floor_artifact_run_from_geometry,
    load_frozen_p07_source,
    verify_floor_artifact_run,
    verify_floor_artifact_run_from_geometry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _frozen_input(tmp_path: Path) -> FrozenP07FloorInput:
    geometry_path = tmp_path / "geometry.npz"
    cells = np.array(
        [
            [x * 0.1 + 0.01, y * 0.1 + 0.01, 0.0]
            for y in range(4)
            for x in range(4)
            if (x, y) != (2, 2)
        ]
    )
    points = cells[np.arange(97_643) % len(cells)].copy()
    points[0, 2] = -0.5
    points[1, 2] = 1.0
    represented = np.ones(97_643, dtype=np.int32)
    represented[0] += 534_961 - 97_643
    np.savez(
        geometry_path,
        points=points,
        colors_rgb=np.zeros((97_643, 3), dtype=np.uint8),
        confidence=np.ones(97_643),
        source_pixel_count=represented,
        source_camera_index=np.arange(97_643, dtype=np.int16) % 4,
        camera_ids=np.asarray(["c1", "c2", "c3", "c4"]),
    )
    rollback_path = tmp_path / "rollback.json"
    _write_json(rollback_path, {"schema_version": "rollback"})
    rerun_path = tmp_path / "selected.rrd"
    rerun_path.write_bytes(b"deterministic-test-rerun")
    rerun_manifest_path = tmp_path / "rerun-manifest.json"
    _write_json(
        rerun_manifest_path,
        {
            "selected_geometry_unchanged": True,
            "source_camera_membership_preserved": True,
            "rerun": {"path": str(rerun_path), "sha256": _sha256(rerun_path)},
        },
    )
    adoption_path = tmp_path / "adoption.json"
    _write_json(
        adoption_path,
        {
            "success": True,
            "selection": "owner-approved-working-facility-geometry-v2",
            "selected_geometry": {
                "path": str(geometry_path),
                "sha256": _sha256(geometry_path),
            },
            "inputs": {
                "D041_v1_rollback_manifest": {
                    "path": str(rollback_path),
                    "sha256": _sha256(rollback_path),
                }
            },
        },
    )
    return FrozenP07FloorInput(
        adoption_manifest_path=adoption_path,
        adoption_manifest_sha256=_sha256(adoption_path),
        selected_geometry_path=geometry_path,
        selected_geometry_sha256=_sha256(geometry_path),
        rollback_manifest_path=rollback_path,
        rollback_manifest_sha256=_sha256(rollback_path),
        final_rerun_manifest_path=rerun_manifest_path,
        final_rerun_manifest_sha256=_sha256(rerun_manifest_path),
        final_rerun_path=rerun_path,
        final_rerun_sha256=_sha256(rerun_path),
    )


def _config() -> FloorProcessingConfig:
    return FloorProcessingConfig()


def test_artifact_run_repeats_to_identical_npz_and_manifest_hashes(
    tmp_path: Path,
) -> None:
    contract = _frozen_input(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = create_floor_artifact_run(contract, _config(), first)
    second_manifest = create_floor_artifact_run(contract, _config(), second)
    assert first_manifest["manifest"]["sha256"] == second_manifest["manifest"]["sha256"]
    first_artifacts = first_manifest["artifacts"]
    second_artifacts = second_manifest["artifacts"]
    assert isinstance(first_artifacts, dict) and isinstance(second_artifacts, dict)
    assert {name: record["sha256"] for name, record in first_artifacts.items()} == {
        name: record["sha256"] for name, record in second_artifacts.items()
    }
    verification = verify_floor_artifact_run(first, contract)
    assert verification["status"] == "passed"
    assert verification["deterministic_recomputation_exact"] is True
    with np.load(first / "authoritative_floor_plane.npz", allow_pickle=False) as plane:
        assert np.all(plane["vertices_xyz_metres"][:, 2] == 0.0)
        assert plane["vertices_xyz_metres"].shape == (4, 3)
        assert plane["triangle_indices"].shape == (2, 3)
    assert verification["original_points_removed"] == 0
    assert verification["original_point_colors_modified"] is False


def test_frozen_source_rejects_changed_identity_and_output_overwrite(
    tmp_path: Path,
) -> None:
    contract = _frozen_input(tmp_path)
    output = tmp_path / "run"
    create_floor_artifact_run(contract, _config(), output)
    with pytest.raises(P08FloorError, match="already exists"):
        create_floor_artifact_run(contract, _config(), output)
    contract.final_rerun_path.write_bytes(b"changed")
    with pytest.raises(P08FloorError, match="identity mismatch"):
        load_frozen_p07_source(contract)


def test_contract_rejects_non_sha256_identity(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"x")
    with pytest.raises(P08FloorError, match="lowercase SHA-256"):
        FrozenP07FloorInput(
            path,
            "BAD",
            path,
            "0" * 64,
            path,
            "0" * 64,
            path,
            "0" * 64,
            path,
            "0" * 64,
        )


def test_dynamic_scene_floor_is_exact_z_zero_and_preserves_variable_roster(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene-combined.npz"
    np.savez_compressed(
        source,
        points=np.array(
            [[-2.0, 1.0, -0.4], [4.0, 7.0, 2.5], [1.0, 4.0, 0.2]],
            dtype=np.float64,
        ),
        colors_rgb=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
        confidence=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        source_pixel_count=np.array([2, 3, 4], dtype=np.int32),
        source_camera_index=np.array([0, 1, 2], dtype=np.int16),
        camera_ids=np.asarray(["north", "south", "loading-bay"]),
    )
    source_sha256 = _sha256(source)
    run = tmp_path / "dynamic-floor"

    result = create_floor_artifact_run_from_geometry(
        source, source_sha256, FloorProcessingConfig(), run
    )
    verification = verify_floor_artifact_run_from_geometry(run, source, source_sha256)

    assert result["scene_update_dynamic_source"] is True
    assert result["source"]["selected_geometry"]["sha256"] == source_sha256
    assert verification["status"] == "passed"
    assert verification["original_points_removed"] == 0
    with np.load(run / "authoritative_floor_plane.npz", allow_pickle=False) as plane:
        assert np.array_equal(
            plane["vertices_xyz_metres"][:, 2], np.zeros(4, dtype=np.float64)
        )
        assert plane["vertices_xyz_metres"][:, :2].tolist() == [
            [-2.0, 1.0],
            [4.0, 1.0],
            [4.0, 7.0],
            [-2.0, 7.0],
        ]


def test_dynamic_scene_floor_rejects_stale_source_before_creating_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene-combined.npz"
    np.savez_compressed(
        source,
        points=np.array([[0.0, 0.0, -1.0], [2.0, 2.0, 1.0]], dtype=np.float64),
        colors_rgb=np.zeros((2, 3), dtype=np.uint8),
        confidence=np.ones(2, dtype=np.float64),
        source_pixel_count=np.ones(2, dtype=np.int32),
        source_camera_index=np.zeros(2, dtype=np.int16),
        camera_ids=np.asarray(["camera-a"]),
    )
    stale_sha256 = _sha256(source)
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(P08FloorError, match="identity mismatch"):
        create_floor_artifact_run_from_geometry(
            source, stale_sha256, FloorProcessingConfig(), tmp_path / "rejected"
        )
    assert not (tmp_path / "rejected").exists()
