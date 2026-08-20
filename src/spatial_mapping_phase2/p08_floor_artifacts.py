"""Immutable, hash-bound P08 floor artifact generation and verification."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p08_floor import (
    FLOOR_DERIVATIVE_AUTHORITY,
    FLOOR_PLANE_COLOR_RGBA,
    FLOOR_PLANE_OPACITY,
    FloorProcessingConfig,
    FloorProcessingResult,
    FloorSourceGeometry,
    P08FloorError,
    process_floor,
)

Array = NDArray[Any]


@dataclass(frozen=True)
class FrozenP07FloorInput:
    """Exact immutable P07 v2 source, v1 rollback, and final Rerun identities."""

    adoption_manifest_path: Path
    adoption_manifest_sha256: str
    selected_geometry_path: Path
    selected_geometry_sha256: str
    rollback_manifest_path: Path
    rollback_manifest_sha256: str
    final_rerun_manifest_path: Path
    final_rerun_manifest_sha256: str
    final_rerun_path: Path
    final_rerun_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "adoption_manifest_sha256",
            "selected_geometry_sha256",
            "rollback_manifest_sha256",
            "final_rerun_manifest_sha256",
            "final_rerun_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise P08FloorError(f"{name} must be a lowercase SHA-256")

    def source_identities(self) -> dict[str, dict[str, Any]]:
        return {
            "adoption_manifest": _identity(self.adoption_manifest_path),
            "selected_geometry": _identity(self.selected_geometry_path),
            "rollback_manifest": _identity(self.rollback_manifest_path),
            "final_rerun_manifest": _identity(self.final_rerun_manifest_path),
            "final_rerun": _identity(self.final_rerun_path),
        }


def load_frozen_p07_source(contract: FrozenP07FloorInput) -> FloorSourceGeometry:
    """Rehash the full P07 boundary and load the selected source geometry."""

    required = (
        (
            contract.adoption_manifest_path,
            contract.adoption_manifest_sha256,
            "adoption manifest",
        ),
        (
            contract.selected_geometry_path,
            contract.selected_geometry_sha256,
            "selected geometry",
        ),
        (
            contract.rollback_manifest_path,
            contract.rollback_manifest_sha256,
            "v1 rollback",
        ),
        (
            contract.final_rerun_manifest_path,
            contract.final_rerun_manifest_sha256,
            "final Rerun manifest",
        ),
        (
            contract.final_rerun_path,
            contract.final_rerun_sha256,
            "final Rerun recording",
        ),
    )
    for path, expected, label in required:
        _require_hash(path, expected, label)
    adoption = _read_json(contract.adoption_manifest_path)
    selected = _mapping(adoption, "selected_geometry")
    if (
        adoption.get("selection") != "owner-approved-working-facility-geometry-v2"
        or adoption.get("success") is not True
        or selected.get("sha256") != contract.selected_geometry_sha256
        or Path(str(selected.get("path"))).resolve()
        != contract.selected_geometry_path.resolve()
    ):
        raise P08FloorError("P07 adoption no longer selects the configured v2 geometry")
    rollback = _mapping(_mapping(adoption, "inputs"), "D041_v1_rollback_manifest")
    if rollback.get("sha256") != contract.rollback_manifest_sha256:
        raise P08FloorError("P07 adoption rollback identity changed")
    final_rerun = _read_json(contract.final_rerun_manifest_path)
    rerun_record = _mapping(final_rerun, "rerun")
    if (
        final_rerun.get("selected_geometry_unchanged") is not True
        or final_rerun.get("source_camera_membership_preserved") is not True
        or rerun_record.get("sha256") != contract.final_rerun_sha256
    ):
        raise P08FloorError("final P07 Rerun boundary no longer preserves selected v2")
    with np.load(contract.selected_geometry_path, allow_pickle=False) as archive:
        source = FloorSourceGeometry(
            points=np.asarray(archive["points"]),
            colors_rgb=np.asarray(archive["colors_rgb"]),
            confidence=np.asarray(archive["confidence"]),
            source_pixel_count=np.asarray(archive["source_pixel_count"]),
            source_camera_index=np.asarray(archive["source_camera_index"]),
            camera_ids=tuple(
                str(value) for value in np.asarray(archive["camera_ids"]).tolist()
            ),
        )
    if source.point_count != 97_643 or source.represented_source_pixel_count != 534_961:
        raise P08FloorError("frozen P07 v2 source counts changed")
    return source


def create_floor_artifact_run(
    contract: FrozenP07FloorInput,
    config: FloorProcessingConfig,
    output_directory: Path,
) -> dict[str, Any]:
    """Create one non-overwriting deterministic floor derivative and manifest."""

    source = load_frozen_p07_source(contract)
    result = process_floor(source, config)
    output = output_directory.resolve()
    if output.exists():
        raise P08FloorError("P08 floor output directory already exists")
    output.mkdir(parents=True)
    payloads = _artifact_payloads(result)
    artifact_records: dict[str, dict[str, Any]] = {}
    for name in sorted(payloads):
        path = output / f"{name}.npz"
        _write_deterministic_npz(path, payloads[name])
        artifact_records[name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "byte_count": path.stat().st_size,
            "array_sha256": {
                key: _array_sha256(value)
                for key, value in sorted(payloads[name].items())
            },
        }
    manifest = {
        "schema_version": "p08-authoritative-floor-plane-artifacts-v2",
        "selection": FLOOR_DERIVATIVE_AUTHORITY,
        "success": True,
        "source": contract.source_identities(),
        "config": config.to_dict(),
        "summary": result.summary(),
        "artifacts": artifact_records,
        "floor_plane_visual_rgba": list(FLOOR_PLANE_COLOR_RGBA),
        "floor_plane_visual_opacity": FLOOR_PLANE_OPACITY,
        "floor_surface_authoritative": True,
        "floor_surface_geometry": (
            "four vertices and two triangles; continuous mathematical rectangle"
        ),
        "generated_point_count": 0,
        "floor_plane_z_metres": 0.0,
        "original_points_removed": 0,
        "original_point_colors_modified": False,
        "P07_v2_modified": False,
        "P07_v1_rollback_modified": False,
        "camera_pose_or_intrinsic_modified": False,
        "unsupported_non_floor_surface_completed": False,
        "accepted_geometry": False,
        "survey_grade_XYZ_accuracy": False,
        "authority": result.authority,
        "intended_use": result.intended_use,
    }
    manifest_path = output / "floor-completion-manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    return {**manifest, "manifest": _identity(manifest_path)}


def verify_floor_artifact_run(
    run_directory: Path, contract: FrozenP07FloorInput
) -> dict[str, Any]:
    """Recompute the floor result and require exact deterministic artifact identities."""

    run = run_directory.resolve()
    manifest_path = run / "floor-completion-manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("success") is not True
        or manifest.get("selection") != FLOOR_DERIVATIVE_AUTHORITY
        or manifest.get("P07_v2_modified") is not False
        or manifest.get("P07_v1_rollback_modified") is not False
        or manifest.get("accepted_geometry") is not False
    ):
        raise P08FloorError(
            "floor artifact manifest escaped the P08 authority boundary"
        )
    source_identities = contract.source_identities()
    if manifest.get("source") != source_identities:
        raise P08FloorError("floor artifact source identities changed")
    config_payload = manifest.get("config")
    if not isinstance(config_payload, dict):
        raise P08FloorError("floor artifact configuration is malformed")
    config = FloorProcessingConfig(**config_payload)
    source = load_frozen_p07_source(contract)
    result = process_floor(source, config)
    expected_payloads = _artifact_payloads(result)
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or set(records) != set(expected_payloads):
        raise P08FloorError("floor artifact inventory is incomplete")
    for name, expected_arrays in expected_payloads.items():
        record = records.get(name)
        if not isinstance(record, dict) or record.get("path") != f"{name}.npz":
            raise P08FloorError(f"floor artifact record is malformed: {name}")
        path = run / str(record["path"])
        _require_hash(path, str(record.get("sha256")), name)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_arrays):
                raise P08FloorError(f"floor artifact arrays changed: {name}")
            for key, expected in expected_arrays.items():
                actual = np.asarray(archive[key])
                equal = (
                    np.array_equal(actual, expected, equal_nan=True)
                    if np.issubdtype(actual.dtype, np.inexact)
                    else np.array_equal(actual, expected)
                )
                if not equal:
                    raise P08FloorError(f"floor artifact array mismatch: {name}/{key}")
    if manifest.get("summary") != result.summary():
        raise P08FloorError("floor artifact summary does not match recomputation")
    return {
        "schema_version": "p08-authoritative-floor-plane-verification-v2",
        "status": "passed",
        "manifest": _identity(manifest_path),
        "source": source_identities,
        "artifact_count": len(expected_payloads),
        "summary": result.summary(),
        "deterministic_recomputation_exact": True,
        "original_source_identity_exact": True,
        "original_points_removed": 0,
        "original_point_colors_modified": False,
        "generated_point_count": 0,
        "authoritative_floor_plane_at_z_zero": True,
        "authoritative_floor_plane_matches_source_xy_box": True,
        "P07_v2_modified": False,
        "P07_v1_rollback_modified": False,
        "accepted_geometry": False,
    }


def _artifact_payloads(result: FloorProcessingResult) -> dict[str, dict[str, Array]]:
    return {
        "authoritative_floor_plane": {
            "vertices_xyz_metres": np.asarray(result.plane_vertices_xyz_metres),
            "triangle_indices": np.asarray(result.plane_triangle_indices),
            "source_bounds_xyz_metres": np.asarray(result.source_bounds_xyz_metres),
            "visual_rgba": np.asarray(FLOOR_PLANE_COLOR_RGBA, dtype=np.uint8),
            "visual_opacity": np.asarray(FLOOR_PLANE_OPACITY, dtype=np.float32),
        }
    }


def _write_deterministic_npz(path: Path, arrays: dict[str, Array]) -> None:
    if path.exists():
        raise P08FloorError(f"immutable artifact already exists: {path}")
    with path.open("xb") as destination:
        with zipfile.ZipFile(destination, mode="w") as archive:
            for key, value in sorted(arrays.items()):
                buffer = io.BytesIO()
                np.lib.format.write_array(  # type: ignore[no-untyped-call]
                    buffer, np.asarray(value), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise P08FloorError(f"immutable manifest already exists: {path}") from error


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise P08FloorError(f"required artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise P08FloorError(f"{label} identity mismatch: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P08FloorError(f"required JSON is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise P08FloorError(f"required JSON object is malformed: {path}")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise P08FloorError(f"required mapping is malformed: {key}")
    return result
