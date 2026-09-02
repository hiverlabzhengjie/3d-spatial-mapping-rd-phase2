"""Strict, secret-free deployment inputs for the isolated XR02 worker."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatial_mapping_phase2.xr02_wp4 import WP4Hashes, WP4Paths

SCHEMA_VERSION = "xr02-worker-deployment-v1"
_PATH_FIELDS = (
    "operator_state",
    "p06_calibration_manifest",
    "p07_geometry_manifest",
    "p08_floor_manifest",
    "p08_floor",
    "detector_model",
    "reid_model",
    "environment_file",
    "camera_policy",
    "ultralytics_config",
)
_HASH_FIELDS = (
    "p06",
    "p07",
    "p08_floor_manifest",
    "p08_floor",
    "detector",
    "reid",
)
_KNOWN_FIELDS = {"schema_version", "hashes", "wp2_overlay", *_PATH_FIELDS}


class XR02DeploymentError(ValueError):
    """Raised when the worker deployment manifest is ambiguous or malformed."""


@dataclass(frozen=True, slots=True)
class XR02DeploymentHashes:
    p06: str
    p07: str
    p08_floor_manifest: str
    p08_floor: str
    detector: str
    reid: str


@dataclass(frozen=True, slots=True)
class XR02Deployment:
    source_path: Path
    operator_state: Path
    p06_calibration_manifest: Path
    p07_geometry_manifest: Path
    p08_floor_manifest: Path
    p08_floor: Path
    detector_model: Path
    reid_model: Path
    environment_file: Path
    camera_policy: Path
    ultralytics_config: Path
    hashes: XR02DeploymentHashes
    wp2_overlay: Path | None = None

    def wp4_paths(self, output_root: Path) -> WP4Paths:
        from spatial_mapping_phase2.xr02_wp4 import WP4Paths

        return WP4Paths(
            operator_state=self.operator_state,
            p06=self.p06_calibration_manifest,
            p07=self.p07_geometry_manifest,
            p08_floor_manifest=self.p08_floor_manifest,
            p08_floor=self.p08_floor,
            detector=self.detector_model,
            reid=self.reid_model,
            output_root=output_root.resolve(),
            environment_file=self.environment_file,
            camera_policy=self.camera_policy,
        )

    def wp4_hashes(self) -> WP4Hashes:
        from spatial_mapping_phase2.xr02_wp4 import WP4Hashes

        return WP4Hashes(
            p06=self.hashes.p06,
            p07=self.hashes.p07,
            p08_floor_manifest=self.hashes.p08_floor_manifest,
            p08_floor=self.hashes.p08_floor,
            detector=self.hashes.detector,
            reid=self.hashes.reid,
        )


def load_xr02_deployment(path: Path) -> XR02Deployment:
    """Load a path-relative manifest without reading secrets or mutable runtime state."""

    source = path.resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XR02DeploymentError(f"cannot read XR02 deployment manifest: {source}") from error
    if not isinstance(value, dict):
        raise XR02DeploymentError("XR02 deployment manifest root must be an object")
    unknown = sorted(set(value) - _KNOWN_FIELDS)
    if unknown:
        raise XR02DeploymentError("unknown XR02 deployment fields: " + ", ".join(unknown))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise XR02DeploymentError(f"XR02 deployment schema must be {SCHEMA_VERSION}")
    missing = sorted(field for field in _PATH_FIELDS if field not in value)
    if missing:
        raise XR02DeploymentError("missing XR02 deployment fields: " + ", ".join(missing))

    base = source.parent
    paths = {field: _manifest_path(value[field], field, base) for field in _PATH_FIELDS}
    raw_hashes = value.get("hashes")
    if not isinstance(raw_hashes, dict):
        raise XR02DeploymentError("hashes must be an object")
    unknown_hashes = sorted(set(raw_hashes) - set(_HASH_FIELDS))
    missing_hashes = sorted(set(_HASH_FIELDS) - set(raw_hashes))
    if unknown_hashes:
        raise XR02DeploymentError("unknown XR02 hash fields: " + ", ".join(unknown_hashes))
    if missing_hashes:
        raise XR02DeploymentError("missing XR02 hash fields: " + ", ".join(missing_hashes))
    hashes = XR02DeploymentHashes(
        **{field: _sha256(raw_hashes[field], f"hashes.{field}") for field in _HASH_FIELDS}
    )
    overlay = value.get("wp2_overlay")
    return XR02Deployment(
        source_path=source,
        operator_state=paths["operator_state"],
        p06_calibration_manifest=paths["p06_calibration_manifest"],
        p07_geometry_manifest=paths["p07_geometry_manifest"],
        p08_floor_manifest=paths["p08_floor_manifest"],
        p08_floor=paths["p08_floor"],
        detector_model=paths["detector_model"],
        reid_model=paths["reid_model"],
        environment_file=paths["environment_file"],
        camera_policy=paths["camera_policy"],
        ultralytics_config=paths["ultralytics_config"],
        hashes=hashes,
        wp2_overlay=(None if overlay is None else _manifest_path(overlay, "wp2_overlay", base)),
    )


def _manifest_path(value: Any, field: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise XR02DeploymentError(f"{field} must be a non-empty path string")
    path = Path(value.strip())
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise XR02DeploymentError(f"{field} must be a 64-character SHA-256")
    return value.lower()
