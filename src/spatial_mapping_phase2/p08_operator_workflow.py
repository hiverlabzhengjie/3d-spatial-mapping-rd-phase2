"""Operator-facing camera summaries and fixed workflow action adapters for P08."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.p08_workflow import P08WorkflowError
from spatial_mapping_phase2.runtime_environment import repository_source_environment
from spatial_mapping_phase2.xr03_da3_policy import SceneDa3Cohort, SceneDa3PolicyError


@dataclass(frozen=True)
class OperatorWorkflowConfig:
    """Credential-free configuration for the current human workflow."""

    camera_input_manifest_path: Path
    camera_input_manifest_sha256: str
    geometry_rerun_artifact_id: str
    floor_rerun_artifact_id: str
    geometry_approved: bool = False
    floor_approved: bool = False
    schema_version: str = "p08-operator-workflow-v1"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OperatorWorkflowConfig:
        if value.get("schema_version") != "p08-operator-workflow-v1":
            raise P08WorkflowError("unsupported operator workflow profile")
        path = Path(_string(value, "camera_input_manifest_path")).resolve()
        expected = _string(value, "camera_input_manifest_sha256")
        if len(expected) != 64:
            raise P08WorkflowError("camera input manifest SHA-256 is malformed")
        for key in ("geometry_approved", "floor_approved"):
            if not isinstance(value.get(key), bool):
                raise P08WorkflowError(f"{key} must be boolean")
        return cls(
            camera_input_manifest_path=path,
            camera_input_manifest_sha256=expected,
            geometry_rerun_artifact_id=_string(value, "geometry_rerun_artifact_id"),
            floor_rerun_artifact_id=_string(value, "floor_rerun_artifact_id"),
            geometry_approved=bool(value["geometry_approved"]),
            floor_approved=bool(value["floor_approved"]),
        )


class CameraInputSummaryAdapter:
    """Read and validate the exact intrinsics and poses used by static reconstruction."""

    def __init__(self, config: OperatorWorkflowConfig) -> None:
        self.config = config

    def summaries(self, camera_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        path = self.config.camera_input_manifest_path
        if not path.is_file():
            raise P08WorkflowError("camera calibration summary is missing")
        if _sha256(path) != self.config.camera_input_manifest_sha256:
            raise P08WorkflowError("camera calibration summary changed; review calibration again")
        manifest = _read_json(path)
        records = manifest.get("cameras")
        if not isinstance(records, list):
            raise P08WorkflowError("camera calibration summary has no camera records")
        by_id = {
            str(record.get("camera_id")): record for record in records if isinstance(record, dict)
        }
        if set(by_id) != set(camera_ids):
            raise P08WorkflowError("camera calibration summary does not match this scene")
        return tuple(self._camera_summary(camera_id, by_id[camera_id]) for camera_id in camera_ids)

    @staticmethod
    def _camera_summary(camera_id: str, record: dict[str, Any]) -> dict[str, Any]:
        intrinsics = _object(record, "intrinsics")
        seed = _object(record, "seed_transform")
        transform = seed.get("T_world_from_camera")
        if (
            not isinstance(transform, list)
            or len(transform) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in transform)
        ):
            raise P08WorkflowError(f"{camera_id} world pose is malformed")
        matrix = [[float(value) for value in row] for row in transform]
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise P08WorkflowError(f"{camera_id} world pose is non-finite")
        rotation = [row[:3] for row in matrix[:3]]
        yaw, pitch, roll = _euler_zyx_degrees(rotation)
        distortion = intrinsics.get("distortion")
        if not isinstance(distortion, list) or any(
            not isinstance(value, int | float) for value in distortion
        ):
            raise P08WorkflowError(f"{camera_id} distortion model is malformed")
        return {
            "camera_id": camera_id,
            "ready": True,
            "status": "Ready",
            "intrinsics": {
                "model": _string(intrinsics, "model"),
                "resolution": [int(intrinsics["width_pixels"]), int(intrinsics["height_pixels"])],
                "fx_pixels": float(intrinsics["fx_pixels"]),
                "fy_pixels": float(intrinsics["fy_pixels"]),
                "cx_pixels": float(intrinsics["cx_pixels"]),
                "cy_pixels": float(intrinsics["cy_pixels"]),
                "distortion": [float(value) for value in distortion],
            },
            "pose": {
                "frame": "world",
                "transform": "T_world_from_camera",
                "position_metres": [matrix[index][3] for index in range(3)],
                "orientation_zyx_degrees": {"yaw": yaw, "pitch": pitch, "roll": roll},
                "matrix": matrix,
            },
        }


@dataclass(frozen=True)
class ReconstructionWorkflowAdapter:
    """Run one complete scene cohort through DA3 and deterministic geometry export."""

    python_executable: Path
    repository_root: Path
    p06_run_directory: Path | None
    source_directory: Path
    checkpoint_directory: Path
    d041_manifest_path: Path | None
    output_root: Path
    expected_geometry_sha256: str | None = None
    process_resolution: int | None = None
    input_readiness: Callable[[], Sequence[str]] | None = None
    supports_scene_camera_policy: bool = True

    def readiness_errors(self) -> tuple[str, ...]:
        checks = [
            (self.python_executable, "Python runtime", True),
            (
                self.repository_root / "src" / "spatial_mapping_phase2",
                "repository package source",
                False,
            ),
            (
                self.repository_root / "scripts" / "run_p07_all4_da3_diagnostic.py",
                "DA3 runner",
                True,
            ),
            (
                self.repository_root / "scripts" / "export_p07_all4_da3_cloud.py",
                "geometry exporter",
                True,
            ),
            (
                self.repository_root / "scripts" / "verify_p07_all4_da3_diagnostic.py",
                "geometry verifier",
                True,
            ),
            (self.source_directory, "DA3 source", False),
            (self.checkpoint_directory, "DA3 checkpoint", False),
        ]
        if self.p06_run_directory is not None:
            checks.append((self.p06_run_directory, "selected reconstruction inputs", False))
        if self.d041_manifest_path is not None:
            checks.append((self.d041_manifest_path, "rollback manifest", True))
        errors: list[str] = []
        for path, label, must_be_file in checks:
            exists = path.is_file() if must_be_file else path.is_dir()
            if not exists:
                errors.append(f"{label} is missing")
        if self.input_readiness is not None:
            errors.extend(str(error) for error in self.input_readiness())
        return tuple(errors)

    def run(
        self,
        job_id: str,
        cancel_event: threading.Event,
        *,
        input_run_directory: Path | None = None,
        process_resolution: int | None = None,
        camera_ids: Sequence[str] | None = None,
        camera_policy_sha256: str | None = None,
    ) -> dict[str, Any]:
        errors = self.readiness_errors()
        if errors:
            raise P08WorkflowError("; ".join(errors))
        selected_source = input_run_directory or self.p06_run_directory
        if selected_source is None:
            raise P08WorkflowError("scene reconstruction inputs have not been prepared")
        selected_inputs = selected_source.resolve()
        input_manifest = selected_inputs / "input-manifest.json"
        run_manifest = selected_inputs / "run-manifest.json"
        if not input_manifest.is_file() or not run_manifest.is_file():
            raise P08WorkflowError("selected reconstruction input manifests are missing")
        output = self.output_root.resolve() / job_id
        if output.exists():
            raise P08WorkflowError("reconstruction output already exists; use a new run name")
        output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_command = [
            str(self.python_executable),
            str(self.repository_root / "scripts" / "run_p07_all4_da3_diagnostic.py"),
            "--p06-run-dir",
            str(selected_inputs),
            "--source-dir",
            str(self.source_directory),
            "--checkpoint-dir",
            str(self.checkpoint_directory),
            "--output-dir",
            str(output),
            "--input-manifest-sha256",
            _sha256(input_manifest),
            "--run-manifest-sha256",
            _sha256(run_manifest),
        ]
        cohort: SceneDa3Cohort | None = None
        if camera_ids is not None:
            try:
                cohort = SceneDa3Cohort.build(camera_ids, camera_policy_sha256 or "")
            except SceneDa3PolicyError as error:
                raise P08WorkflowError(str(error)) from error
            diagnostic_command.extend(cohort.cli_arguments())
        elif camera_policy_sha256 is not None:
            raise P08WorkflowError(
                "camera-policy SHA-256 cannot be supplied without a scene roster"
            )
        effective_resolution = (
            process_resolution if process_resolution is not None else self.process_resolution
        )
        if effective_resolution is not None:
            diagnostic_command.extend(("--process-resolution", str(effective_resolution)))
        verifier_command = [
            str(self.python_executable),
            str(self.repository_root / "scripts" / "verify_p07_all4_da3_diagnostic.py"),
            "--run-dir",
            str(output),
            "--output",
            str(output / "verification.json"),
        ]
        if cohort is None:
            if self.d041_manifest_path is None:
                raise P08WorkflowError("historical reconstruction requires its rollback manifest")
            verifier_command.extend(("--d041-manifest", str(self.d041_manifest_path)))
        commands = (
            tuple(diagnostic_command),
            (
                str(self.python_executable),
                str(self.repository_root / "scripts" / "export_p07_all4_da3_cloud.py"),
                "--diagnostic-run-dir",
                str(output),
            ),
            tuple(verifier_command),
        )
        started = time.perf_counter()
        for command in commands:
            _run_process(command, self.repository_root, cancel_event)
        geometry_manifest = _read_json(output / "geometry-manifest.json")
        combined = _object(geometry_manifest, "combined")
        rerun = _object(geometry_manifest, "rerun")
        geometry_hash = _string(combined, "sha256")
        if (
            input_run_directory is None
            and self.expected_geometry_sha256 is not None
            and geometry_hash != self.expected_geometry_sha256
        ):
            raise P08WorkflowError(
                "static reconstruction completed but differs from the selected geometry; "
                "inspect it as a new candidate"
            )
        return {
            "output_directory": str(output),
            "inference_manifest": _identity(output / "inference-manifest.json"),
            "geometry_manifest": _identity(output / "geometry-manifest.json"),
            "verification": _identity(output / "verification.json"),
            "combined_geometry": {
                "path": _string(combined, "path"),
                "sha256": geometry_hash,
                "point_count": int(combined["point_count"]),
            },
            "rerun": {
                "path": _string(rerun, "path"),
                "sha256": _string(rerun, "sha256"),
                "byte_count": int(rerun["byte_count"]),
            },
            "matches_selected_geometry": self.expected_geometry_sha256 == geometry_hash,
            "fresh_scene_update_input": input_run_directory is not None,
            "scene_camera_ids": list(camera_ids) if camera_ids is not None else None,
            "camera_policy_sha256": camera_policy_sha256,
            "da3_cohort_policy": (
                "all-enabled-cameras-per-scene-joint"
                if camera_ids is not None and len(camera_ids) > 1
                else "single-camera-single-view"
                if camera_ids is not None
                else "historical-all-four"
            ),
            "scene_da3_cohort": None if cohort is None else cohort.to_dict(),
            "process_resolution": effective_resolution,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def ensure_geometry_review(
        self, run_directory: Path, cancel_event: threading.Event
    ) -> dict[str, Any]:
        """Return a camera-rich Rerun, deriving one without repeating DA3 when necessary."""

        geometry_manifest_path = run_directory / "geometry-manifest.json"
        geometry_manifest = _read_json(geometry_manifest_path)
        camera_count = len(geometry_manifest.get("camera_order", ()))
        rerun = _object(geometry_manifest, "rerun")
        context_counts = (
            rerun.get("camera_frustum_count"),
            rerun.get("camera_rgb_image_plane_count"),
            rerun.get("camera_orientation_axis_count"),
            rerun.get("fixed_camera_centre_count"),
            rerun.get("world_space_camera_label_count"),
        )
        if camera_count > 0 and all(value == camera_count for value in context_counts):
            return {
                "manifest": _identity(geometry_manifest_path),
                "rerun": self._verified_rerun_identity(rerun),
                "reused_existing_preview": True,
                "derived_without_da3_rerun": False,
            }

        manifest_path = run_directory / "geometry-review-manifest-v2.json"
        rerun_path = run_directory / "all-four-da3-combined-geometry-review-v2.rrd"
        if manifest_path.exists() or rerun_path.exists():
            if not manifest_path.is_file() or not rerun_path.is_file():
                raise P08WorkflowError(
                    "partial geometry review output exists; preserve it and use a new run"
                )
            return self._verified_geometry_review(manifest_path, rerun_path, reused=True)

        _run_process(
            (
                str(self.python_executable),
                str(self.repository_root / "scripts" / "export_p08_geometry_review_rerun.py"),
                "--run-dir",
                str(run_directory),
            ),
            self.repository_root,
            cancel_event,
        )
        return self._verified_geometry_review(manifest_path, rerun_path, reused=False)

    def _verified_geometry_review(
        self, manifest_path: Path, rerun_path: Path, *, reused: bool
    ) -> dict[str, Any]:
        manifest = _read_json(manifest_path)
        if manifest.get("success") is not True:
            raise P08WorkflowError("geometry review manifest is not successful")
        camera_count = len(manifest.get("camera_order", ()))
        count_keys = (
            "camera_frustum_count",
            "camera_rgb_image_plane_count",
            "camera_orientation_axis_count",
            "fixed_camera_centre_count",
            "world_space_camera_label_count",
        )
        if camera_count <= 0 or any(manifest.get(key) != camera_count for key in count_keys):
            raise P08WorkflowError("geometry review camera context is incomplete")
        rerun = self._verified_rerun_identity(_object(manifest, "rerun"))
        if Path(_string(rerun, "path")).resolve() != rerun_path.resolve():
            raise P08WorkflowError("geometry review Rerun path changed")
        return {
            "manifest": _identity(manifest_path),
            "rerun": rerun,
            "reused_existing_preview": reused,
            "derived_without_da3_rerun": True,
        }

    @staticmethod
    def _verified_rerun_identity(record: dict[str, Any]) -> dict[str, Any]:
        path = Path(_string(record, "path")).resolve()
        expected_sha256 = _string(record, "sha256")
        expected_bytes = int(record["byte_count"])
        if (
            not path.is_file()
            or _sha256(path) != expected_sha256
            or path.stat().st_size != expected_bytes
        ):
            raise P08WorkflowError("geometry review Rerun identity mismatch")
        return {
            "path": str(path),
            "sha256": expected_sha256,
            "byte_count": expected_bytes,
        }


@dataclass(frozen=True)
class FloorPreviewWorkflowAdapter:
    """Build a floor Rerun recording from one completed floor job."""

    python_executable: Path
    repository_root: Path
    floor_contract_path: Path | None

    def run(self, run_directory: Path, cancel_event: threading.Event) -> dict[str, Any]:
        if self.floor_contract_path is None:
            raise P08WorkflowError("historical floor preview requires its frozen source contract")
        manifest_path = run_directory / "floor-rerun-manifest-v4.json"
        rerun_path = run_directory / "candidate-working-facility-geometry-v3-floor-context-v4.rrd"
        if manifest_path.exists() or rerun_path.exists():
            if not manifest_path.is_file() or not rerun_path.is_file():
                raise P08WorkflowError(
                    "floor preview artifacts are incomplete; generate a new floor result"
                )
            return self._verified_preview(manifest_path, rerun_path, reused=True)
        command = (
            str(self.python_executable),
            str(self.repository_root / "scripts" / "export_p08_floor_rerun.py"),
            "--contract",
            str(self.floor_contract_path),
            "--run-dir",
            str(run_directory),
        )
        _run_process(command, self.repository_root, cancel_event)
        return self._verified_preview(manifest_path, rerun_path, reused=False)

    def run_for_geometry(
        self,
        run_directory: Path,
        geometry_manifest_path: Path,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        manifest_path = run_directory / "floor-rerun-manifest-v4.json"
        rerun_path = run_directory / "candidate-working-facility-geometry-v3-floor-context-v4.rrd"
        if manifest_path.exists() or rerun_path.exists():
            if not manifest_path.is_file() or not rerun_path.is_file():
                raise P08WorkflowError(
                    "scene-update floor preview artifacts are incomplete; preserve the run"
                )
            return self._verified_preview(manifest_path, rerun_path, reused=True)
        command = (
            str(self.python_executable),
            str(self.repository_root / "scripts" / "export_p08_floor_rerun.py"),
            "--geometry-manifest",
            str(geometry_manifest_path.resolve()),
            "--run-dir",
            str(run_directory.resolve()),
        )
        _run_process(command, self.repository_root, cancel_event)
        return self._verified_preview(manifest_path, rerun_path, reused=False)

    @staticmethod
    def _verified_preview(
        manifest_path: Path, rerun_path: Path, *, reused: bool
    ) -> dict[str, Any]:
        manifest = _read_json(manifest_path)
        rerun = _object(manifest, "rerun")
        recorded_path = Path(_string(rerun, "path")).resolve()
        recorded_sha256 = _string(rerun, "sha256")
        recorded_bytes = int(rerun["byte_count"])
        if (
            recorded_path != rerun_path.resolve()
            or _sha256(rerun_path) != recorded_sha256
            or rerun_path.stat().st_size != recorded_bytes
        ):
            raise P08WorkflowError("existing floor preview identity is stale")
        return {
            "manifest": _identity(manifest_path),
            "rerun": {
                "path": str(recorded_path),
                "sha256": recorded_sha256,
                "byte_count": recorded_bytes,
            },
            "reused_existing_preview": reused,
        }


def _run_process(
    command: tuple[str, ...], working_directory: Path, cancel_event: threading.Event
) -> None:
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as process_log:
        process = subprocess.Popen(
            list(command),
            cwd=working_directory,
            env=repository_source_environment(working_directory),
            stdout=process_log,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            close_fds=True,
        )
        while process.poll() is None:
            if cancel_event.wait(0.1):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise P08WorkflowError("job cancelled")
        if process.returncode != 0:
            process_log.seek(0)
            message = process_log.read()[-1500:].strip() or "workflow process failed"
            raise P08WorkflowError(message)


def _euler_zyx_degrees(rotation: list[list[float]]) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        yaw = math.atan2(rotation[1][0], rotation[0][0])
        roll = math.atan2(rotation[2][1], rotation[2][2])
    else:
        yaw = math.atan2(-rotation[0][1], rotation[1][1])
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P08WorkflowError(f"workflow input is unreadable: {path.name}") from error
    if not isinstance(value, dict):
        raise P08WorkflowError(f"workflow input must be an object: {path.name}")
    return value


def _object(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise P08WorkflowError(f"{key} must be an object")
    return result


def _string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise P08WorkflowError(f"{key} must be a non-blank string")
    return result.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }
