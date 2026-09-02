"""Shared P02-P08 scene, phase, artifact, job, and safe-launch contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from spatial_mapping_phase2.p08_artifact_catalog import (
    MILESTONE_INDEX,
    ArtifactCatalogError,
    SceneArtifactCatalog,
    milestone_definition,
    milestone_for_artifact,
)
from spatial_mapping_phase2.p08_floor import FloorProcessingConfig
from spatial_mapping_phase2.p08_floor_artifacts import (
    FrozenP07FloorInput,
    create_floor_artifact_run,
    create_floor_artifact_run_from_geometry,
    verify_floor_artifact_run,
    verify_floor_artifact_run_from_geometry,
)
from spatial_mapping_phase2.p08_scene_updates import (
    AdoptedSceneResult,
    SceneUpdateError,
    SceneUpdateRepository,
    SceneUpdateSchedule,
    SceneUpdateScheduler,
    UpdateMode,
)
from spatial_mapping_phase2.xr03_camera_policy import (
    CameraPolicyError,
    CameraPolicyRepository,
    SceneCameraPolicy,
    policy_from_changes,
)

IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
PHASE_ORDER = ("P02", "P03", "P04", "P05", "P06", "P07", "P08")
SATISFIED_PREREQUISITE_STATES = frozenset(
    {"ready", "running", "complete", "provisional", "rejected"}
)
OPERATOR_SURFACE_IDS = frozenset({"facility", "capture"})


class P08WorkflowError(ValueError):
    """Raised when workflow configuration or an action violates the P08 contract."""


class PhaseState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    RUNNING = "running"
    COMPLETE = "complete"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"
    FAILED = "failed"
    STALE = "stale"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowCapabilityState(StrEnum):
    """Availability of one operator-facing workflow surface for the active scene."""

    AVAILABLE = "available"
    BLOCKED = "blocked"
    NOT_PROVISIONED = "not_provisioned"
    UNHEALTHY = "unhealthy"
    READY = "ready"


@dataclass(frozen=True)
class WorkflowCapability:
    """Small, explicit contract for an operator-facing scene capability."""

    state: WorkflowCapabilityState
    reason_code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "state": self.state.value,
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class CameraConfig:
    """Credential-free camera roster entry."""

    camera_id: str
    display_name: str
    endpoint_environment_key: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.camera_id, "camera_id")
        if not self.display_name.strip():
            raise P08WorkflowError("camera display_name must not be blank")
        if self.endpoint_environment_key is not None and (
            not self.endpoint_environment_key
            or any(character.isspace() for character in self.endpoint_environment_key)
        ):
            raise P08WorkflowError("endpoint environment key is malformed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "display_name": self.display_name,
            "endpoint_environment_key": self.endpoint_environment_key,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class WorkspaceReference:
    reference_id: str
    phase_id: str
    path: Path
    kind: str
    required_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.reference_id, "workspace reference_id")
        _require_phase(self.phase_id)
        if not self.path.is_absolute():
            raise P08WorkflowError("workspace reference path must be absolute")
        if not self.kind.strip():
            raise P08WorkflowError("workspace reference kind must not be blank")
        if any(Path(name).name != name for name in self.required_files):
            raise P08WorkflowError("required workspace files must be plain filenames")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "phase_id": self.phase_id,
            "path": str(self.path),
            "kind": self.kind,
            "required_files": list(self.required_files),
        }


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    phase_id: str
    kind: str
    path: Path
    sha256: str
    authority: str
    selected: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id, "artifact_id")
        _require_phase(self.phase_id)
        if not self.kind.strip() or not self.authority.strip():
            raise P08WorkflowError("artifact kind and authority must not be blank")
        if not self.path.is_absolute():
            raise P08WorkflowError("artifact path must be absolute")
        _require_sha256(self.sha256, "artifact sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "phase_id": self.phase_id,
            "kind": self.kind,
            "path": str(self.path),
            "sha256": self.sha256,
            "authority": self.authority,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class PhaseRecord:
    phase_id: str
    state: PhaseState
    message: str
    prerequisites: tuple[str, ...] = ()
    workspace_reference_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_phase(self.phase_id)
        if not self.message.strip():
            raise P08WorkflowError("phase message must not be blank")
        if any(value not in PHASE_ORDER for value in self.prerequisites):
            raise P08WorkflowError("phase prerequisite is unknown")
        if self.phase_id in self.prerequisites:
            raise P08WorkflowError("phase cannot depend on itself")

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "state": self.state.value,
            "message": self.message,
            "prerequisites": list(self.prerequisites),
            "workspace_reference_ids": list(self.workspace_reference_ids),
            "artifact_ids": list(self.artifact_ids),
        }


@dataclass(frozen=True)
class SceneWorkspace:
    """Versioned credential-free project/scene descriptor with a variable camera roster."""

    project_id: str
    scene_id: str
    display_name: str
    artifact_root: Path
    cameras: tuple[CameraConfig, ...]
    phases: tuple[PhaseRecord, ...]
    workspace_references: tuple[WorkspaceReference, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    schema_version: str = "p08-scene-workspace-v1"

    def __post_init__(self) -> None:
        _require_identifier(self.project_id, "project_id")
        _require_identifier(self.scene_id, "scene_id")
        if not self.display_name.strip() or not self.artifact_root.is_absolute():
            raise P08WorkflowError("scene display name and absolute artifact root are required")
        if not self.cameras or len({camera.camera_id for camera in self.cameras}) != len(
            self.cameras
        ):
            raise P08WorkflowError("scene camera roster must be non-empty and unique")
        phase_ids = tuple(record.phase_id for record in self.phases)
        if phase_ids != PHASE_ORDER:
            raise P08WorkflowError("scene must declare P02-P08 phases in canonical order")
        workspace_ids = {reference.reference_id for reference in self.workspace_references}
        artifact_ids = {artifact.artifact_id for artifact in self.artifacts}
        if len(workspace_ids) != len(self.workspace_references):
            raise P08WorkflowError("workspace reference IDs must be unique")
        if len(artifact_ids) != len(self.artifacts):
            raise P08WorkflowError("artifact IDs must be unique")
        for record in self.phases:
            if not set(record.workspace_reference_ids) <= workspace_ids:
                raise P08WorkflowError(f"{record.phase_id} references an unknown workspace")
            if not set(record.artifact_ids) <= artifact_ids:
                raise P08WorkflowError(f"{record.phase_id} references an unknown artifact")
        if self.schema_version != "p08-scene-workspace-v1":
            raise P08WorkflowError("unsupported scene workspace schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "scene_id": self.scene_id,
            "display_name": self.display_name,
            "artifact_root": str(self.artifact_root),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "phases": [record.to_dict() for record in self.phases],
            "workspace_references": [
                reference.to_dict() for reference in self.workspace_references
            ],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneWorkspace:
        return cls(
            project_id=_string(value, "project_id"),
            scene_id=_string(value, "scene_id"),
            display_name=_string(value, "display_name"),
            artifact_root=Path(_string(value, "artifact_root")),
            cameras=tuple(
                CameraConfig(
                    camera_id=_string(item, "camera_id"),
                    display_name=_string(item, "display_name"),
                    endpoint_environment_key=_optional_string(
                        item.get("endpoint_environment_key")
                    ),
                    enabled=_boolean(item, "enabled"),
                )
                for item in _object_list(value, "cameras")
            ),
            phases=tuple(
                PhaseRecord(
                    phase_id=_string(item, "phase_id"),
                    state=PhaseState(_string(item, "state")),
                    message=_string(item, "message"),
                    prerequisites=_string_tuple(item, "prerequisites"),
                    workspace_reference_ids=_string_tuple(item, "workspace_reference_ids"),
                    artifact_ids=_string_tuple(item, "artifact_ids"),
                )
                for item in _object_list(value, "phases")
            ),
            workspace_references=tuple(
                WorkspaceReference(
                    reference_id=_string(item, "reference_id"),
                    phase_id=_string(item, "phase_id"),
                    path=Path(_string(item, "path")),
                    kind=_string(item, "kind"),
                    required_files=_string_tuple(item, "required_files"),
                )
                for item in _object_list(value, "workspace_references")
            ),
            artifacts=tuple(
                ArtifactReference(
                    artifact_id=_string(item, "artifact_id"),
                    phase_id=_string(item, "phase_id"),
                    kind=_string(item, "kind"),
                    path=Path(_string(item, "path")),
                    sha256=_string(item, "sha256"),
                    authority=_string(item, "authority"),
                    selected=_boolean(item, "selected"),
                )
                for item in _object_list(value, "artifacts")
            ),
            schema_version=_string(value, "schema_version"),
        )


class SceneWorkspaceRepository:
    """Non-overwriting scene descriptor and immutable action-run manifest repository."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.scene_path = self.root / "scene.json"
        self.runs_directory = self.root / "runs"
        self.operator_state_path = self.root / "operator-state.json"
        self.operator_state_archive = self.root / "operator-state-archive"
        self.artifact_catalog_path = self.root / "artifact-catalog.sqlite3"
        self.camera_policy_path = self.root / "camera-policy.sqlite3"
        self.calibration_workflow_path = self.root / "calibration-workflow.sqlite3"

    def create(self, scene: SceneWorkspace) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_directory.mkdir(exist_ok=True)
        _write_json_exclusive(self.scene_path, scene.to_dict())
        return self.scene_path

    def load(self) -> SceneWorkspace:
        return SceneWorkspace.from_dict(_read_json(self.scene_path))

    def write_run_manifest(self, run_id: str, payload: Mapping[str, Any]) -> Path:
        path = self.ensure_run_id_available(run_id)
        body = {
            "schema_version": "p08-workflow-action-run-v1",
            "project_id": self.load().project_id,
            "scene_id": self.load().scene_id,
            "run_id": run_id,
            **dict(payload),
        }
        _write_json_exclusive(path, body)
        return path

    def ensure_run_id_available(self, run_id: str) -> Path:
        _require_identifier(run_id, "run_id")
        self.runs_directory.mkdir(parents=True, exist_ok=True)
        path = self.runs_directory / f"{run_id}.json"
        if path.exists():
            raise P08WorkflowError("workflow run_id already exists")
        return path

    def read_operator_state(self) -> dict[str, Any] | None:
        if not self.operator_state_path.exists():
            return None
        return _read_json(self.operator_state_path)

    def write_operator_state(self, payload: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.operator_state_path, payload)
        return self.operator_state_path

    def archive_operator_state(self, archive_id: str, payload: Mapping[str, Any]) -> Path:
        _require_identifier(archive_id, "operator state archive_id")
        self.operator_state_archive.mkdir(parents=True, exist_ok=True)
        path = self.operator_state_archive / f"{archive_id}.json"
        _write_json_exclusive(path, payload)
        return path


@dataclass(frozen=True)
class PhaseStatus:
    phase_id: str
    state: PhaseState
    message: str
    reasons: tuple[str, ...]
    workspace_reference_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "state": self.state.value,
            "message": self.message,
            "reasons": list(self.reasons),
            "workspace_reference_ids": list(self.workspace_reference_ids),
            "artifact_ids": list(self.artifact_ids),
        }


class PhaseAdapter(Protocol):
    def inspect(
        self,
        scene: SceneWorkspace,
        record: PhaseRecord,
        workspace_by_id: Mapping[str, WorkspaceReference],
        artifact_by_id: Mapping[str, ArtifactReference],
    ) -> tuple[str, ...]: ...


class FilesystemPhaseAdapter:
    """Read-only compatibility adapter for existing workspaces and artifacts."""

    def inspect(
        self,
        scene: SceneWorkspace,
        record: PhaseRecord,
        workspace_by_id: Mapping[str, WorkspaceReference],
        artifact_by_id: Mapping[str, ArtifactReference],
    ) -> tuple[str, ...]:
        del scene
        reasons: list[str] = []
        for reference_id in record.workspace_reference_ids:
            reference = workspace_by_id[reference_id]
            if not reference.path.is_dir():
                reasons.append(f"workspace missing: {reference.reference_id}")
                continue
            for filename in reference.required_files:
                if not (reference.path / filename).is_file():
                    reasons.append(
                        f"workspace prerequisite missing: {reference.reference_id}/{filename}"
                    )
        for artifact_id in record.artifact_ids:
            artifact = artifact_by_id[artifact_id]
            if not artifact.path.is_file():
                reasons.append(f"artifact missing: {artifact.artifact_id}")
            elif _sha256(artifact.path) != artifact.sha256:
                reasons.append(f"artifact identity stale: {artifact.artifact_id}")
        return tuple(reasons)


@dataclass
class _MutableJob:
    job_id: str
    phase_id: str
    action: str
    state: JobState
    cancel_event: threading.Event
    submitted_at_utc: str
    completed_at_utc: str | None = None
    result: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    future: Future[None] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "phase_id": self.phase_id,
            "action": self.action,
            "state": self.state.value,
            "submitted_at_utc": self.submitted_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "result": self.result,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "cancellation_requested": self.cancel_event.is_set(),
        }


JobOperation = Callable[[threading.Event], Mapping[str, Any]]


class BoundedJobManager:
    """Instance-scoped bounded jobs with explicit status, cancellation, and redacted errors."""

    def __init__(self, maximum_workers: int = 1, maximum_outstanding_jobs: int = 4) -> None:
        if maximum_workers < 1 or maximum_outstanding_jobs < maximum_workers:
            raise P08WorkflowError("job capacity is invalid")
        self._maximum_outstanding_jobs = maximum_outstanding_jobs
        self._executor = ThreadPoolExecutor(
            max_workers=maximum_workers, thread_name_prefix="p08-workflow"
        )
        self._jobs: dict[str, _MutableJob] = {}
        self._lock = threading.RLock()

    def submit(
        self, job_id: str, phase_id: str, action: str, operation: JobOperation
    ) -> dict[str, Any]:
        _require_identifier(job_id, "job_id")
        _require_phase(phase_id)
        if not action.strip():
            raise P08WorkflowError("job action must not be blank")
        with self._lock:
            if job_id in self._jobs:
                raise P08WorkflowError("job_id already exists")
            outstanding = sum(
                job.state in {JobState.QUEUED, JobState.RUNNING} for job in self._jobs.values()
            )
            if outstanding >= self._maximum_outstanding_jobs:
                raise P08WorkflowError("bounded job capacity is full")
            job = _MutableJob(
                job_id=job_id,
                phase_id=phase_id,
                action=action,
                state=JobState.QUEUED,
                cancel_event=threading.Event(),
                submitted_at_utc=datetime.now(UTC).isoformat(),
            )
            self._jobs[job_id] = job
            job.future = self._executor.submit(self._execute, job, operation)
            return job.snapshot()

    def _execute(self, job: _MutableJob, operation: JobOperation) -> None:
        with self._lock:
            if job.cancel_event.is_set():
                job.state = JobState.CANCELLED
                job.completed_at_utc = datetime.now(UTC).isoformat()
                return
            job.state = JobState.RUNNING
        try:
            result = dict(operation(job.cancel_event))
            with self._lock:
                if job.cancel_event.is_set():
                    job.state = JobState.CANCELLED
                else:
                    job.result = result
                    job.state = JobState.COMPLETE
                job.completed_at_utc = datetime.now(UTC).isoformat()
        except Exception as error:
            with self._lock:
                job.error_type = type(error).__name__
                job.error_message = _redact_error(str(error))
                job.state = JobState.CANCELLED if job.cancel_event.is_set() else JobState.FAILED
                job.completed_at_utc = datetime.now(UTC).isoformat()

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._require_job(job_id)
            if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
                return job.snapshot()
            job.cancel_event.set()
            if job.future is not None and job.future.cancel():
                job.state = JobState.CANCELLED
                job.completed_at_utc = datetime.now(UTC).isoformat()
            return job.snapshot()

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._require_job(job_id).snapshot()

    def list(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._jobs[key].snapshot() for key in sorted(self._jobs))

    def clear_terminal(self) -> tuple[dict[str, Any], ...]:
        """Remove terminal in-memory jobs after their immutable records are archived."""
        with self._lock:
            if any(
                job.state in {JobState.QUEUED, JobState.RUNNING} for job in self._jobs.values()
            ):
                raise P08WorkflowError("wait for the active workflow action before starting over")
            snapshots = tuple(self._jobs[key].snapshot() for key in sorted(self._jobs))
            self._jobs.clear()
            return snapshots

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _require_job(self, job_id: str) -> _MutableJob:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise P08WorkflowError("unknown job_id") from error


LaunchCallable = Callable[[Sequence[str]], Any]


class SafeRerunLauncher:
    """Launch one selected hash-verified Rerun artifact through fixed local configuration."""

    def __init__(
        self,
        viewer_executable: Path,
        allowed_artifact_roots: Sequence[Path],
        launch: LaunchCallable | None = None,
    ) -> None:
        self.viewer_executable = _resolve_rerun_viewer(viewer_executable)
        self.allowed_artifact_roots = tuple(path.resolve() for path in allowed_artifact_roots)
        if not self.viewer_executable.is_file():
            raise P08WorkflowError("configured Rerun viewer executable is missing")
        if not self.allowed_artifact_roots:
            raise P08WorkflowError("at least one allowed Rerun artifact root is required")
        self._launch = launch or self._default_launch

    def launch_selected(self, scene: SceneWorkspace, artifact_id: str) -> dict[str, Any]:
        artifacts = {artifact.artifact_id: artifact for artifact in scene.artifacts}
        try:
            artifact = artifacts[artifact_id]
        except KeyError as error:
            raise P08WorkflowError("unknown Rerun artifact_id") from error
        if artifact.kind != "rerun-recording" or not artifact.selected:
            raise P08WorkflowError("Rerun launch requires a selected recording artifact")
        return self.launch_artifact(artifact)

    def launch_artifact(self, artifact: ArtifactReference) -> dict[str, Any]:
        """Launch one already-selected artifact, including a runtime-generated recording."""

        if artifact.kind != "rerun-recording" or not artifact.selected:
            raise P08WorkflowError("Rerun launch requires a selected recording artifact")
        path = artifact.path.resolve()
        if path.suffix.lower() != ".rrd" or not path.is_file():
            raise P08WorkflowError("selected Rerun artifact must be an existing .rrd file")
        if not any(path.is_relative_to(root) for root in self.allowed_artifact_roots):
            raise P08WorkflowError("selected Rerun artifact is outside configured allowed roots")
        if _sha256(path) != artifact.sha256:
            raise P08WorkflowError("selected Rerun artifact identity is stale")
        process = self._launch((str(self.viewer_executable), "--port", "0", str(path)))
        process_id = getattr(process, "pid", None)
        return {
            "artifact_id": artifact.artifact_id,
            "path": str(path),
            "sha256": artifact.sha256,
            "viewer": str(self.viewer_executable),
            "process_id": process_id if isinstance(process_id, int) else None,
            "status": "launched",
        }

    def with_allowed_root(self, root: Path) -> SafeRerunLauncher:
        """Return an equivalent launcher that also permits one managed-scene artifact root."""

        resolved = root.resolve()
        roots = (*self.allowed_artifact_roots, resolved)
        return SafeRerunLauncher(self.viewer_executable, roots, self._launch)

    @staticmethod
    def _default_launch(arguments: Sequence[str]) -> subprocess.Popen[bytes]:
        startup_info = None
        if hasattr(subprocess, "STARTUPINFO"):
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = 1
        return subprocess.Popen(
            list(arguments),
            shell=False,
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startup_info,
        )


@dataclass(frozen=True)
class FloorWorkflowAdapter:
    """Shared CLI/UI adapter over the exact floor artifact service."""

    contract: FrozenP07FloorInput | None
    config: FloorProcessingConfig
    output_root: Path

    def run(self, job_id: str, cancel_event: threading.Event) -> dict[str, Any]:
        if self.contract is None:
            raise P08WorkflowError(
                "historical floor processing requires its frozen source contract"
            )
        if cancel_event.is_set():
            raise P08WorkflowError("floor job cancelled before execution")
        output = self.output_root.resolve() / job_id
        result = create_floor_artifact_run(self.contract, self.config, output)
        verification = verify_floor_artifact_run(output, self.contract)
        verification_path = output / "verification.json"
        _write_json_exclusive(verification_path, verification)
        if cancel_event.is_set():
            raise P08WorkflowError("floor job cancelled after artifact generation")
        manifest = result.get("manifest")
        if not isinstance(manifest, dict):
            raise P08WorkflowError("floor artifact service returned no manifest identity")
        return {
            "output_directory": str(output),
            "manifest": manifest,
            "verification": _identity(verification_path),
            "verification_status": verification["status"],
            "summary": verification["summary"],
        }

    def run_for_source(
        self,
        job_id: str,
        source_path: Path,
        source_sha256: str,
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Run the unchanged floor policy against a new immutable scene-update geometry."""

        if cancel_event.is_set():
            raise P08WorkflowError("floor job cancelled before execution")
        output = self.output_root.resolve() / job_id
        result = create_floor_artifact_run_from_geometry(
            source_path.resolve(), source_sha256, self.config, output
        )
        verification = verify_floor_artifact_run_from_geometry(
            output, source_path.resolve(), source_sha256
        )
        verification_path = output / "verification.json"
        _write_json_exclusive(verification_path, verification)
        if cancel_event.is_set():
            raise P08WorkflowError("floor job cancelled after artifact generation")
        manifest = result.get("manifest")
        if not isinstance(manifest, dict):
            raise P08WorkflowError("floor artifact service returned no manifest identity")
        return {
            "output_directory": str(output),
            "manifest": manifest,
            "verification": _identity(verification_path),
            "verification_status": verification["status"],
            "summary": verification["summary"],
            "source_geometry": {"path": str(source_path.resolve()), "sha256": source_sha256},
        }


@dataclass(frozen=True)
class _CalibrationReadiness:
    cameras: tuple[dict[str, Any], ...]
    details: dict[str, Any] | None
    error: str | None

    @property
    def ready(self) -> bool:
        return bool(self.cameras) and all(bool(camera.get("ready")) for camera in self.cameras)


@dataclass(frozen=True)
class _ReconstructionReadiness:
    errors: tuple[str, ...]
    lens_policy_ready: bool
    overlap_policy_ready: bool
    result_current: bool


@dataclass(frozen=True)
class _OperatorReviewState:
    geometry_approved: bool
    floor_approved: bool
    geometry_artifact_id: str | None
    floor_artifact_id: str | None
    latest_approved_floor_artifact_id: str | None
    current_floor_job_id: str | None
    current_floor_output_directory: Path | None
    previewed_targets: frozenset[str]


@dataclass
class WorkflowService:
    repository: SceneWorkspaceRepository
    jobs: BoundedJobManager
    adapters: Mapping[str, PhaseAdapter] = field(default_factory=dict)
    floor_adapter: FloorWorkflowAdapter | None = None
    rerun_launcher: SafeRerunLauncher | None = None
    operator_config: Any | None = None
    camera_summary_adapter: Any | None = None
    reconstruction_adapter: Any | None = None
    calibration_adapter: Any | None = None
    floor_preview_adapter: Any | None = None
    scene_update_adapter: Any | None = None
    live_operations_adapter: Any | None = None
    scene_evidence_adapter: Any | None = None
    resource_coordinator: Any | None = None
    resource_scene_uuid: str | None = None
    scene_display_name_provider: Callable[[], str] | None = None
    operator_surface_ids: frozenset[str] = field(default_factory=frozenset)
    _workflow_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _runtime_artifacts: dict[str, ArtifactReference] = field(
        default_factory=dict, init=False, repr=False
    )
    _geometry_approved: bool = field(default=False, init=False, repr=False)
    _floor_approved: bool = field(default=False, init=False, repr=False)
    _previewed_targets: set[str] = field(default_factory=set, init=False, repr=False)
    _operator_session_id: str = field(default="initial", init=False, repr=False)
    _operator_state_revision: int = field(default=0, init=False, repr=False)
    _active_geometry_artifact_id: str | None = field(default=None, init=False, repr=False)
    _active_floor_artifact_id: str | None = field(default=None, init=False, repr=False)
    _latest_approved_floor_artifact_id: str | None = field(default=None, init=False, repr=False)
    _current_floor_job_id: str | None = field(default=None, init=False, repr=False)
    _current_floor_output_directory: Path | None = field(default=None, init=False, repr=False)
    _geometry_source_path: Path | None = field(default=None, init=False, repr=False)
    _geometry_source_sha256: str | None = field(default=None, init=False, repr=False)
    _geometry_input_lineage_sha256: str | None = field(default=None, init=False, repr=False)
    _operator_state_enabled: bool = field(default=False, init=False, repr=False)
    _artifact_catalog: SceneArtifactCatalog | None = field(default=None, init=False, repr=False)
    _camera_policy: CameraPolicyRepository | None = field(default=None, init=False, repr=False)
    _artifact_catalog_initialized: bool = field(default=False, init=False, repr=False)
    _catalog_input_controls_enabled: bool = field(default=False, init=False, repr=False)
    _scene_update_scheduler: SceneUpdateScheduler | None = field(
        default=None, init=False, repr=False
    )
    _live_resource_lease_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.operator_surface_ids = frozenset(self.operator_surface_ids)
        unsupported_surfaces = self.operator_surface_ids - OPERATOR_SURFACE_IDS
        if unsupported_surfaces:
            values = ", ".join(sorted(unsupported_surfaces))
            raise P08WorkflowError(f"unsupported operator surface IDs: {values}")
        self._catalog_input_controls_enabled = self.operator_config is not None
        self._operator_state_enabled = self.operator_config is not None or any(
            adapter is not None
            for adapter in (
                self.calibration_adapter,
                self.reconstruction_adapter,
                self.floor_adapter,
                self.floor_preview_adapter,
            )
        )
        if self._operator_state_enabled:
            state = self.repository.read_operator_state()
            if state is None:
                if self.operator_config is not None:
                    self._initialize_operator_state()
                self._persist_operator_state()
            else:
                self._restore_operator_state(state)
        scene = self.repository.load()
        enabled_camera_ids = tuple(camera.camera_id for camera in scene.cameras if camera.enabled)
        self._camera_policy = CameraPolicyRepository(
            self.repository.camera_policy_path,
            scene.project_id,
            scene.scene_id,
            enabled_camera_ids,
        )
        self._artifact_catalog = SceneArtifactCatalog(
            self.repository.artifact_catalog_path,
            scene.project_id,
            scene.scene_id,
            (
                scene.artifact_root,
                *(reference.path for reference in scene.workspace_references),
                *(
                    tuple(self.scene_evidence_adapter.allowed_artifact_roots)
                    if self.scene_evidence_adapter is not None
                    else ()
                ),
            ),
        )
        self._artifact_catalog_initialized = self._artifact_catalog.has_versions()
        self._sync_artifact_catalog()
        selected_input = self._artifact_catalog.selected("reconstruction-input")
        if selected_input is not None and selected_input["kind"] == "da3-input-manifest":
            self._configure_reconstruction_input(selected_input)
        self._migrate_overlap_independent_geometry_lineage()
        self._scene_update_scheduler = SceneUpdateScheduler(
            SceneUpdateRepository(self.repository.root / "scene-updates.json"),
            self._submit_scheduled_scene_update,
            self._scene_update_pipeline_busy,
            start_thread=False,
        )
        if self._floor_approved:
            self._unlock_scene_updates_from_current_result()

    def close(self) -> None:
        scheduler = self._scene_update_scheduler
        if scheduler is not None:
            scheduler.close()
        live_operations = self.live_operations_adapter
        if live_operations is not None and hasattr(live_operations, "close"):
            live_operations.close()
        self._release_live_resource()
        self.jobs.close()

    def enable_live_operations(self, adapter: Any) -> None:
        """Attach the isolated XR02 worker without starting cameras or a run."""

        if self.live_operations_adapter is not None:
            raise P08WorkflowError("Live operations are already configured")
        self.live_operations_adapter = adapter

    def enable_resource_coordination(
        self,
        coordinator: Any,
        scene_uuid: str,
        display_name_provider: Callable[[], str] | None = None,
    ) -> None:
        """Share one explicit heavy-operation lease across independent scene runtimes."""

        if self.resource_coordinator is not None:
            raise P08WorkflowError("scene resource coordination is already configured")
        self.resource_coordinator = coordinator
        self.resource_scene_uuid = scene_uuid
        self.scene_display_name_provider = display_name_provider

    def enable_scene_updates(self, adapter: Any) -> None:
        """Attach the live adapter only after RTSP capture has initialized successfully."""

        self.scene_update_adapter = adapter
        self._require_scene_update_scheduler().start()

    def _initialize_operator_state(self) -> None:
        config = self.operator_config
        if config is None:
            raise P08WorkflowError("operator workflow is not configured")
        self._geometry_approved = bool(config.geometry_approved)
        self._floor_approved = bool(config.floor_approved)
        self._active_geometry_artifact_id = str(config.geometry_rerun_artifact_id)
        self._active_floor_artifact_id = str(config.floor_rerun_artifact_id)
        self._latest_approved_floor_artifact_id = str(config.floor_rerun_artifact_id)
        if self.floor_adapter is not None and self.floor_adapter.contract is not None:
            self._geometry_source_path = self.floor_adapter.contract.selected_geometry_path
            self._geometry_source_sha256 = self.floor_adapter.contract.selected_geometry_sha256

    def _restore_operator_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "p08-operator-session-v1":
            raise P08WorkflowError("unsupported operator session state")
        scene = self.repository.load()
        if state.get("project_id") != scene.project_id or state.get("scene_id") != scene.scene_id:
            raise P08WorkflowError("operator session does not match this workflow scene")
        session_id = _required_mapping_string(state, "session_id")
        _require_identifier(session_id, "operator session_id")
        revision = state.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise P08WorkflowError("operator session revision is malformed")
        self._operator_session_id = session_id
        self._operator_state_revision = revision
        self._geometry_approved = _boolean(state, "geometry_approved")
        self._floor_approved = _boolean(state, "floor_approved")
        self._previewed_targets = set(_string_tuple(state, "previewed_targets"))
        if not self._previewed_targets.issubset({"geometry", "floor"}):
            raise P08WorkflowError("operator preview targets are malformed")
        self._active_geometry_artifact_id = _optional_string(
            state.get("active_geometry_artifact_id")
        )
        self._active_floor_artifact_id = _optional_string(state.get("active_floor_artifact_id"))
        self._latest_approved_floor_artifact_id = _optional_string(
            state.get("latest_approved_floor_artifact_id")
        )
        self._current_floor_job_id = _optional_string(state.get("current_floor_job_id"))
        floor_output = _optional_string(state.get("current_floor_output_directory"))
        self._current_floor_output_directory = Path(floor_output) if floor_output else None
        geometry_source = _optional_string(state.get("geometry_source_path"))
        self._geometry_source_path = Path(geometry_source) if geometry_source else None
        self._geometry_source_sha256 = _optional_string(state.get("geometry_source_sha256"))
        if self._geometry_source_sha256 is not None:
            _require_sha256(self._geometry_source_sha256, "geometry source sha256")
        self._geometry_input_lineage_sha256 = _optional_string(
            state.get("geometry_input_lineage_sha256")
        )
        if self._geometry_input_lineage_sha256 is not None:
            _require_sha256(
                self._geometry_input_lineage_sha256, "geometry input lineage sha256"
            )
        for value in _object_list(state, "runtime_artifacts"):
            artifact = ArtifactReference(
                artifact_id=_string(value, "artifact_id"),
                phase_id=_string(value, "phase_id"),
                kind=_string(value, "kind"),
                path=Path(_string(value, "path")),
                sha256=_string(value, "sha256"),
                authority=_string(value, "authority"),
                selected=_boolean(value, "selected"),
            )
            self._runtime_artifacts[artifact.artifact_id] = artifact

    def _operator_state_payload(self) -> dict[str, Any]:
        scene = self.repository.load()
        return {
            "schema_version": "p08-operator-session-v1",
            "project_id": scene.project_id,
            "scene_id": scene.scene_id,
            "session_id": self._operator_session_id,
            "revision": self._operator_state_revision,
            "geometry_approved": self._geometry_approved,
            "floor_approved": self._floor_approved,
            "previewed_targets": sorted(self._previewed_targets),
            "active_geometry_artifact_id": self._active_geometry_artifact_id,
            "active_floor_artifact_id": self._active_floor_artifact_id,
            "latest_approved_floor_artifact_id": self._latest_approved_floor_artifact_id,
            "current_floor_job_id": self._current_floor_job_id,
            "current_floor_output_directory": (
                str(self._current_floor_output_directory)
                if self._current_floor_output_directory is not None
                else None
            ),
            "geometry_source_path": (
                str(self._geometry_source_path) if self._geometry_source_path is not None else None
            ),
            "geometry_source_sha256": self._geometry_source_sha256,
            "geometry_input_lineage_sha256": self._geometry_input_lineage_sha256,
            "runtime_artifacts": [
                artifact.to_dict()
                for artifact in sorted(
                    self._runtime_artifacts.values(), key=lambda item: item.artifact_id
                )
            ],
        }

    def _persist_operator_state(self) -> None:
        self._operator_state_revision += 1
        self.repository.write_operator_state(self._operator_state_payload())
        if self._artifact_catalog is not None:
            self._sync_artifact_catalog()

    def status(self) -> dict[str, Any]:
        scene = self.repository.load()
        statuses = self._phase_statuses(scene)
        return {
            "schema_version": "p08-workflow-status-v1",
            "project_id": scene.project_id,
            "scene_id": scene.scene_id,
            "display_name": (
                self.scene_display_name_provider()
                if self.scene_display_name_provider is not None
                else scene.display_name
            ),
            "camera_roster": [camera.to_dict() for camera in scene.cameras],
            "phases": [status.to_dict() for status in statuses],
            "jobs": list(self.jobs.list()),
            "artifacts": [artifact.to_dict() for artifact in self._all_artifacts(scene)],
            "operator": self.operator_status(scene, statuses),
            "scene_updates": self.scene_update_status(),
            "camera_policy": self.camera_policy_status(),
            "live_operations": self.live_operations_status(),
            "scene_inputs": (
                self.scene_evidence_adapter.status()
                if self.scene_evidence_adapter is not None
                else None
            ),
            "shared_processing": self._resource_status(),
        }

    def _resource_status(self) -> dict[str, Any]:
        coordinator = self.resource_coordinator
        if coordinator is None:
            return {"configured": False, "active": None, "queue": []}
        return {"configured": True, **dict(coordinator.resource_status())}

    def scene_update_status(self) -> dict[str, Any]:
        scheduler = self._require_scene_update_scheduler()
        status = scheduler.status()
        status["available"] = self.scene_update_adapter is not None
        status["pipeline_busy"] = self._scene_update_pipeline_busy()
        status["rollback_results"] = [result.to_dict() for result in scheduler.rollback_choices()]
        return status

    def live_operations_status(self) -> dict[str, Any]:
        operator = self.operator_status()
        live_capability = operator["workflow_capabilities"]["live_operations"]
        blockers: list[str] = []
        if not operator["floor"]["approved"]:
            blockers.append("Approve the current final result")
        if not operator["xr02_overlap_ready"]:
            blockers.append("Save the Final Review camera-overlap declarations")
        if not operator["inputs_ready"]:
            blockers.append("Resolve the current camera/calibration input issues")
        if self._scene_update_pipeline_busy():
            blockers.append("Wait for the active spatial workflow action")
        resource = self._resource_status()
        active_resource = resource.get("active")
        if (
            isinstance(active_resource, Mapping)
            and active_resource.get("scene_uuid") != self.resource_scene_uuid
        ):
            blockers.append("Wait for heavy processing in the other open scene")
        if resource.get("queue"):
            blockers.append("Wait for the queued scene processing action")
        adapter = self.live_operations_adapter
        if adapter is None:
            missing_runtime_message = (
                "Live operations are not set up for this scene"
                if live_capability["state"] == WorkflowCapabilityState.NOT_PROVISIONED.value
                else "Restart with the configured XR02 worker"
            )
            return {
                "schema": "xr03.integrated_live_operations.v1",
                "available": False,
                "eligible": False,
                "blockers": [*blockers, missing_runtime_message],
                "worker": None,
            }
        try:
            worker = adapter.status()
        except Exception as error:
            return {
                "schema": "xr03.integrated_live_operations.v1",
                "available": False,
                "eligible": False,
                "blockers": [*blockers, f"XR02 worker unavailable: {type(error).__name__}"],
                "worker": None,
            }
        return {
            "schema": "xr03.integrated_live_operations.v1",
            "available": True,
            "eligible": not blockers and worker.get("operator_state") == "ready",
            "blockers": blockers,
            "worker": worker,
        }

    def start_live_operations(self, mode: str) -> dict[str, Any]:
        if mode not in {"live", "recording"}:
            raise P08WorkflowError("Live operations mode must be live or recording")
        adapter = self._require_live_operations()
        status = self.live_operations_status()
        blockers = status["blockers"]
        if blockers:
            raise P08WorkflowError("Live operations are blocked: " + "; ".join(blockers))
        lease_id = self._acquire_live_resource(mode)
        with self._workflow_lock:
            try:
                self._require_pipeline_idle()
                result = adapter.start_live() if mode == "live" else adapter.start_recording()
            except Exception:
                self._release_live_resource(lease_id)
                raise
        return dict(result)

    def stop_live_operations(self) -> dict[str, Any]:
        adapter = self._require_live_operations()
        before = adapter.status()
        result = dict(adapter.stop(reason="operator"))
        self._release_live_resource()
        if before.get("active_mode") == "recording":
            self._start_deferred_update_after_recording()
        return result

    def _acquire_live_resource(self, operation: str) -> str | None:
        coordinator = self.resource_coordinator
        scene_uuid = self.resource_scene_uuid
        if coordinator is None or scene_uuid is None:
            return None
        lease_id = str(coordinator.acquire_resource_now(scene_uuid, operation))
        self._live_resource_lease_id = lease_id
        return lease_id

    def _release_live_resource(self, lease_id: str | None = None) -> None:
        coordinator = self.resource_coordinator
        current = lease_id or self._live_resource_lease_id
        if coordinator is None or current is None:
            return
        coordinator.release_resource(current)
        if current == self._live_resource_lease_id:
            self._live_resource_lease_id = None

    def _coordinated_operation(
        self,
        request_id: str,
        action: str,
        cancel_event: threading.Event,
        operation: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        coordinator = self.resource_coordinator
        scene_uuid = self.resource_scene_uuid
        if coordinator is None or scene_uuid is None:
            return operation()
        return cast(
            Mapping[str, Any],
            coordinator.run_with_resource(
                scene_uuid,
                request_id,
                action,
                cancel_event,
                operation,
            ),
        )

    def live_operations_action(
        self,
        action: str,
        *,
        session_id: str | None = None,
        label: str | None = None,
        confirmation: str | None = None,
    ) -> dict[str, Any]:
        adapter = self._require_live_operations()
        if action == "open-rerun":
            return dict(adapter.open_rerun())
        if action == "reset-trails":
            return dict(adapter.reset_trails())
        if action == "export":
            return dict(adapter.export_evidence_snapshot())
        if not session_id:
            raise P08WorkflowError("session_id is required")
        if action == "view-recording":
            return dict(adapter.view_recording(session_id))
        if action == "save-recording":
            if not label:
                raise P08WorkflowError("recording label is required")
            return dict(adapter.save_recording(session_id, label))
        if action == "delete-recording":
            if not confirmation:
                raise P08WorkflowError("recording deletion confirmation is required")
            return dict(adapter.delete_recording(session_id, confirmation))
        raise P08WorkflowError("unknown Live operations action")

    def cancel_live_auto_resume(self) -> dict[str, Any]:
        try:
            scheduler = self._require_scene_update_scheduler()
            scheduler.cancel_live_resume()
            return self.live_operations_status()
        except SceneUpdateError as error:
            raise P08WorkflowError(str(error)) from error

    def acknowledge_live_operations_warning(self, warning_id: str) -> dict[str, Any]:
        self._require_scene_update_scheduler().acknowledge_warning(warning_id)
        return self.scene_update_status()

    def _require_live_operations(self) -> Any:
        adapter = self.live_operations_adapter
        if adapter is None:
            raise P08WorkflowError("Live operations are not configured in this launch")
        return adapter

    def _require_live_operations_inactive(self) -> None:
        adapter = self.live_operations_adapter
        if adapter is None:
            return
        try:
            status = adapter.status()
        except Exception as error:
            raise P08WorkflowError("XR02 worker state is unavailable") from error
        if status.get("active") is True:
            raise P08WorkflowError("Stop Live operations before changing the active spatial scene")

    def configure_scene_updates(self, value: Mapping[str, Any]) -> dict[str, Any]:
        try:
            schedule = SceneUpdateSchedule.from_dict(value)
            return self._require_scene_update_scheduler().configure(schedule)
        except SceneUpdateError as error:
            raise P08WorkflowError(str(error)) from error

    def start_scene_update(
        self, update_id: str, mode: UpdateMode = UpdateMode.MANUAL
    ) -> dict[str, Any]:
        adapter = self.scene_update_adapter
        if adapter is None:
            raise P08WorkflowError("fresh RTSP scene updates are not configured in this launch")
        scheduler = self._require_scene_update_scheduler()
        state = scheduler.status()
        if state["unlocked_at_utc"] is None:
            raise P08WorkflowError("approve the first final result before using scene updates")
        operator = self.operator_status()
        if not operator["inputs_ready"]:
            raise P08WorkflowError("current facility and camera inputs require attention")
        configured = SceneUpdateSchedule.from_dict(state["schedule"])
        if mode is UpdateMode.MANUAL:
            schedule = SceneUpdateSchedule(
                enabled=False,
                mode=UpdateMode.MANUAL,
                timezone=configured.timezone,
                median_frame_count=configured.median_frame_count,
                median_spacing_seconds=configured.median_spacing_seconds,
            )
        else:
            if not configured.enabled or configured.mode is not mode:
                raise P08WorkflowError("scheduled scene update mode is no longer enabled")
            schedule = configured
        if mode is UpdateMode.MANUAL:
            self._require_live_operations_inactive()
        with self._workflow_lock:
            self._require_pipeline_idle()
            self.repository.ensure_run_id_available(update_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            try:
                result = dict(adapter.run(update_id, schedule, cancel_event))
                if result.get("complete_chain") is not True:
                    raise P08WorkflowError("scene update did not complete every required step")
                adopted = self._scene_result_from_pipeline(update_id, mode, result)
                if mode is UpdateMode.MANUAL:
                    scheduler.record_candidate(adopted)
                else:
                    self._adopt_scene_update_result(adopted)
                    scheduler.adopt_scheduled(adopted)
                run_path = self.repository.write_run_manifest(
                    update_id,
                    {
                        "phase_id": "P08",
                        "action": "full-scene-update",
                        "trigger_mode": mode.value,
                        "auto_adopted": mode is not UpdateMode.MANUAL,
                        "result": result,
                        "immutable": True,
                    },
                )
                return {
                    **result,
                    "adoption_state": "candidate" if mode is UpdateMode.MANUAL else "adopted",
                    "workflow_run_manifest": _identity(run_path),
                }
            except Exception as error:
                scheduler.record_failure(update_id, str(error))
                raise

        with self._workflow_lock:
            return self.jobs.submit(
                update_id,
                "P08",
                "full-scene-update",
                lambda cancel_event: self._coordinated_operation(
                    update_id,
                    "full-scene-update",
                    cancel_event,
                    lambda: operation(cancel_event),
                ),
            )

    def launch_scene_update_candidate(self, action_id: str, result_id: str) -> dict[str, Any]:
        scheduler = self._require_scene_update_scheduler()
        state = scheduler.status()
        candidate = state.get("manual_candidate")
        if not isinstance(candidate, dict) or candidate.get("result_id") != result_id:
            raise P08WorkflowError("manual scene-update candidate is unavailable")
        artifact = self._artifact_from_scene_result(
            candidate["final_artifact"], result_id, "final"
        )
        if self.rerun_launcher is None:
            raise P08WorkflowError("Rerun launcher is not configured")
        self.repository.ensure_run_id_available(action_id)
        launched = self.rerun_launcher.launch_artifact(artifact)
        scheduler.mark_candidate_previewed(result_id)
        self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "open-scene-update-candidate",
                "result": launched,
                "immutable": True,
            },
        )
        return launched

    def adopt_manual_scene_update(self, action_id: str, result_id: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        self.repository.ensure_run_id_available(action_id)
        try:
            result = self._require_scene_update_scheduler().adopt_candidate(result_id)
        except SceneUpdateError as error:
            raise P08WorkflowError(str(error)) from error
        self._adopt_scene_update_result(result)
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "adopt-manual-scene-update",
                "result": {"result_id": result_id, "status": "approved"},
                "immutable": True,
            },
        )
        return {"result_id": result_id, "status": "approved", "manifest": _identity(run_path)}

    def rollback_scene_update(self, action_id: str, result_id: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        self.repository.ensure_run_id_available(action_id)
        try:
            result = self._require_scene_update_scheduler().rollback(result_id)
        except SceneUpdateError as error:
            raise P08WorkflowError(str(error)) from error
        self._adopt_scene_update_result(result)
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "rollback-scene-update",
                "result": {"result_id": result_id, "status": "restored"},
                "immutable": True,
            },
        )
        return {"result_id": result_id, "status": "restored", "manifest": _identity(run_path)}

    def _scene_result_from_pipeline(
        self, update_id: str, mode: UpdateMode, result: Mapping[str, Any]
    ) -> AdoptedSceneResult:
        geometry_artifact = dict(_required_mapping(result, "geometry_artifact"))
        final_artifact = dict(_required_mapping(result, "final_artifact"))
        geometry_artifact["artifact_id"] = f"{update_id}-geometry"
        final_artifact["artifact_id"] = f"{update_id}-final"
        return AdoptedSceneResult(
            result_id=update_id,
            adopted_at_utc=datetime.now(UTC).isoformat(),
            trigger_mode=mode,
            geometry_artifact=geometry_artifact,
            final_artifact=final_artifact,
            geometry_source=dict(_required_mapping(result, "geometry_source")),
            floor_job_id=_required_mapping_string(result, "floor_job_id"),
            floor_output_directory=_required_mapping_string(result, "floor_output_directory"),
        )

    def _adopt_scene_update_result(self, result: AdoptedSceneResult) -> None:
        geometry = self._artifact_from_scene_result(
            result.geometry_artifact, result.result_id, "geometry"
        )
        final = self._artifact_from_scene_result(result.final_artifact, result.result_id, "final")
        source_path = Path(_required_mapping_string(result.geometry_source, "path"))
        source_sha256 = _required_mapping_string(result.geometry_source, "sha256")
        _require_sha256(source_sha256, "scene update geometry source")
        with self._workflow_lock:
            self._runtime_artifacts[geometry.artifact_id] = geometry
            self._runtime_artifacts[final.artifact_id] = final
            self._active_geometry_artifact_id = geometry.artifact_id
            self._active_floor_artifact_id = final.artifact_id
            self._latest_approved_floor_artifact_id = final.artifact_id
            self._geometry_source_path = source_path
            self._geometry_source_sha256 = source_sha256
            self._current_floor_job_id = result.floor_job_id
            self._current_floor_output_directory = Path(result.floor_output_directory)
            self._geometry_approved = True
            self._floor_approved = True
            self._previewed_targets.clear()
            self._persist_operator_state()
            catalog = self._require_artifact_catalog()
            catalog.record_event(
                "approved",
                artifact_id=geometry.artifact_id,
                milestone_key="geometry-review",
                detail={"scene_update_result_id": result.result_id},
            )
            catalog.record_event(
                "approved",
                artifact_id=final.artifact_id,
                milestone_key="final-review",
                detail={"scene_update_result_id": result.result_id},
            )

    def _artifact_from_scene_result(
        self, value: Mapping[str, Any], result_id: str, suffix: str
    ) -> ArtifactReference:
        artifact_id = str(value.get("artifact_id") or f"{result_id}-{suffix}")
        return ArtifactReference(
            artifact_id=artifact_id,
            phase_id="P07" if suffix == "geometry" else "P08",
            kind="rerun-recording",
            path=Path(_required_mapping_string(value, "path")),
            sha256=_required_mapping_string(value, "sha256"),
            authority="adopted XR01 static scene update",
            selected=True,
        )

    def _unlock_scene_updates_from_current_result(self) -> None:
        scheduler = self._require_scene_update_scheduler()
        if scheduler.status()["unlocked_at_utc"] is not None:
            return
        scene = self.repository.load()
        artifacts = {artifact.artifact_id: artifact for artifact in self._all_artifacts(scene)}
        geometry = artifacts.get(str(self._active_geometry_artifact_id))
        final = artifacts.get(str(self._active_floor_artifact_id))
        if (
            geometry is None
            or final is None
            or self._geometry_source_path is None
            or self._geometry_source_sha256 is None
        ):
            return
        floor_directory = self._current_floor_output_directory or final.path.parent
        result = AdoptedSceneResult(
            result_id="initial-manual-approved",
            adopted_at_utc=datetime.now(UTC).isoformat(),
            trigger_mode=UpdateMode.MANUAL,
            geometry_artifact={**geometry.to_dict(), "byte_count": geometry.path.stat().st_size},
            final_artifact={**final.to_dict(), "byte_count": final.path.stat().st_size},
            geometry_source={
                "path": str(self._geometry_source_path),
                "sha256": self._geometry_source_sha256,
            },
            floor_job_id=self._current_floor_job_id or "initial-floor-result",
            floor_output_directory=str(floor_directory),
            initial_manual_result=True,
        )
        scheduler.unlock(result)

    def _require_scene_update_scheduler(self) -> SceneUpdateScheduler:
        scheduler = self._scene_update_scheduler
        if scheduler is None:
            raise P08WorkflowError("scene update scheduler is unavailable")
        return scheduler

    def _submit_scheduled_scene_update(self, update_id: str, mode: UpdateMode) -> None:
        scheduler = self._require_scene_update_scheduler()
        adapter = self.live_operations_adapter
        previous_session_id: str | None = None
        if adapter is not None:
            status = adapter.status()
            if status.get("active_mode") == "recording":
                scheduler.defer_for_recording(update_id, mode)
                return
            if status.get("active_mode") == "live":
                previous = status.get("active_session_id")
                previous_session_id = str(previous) if previous else None
                scheduler.set_live_coordination(
                    "pausing_live",
                    update_id=update_id,
                    previous_session_id=previous_session_id,
                    resume_requested=True,
                    message="Live Service is pausing for the scheduled scene update",
                )
                try:
                    adapter.stop(reason="scheduled_scene_update")
                    self._release_live_resource()
                except Exception as error:
                    scheduler.set_live_coordination(
                        "pause_failed",
                        update_id=update_id,
                        previous_session_id=previous_session_id,
                        message=f"Live Service could not pause: {type(error).__name__}",
                    )
                    scheduler.warn_live_coordination_failure(
                        update_id,
                        "The scheduled scene update could not start because Live Service did "
                        "not stop cleanly.",
                    )
                    raise
                scheduler.warn_live_paused_for_update(update_id)
                scheduler.set_live_coordination(
                    "scene_update_running",
                    update_id=update_id,
                    previous_session_id=previous_session_id,
                    resume_requested=True,
                    message="Scheduled scene update is running; Live will resume automatically",
                )
        try:
            job = self.start_scene_update(update_id, mode)
        except Exception:
            if previous_session_id is not None:
                self._resume_live_after_submission_failure(update_id, previous_session_id)
            raise
        if job.get("state") not in {JobState.QUEUED.value, JobState.RUNNING.value}:
            if previous_session_id is not None:
                self._resume_live_after_submission_failure(update_id, previous_session_id)
            raise P08WorkflowError("scheduled scene update did not enter the bounded job queue")
        if previous_session_id is not None:
            thread = threading.Thread(
                target=self._resume_live_after_scheduled_update,
                args=(update_id, previous_session_id),
                name=f"xr03-live-resume-{update_id}",
                daemon=True,
            )
            thread.start()

    def _resume_live_after_submission_failure(
        self, update_id: str, previous_session_id: str
    ) -> None:
        scheduler = self._require_scene_update_scheduler()
        if not scheduler.live_resume_requested(update_id):
            return
        lease_id: str | None = None
        try:
            adapter = self._require_live_operations()
            lease_id = self._acquire_live_resource("live")
            adapter.start_live(
                resumed_from_session_id=previous_session_id,
                scene_update_id=update_id,
            )
            scheduler.set_live_coordination(
                "live_resumed",
                update_id=update_id,
                previous_session_id=previous_session_id,
                message="Live Service resumed because the scheduled scene update did not start",
            )
            scheduler.warn_update_failed_live_resumed(update_id)
        except Exception as error:
            self._release_live_resource(lease_id)
            scheduler.set_live_coordination(
                "resume_failed",
                update_id=update_id,
                previous_session_id=previous_session_id,
                message=f"Live automatic restart failed: {type(error).__name__}",
            )
            scheduler.warn_live_coordination_failure(
                update_id,
                "The scheduled scene update did not start and Live Service could not be "
                "restarted automatically.",
            )

    def _resume_live_after_scheduled_update(
        self, update_id: str, previous_session_id: str
    ) -> None:
        scheduler = self._require_scene_update_scheduler()
        while True:
            job = self.jobs.status(update_id)
            state = str(job["state"])
            if state in {
                JobState.COMPLETE.value,
                JobState.FAILED.value,
                JobState.CANCELLED.value,
            }:
                break
            time.sleep(0.25)
        if not scheduler.live_resume_requested(update_id):
            scheduler.set_live_coordination(
                "resume_cancelled",
                update_id=update_id,
                previous_session_id=previous_session_id,
                message="Automatic Live restart was cancelled by the operator",
            )
            return
        lease_id: str | None = None
        try:
            adapter = self._require_live_operations()
            lease_id = self._acquire_live_resource("live")
            adapter.start_live(
                resumed_from_session_id=previous_session_id,
                scene_update_id=update_id,
            )
            scheduler.set_live_coordination(
                "live_resumed",
                update_id=update_id,
                previous_session_id=previous_session_id,
                message=(
                    "Live Service resumed on the updated scene"
                    if state == JobState.COMPLETE.value
                    else "Live Service resumed on the previous accepted scene"
                ),
            )
            if state != JobState.COMPLETE.value:
                scheduler.warn_update_failed_live_resumed(update_id)
        except Exception as error:
            self._release_live_resource(lease_id)
            scheduler.set_live_coordination(
                "resume_failed",
                update_id=update_id,
                previous_session_id=previous_session_id,
                message=f"Live automatic restart failed: {type(error).__name__}",
            )

    def _start_deferred_update_after_recording(self) -> None:
        scheduler = self._require_scene_update_scheduler()
        deferred = scheduler.deferred_update()
        if deferred is None:
            return
        update_id, mode = deferred
        try:
            self._submit_scheduled_scene_update(update_id, mode)
        except Exception:
            scheduler.warn_deferred_update_start_failed(update_id)
            return
        scheduler.clear_deferred_update(update_id)

    def _scene_update_pipeline_busy(self) -> bool:
        return any(
            job["state"] in {JobState.QUEUED.value, JobState.RUNNING.value}
            and job["action"]
            in {
                "all-camera-static-reconstruction",
                "floor-completion",
                "build-and-open-floor-preview",
                "full-scene-update",
            }
            for job in self.jobs.list()
        )

    def artifact_catalog_status(self) -> dict[str, Any]:
        self._sync_artifact_catalog()
        catalog = self._require_artifact_catalog()
        calibration_history = (
            self.calibration_adapter.history() if self.calibration_adapter is not None else None
        )
        return {
            **catalog.status(),
            "camera_policy": self.camera_policy_status(),
            "calibration_history": calibration_history,
        }

    def camera_policy_status(self) -> dict[str, Any]:
        return self._require_camera_policy().status()

    def calibration_status(self) -> dict[str, Any]:
        adapter = self.calibration_adapter
        if adapter is None:
            raise P08WorkflowError("integrated camera calibration is not configured")
        try:
            policy = self._require_camera_policy().active(require_lens=True)
            result = adapter.status(policy)
            if not isinstance(result, dict):
                raise P08WorkflowError("calibration adapter returned a malformed status")
            return result
        except (CameraPolicyError, ValueError) as error:
            raise P08WorkflowError(str(error)) from error

    def determine_calibration_intrinsics(self, action_id: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        adapter = self.calibration_adapter
        if adapter is None:
            raise P08WorkflowError("integrated camera calibration is not configured")
        self.repository.ensure_run_id_available(action_id)
        try:
            with self._workflow_lock:
                self._require_pipeline_idle()
                result = adapter.determine_intrinsics(
                    self._require_camera_policy().active(require_lens=True)
                )
        except (CameraPolicyError, ValueError) as error:
            raise P08WorkflowError(str(error)) from error
        return self._record_calibration_action(action_id, "determine-intrinsics", result)

    def calibrate_camera(self, action_id: str, camera_id: str) -> dict[str, Any]:
        return self._run_calibration_action(
            action_id,
            "calibrate-camera",
            lambda adapter, policy: adapter.calibrate_camera(camera_id, policy),
        )

    def review_camera_calibration(
        self, action_id: str, camera_id: str, attempt_sha256: str
    ) -> dict[str, Any]:
        return self._run_calibration_action(
            action_id,
            "review-camera-calibration",
            lambda adapter, policy: adapter.review_camera(camera_id, attempt_sha256, policy),
        )

    def override_camera_calibration(
        self,
        action_id: str,
        camera_id: str,
        attempt_sha256: str,
        reason: str,
        acknowledged: bool,
    ) -> dict[str, Any]:
        return self._run_calibration_action(
            action_id,
            "override-camera-calibration",
            lambda adapter, policy: adapter.override_camera(
                camera_id, attempt_sha256, reason, acknowledged, policy
            ),
        )

    def _run_calibration_action(
        self, action_id: str, action: str, operation: Any
    ) -> dict[str, Any]:
        self._require_live_operations_inactive()
        adapter = self.calibration_adapter
        if adapter is None:
            raise P08WorkflowError("integrated camera calibration is not configured")
        self.repository.ensure_run_id_available(action_id)
        try:
            with self._workflow_lock:
                self._require_pipeline_idle()
                policy = self._require_camera_policy().active(require_lens=True)
                result = operation(adapter, policy)
        except (CameraPolicyError, ValueError) as error:
            raise P08WorkflowError(str(error)) from error
        return self._record_calibration_action(action_id, action, result)

    def _record_calibration_action(
        self, action_id: str, action: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "XR03",
                "action": action,
                "result": dict(result),
                "immutable": True,
            },
        )
        return {**dict(result), "workflow_run_manifest": _identity(run_path)}

    def camera_policy_impact(self, value: Mapping[str, Any]) -> dict[str, Any]:
        try:
            policy = policy_from_changes(self._require_camera_policy(), value)
            return self._require_camera_policy().impact(policy)
        except CameraPolicyError as error:
            raise P08WorkflowError(str(error)) from error

    def apply_camera_policy(
        self,
        action_id: str,
        value: Mapping[str, Any],
        *,
        expected_revision: int | None,
        confirm_impacts: bool,
    ) -> dict[str, Any]:
        self._require_live_operations_inactive()
        self.repository.ensure_run_id_available(action_id)
        try:
            policy = policy_from_changes(self._require_camera_policy(), value)
            with self._workflow_lock:
                self._require_pipeline_idle()
                result = self._require_camera_policy().apply(
                    action_id,
                    policy,
                    expected_revision=expected_revision,
                    confirm_impacts=confirm_impacts,
                )
        except CameraPolicyError as error:
            raise P08WorkflowError(str(error)) from error
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "XR03",
                "action": "apply-camera-policy",
                "result": result,
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def rollback_camera_policy(
        self,
        action_id: str,
        target_revision: int,
        *,
        expected_revision: int,
        confirm_impacts: bool,
    ) -> dict[str, Any]:
        self._require_live_operations_inactive()
        self.repository.ensure_run_id_available(action_id)
        try:
            with self._workflow_lock:
                self._require_pipeline_idle()
                result = self._require_camera_policy().rollback(
                    action_id,
                    target_revision,
                    expected_revision=expected_revision,
                    confirm_impacts=confirm_impacts,
                )
        except CameraPolicyError as error:
            raise P08WorkflowError(str(error)) from error
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "XR03",
                "action": "rollback-camera-policy",
                "result": result,
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def artifact_selection_impact(self, artifact_id: str) -> dict[str, Any]:
        try:
            return self._require_artifact_catalog().selection_impact(artifact_id)
        except ArtifactCatalogError as error:
            raise P08WorkflowError(str(error)) from error

    def select_artifact_version(
        self,
        action_id: str,
        artifact_id: str,
        *,
        confirm_impacts: bool,
    ) -> dict[str, Any]:
        self._require_live_operations_inactive()
        self.repository.ensure_run_id_available(action_id)
        with self._workflow_lock:
            self._require_pipeline_idle()
            try:
                result = self._require_artifact_catalog().select(
                    artifact_id, confirm_impacts=confirm_impacts
                )
            except ArtifactCatalogError as error:
                raise P08WorkflowError(str(error)) from error
            selected = self._require_artifact_catalog().selected(result["milestone_key"])
            if selected is None:
                raise P08WorkflowError("selected artifact version could not be restored")
            self._apply_catalog_selection(selected)
            self._persist_operator_state()
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "select-artifact-version",
                "result": result,
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def verify_artifact_version(self, action_id: str, artifact_id: str) -> dict[str, Any]:
        self.repository.ensure_run_id_available(action_id)
        try:
            result = self._require_artifact_catalog().verify(artifact_id)
        except ArtifactCatalogError as error:
            raise P08WorkflowError(str(error)) from error
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "verify-artifact-version",
                "result": result,
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def archive_artifact_version(
        self, action_id: str, artifact_id: str, *, archived: bool
    ) -> dict[str, Any]:
        self.repository.ensure_run_id_available(action_id)
        try:
            result = self._require_artifact_catalog().set_archived(artifact_id, archived)
        except ArtifactCatalogError as error:
            raise P08WorkflowError(str(error)) from error
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "archive-artifact-version" if archived else "restore-artifact-version",
                "result": {**result, "physical_file_preserved": True},
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def artifact_deletion_impact(self, artifact_id: str) -> dict[str, Any]:
        try:
            return self._require_artifact_catalog().deletion_impact(artifact_id)
        except ArtifactCatalogError as error:
            raise P08WorkflowError(str(error)) from error

    def artifact_batch_deletion_impact(self, artifact_ids: Sequence[str]) -> dict[str, Any]:
        try:
            return self._require_artifact_catalog().batch_deletion_impact(artifact_ids)
        except ArtifactCatalogError as error:
            raise P08WorkflowError(str(error)) from error

    def delete_artifact_version(
        self,
        action_id: str,
        artifact_id: str,
        *,
        deletion_token: str,
    ) -> dict[str, Any]:
        self.repository.ensure_run_id_available(action_id)
        with self._workflow_lock:
            self._require_pipeline_idle()
            try:
                result = self._require_artifact_catalog().delete_permanently(
                    artifact_id, deletion_token
                )
            except ArtifactCatalogError as error:
                raise P08WorkflowError(str(error)) from error
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "delete-artifact-version-permanently",
                "result": result,
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def delete_artifact_versions(
        self,
        action_id: str,
        deletion_tokens: Mapping[str, str],
    ) -> dict[str, Any]:
        self.repository.ensure_run_id_available(action_id)
        with self._workflow_lock:
            self._require_pipeline_idle()
            try:
                result = self._require_artifact_catalog().delete_batch_permanently(deletion_tokens)
            except ArtifactCatalogError as error:
                raise P08WorkflowError(str(error)) from error
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "delete-artifact-versions-permanently",
                "result": result,
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def operator_status(
        self,
        scene: SceneWorkspace | None = None,
        statuses: tuple[PhaseStatus, ...] | None = None,
    ) -> dict[str, Any]:
        scene = scene or self.repository.load()
        statuses = statuses or self._phase_statuses(scene)
        phase_by_id = {status.phase_id: status for status in statuses}
        camera_ids = tuple(camera.camera_id for camera in scene.cameras if camera.enabled)
        calibration = self._operator_calibration_readiness(camera_ids)
        reconstruction = self._operator_reconstruction_readiness(camera_ids)
        capabilities = self._operator_workflow_capabilities(statuses)
        input_issues = tuple(
            dict.fromkeys(
                reconstruction.errors + ((calibration.error,) if calibration.error else ())
            )
        )
        artifacts = {artifact.artifact_id: artifact for artifact in self._all_artifacts(scene)}
        jobs = self.jobs.list()
        active_job = _active_operator_job(jobs)
        pipeline_busy = active_job is not None
        review = self._operator_review_state()
        current_result_ready = calibration.ready and reconstruction.result_current
        geometry_available = review.geometry_artifact_id in artifacts
        floor_available = review.floor_artifact_id in artifacts
        latest_approved_floor_available = review.latest_approved_floor_artifact_id in artifacts
        floor_open_artifact_id = (
            review.floor_artifact_id
            if floor_available
            else review.latest_approved_floor_artifact_id
        )
        steps = _operator_steps(
            phase_by_id,
            calibration,
            reconstruction,
            capabilities,
            review,
            geometry_available,
            floor_available,
            active_job,
        )
        return {
            "steps": list(steps),
            "cameras": list(calibration.cameras),
            "camera_summary_error": calibration.error,
            "calibration": calibration.details,
            "calibration_warnings": (
                []
                if calibration.details is None
                else list(calibration.details.get("warnings", ()))
            ),
            "workflow_capabilities": {
                capability_id: capability.to_dict()
                for capability_id, capability in capabilities.items()
            },
            "inputs_ready": calibration.ready and not reconstruction.errors,
            "lens_policy_ready": reconstruction.lens_policy_ready,
            "xr02_overlap_ready": reconstruction.overlap_policy_ready,
            "input_issues": list(input_issues),
            "session": {
                "can_start_fresh": not pipeline_busy,
                "has_activity": bool(jobs),
            },
            "active_action": (
                {
                    "action": active_job["action"],
                    "state": active_job["state"],
                }
                if active_job is not None
                else None
            ),
            "geometry": {
                "available": geometry_available,
                "approved": review.geometry_approved and current_result_ready,
                "previewed": "geometry" in review.previewed_targets,
                "artifact_id": review.geometry_artifact_id,
                "can_run": calibration.ready and not reconstruction.errors and not pipeline_busy,
                "can_preview": geometry_available and not pipeline_busy,
            },
            "floor": {
                "available": floor_available,
                "approved": review.floor_approved and current_result_ready,
                "previewed": "floor" in review.previewed_targets,
                "artifact_id": review.floor_artifact_id,
                "open_artifact_id": floor_open_artifact_id,
                "opening_latest_approved": not floor_available and latest_approved_floor_available,
                "current_floor_job_id": review.current_floor_job_id,
                "can_generate": review.geometry_approved
                and current_result_ready
                and self.floor_adapter is not None
                and not pipeline_busy,
                "can_build_preview": review.current_floor_job_id is not None
                and review.current_floor_output_directory is not None
                and current_result_ready
                and not pipeline_busy,
                "can_preview": floor_open_artifact_id in artifacts and not pipeline_busy,
            },
        }

    def _operator_workflow_capabilities(
        self, statuses: tuple[PhaseStatus, ...]
    ) -> dict[str, WorkflowCapability]:
        """Describe configured operator surfaces without inferring scientific readiness.

        A managed scene may intentionally have only Facility and Capture provisioned.  That is
        different from a live-camera fault or a failed calibration, so the browser needs a
        stable, scene-local reason rather than reverse-engineering missing adapter errors.
        """

        phase_by_id = {status.phase_id: status for status in statuses}

        def surface_available(surface_id: str, phase_id: str) -> bool:
            return (
                surface_id in self.operator_surface_ids
                or phase_by_id[phase_id].state is not PhaseState.UNAVAILABLE
            )

        calibration_configured = (
            self.calibration_adapter is not None or self.camera_summary_adapter is not None
        )
        reconstruction_configured = self.reconstruction_adapter is not None
        floor_configured = self.floor_adapter is not None
        final_review_configured = (
            reconstruction_configured
            and floor_configured
            and self.floor_preview_adapter is not None
            and self.rerun_launcher is not None
        )
        live_configured = self.live_operations_adapter is not None
        updates_configured = self.scene_update_adapter is not None

        return {
            "facility": _workflow_capability(
                surface_available("facility", "P02"),
                "facility-registration-not-provisioned",
                "Facility registration is not set up for this scene.",
            ),
            "capture": _workflow_capability(
                surface_available("capture", "P03"),
                "capture-not-provisioned",
                "Live capture is not set up for this scene.",
            ),
            "calibration": _workflow_capability(
                calibration_configured,
                "scene-calibration-not-provisioned",
                "Set up calibration inputs for this scene before pose work can begin.",
            ),
            "reconstruction": _workflow_capability(
                reconstruction_configured,
                "scene-reconstruction-not-provisioned",
                "Set up scene calibration and reconstruction before DA3 processing can begin.",
            ),
            "floor": _workflow_capability(
                floor_configured,
                "scene-floor-not-provisioned",
                "Add a scene-specific point cloud before floor refinement can begin.",
            ),
            "final_review": _workflow_capability(
                final_review_configured,
                "scene-final-review-not-provisioned",
                "Create a scene-specific floor result before final review.",
            ),
            "live_operations": _workflow_capability(
                live_configured,
                "scene-live-operations-not-provisioned",
                "Complete this scene's final review before Live operations can be configured.",
            ),
            "scene_updates": _workflow_capability(
                updates_configured,
                "scene-updates-not-provisioned",
                "Complete this scene's final review before scene updates can be configured.",
            ),
        }

    def _operator_calibration_readiness(
        self, camera_ids: tuple[str, ...]
    ) -> _CalibrationReadiness:
        if self.calibration_adapter is not None:
            try:
                policy = self._require_camera_policy().active(require_lens=True)
                result = self.calibration_adapter.status(policy)
                if not isinstance(result, dict):
                    raise P08WorkflowError("calibration adapter returned a malformed status")
                return _CalibrationReadiness(tuple(result["cameras"]), result, None)
            except (CameraPolicyError, ValueError) as error:
                return _CalibrationReadiness((), None, str(error))
        if self.camera_summary_adapter is not None:
            try:
                cameras = self.camera_summary_adapter.summaries(camera_ids)
                return _CalibrationReadiness(cameras, None, None)
            except P08WorkflowError as error:
                return _CalibrationReadiness((), None, str(error))
        return _CalibrationReadiness(
            (), None, "Final camera calibration summary is not configured"
        )

    def _operator_reconstruction_readiness(
        self, camera_ids: tuple[str, ...]
    ) -> _ReconstructionReadiness:
        errors = (
            tuple(self.reconstruction_adapter.readiness_errors())
            if self.reconstruction_adapter is not None
            else ("Static reconstruction is not configured",)
        )
        policy_required = getattr(
            self.reconstruction_adapter, "supports_scene_camera_policy", False
        )
        lens_ready = not policy_required
        overlap_ready = False
        if policy_required:
            try:
                policy = self._require_camera_policy().active(require_lens=True)
                lens_ready = True
                overlap_ready = policy.overlap_complete
                if policy.camera_ids != camera_ids:
                    lens_ready = False
                    errors += ("Active camera policy differs from the enabled scene roster",)
            except CameraPolicyError as error:
                errors += (str(error),)
        lineage_error = self._geometry_lineage_error()
        if lineage_error is not None:
            errors += (lineage_error,)
        return _ReconstructionReadiness(
            errors + self._catalog_workflow_input_issues(),
            lens_ready,
            overlap_ready,
            lineage_error is None,
        )

    def _operator_review_state(self) -> _OperatorReviewState:
        with self._workflow_lock:
            return _OperatorReviewState(
                geometry_approved=self._geometry_approved,
                floor_approved=self._floor_approved,
                geometry_artifact_id=self._active_geometry_artifact_id,
                floor_artifact_id=self._active_floor_artifact_id,
                latest_approved_floor_artifact_id=self._latest_approved_floor_artifact_id,
                current_floor_job_id=self._current_floor_job_id,
                current_floor_output_directory=self._current_floor_output_directory,
                previewed_targets=frozenset(self._previewed_targets),
            )

    def _phase_statuses(self, scene: SceneWorkspace) -> tuple[PhaseStatus, ...]:
        workspace_by_id = {
            reference.reference_id: reference for reference in scene.workspace_references
        }
        artifact_by_id = {artifact.artifact_id: artifact for artifact in scene.artifacts}
        resolved: dict[str, PhaseStatus] = {}
        default_adapter = FilesystemPhaseAdapter()
        active_jobs = self.jobs.list()
        for record in scene.phases:
            blocked = tuple(
                prerequisite
                for prerequisite in record.prerequisites
                if prerequisite not in resolved
                or resolved[prerequisite].state.value not in SATISFIED_PREREQUISITE_STATES
            )
            adapter = self.adapters.get(record.phase_id, default_adapter)
            reasons = adapter.inspect(scene, record, workspace_by_id, artifact_by_id)
            phase_jobs = [job for job in active_jobs if job["phase_id"] == record.phase_id]
            if any(
                job["state"] in {JobState.QUEUED.value, JobState.RUNNING.value}
                for job in phase_jobs
            ):
                state = PhaseState.RUNNING
                message = "A bounded phase job is active."
            elif blocked:
                state = PhaseState.UNAVAILABLE
                message = f"Prerequisites unavailable: {', '.join(blocked)}"
            elif reasons:
                state = PhaseState.STALE
                message = "Referenced workspace or artifact identity requires attention."
            else:
                resolved_state = getattr(adapter, "resolved_state", None)
                state = resolved_state() if callable(resolved_state) else record.state
                resolved_message = getattr(adapter, "resolved_message", None)
                message = resolved_message() if callable(resolved_message) else record.message
            resolved[record.phase_id] = PhaseStatus(
                record.phase_id,
                state,
                message,
                tuple(reasons) + tuple(f"prerequisite unavailable: {item}" for item in blocked),
                record.workspace_reference_ids,
                record.artifact_ids,
            )
        return tuple(resolved[phase_id] for phase_id in PHASE_ORDER)

    def start_floor_job(self, job_id: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        floor_adapter = self.floor_adapter
        if floor_adapter is None:
            raise P08WorkflowError("floor processing adapter is not configured")
        if not self.operator_status()["inputs_ready"]:
            raise P08WorkflowError("Resolve the current scene inputs before refining the floor")
        with self._workflow_lock:
            self._require_pipeline_idle()
            if not self._geometry_approved:
                raise P08WorkflowError(
                    "Approve the reconstructed geometry before refining the floor"
                )
            geometry_source_path = self._geometry_source_path
            geometry_source_sha256 = self._geometry_source_sha256
            contract = floor_adapter.contract
            if geometry_source_path is None or geometry_source_sha256 is None:
                raise P08WorkflowError("current reconstructed geometry is unavailable")
            if contract is not None and (
                geometry_source_sha256 != contract.selected_geometry_sha256
                or geometry_source_path.resolve() != contract.selected_geometry_path.resolve()
            ):
                raise P08WorkflowError(
                    "current reconstructed geometry does not match the floor source contract"
                )
            lineage_error = self._geometry_lineage_error()
            if lineage_error is not None:
                raise P08WorkflowError(lineage_error)
            self.repository.ensure_run_id_available(job_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            result = (
                floor_adapter.run(job_id, cancel_event)
                if contract is not None
                else floor_adapter.run_for_source(
                    job_id,
                    geometry_source_path,
                    geometry_source_sha256,
                    cancel_event,
                )
            )
            result = {
                **result,
                "source_geometry": {
                    "path": str(geometry_source_path),
                    "sha256": geometry_source_sha256,
                },
            }
            run_path = self.repository.write_run_manifest(
                job_id,
                {
                    "phase_id": "P08",
                    "action": "floor-completion",
                    "result": result,
                    "immutable": True,
                },
            )
            output_directory = _required_mapping_string(result, "output_directory")
            with self._workflow_lock:
                self._current_floor_job_id = job_id
                self._current_floor_output_directory = Path(output_directory)
                self._active_floor_artifact_id = None
                self._floor_approved = False
                self._previewed_targets.discard("floor")
                self._persist_operator_state()
                self._require_artifact_catalog().record_event(
                    "generated",
                    milestone_key="floor-refined-geometry",
                    detail={"workflow_job_id": job_id, "output_directory": output_directory},
                )
            return {**result, "workflow_run_manifest": _identity(run_path)}

        with self._workflow_lock:
            return self.jobs.submit(
                job_id,
                "P08",
                "floor-completion",
                lambda cancel_event: self._coordinated_operation(
                    job_id,
                    "floor-completion",
                    cancel_event,
                    lambda: operation(cancel_event),
                ),
            )

    def start_reconstruction_job(self, job_id: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        adapter = self.reconstruction_adapter
        if adapter is None:
            raise P08WorkflowError("static reconstruction is not configured")
        operator = self.operator_status()
        if not operator["inputs_ready"]:
            raise P08WorkflowError("Resolve every camera input before static reconstruction")
        scene_camera_ids: tuple[str, ...] | None = None
        camera_policy_sha256: str | None = None
        if getattr(adapter, "supports_scene_camera_policy", False):
            try:
                policy = self._require_camera_policy().active(require_lens=True)
            except CameraPolicyError as error:
                raise P08WorkflowError(str(error)) from error
            scene = self.repository.load()
            scene_camera_ids = tuple(
                camera.camera_id for camera in scene.cameras if camera.enabled
            )
            if policy.camera_ids != scene_camera_ids:
                raise P08WorkflowError(
                    "active camera policy differs from the enabled scene roster"
                )
            camera_policy_sha256 = policy.sha256
        with self._workflow_lock:
            self._require_pipeline_idle()
            self.repository.ensure_run_id_available(job_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            input_lineage = self._current_managed_input_lineage_sha256()
            if self.scene_evidence_adapter is not None and input_lineage is None:
                raise P08WorkflowError("current scene inputs are not ready for reconstruction")
            calibrated_input_directory: Path | None = None
            if self.calibration_adapter is not None:
                if scene_camera_ids is None:
                    raise P08WorkflowError(
                        "integrated calibration requires scene-policy-aware reconstruction"
                    )
                try:
                    active_policy = self._require_camera_policy().active(require_lens=True)
                    baseline_directory = getattr(adapter, "p06_run_directory", None)
                    calibrated_input_directory = (
                        self.calibration_adapter.prepare_reconstruction_inputs(
                            active_policy,
                            baseline_directory,
                            self.repository.root / "calibrated-reconstruction-inputs" / job_id,
                        )
                    )
                except (CameraPolicyError, ValueError) as error:
                    raise P08WorkflowError(str(error)) from error
            if scene_camera_ids is None:
                result = adapter.run(job_id, cancel_event)
            else:
                result = adapter.run(
                    job_id,
                    cancel_event,
                    input_run_directory=calibrated_input_directory,
                    camera_ids=scene_camera_ids,
                    camera_policy_sha256=camera_policy_sha256,
                )
            if input_lineage != self._current_managed_input_lineage_sha256():
                raise P08WorkflowError(
                    "scene inputs changed while reconstruction was running; preserve the output "
                    "and start a new run"
                )
            rerun = result.get("rerun")
            combined = result.get("combined_geometry")
            if not isinstance(rerun, Mapping) or not isinstance(combined, Mapping):
                raise P08WorkflowError("reconstruction returned incomplete preview metadata")
            artifact = ArtifactReference(
                artifact_id=job_id,
                phase_id="P07",
                kind="rerun-recording",
                path=Path(_required_mapping_string(rerun, "path")),
                sha256=_required_mapping_string(rerun, "sha256"),
                authority="current generated reconstruction preview",
                selected=True,
            )
            run_path = self.repository.write_run_manifest(
                job_id,
                {
                    "phase_id": "P06",
                    "action": "all-camera-static-reconstruction",
                    "result": result,
                    "immutable": True,
                },
            )
            with self._workflow_lock:
                self._runtime_artifacts[artifact.artifact_id] = artifact
                self._active_geometry_artifact_id = artifact.artifact_id
                self._active_floor_artifact_id = None
                self._current_floor_job_id = None
                self._current_floor_output_directory = None
                self._geometry_source_path = Path(_required_mapping_string(combined, "path"))
                self._geometry_source_sha256 = _required_mapping_string(combined, "sha256")
                self._geometry_input_lineage_sha256 = input_lineage
                self._geometry_approved = False
                self._floor_approved = False
                self._previewed_targets.clear()
                self._persist_operator_state()
                self._require_artifact_catalog().record_event(
                    "generated",
                    artifact_id=artifact.artifact_id,
                    milestone_key="geometry-review",
                    detail={"workflow_job_id": job_id},
                )
            return {**result, "workflow_run_manifest": _identity(run_path)}

        with self._workflow_lock:
            return self.jobs.submit(
                job_id,
                "P06",
                "all-camera-static-reconstruction",
                lambda cancel_event: self._coordinated_operation(
                    job_id,
                    "all-camera-static-reconstruction",
                    cancel_event,
                    lambda: operation(cancel_event),
                ),
            )

    def start_floor_preview_job(self, job_id: str, floor_job_id: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        if not self.operator_status()["inputs_ready"]:
            raise P08WorkflowError("Resolve the current scene inputs before final review")
        adapter = self.floor_preview_adapter
        launcher = self.rerun_launcher
        if adapter is None or launcher is None:
            raise P08WorkflowError("floor preview is not configured")
        with self._workflow_lock:
            self._require_pipeline_idle()
            if (
                floor_job_id != self._current_floor_job_id
                or self._current_floor_output_directory is None
            ):
                raise P08WorkflowError("Select the current completed floor result")
            output = self._current_floor_output_directory
            geometry_source = self._geometry_source_path
            floor_adapter = self.floor_adapter
            dynamic_geometry_manifest: Path | None = None
            if floor_adapter is not None and floor_adapter.contract is None:
                if geometry_source is None:
                    raise P08WorkflowError("current reconstructed geometry is unavailable")
                dynamic_geometry_manifest = (
                    geometry_source.parent.parent / "geometry-manifest.json"
                )
                if not dynamic_geometry_manifest.is_file():
                    raise P08WorkflowError("current geometry manifest is unavailable")
            self.repository.ensure_run_id_available(job_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            preview = (
                adapter.run(output, cancel_event)
                if dynamic_geometry_manifest is None
                else adapter.run_for_geometry(
                    output, dynamic_geometry_manifest, cancel_event
                )
            )
            rerun = preview.get("rerun")
            if not isinstance(rerun, dict):
                raise P08WorkflowError("floor preview builder returned no recording")
            artifact = ArtifactReference(
                artifact_id=job_id,
                phase_id="P08",
                kind="rerun-recording",
                path=Path(_required_mapping_string(rerun, "path")),
                sha256=_required_mapping_string(rerun, "sha256"),
                authority="current generated floor preview",
                selected=True,
            )
            with self._workflow_lock:
                self._runtime_artifacts[artifact.artifact_id] = artifact
                self._active_floor_artifact_id = artifact.artifact_id
                self._floor_approved = False
                self._previewed_targets.discard("floor")
                self._persist_operator_state()
                self._require_artifact_catalog().record_event(
                    "generated",
                    artifact_id=artifact.artifact_id,
                    milestone_key="final-review",
                    detail={"workflow_job_id": job_id},
                )
            launched = launcher.launch_artifact(artifact)
            with self._workflow_lock:
                self._previewed_targets.add("floor")
                self._persist_operator_state()
            run_path = self.repository.write_run_manifest(
                job_id,
                {
                    "phase_id": "P08",
                    "action": "build-and-open-floor-preview",
                    "result": {**preview, "launch": launched},
                    "immutable": True,
                },
            )
            return {**preview, "launch": launched, "workflow_run_manifest": _identity(run_path)}

        with self._workflow_lock:
            return self.jobs.submit(
                job_id,
                "P08",
                "build-and-open-floor-preview",
                lambda cancel_event: self._coordinated_operation(
                    job_id,
                    "build-and-open-floor-preview",
                    cancel_event,
                    lambda: operation(cancel_event),
                ),
            )

    def launch_rerun(self, action_id: str, artifact_id: str) -> dict[str, Any]:
        if self.rerun_launcher is None:
            raise P08WorkflowError("Rerun launcher is not configured")
        self.repository.ensure_run_id_available(action_id)
        scene = self.repository.load()
        artifacts = {artifact.artifact_id: artifact for artifact in self._all_artifacts(scene)}
        try:
            artifact = artifacts[artifact_id]
        except KeyError as error:
            raise P08WorkflowError("unknown Rerun artifact_id") from error
        target = self._target_for_artifact(artifact_id)
        if target == "geometry":
            artifact = self._ensure_camera_rich_geometry_review(artifact)
        result = self.rerun_launcher.launch_artifact(artifact)
        if target is not None:
            with self._workflow_lock:
                self._previewed_targets.add(target)
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "launch-selected-rerun",
                "result": result,
                "immutable": True,
            },
        )
        if target is not None:
            with self._workflow_lock:
                self._persist_operator_state()
                self._require_artifact_catalog().record_event(
                    "preview-opened",
                    artifact_id=artifact.artifact_id,
                    milestone_key=("geometry-review" if target == "geometry" else "final-review"),
                    detail={"target": target},
                )
        return {**result, "workflow_run_manifest": _identity(run_path)}

    def _ensure_camera_rich_geometry_review(
        self, artifact: ArtifactReference
    ) -> ArtifactReference:
        adapter = self.reconstruction_adapter
        geometry_source = self._geometry_source_path
        ensure = getattr(adapter, "ensure_geometry_review", None)
        if ensure is None or geometry_source is None:
            return artifact
        review = ensure(geometry_source.parent.parent, threading.Event())
        rerun = review.get("rerun")
        if not isinstance(rerun, Mapping):
            raise P08WorkflowError("geometry review builder returned no recording")
        review_path = Path(_required_mapping_string(rerun, "path"))
        review_sha256 = _required_mapping_string(rerun, "sha256")
        if review_path.resolve() == artifact.path.resolve() and review_sha256 == artifact.sha256:
            return artifact
        review_artifact = ArtifactReference(
            artifact_id=_geometry_review_artifact_id(geometry_source.parent.parent.name),
            phase_id="P07",
            kind="rerun-recording",
            path=review_path,
            sha256=review_sha256,
            authority="camera-rich current reconstruction review",
            selected=True,
        )
        with self._workflow_lock:
            self._runtime_artifacts[review_artifact.artifact_id] = review_artifact
            self._active_geometry_artifact_id = review_artifact.artifact_id
            self._persist_operator_state()
        return review_artifact

    def approve_result(self, action_id: str, target: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        if target not in {"geometry", "floor"}:
            raise P08WorkflowError("approval target must be geometry or floor")
        if not self.operator_status()["inputs_ready"]:
            raise P08WorkflowError("Resolve the current scene inputs before approval")
        self.repository.ensure_run_id_available(action_id)
        with self._workflow_lock:
            already_approved = (
                self._geometry_approved if target == "geometry" else self._floor_approved
            )
            if target not in self._previewed_targets and not already_approved:
                raise P08WorkflowError(f"Open the {target} preview before approving it")
            if target == "geometry":
                self._geometry_approved = True
            else:
                if not self._geometry_approved:
                    raise P08WorkflowError("Approve the geometry before approving the floor")
                self._floor_approved = True
                self._latest_approved_floor_artifact_id = self._active_floor_artifact_id
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P07" if target == "geometry" else "P08",
                "action": f"approve-{target}",
                "result": {"target": target, "status": "approved"},
                "immutable": True,
            },
        )
        with self._workflow_lock:
            self._persist_operator_state()
            active_artifact_id = (
                self._active_geometry_artifact_id
                if target == "geometry"
                else self._active_floor_artifact_id
            )
            self._require_artifact_catalog().record_event(
                "approved",
                artifact_id=active_artifact_id,
                milestone_key="geometry-review" if target == "geometry" else "final-review",
                detail={"target": target},
            )
        if target == "floor":
            self._unlock_scene_updates_from_current_result()
        return {
            "target": target,
            "status": "approved",
            "workflow_run_manifest": _identity(run_path),
        }

    def start_fresh_operator_session(self, action_id: str, session_id: str) -> dict[str, Any]:
        self._require_live_operations_inactive()
        if not self._operator_state_enabled:
            raise P08WorkflowError("operator workflow is not configured")
        _require_identifier(session_id, "operator session_id")
        self.repository.ensure_run_id_available(action_id)
        with self._workflow_lock:
            self._require_pipeline_idle()
            terminal_jobs = self.jobs.clear_terminal()
            archive = self.repository.archive_operator_state(
                action_id,
                {
                    **self._operator_state_payload(),
                    "archived_terminal_jobs": list(terminal_jobs),
                },
            )
            latest_approved = self._latest_approved_floor_artifact_id
            preserved_runtime_artifact = self._runtime_artifacts.get(latest_approved or "")
            self._operator_session_id = session_id
            self._operator_state_revision = 0
            self._active_geometry_artifact_id = None
            self._active_floor_artifact_id = None
            self._latest_approved_floor_artifact_id = latest_approved
            self._current_floor_job_id = None
            self._current_floor_output_directory = None
            self._geometry_source_path = None
            self._geometry_source_sha256 = None
            self._geometry_input_lineage_sha256 = None
            self._geometry_approved = False
            self._floor_approved = False
            self._previewed_targets.clear()
            self._runtime_artifacts = (
                {preserved_runtime_artifact.artifact_id: preserved_runtime_artifact}
                if preserved_runtime_artifact is not None
                else {}
            )
            self._persist_operator_state()
            self._require_artifact_catalog().record_event(
                "fresh-session-started",
                detail={"session_id": session_id, "preserved_artifacts": True},
            )
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "start-fresh-operator-session",
                "result": {
                    "session_id": session_id,
                    "archived_state": _identity(archive),
                    "preserved_artifacts": True,
                },
                "immutable": True,
            },
        )
        return {
            "status": "ready",
            "session_id": session_id,
            "archived_state": _identity(archive),
            "workflow_run_manifest": _identity(run_path),
        }

    def _all_artifacts(self, scene: SceneWorkspace) -> tuple[ArtifactReference, ...]:
        with self._workflow_lock:
            scene_input_artifacts = (
                tuple(self.scene_evidence_adapter.artifacts())
                if self.scene_evidence_adapter is not None
                else ()
            )
            return (
                *scene.artifacts,
                *self._runtime_artifacts.values(),
                *scene_input_artifacts,
            )

    def _require_artifact_catalog(self) -> SceneArtifactCatalog:
        if self._artifact_catalog is None:
            raise P08WorkflowError("artifact version catalog is not configured")
        return self._artifact_catalog

    def _require_camera_policy(self) -> CameraPolicyRepository:
        if self._camera_policy is None:
            raise P08WorkflowError("camera policy repository is not configured")
        return self._camera_policy

    def _sync_artifact_catalog(self) -> None:
        catalog = self._artifact_catalog
        if catalog is None:
            return
        scene = self.repository.load()
        self._sync_floor_plan_source(catalog, scene)
        for artifact in self._all_artifacts(scene):
            milestone_key = milestone_for_artifact(
                artifact.artifact_id, artifact.phase_id, artifact.kind
            )
            if artifact.kind == "rerun-recording":
                self._sync_companion_artifacts(catalog, scene, artifact)
            definition = milestone_definition(milestone_key)
            provider_metadata = (
                dict(self.scene_evidence_adapter.artifact_metadata(artifact))
                if self.scene_evidence_adapter is not None
                else {}
            )
            selectable = bool(provider_metadata.get("selectable", True)) and artifact.kind not in {
                "da3-run-manifest",
                "geometry-adoption-manifest",
            }
            selected = self._catalog_artifact_is_selected(artifact, milestone_key)
            parent = self._selected_parent_artifact_id(catalog, milestone_key)
            canonical_id = catalog.register(
                artifact_id=artifact.artifact_id,
                milestone_key=milestone_key,
                phase_id=artifact.phase_id,
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                display_name=artifact.path.name,
                significance=definition.significance,
                selected=selected,
                metadata={
                    "workflow_artifact_id": artifact.artifact_id,
                    "selectable": selectable,
                    **provider_metadata,
                },
                parent_artifact_ids=(parent,) if parent else (),
            )
            retention = self._baseline_retention(scene, artifact)
            if retention is not None:
                retention_class, reason = retention
                catalog.protect_retention(
                    canonical_id,
                    retention_class,
                    reason,
                    source="scene.json selected artifact",
                )
        self._sync_required_rollback(catalog)
        self._discover_past_artifact_versions(catalog, scene)
        self._protect_available_predecessor_copies(catalog, scene)
        self._artifact_catalog_initialized = True

    def _discover_past_artifact_versions(
        self, catalog: SceneArtifactCatalog, scene: SceneWorkspace
    ) -> None:
        workspace_by_kind: dict[str, list[WorkspaceReference]] = {}
        for reference in scene.workspace_references:
            workspace_by_kind.setdefault(reference.kind, []).append(reference)

        for reference in workspace_by_kind.get("p02-registration-workspace", []):
            for path in sorted((reference.path / "exports").glob("*.json")):
                self._register_discovered_file(
                    catalog,
                    path,
                    "facility-registration",
                    "P02",
                    "facility-registration",
                    selectable=True,
                    selected=self._initial_scene_path_is_selected(scene, path),
                )

        if self.scene_evidence_adapter is not None:
            facility_root = self.repository.root.parent / "facility-registration"
            selected_managed_paths = {
                artifact.path.resolve()
                for artifact in self.scene_evidence_adapter.artifacts()
                if artifact.selected
            }
            for path in sorted((facility_root / "exports").glob("*.json")):
                self._register_discovered_file(
                    catalog,
                    path,
                    "facility-registration",
                    "P02",
                    "facility-registration",
                    selectable=True,
                    selected=path.resolve() in selected_managed_paths,
                    metadata={"managed_scene_input": True},
                )

        bundle_root = scene.artifact_root / "captures" / "p03" / "temporal_bundles"
        for path in sorted(bundle_root.glob("*/bundle.json")):
            self._register_discovered_file(
                catalog,
                path,
                "capture-bundle",
                "P03",
                "capture-bundle",
                selectable=True,
                selected=self._initial_scene_path_is_selected(scene, path),
            )

        managed_bundle_root = scene.artifact_root / "captures" / "managed-scene" / "sessions"
        selected_managed_paths = (
            {
                artifact.path.resolve()
                for artifact in self.scene_evidence_adapter.artifacts()
                if artifact.selected
            }
            if self.scene_evidence_adapter is not None
            else set()
        )
        for path in sorted(managed_bundle_root.glob("*/bundles/*.json")):
            try:
                bundle_payload = _read_json(path)
            except P08WorkflowError:
                bundle_payload = {}
            complete = bundle_payload.get(
                "status"
            ) == "complete-roster" and not bundle_payload.get("missing_camera_ids")
            self._register_discovered_file(
                catalog,
                path,
                "capture-bundle",
                "P03",
                "capture-bundle",
                selectable=complete,
                selected=path.resolve() in selected_managed_paths,
                metadata={
                    "managed_scene_input": True,
                    "valid_for_downstream": complete,
                    "missing_camera_ids": bundle_payload.get("missing_camera_ids", []),
                },
            )

        calibration_references = workspace_by_kind.get(
            "p04-calibration-workspace", []
        ) + workspace_by_kind.get("p05-calibration-workspace", [])
        for reference in calibration_references:
            for path in sorted((reference.path / "exports").glob("*.json")):
                self._register_discovered_file(
                    catalog,
                    path,
                    "calibration-correspondence",
                    reference.phase_id,
                    "camera-correspondence-export",
                    selectable=False,
                    selected=False,
                    metadata={"workspace_reference_id": reference.reference_id},
                )

        run_root = scene.artifact_root / "runs"
        for path in sorted(run_root.glob("p06-*/input-manifest.json")):
            self._register_discovered_file(
                catalog,
                path,
                "reconstruction-input",
                "P06",
                "da3-input-manifest",
                selectable=True,
                selected=self._initial_scene_path_is_selected(scene, path),
            )
        for path in sorted(run_root.glob("p06-*/run-manifest.json")):
            self._register_discovered_file(
                catalog,
                path,
                "reconstruction-input",
                "P06",
                "da3-run-manifest",
                selectable=False,
                selected=False,
            )

        for run_directory in sorted(run_root.glob("reconstruction-*")):
            if not run_directory.is_dir():
                continue
            for path in sorted((run_directory / "geometry").glob("*.npz")):
                selected = (
                    self._geometry_source_path is not None
                    and path.resolve() == self._geometry_source_path.resolve()
                )
                self._register_discovered_file(
                    catalog,
                    path,
                    "reconstructed-geometry",
                    "P07",
                    "point-cloud-npz",
                    selectable=True,
                    selected=selected,
                    metadata={"run_directory": str(run_directory)},
                )
            for path in sorted(run_directory.glob("*.rrd")):
                selected = any(
                    artifact.artifact_id == self._active_geometry_artifact_id
                    and artifact.path.resolve() == path.resolve()
                    for artifact in self._runtime_artifacts.values()
                )
                self._register_discovered_file(
                    catalog,
                    path,
                    "geometry-review",
                    "P07",
                    "rerun-recording",
                    selectable=True,
                    selected=selected,
                    metadata={"run_directory": str(run_directory)},
                )

        for run_directory in sorted(run_root.glob("floor-*")):
            if not run_directory.is_dir():
                continue
            companions = (
                (
                    "floor-completion-manifest.json",
                    "floor-refined-geometry",
                    "floor-completion-manifest",
                    True,
                ),
                (
                    "authoritative_floor_plane.npz",
                    "floor-refined-geometry",
                    "floor-plane-npz",
                    False,
                ),
                ("verification.json", "floor-verification", "floor-verification", True),
                (
                    "candidate-working-facility-geometry-v3-floor-context-v4.rrd",
                    "final-review",
                    "rerun-recording",
                    True,
                ),
            )
            for filename, milestone_key, kind, selectable in companions:
                path = run_directory / filename
                if not path.is_file():
                    continue
                selected = selectable and (
                    self._current_floor_output_directory is not None
                    and run_directory.resolve() == self._current_floor_output_directory.resolve()
                    and (
                        milestone_key != "final-review"
                        or any(
                            artifact.artifact_id == self._active_floor_artifact_id
                            and artifact.path.resolve() == path.resolve()
                            for artifact in self._runtime_artifacts.values()
                        )
                    )
                )
                self._register_discovered_file(
                    catalog,
                    path,
                    milestone_key,
                    "P08",
                    kind,
                    selectable=selectable,
                    selected=selected,
                    metadata={"run_directory": str(run_directory)},
                )

    def _register_discovered_file(
        self,
        catalog: SceneArtifactCatalog,
        path: Path,
        milestone_key: str,
        phase_id: str,
        kind: str,
        *,
        selectable: bool,
        selected: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        sha256 = catalog.known_sha256(milestone_key, path) or _sha256(path)
        definition = milestone_definition(milestone_key)
        parent = self._selected_parent_artifact_id(catalog, milestone_key)
        catalog.register(
            artifact_id=_catalog_id(kind, path, sha256),
            milestone_key=milestone_key,
            phase_id=phase_id,
            kind=kind,
            path=path,
            sha256=sha256,
            display_name=path.name,
            significance=definition.significance,
            selected=selected,
            metadata={"selectable": selectable, **dict(metadata or {})},
            parent_artifact_ids=(parent,) if parent else (),
        )

    def _initial_scene_path_is_selected(self, scene: SceneWorkspace, path: Path) -> bool:
        return not self._artifact_catalog_initialized and any(
            artifact.selected and artifact.path.resolve() == path.resolve()
            for artifact in scene.artifacts
        )

    def _sync_floor_plan_source(
        self, catalog: SceneArtifactCatalog, scene: SceneWorkspace
    ) -> None:
        registrations = [
            item for item in self._all_artifacts(scene) if item.kind == "facility-registration"
        ]
        if not registrations:
            return
        try:
            payload = _read_json(registrations[-1].path)
            plan = payload.get("plan")
            if not isinstance(plan, Mapping):
                return
            filename = _required_mapping_string(plan, "original_filename")
            sha256 = _required_mapping_string(plan, "source_sha256")
        except (OSError, P08WorkflowError):
            return
        candidates = sorted((scene.artifact_root / "runs").glob(f"p02-*/inputs/{filename}"))
        managed_sources = registrations[-1].path.parent.parent / "sources"
        candidates.extend(
            sorted(path for path in managed_sources.glob(f"{sha256}.*") if path.is_file())
        )
        source = next((path for path in reversed(candidates) if _sha256(path) == sha256), None)
        if source is None:
            return
        definition = milestone_definition("floor-plan-source")
        artifact_id = catalog.register(
            artifact_id=f"floor-plan-{sha256[:12]}",
            milestone_key="floor-plan-source",
            phase_id="P02",
            kind="floor-plan-pdf",
            path=source,
            sha256=sha256,
            display_name=filename,
            significance=definition.significance,
            selected=registrations[-1].selected,
            metadata={
                "selectable": True,
                "source_revision": payload.get("source_revision"),
                "managed_scene_input": registrations[-1].authority.startswith(
                    "managed scene-local"
                ),
            },
        )
        catalog.protect_retention(
            artifact_id,
            "accepted-predecessor",
            "The accepted floor-plan source defines the physical scene and all later work.",
            source="selected P02 facility registration",
        )

    def _sync_companion_artifacts(
        self,
        catalog: SceneArtifactCatalog,
        scene: SceneWorkspace,
        artifact: ArtifactReference,
    ) -> None:
        run_directory = artifact.path.parent
        baseline_retention = self._baseline_retention(scene, artifact)
        if artifact.phase_id == "P07":
            candidates = sorted((run_directory / "geometry").glob("*.npz"))
            for path in candidates:
                sha256 = catalog.known_sha256("reconstructed-geometry", path) or _sha256(path)
                selected = (
                    self._geometry_source_path is not None
                    and path.resolve() == self._geometry_source_path.resolve()
                    and sha256 == self._geometry_source_sha256
                )
                definition = milestone_definition("reconstructed-geometry")
                artifact_id = catalog.register(
                    artifact_id=_catalog_id("geometry", path, sha256),
                    milestone_key="reconstructed-geometry",
                    phase_id="P07",
                    kind="point-cloud-npz",
                    path=path,
                    sha256=sha256,
                    display_name=path.name,
                    significance=definition.significance,
                    selected=selected,
                    metadata={"selectable": True, "run_directory": str(run_directory)},
                    parent_artifact_ids=tuple(
                        value
                        for value in (
                            self._selected_parent_artifact_id(catalog, "reconstructed-geometry"),
                        )
                        if value
                    ),
                )
                if baseline_retention is not None:
                    retention_class, reason = baseline_retention
                    catalog.protect_retention(
                        artifact_id,
                        retention_class,
                        f"{reason} This is its exact point-cloud companion.",
                        source="selected P07 Rerun companion",
                    )
            return
        companions = (
            (
                "floor-completion-manifest.json",
                "floor-refined-geometry",
                "floor-completion-manifest",
            ),
            ("authoritative_floor_plane.npz", "floor-refined-geometry", "floor-plane-npz"),
            ("verification.json", "floor-verification", "floor-verification"),
        )
        for filename, milestone_key, kind in companions:
            path = run_directory / filename
            if not path.is_file():
                continue
            sha256 = catalog.known_sha256(milestone_key, path) or _sha256(path)
            selected = (
                self._current_floor_output_directory is not None
                and run_directory.resolve() == self._current_floor_output_directory.resolve()
                and kind != "floor-plane-npz"
            )
            definition = milestone_definition(milestone_key)
            artifact_id = catalog.register(
                artifact_id=_catalog_id(kind, path, sha256),
                milestone_key=milestone_key,
                phase_id="P08",
                kind=kind,
                path=path,
                sha256=sha256,
                display_name=path.name,
                significance=definition.significance,
                selected=selected,
                metadata={
                    "selectable": kind != "floor-plane-npz",
                    "run_directory": str(run_directory),
                },
                parent_artifact_ids=tuple(
                    value
                    for value in (self._selected_parent_artifact_id(catalog, milestone_key),)
                    if value
                ),
            )
            if baseline_retention is not None:
                retention_class, reason = baseline_retention
                catalog.protect_retention(
                    artifact_id,
                    retention_class,
                    f"{reason} This is a required floor-result companion.",
                    source="selected P08 Rerun companion",
                )

    @staticmethod
    def _baseline_retention(
        scene: SceneWorkspace, artifact: ArtifactReference
    ) -> tuple[str, str] | None:
        baseline = next(
            (
                item
                for item in scene.artifacts
                if item.selected and item.artifact_id == artifact.artifact_id
            ),
            None,
        )
        if baseline is None:
            return None
        if baseline.phase_id == "P08":
            return (
                "selected-authority",
                "This artifact is a selected P08 authority record in the frozen scene baseline.",
            )
        return (
            "accepted-predecessor",
            "This accepted predecessor input is part of the frozen P02-P07 authority chain.",
        )

    def _sync_required_rollback(self, catalog: SceneArtifactCatalog) -> None:
        floor_adapter = self.floor_adapter
        contract = getattr(floor_adapter, "contract", None)
        if contract is None:
            return
        path = Path(contract.rollback_manifest_path)
        sha256 = str(contract.rollback_manifest_sha256)
        artifact_id = catalog.register(
            artifact_id="p07-v1-required-rollback-manifest",
            milestone_key="reconstructed-geometry",
            phase_id="P07",
            kind="geometry-rollback-manifest",
            path=path,
            sha256=sha256,
            display_name=path.name,
            significance="Required rollback identity for the accepted P07 v2 source.",
            selected=False,
            metadata={"selectable": False, "rollback_for": "P07-v2"},
        )
        catalog.protect_retention(
            artifact_id,
            "required-rollback",
            "D042 requires this exact P07 v1 manifest as the rollback identity.",
            source="P08 source contract",
        )

    @staticmethod
    def _protect_available_predecessor_copies(
        catalog: SceneArtifactCatalog, scene: SceneWorkspace
    ) -> None:
        """Preserve exact bytes when a frozen predecessor path is already unavailable."""

        unavailable_hashes = {
            artifact.sha256
            for artifact in scene.artifacts
            if artifact.selected and artifact.phase_id != "P08" and not artifact.path.is_file()
        }
        if not unavailable_hashes:
            return
        for milestone in catalog.status()["milestones"]:
            for version in milestone["versions"]:
                if version["sha256"] in unavailable_hashes and version["lifecycle"] == "available":
                    catalog.protect_retention(
                        str(version["artifact_id"]),
                        "accepted-predecessor",
                        "This is a retained exact-content copy of an accepted predecessor whose "
                        "original path is unavailable. Protection preserves bytes only and does "
                        "not substitute or amend authority.",
                        source="frozen predecessor identity preservation",
                    )

    def _catalog_artifact_is_selected(
        self, artifact: ArtifactReference, milestone_key: str
    ) -> bool:
        if milestone_key in {
            "facility-registration",
            "capture-bundle",
        } and artifact.authority.startswith("managed scene-local"):
            return artifact.selected
        if milestone_key == "geometry-review":
            return (
                artifact.artifact_id == self._active_geometry_artifact_id
                if self._active_geometry_artifact_id is not None
                else artifact.selected and not self._artifact_catalog_initialized
            )
        if milestone_key == "final-review":
            return (
                artifact.artifact_id == self._active_floor_artifact_id
                if self._active_floor_artifact_id is not None
                else artifact.selected and not self._artifact_catalog_initialized
            )
        if milestone_key == "reconstructed-geometry" and artifact.kind == "point-cloud-npz":
            return (
                self._geometry_source_path is not None
                and artifact.path.resolve() == self._geometry_source_path.resolve()
                and artifact.sha256 == self._geometry_source_sha256
            )
        if milestone_key in {"floor-refined-geometry", "floor-verification"}:
            if self._current_floor_output_directory is not None:
                return (
                    artifact.path.parent.resolve()
                    == self._current_floor_output_directory.resolve()
                )
        return artifact.selected and not self._artifact_catalog_initialized

    def _selected_parent_artifact_id(
        self, catalog: SceneArtifactCatalog, milestone_key: str
    ) -> str | None:
        index = MILESTONE_INDEX[milestone_key]
        if index == 0:
            return None
        preceding_key = tuple(MILESTONE_INDEX)[index - 1]
        selected = catalog.selected(preceding_key)
        return str(selected["artifact_id"]) if selected else None

    def _apply_catalog_selection(self, selected: Mapping[str, Any]) -> None:
        milestone_key = _required_mapping_string(selected, "milestone_key")
        artifact_id = _required_mapping_string(selected, "artifact_id")
        path = Path(_required_mapping_string(selected, "path"))
        sha256 = _required_mapping_string(selected, "sha256")
        phase_id = _required_mapping_string(selected, "phase_id")
        kind = _required_mapping_string(selected, "kind")
        if (
            milestone_key in {"facility-registration", "capture-bundle"}
            and self.scene_evidence_adapter is not None
        ):
            try:
                self.scene_evidence_adapter.select_artifact(
                    ArtifactReference(
                        artifact_id,
                        phase_id,
                        kind,
                        path,
                        sha256,
                        "managed scene-local catalog selection",
                        True,
                    )
                )
            except ValueError as error:
                raise P08WorkflowError(str(error)) from error
        index = MILESTONE_INDEX[milestone_key]
        if index <= MILESTONE_INDEX["reconstruction-input"]:
            self._active_geometry_artifact_id = None
            self._geometry_source_path = None
            self._geometry_source_sha256 = None
            self._clear_geometry_and_floor_review_state()
            if milestone_key == "reconstruction-input":
                self._configure_reconstruction_input(selected)
        elif milestone_key == "reconstructed-geometry":
            if kind != "point-cloud-npz":
                raise P08WorkflowError("select a point-cloud version for combined geometry")
            self._geometry_source_path = path
            self._geometry_source_sha256 = sha256
            self._active_geometry_artifact_id = None
            self._clear_geometry_and_floor_review_state()
        elif milestone_key == "geometry-review":
            artifact = ArtifactReference(
                artifact_id, phase_id, kind, path, sha256, "selected catalog version", True
            )
            self._runtime_artifacts[artifact_id] = artifact
            self._active_geometry_artifact_id = artifact_id
            self._clear_geometry_and_floor_review_state(
                clear_geometry_artifact=False, clear_geometry_lineage=False
            )
        elif milestone_key == "floor-refined-geometry":
            self._current_floor_output_directory = path.parent
            self._current_floor_job_id = path.parent.name
            self._active_floor_artifact_id = None
            self._floor_approved = False
            self._previewed_targets.discard("floor")
        elif milestone_key == "floor-verification":
            self._floor_approved = False
            self._previewed_targets.discard("floor")
        elif milestone_key == "final-review":
            artifact = ArtifactReference(
                artifact_id, phase_id, kind, path, sha256, "selected catalog version", True
            )
            self._runtime_artifacts[artifact_id] = artifact
            self._active_floor_artifact_id = artifact_id
            self._floor_approved = False
            self._previewed_targets.discard("floor")

    def _configure_reconstruction_input(self, selected: Mapping[str, Any]) -> None:
        path = Path(_required_mapping_string(selected, "path"))
        sha256 = _required_mapping_string(selected, "sha256")
        if selected.get("kind") != "da3-input-manifest":
            raise P08WorkflowError("select an input manifest for static reconstruction")
        camera_adapter = self.camera_summary_adapter
        camera_config = getattr(camera_adapter, "config", None)
        if camera_adapter is not None and camera_config is not None:
            camera_adapter.config = replace(
                camera_config,
                camera_input_manifest_path=path,
                camera_input_manifest_sha256=sha256,
            )
        reconstruction_adapter = self.reconstruction_adapter
        if reconstruction_adapter is not None and hasattr(
            reconstruction_adapter, "p06_run_directory"
        ):
            self.reconstruction_adapter = replace(
                reconstruction_adapter,
                p06_run_directory=path.parent,
                expected_geometry_sha256=None,
            )

    def _catalog_workflow_input_issues(self) -> tuple[str, ...]:
        if not self._catalog_input_controls_enabled or self._artifact_catalog is None:
            return ()
        required = (
            "floor-plan-source",
            "facility-registration",
            "capture-bundle",
            "calibration-correspondence",
            "calibration-pose-registry",
            "reconstruction-input",
        )
        missing = [
            milestone_definition(key).title
            for key in required
            if self._artifact_catalog.selected(key) is None
        ]
        if not missing:
            return ()
        return ("Select or generate a current version for: " + ", ".join(missing),)

    def _clear_geometry_and_floor_review_state(
        self,
        *,
        clear_geometry_artifact: bool = True,
        clear_geometry_lineage: bool = True,
    ) -> None:
        if clear_geometry_artifact:
            self._active_geometry_artifact_id = None
        self._active_floor_artifact_id = None
        self._current_floor_job_id = None
        self._current_floor_output_directory = None
        self._geometry_approved = False
        self._floor_approved = False
        self._previewed_targets.clear()
        if clear_geometry_lineage:
            self._geometry_input_lineage_sha256 = None

    def _managed_input_lineage_sha256(
        self, policy: SceneCameraPolicy, *, legacy_full_policy: bool = False
    ) -> str | None:
        if self.scene_evidence_adapter is None or self.calibration_adapter is None:
            return None
        try:
            evidence = self.scene_evidence_adapter.status()
            facility = _required_mapping(evidence, "facility")
            capture = _required_mapping(evidence, "capture")
            if facility.get("ready") is not True or capture.get("ready") is not True:
                return None
            facility_identity = _required_mapping(facility, "current_export")
            capture_identity = _required_mapping(capture, "current_bundle")
            calibration = self.calibration_adapter.status(policy)
            if calibration.get("all_cameras_ready") is not True:
                return None
            intrinsic_batch = _required_mapping(calibration, "intrinsic_batch")
            camera_records = calibration.get("cameras")
            if not isinstance(camera_records, list):
                return None
            cameras: list[dict[str, str]] = []
            for camera in camera_records:
                if not isinstance(camera, Mapping):
                    return None
                attempt = _required_mapping(camera, "attempt")
                cameras.append(
                    {
                        "camera_id": _required_mapping_string(camera, "camera_id"),
                        "readiness": _required_mapping_string(camera, "readiness"),
                        "attempt_sha256": _required_mapping_string(
                            attempt, "payload_sha256"
                        ),
                    }
                )
            payload = {
                "facility_export_sha256": _required_mapping_string(
                    facility_identity, "sha256"
                ),
                "capture_bundle_sha256": _required_mapping_string(
                    capture_identity, "sha256"
                ),
                (
                    "camera_policy_sha256"
                    if legacy_full_policy
                    else "intrinsic_policy_sha256"
                ): (
                    policy.sha256
                    if legacy_full_policy
                    else policy.intrinsic_policy_sha256
                ),
                "intrinsic_batch_sha256": _required_mapping_string(
                    intrinsic_batch, "payload_sha256"
                ),
                "cameras": cameras,
            }
        except (CameraPolicyError, P08WorkflowError, ValueError):
            return None
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _current_managed_input_lineage_sha256(self) -> str | None:
        """Hash only current scene inputs that materially determine generated geometry."""

        try:
            policy = self._require_camera_policy().active(require_lens=True)
        except CameraPolicyError:
            return None
        return self._managed_input_lineage_sha256(policy)

    def _migrate_overlap_independent_geometry_lineage(self) -> None:
        """Upgrade a legacy full-policy lineage without accepting changed geometry inputs."""

        expected = self._geometry_input_lineage_sha256
        source_path = self._geometry_source_path
        if expected is None or source_path is None:
            return
        current = self._current_managed_input_lineage_sha256()
        if current is None or current == expected:
            return
        try:
            active_policy = self._require_camera_policy().active(require_lens=True)
        except CameraPolicyError:
            return
        candidates = [active_policy]
        input_manifest = (
            self.repository.root
            / "calibrated-reconstruction-inputs"
            / source_path.parent.parent.name
            / "input-manifest.json"
        )
        if input_manifest.is_file():
            try:
                source_policy_sha256 = _required_mapping_string(
                    _read_json(input_manifest), "camera_policy_sha256"
                )
                source_policy = self._require_camera_policy().by_sha256(source_policy_sha256)
                if source_policy is not None and source_policy not in candidates:
                    candidates.append(source_policy)
            except (CameraPolicyError, P08WorkflowError, ValueError):
                pass
        for candidate in candidates:
            if candidate.intrinsic_policy_sha256 != active_policy.intrinsic_policy_sha256:
                continue
            legacy = self._managed_input_lineage_sha256(candidate, legacy_full_policy=True)
            if legacy == expected:
                self._geometry_input_lineage_sha256 = current
                self._persist_operator_state()
                return

    def _geometry_lineage_error(self) -> str | None:
        expected = self._geometry_input_lineage_sha256
        if expected is None or self._geometry_source_path is None:
            return None
        current = self._current_managed_input_lineage_sha256()
        if current == expected:
            return None
        return (
            "Current Facility, Capture, intrinsic policy, or calibration differs from the "
            "reconstructed geometry"
        )

    def _require_pipeline_idle(self) -> None:
        if any(
            job["state"] in {JobState.QUEUED.value, JobState.RUNNING.value}
            and job["action"]
            in {
                "all-camera-static-reconstruction",
                "floor-completion",
                "build-and-open-floor-preview",
                "full-scene-update",
            }
            for job in self.jobs.list()
        ):
            raise P08WorkflowError("wait for the active workflow action to finish")

    def _configured_artifact_id(self, target: str) -> str | None:
        if self.operator_config is None:
            return None
        return str(
            self.operator_config.geometry_rerun_artifact_id
            if target == "geometry"
            else self.operator_config.floor_rerun_artifact_id
        )

    def _target_for_artifact(self, artifact_id: str) -> str | None:
        with self._workflow_lock:
            if artifact_id == self._active_geometry_artifact_id:
                return "geometry"
            if artifact_id == self._active_floor_artifact_id:
                return "floor"
        return None


def load_frozen_floor_input(value: Mapping[str, Any]) -> FrozenP07FloorInput:
    return FrozenP07FloorInput(
        adoption_manifest_path=Path(_string(value, "adoption_manifest_path")),
        adoption_manifest_sha256=_string(value, "adoption_manifest_sha256"),
        selected_geometry_path=Path(_string(value, "selected_geometry_path")),
        selected_geometry_sha256=_string(value, "selected_geometry_sha256"),
        rollback_manifest_path=Path(_string(value, "rollback_manifest_path")),
        rollback_manifest_sha256=_string(value, "rollback_manifest_sha256"),
        final_rerun_manifest_path=Path(_string(value, "final_rerun_manifest_path")),
        final_rerun_manifest_sha256=_string(value, "final_rerun_manifest_sha256"),
        final_rerun_path=Path(_string(value, "final_rerun_path")),
        final_rerun_sha256=_string(value, "final_rerun_sha256"),
    )


def _workflow_capability(
    configured: bool, missing_reason_code: str, missing_message: str
) -> WorkflowCapability:
    if configured:
        return WorkflowCapability(
            WorkflowCapabilityState.AVAILABLE,
            "configured",
            "This workflow step is configured for this scene.",
        )
    return WorkflowCapability(
        WorkflowCapabilityState.NOT_PROVISIONED,
        missing_reason_code,
        missing_message,
    )


def _active_operator_job(jobs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    active = tuple(
        job
        for job in jobs
        if job["state"] in {JobState.QUEUED.value, JobState.RUNNING.value}
        and job["action"]
        in {
            "all-camera-static-reconstruction",
            "floor-completion",
            "build-and-open-floor-preview",
        }
    )
    return active[-1] if active else None


def _operator_steps(
    phase_by_id: Mapping[str, PhaseStatus],
    calibration: _CalibrationReadiness,
    reconstruction: _ReconstructionReadiness,
    capabilities: Mapping[str, WorkflowCapability],
    review: _OperatorReviewState,
    geometry_available: bool,
    floor_available: bool,
    active_job: Mapping[str, Any] | None,
) -> tuple[dict[str, str], ...]:
    action = None if active_job is None else active_job["action"]
    current_result_ready = calibration.ready and reconstruction.result_current
    early_ready = reconstruction.lens_policy_ready and all(
        phase_by_id[phase_id].state
        not in {PhaseState.UNAVAILABLE, PhaseState.FAILED, PhaseState.STALE}
        for phase_id in ("P02", "P03")
    )
    if action == "all-camera-static-reconstruction":
        reconstruction_state = "running"
    elif review.geometry_approved and current_result_ready:
        reconstruction_state = "approved"
    elif geometry_available:
        reconstruction_state = "ready_for_review"
    elif capabilities["reconstruction"].state is WorkflowCapabilityState.NOT_PROVISIONED:
        reconstruction_state = "blocked"
    elif calibration.ready and not reconstruction.errors:
        reconstruction_state = "ready"
    else:
        reconstruction_state = "blocked"

    if action in {"floor-completion", "build-and-open-floor-preview"}:
        floor_state = "running"
    elif review.floor_approved and current_result_ready:
        floor_state = "approved"
    elif floor_available and review.geometry_approved and current_result_ready:
        floor_state = "ready_for_review"
    elif capabilities["floor"].state is WorkflowCapabilityState.NOT_PROVISIONED:
        floor_state = "blocked"
    elif review.geometry_approved and current_result_ready:
        floor_state = "ready"
    else:
        floor_state = "blocked"

    return (
        _operator_step("setup", "Facility & cameras", "complete" if early_ready else "attention"),
        _operator_step(
            "capture",
            "Capture",
            "complete"
            if early_ready
            else "attention"
            if capabilities["capture"].state is WorkflowCapabilityState.AVAILABLE
            else "blocked",
        ),
        _operator_step(
            "calibration",
            "Calibration & pose",
            "complete"
            if calibration.ready
            else "blocked"
            if capabilities["calibration"].state is WorkflowCapabilityState.NOT_PROVISIONED
            else "attention",
        ),
        _operator_step("reconstruction", "Static reconstruction", reconstruction_state),
        _operator_step("floor", "Floor refinement", floor_state),
        _operator_step(
            "results",
            "Final review",
            "complete"
            if review.floor_approved and current_result_ready
            else "blocked"
            if capabilities["final_review"].state is WorkflowCapabilityState.NOT_PROVISIONED
            else "pending",
        ),
    )


def _operator_step(step_id: str, title: str, state: str) -> dict[str, str]:
    return {"step_id": step_id, "title": title, "state": state}


def _required_mapping_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise P08WorkflowError(f"{key} must be a non-blank string")
    return result.strip()


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise P08WorkflowError(f"{key} must be an object")
    return result


def _require_identifier(value: str, label: str) -> None:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise P08WorkflowError(f"{label} is malformed")


def _require_phase(value: str) -> None:
    if value not in PHASE_ORDER:
        raise P08WorkflowError("phase_id must be one of P02 through P08")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise P08WorkflowError(f"{label} must be a lowercase SHA-256")


def _redact_error(message: str) -> str:
    redacted = re.sub(r"(?i)rtsp://[^\s'\"]+", "rtsp://<redacted>", message)
    redacted = re.sub(r"(?i)(password|passwd|token|secret)=([^\s&]+)", r"\1=<redacted>", redacted)
    return redacted[:1000]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P08WorkflowError(f"workflow JSON is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise P08WorkflowError(f"workflow JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise P08WorkflowError(f"immutable workflow record already exists: {path}") from error


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _resolve_rerun_viewer(path: Path) -> Path:
    """Resolve a Windows virtual-environment shim to Rerun's native viewer binary."""
    resolved = path.resolve()
    if resolved.name.lower() != "rerun.exe":
        return resolved
    candidate = (
        resolved.parent.parent / "Lib" / "site-packages" / "rerun_sdk" / "rerun_cli" / "rerun.exe"
    )
    return candidate.resolve() if candidate.is_file() else resolved


def _geometry_review_artifact_id(run_name: str) -> str:
    suffix = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:8]
    return f"{run_name[:38]}-geometry-review-{suffix}"


def _catalog_id(prefix: str, path: Path, sha256: str) -> str:
    safe_prefix = re.sub(r"[^a-z0-9-]+", "-", prefix.lower()).strip("-")[:28]
    identity = hashlib.sha256(f"{path.resolve()}|{sha256}".encode()).hexdigest()[:12]
    return f"{safe_prefix}-{identity}"


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


def _string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise P08WorkflowError(f"{key} must be a non-blank string")
    return result.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise P08WorkflowError("optional string value is malformed")
    return value


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise P08WorkflowError(f"{key} must be boolean")
    return result


def _object_list(value: Mapping[str, Any], key: str) -> tuple[dict[str, Any], ...]:
    result = value.get(key)
    if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
        raise P08WorkflowError(f"{key} must be a list of objects")
    return tuple(result)


def _string_tuple(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    result = value.get(key)
    if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
        raise P08WorkflowError(f"{key} must be a list of strings")
    return tuple(result)
