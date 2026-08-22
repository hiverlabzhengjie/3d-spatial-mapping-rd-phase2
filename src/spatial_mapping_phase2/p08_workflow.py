"""Shared P02-P08 scene, phase, artifact, job, and safe-launch contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

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
    verify_floor_artifact_run,
)

IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
PHASE_ORDER = ("P02", "P03", "P04", "P05", "P06", "P07", "P08")
SATISFIED_PREREQUISITE_STATES = frozenset(
    {"ready", "running", "complete", "provisional", "rejected"}
)


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
                job_id,
                phase_id,
                action,
                JobState.QUEUED,
                threading.Event(),
            )
            self._jobs[job_id] = job
            job.future = self._executor.submit(self._execute, job, operation)
            return job.snapshot()

    def _execute(self, job: _MutableJob, operation: JobOperation) -> None:
        with self._lock:
            if job.cancel_event.is_set():
                job.state = JobState.CANCELLED
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
        except Exception as error:
            with self._lock:
                job.error_type = type(error).__name__
                job.error_message = _redact_error(str(error))
                job.state = JobState.CANCELLED if job.cancel_event.is_set() else JobState.FAILED

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._require_job(job_id)
            if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
                return job.snapshot()
            job.cancel_event.set()
            if job.future is not None and job.future.cancel():
                job.state = JobState.CANCELLED
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

    contract: FrozenP07FloorInput
    config: FloorProcessingConfig
    output_root: Path

    def run(self, job_id: str, cancel_event: threading.Event) -> dict[str, Any]:
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
    floor_preview_adapter: Any | None = None
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
    _artifact_catalog: SceneArtifactCatalog | None = field(default=None, init=False, repr=False)
    _artifact_catalog_initialized: bool = field(default=False, init=False, repr=False)
    _catalog_input_controls_enabled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._catalog_input_controls_enabled = self.operator_config is not None
        if self.operator_config is not None:
            state = self.repository.read_operator_state()
            if state is None:
                self._initialize_operator_state()
                self._persist_operator_state()
            else:
                self._restore_operator_state(state)
        scene = self.repository.load()
        self._artifact_catalog = SceneArtifactCatalog(
            self.repository.artifact_catalog_path,
            scene.project_id,
            scene.scene_id,
            (
                scene.artifact_root,
                *(reference.path for reference in scene.workspace_references),
            ),
        )
        self._artifact_catalog_initialized = self._artifact_catalog.has_versions()
        self._sync_artifact_catalog()
        selected_input = self._artifact_catalog.selected("reconstruction-input")
        if selected_input is not None and selected_input["kind"] == "da3-input-manifest":
            self._configure_reconstruction_input(selected_input)

    def _initialize_operator_state(self) -> None:
        config = self.operator_config
        if config is None:
            raise P08WorkflowError("operator workflow is not configured")
        self._geometry_approved = bool(config.geometry_approved)
        self._floor_approved = bool(config.floor_approved)
        self._active_geometry_artifact_id = str(config.geometry_rerun_artifact_id)
        self._active_floor_artifact_id = str(config.floor_rerun_artifact_id)
        self._latest_approved_floor_artifact_id = str(config.floor_rerun_artifact_id)
        if self.floor_adapter is not None:
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
            "display_name": scene.display_name,
            "camera_roster": [camera.to_dict() for camera in scene.cameras],
            "phases": [status.to_dict() for status in statuses],
            "jobs": list(self.jobs.list()),
            "artifacts": [artifact.to_dict() for artifact in self._all_artifacts(scene)],
            "operator": self.operator_status(scene, statuses),
        }

    def artifact_catalog_status(self) -> dict[str, Any]:
        self._sync_artifact_catalog()
        catalog = self._require_artifact_catalog()
        return catalog.status()

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
        summary_error: str | None = None
        cameras: tuple[dict[str, Any], ...] = ()
        if self.camera_summary_adapter is not None:
            try:
                cameras = self.camera_summary_adapter.summaries(camera_ids)
            except P08WorkflowError as error:
                summary_error = str(error)
        else:
            summary_error = "Final camera calibration summary is not configured"
        camera_ready = bool(cameras) and all(bool(camera.get("ready")) for camera in cameras)
        reconstruction_errors = (
            tuple(self.reconstruction_adapter.readiness_errors())
            if self.reconstruction_adapter is not None
            else ("Static reconstruction is not configured",)
        )
        reconstruction_errors += self._catalog_workflow_input_issues()
        artifacts = {artifact.artifact_id: artifact for artifact in self._all_artifacts(scene)}
        active_jobs = tuple(
            job
            for job in self.jobs.list()
            if job["state"] in {JobState.QUEUED.value, JobState.RUNNING.value}
            and job["action"]
            in {
                "all-camera-static-reconstruction",
                "floor-completion",
                "build-and-open-floor-preview",
            }
        )
        active_job = active_jobs[-1] if active_jobs else None
        pipeline_busy = active_job is not None
        reconstruction_running = (
            active_job is not None and active_job["action"] == "all-camera-static-reconstruction"
        )
        floor_running = active_job is not None and active_job["action"] in {
            "floor-completion",
            "build-and-open-floor-preview",
        }
        with self._workflow_lock:
            geometry_approved = self._geometry_approved
            floor_approved = self._floor_approved
            geometry_artifact_id = self._active_geometry_artifact_id
            floor_artifact_id = self._active_floor_artifact_id
            latest_approved_floor_artifact_id = self._latest_approved_floor_artifact_id
            current_floor_job_id = self._current_floor_job_id
            current_floor_output_directory = self._current_floor_output_directory
        geometry_available = geometry_artifact_id in artifacts
        floor_available = floor_artifact_id in artifacts
        latest_approved_floor_available = latest_approved_floor_artifact_id in artifacts
        floor_open_artifact_id = (
            floor_artifact_id if floor_available else latest_approved_floor_artifact_id
        )
        early_ready = all(
            phase_by_id[phase_id].state
            not in {PhaseState.UNAVAILABLE, PhaseState.FAILED, PhaseState.STALE}
            for phase_id in ("P02", "P03")
        )
        steps = (
            _operator_step(
                "setup", "Facility & cameras", "complete" if early_ready else "attention"
            ),
            _operator_step("capture", "Capture", "complete" if early_ready else "blocked"),
            _operator_step(
                "calibration", "Calibration & pose", "complete" if camera_ready else "attention"
            ),
            _operator_step(
                "reconstruction",
                "Static reconstruction",
                "running"
                if reconstruction_running
                else (
                    "approved"
                    if geometry_approved
                    else (
                        "ready_for_review"
                        if geometry_available
                        else ("ready" if camera_ready and not reconstruction_errors else "blocked")
                    )
                ),
            ),
            _operator_step(
                "floor",
                "Floor refinement",
                "running"
                if floor_running
                else (
                    "approved"
                    if floor_approved
                    else (
                        "ready_for_review"
                        if floor_available and geometry_approved
                        else ("ready" if geometry_approved else "blocked")
                    )
                ),
            ),
            _operator_step("results", "Final review", "complete" if floor_approved else "pending"),
        )
        return {
            "steps": list(steps),
            "cameras": list(cameras),
            "camera_summary_error": summary_error,
            "inputs_ready": camera_ready and not reconstruction_errors,
            "input_issues": list(reconstruction_errors)
            + ([summary_error] if summary_error else []),
            "session": {
                "can_start_fresh": not pipeline_busy,
                "has_activity": bool(self.jobs.list()),
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
                "approved": geometry_approved,
                "previewed": "geometry" in self._previewed_targets,
                "artifact_id": geometry_artifact_id,
                "can_run": camera_ready and not reconstruction_errors and not pipeline_busy,
                "can_preview": geometry_available and not pipeline_busy,
            },
            "floor": {
                "available": floor_available,
                "approved": floor_approved,
                "previewed": "floor" in self._previewed_targets,
                "artifact_id": floor_artifact_id,
                "open_artifact_id": floor_open_artifact_id,
                "opening_latest_approved": not floor_available and latest_approved_floor_available,
                "current_floor_job_id": current_floor_job_id,
                "can_generate": geometry_approved
                and self.floor_adapter is not None
                and not pipeline_busy,
                "can_build_preview": current_floor_job_id is not None
                and current_floor_output_directory is not None
                and not pipeline_busy,
                "can_preview": floor_open_artifact_id in artifacts and not pipeline_busy,
            },
        }

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
                state = record.state
                message = record.message
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
        floor_adapter = self.floor_adapter
        if floor_adapter is None:
            raise P08WorkflowError("floor processing adapter is not configured")
        with self._workflow_lock:
            self._require_pipeline_idle()
            if self.operator_config is not None and not self._geometry_approved:
                raise P08WorkflowError(
                    "Approve the reconstructed geometry before refining the floor"
                )
            geometry_source_path = self._geometry_source_path
            geometry_source_sha256 = self._geometry_source_sha256
            if (
                geometry_source_path is None
                or geometry_source_sha256 != floor_adapter.contract.selected_geometry_sha256
            ):
                raise P08WorkflowError(
                    "current reconstructed geometry does not match the floor source contract"
                )
            self.repository.ensure_run_id_available(job_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            result = floor_adapter.run(job_id, cancel_event)
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
            return self.jobs.submit(job_id, "P08", "floor-completion", operation)

    def start_reconstruction_job(self, job_id: str) -> dict[str, Any]:
        adapter = self.reconstruction_adapter
        if adapter is None:
            raise P08WorkflowError("static reconstruction is not configured")
        operator = self.operator_status()
        if not operator["inputs_ready"]:
            raise P08WorkflowError("Resolve every camera input before static reconstruction")
        with self._workflow_lock:
            self._require_pipeline_idle()
            self.repository.ensure_run_id_available(job_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            result = adapter.run(job_id, cancel_event)
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
            return self.jobs.submit(job_id, "P06", "all-camera-static-reconstruction", operation)

    def start_floor_preview_job(self, job_id: str, floor_job_id: str) -> dict[str, Any]:
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
            self.repository.ensure_run_id_available(job_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            preview = adapter.run(output, cancel_event)
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
            return self.jobs.submit(job_id, "P08", "build-and-open-floor-preview", operation)

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
        if target not in {"geometry", "floor"}:
            raise P08WorkflowError("approval target must be geometry or floor")
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
        return {
            "target": target,
            "status": "approved",
            "workflow_run_manifest": _identity(run_path),
        }

    def start_fresh_operator_session(self, action_id: str, session_id: str) -> dict[str, Any]:
        if self.operator_config is None:
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
            return (*scene.artifacts, *self._runtime_artifacts.values())

    def _require_artifact_catalog(self) -> SceneArtifactCatalog:
        if self._artifact_catalog is None:
            raise P08WorkflowError("artifact version catalog is not configured")
        return self._artifact_catalog

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
            selectable = artifact.kind not in {
                "da3-run-manifest",
                "geometry-adoption-manifest",
            }
            selected = selectable and self._catalog_artifact_is_selected(artifact, milestone_key)
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
        registrations = [item for item in scene.artifacts if item.kind == "facility-registration"]
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
            selected=not self._artifact_catalog_initialized,
            metadata={"selectable": True, "source_revision": payload.get("source_revision")},
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
            self._clear_geometry_and_floor_review_state(clear_geometry_artifact=False)
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
        self, *, clear_geometry_artifact: bool = True
    ) -> None:
        if clear_geometry_artifact:
            self._active_geometry_artifact_id = None
        self._active_floor_artifact_id = None
        self._current_floor_job_id = None
        self._current_floor_output_directory = None
        self._geometry_approved = False
        self._floor_approved = False
        self._previewed_targets.clear()

    def _require_pipeline_idle(self) -> None:
        if any(
            job["state"] in {JobState.QUEUED.value, JobState.RUNNING.value}
            and job["action"]
            in {
                "all-camera-static-reconstruction",
                "floor-completion",
                "build-and-open-floor-preview",
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


def _operator_step(step_id: str, title: str, state: str) -> dict[str, str]:
    return {"step_id": step_id, "title": title, "state": state}


def _required_mapping_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise P08WorkflowError(f"{key} must be a non-blank string")
    return result.strip()


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
