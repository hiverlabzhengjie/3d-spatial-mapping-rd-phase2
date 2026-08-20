"""Shared P02-P08 scene, phase, artifact, job, and safe-launch contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

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
                    workspace_reference_ids=_string_tuple(
                        item, "workspace_reference_ids"
                    ),
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
                job.state in {JobState.QUEUED, JobState.RUNNING}
                for job in self._jobs.values()
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
        self.viewer_executable = viewer_executable.resolve()
        self.allowed_artifact_roots = tuple(path.resolve() for path in allowed_artifact_roots)
        if not self.viewer_executable.is_file():
            raise P08WorkflowError("configured Rerun viewer executable is missing")
        if not self.allowed_artifact_roots:
            raise P08WorkflowError("at least one allowed Rerun artifact root is required")
        self._launch = launch or self._default_launch

    def launch_selected(
        self, scene: SceneWorkspace, artifact_id: str
    ) -> dict[str, Any]:
        artifacts = {artifact.artifact_id: artifact for artifact in scene.artifacts}
        try:
            artifact = artifacts[artifact_id]
        except KeyError as error:
            raise P08WorkflowError("unknown Rerun artifact_id") from error
        if artifact.kind != "rerun-recording" or not artifact.selected:
            raise P08WorkflowError("Rerun launch requires a selected recording artifact")
        path = artifact.path.resolve()
        if path.suffix.lower() != ".rrd" or not path.is_file():
            raise P08WorkflowError("selected Rerun artifact must be an existing .rrd file")
        if not any(path.is_relative_to(root) for root in self.allowed_artifact_roots):
            raise P08WorkflowError("selected Rerun artifact is outside configured allowed roots")
        if _sha256(path) != artifact.sha256:
            raise P08WorkflowError("selected Rerun artifact identity is stale")
        process = self._launch((str(self.viewer_executable), str(path)))
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
        return subprocess.Popen(list(arguments), shell=False, close_fds=True)


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
        if cancel_event.is_set():
            raise P08WorkflowError("floor job cancelled after artifact generation")
        manifest = result.get("manifest")
        if not isinstance(manifest, dict):
            raise P08WorkflowError("floor artifact service returned no manifest identity")
        return {
            "output_directory": str(output),
            "manifest": manifest,
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
            "artifacts": [artifact.to_dict() for artifact in scene.artifacts],
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
        self.repository.ensure_run_id_available(job_id)

        def operation(cancel_event: threading.Event) -> Mapping[str, Any]:
            result = floor_adapter.run(job_id, cancel_event)
            run_path = self.repository.write_run_manifest(
                job_id,
                {
                    "phase_id": "P08",
                    "action": "floor-completion",
                    "result": result,
                    "immutable": True,
                },
            )
            return {**result, "workflow_run_manifest": _identity(run_path)}

        return self.jobs.submit(job_id, "P08", "floor-completion", operation)

    def launch_rerun(self, action_id: str, artifact_id: str) -> dict[str, Any]:
        if self.rerun_launcher is None:
            raise P08WorkflowError("Rerun launcher is not configured")
        self.repository.ensure_run_id_available(action_id)
        result = self.rerun_launcher.launch_selected(self.repository.load(), artifact_id)
        run_path = self.repository.write_run_manifest(
            action_id,
            {
                "phase_id": "P08",
                "action": "launch-selected-rerun",
                "result": result,
                "immutable": True,
            },
        )
        return {**result, "workflow_run_manifest": _identity(run_path)}


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
