"""Scene-local evidence coordination for managed Facility and Capture tools.

The two tools intentionally own separate append-only stores.  This module gives the combined
workflow a read-only, hash-checked view of their current outputs without promoting calibration,
geometry, or any Office-scene authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.managed_scene_capture import (
    SceneBundleStatus,
    SceneCaptureError,
    SceneCaptureRepository,
    SceneCaptureStatus,
)
from spatial_mapping_phase2.p08_workflow import (
    ArtifactReference,
    PhaseRecord,
    PhaseState,
    SceneWorkspace,
    WorkspaceReference,
)


class ManagedSceneEvidenceError(ValueError):
    """Raised when retained managed-scene evidence is malformed or inconsistent."""


class ManagedSceneEvidenceCoordinator:
    """Resolve current managed-scene inputs from their independent local stores."""

    def __init__(
        self,
        facility_workspace: Path,
        capture_repository: SceneCaptureRepository,
        scene_id: str,
        camera_ids: Sequence[str],
        camera_names: Mapping[str, str] | None = None,
    ) -> None:
        self.facility_workspace = facility_workspace.resolve()
        self.facility_selection_path = self.facility_workspace / "current-export.json"
        self.capture_repository = capture_repository
        self.scene_id = scene_id
        self.camera_ids = tuple(camera_ids)
        self.camera_names = dict(camera_names or {})
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ManagedSceneEvidenceError("managed scene requires a unique camera roster")

    @property
    def allowed_artifact_roots(self) -> tuple[Path, ...]:
        return (self.facility_workspace, self.capture_repository.root)

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": "managed-scene-evidence-status-v1",
            "facility": self.facility_status(),
            "capture": self.capture_status(),
        }

    def facility_status(self) -> dict[str, Any]:
        exports = tuple(
            sorted(
                (path.resolve() for path in (self.facility_workspace / "exports").glob("*.json")),
                key=lambda path: (path.stat().st_mtime_ns, str(path).lower()),
            )
        )
        if not exports:
            return {
                "ready": False,
                "issues": ["Export a Facility & cameras snapshot for this scene."],
                "current_export": None,
                "plan_image": None,
            }
        path = exports[-1]
        if (
            self.facility_selection_path.is_file()
            and self.facility_selection_path.stat().st_mtime_ns >= path.stat().st_mtime_ns
        ):
            try:
                selection = _read_json(self.facility_selection_path, "facility selection")
                if selection.get("schema_version") != "managed-scene-facility-selection-v1":
                    raise ManagedSceneEvidenceError(
                        "The current facility selection has an unsupported schema."
                    )
                relative_path = selection.get("export_relative_path")
                expected_sha256 = selection.get("export_sha256")
                if not isinstance(relative_path, str) or not _is_sha256(expected_sha256):
                    raise ManagedSceneEvidenceError("The current facility selection is malformed.")
                selected_path = (self.facility_workspace / relative_path).resolve()
                if (
                    not selected_path.is_relative_to(
                        (self.facility_workspace / "exports").resolve()
                    )
                    or _sha256(selected_path) != expected_sha256
                ):
                    raise ManagedSceneEvidenceError(
                        "The current facility selection is missing or changed."
                    )
                path = selected_path
            except (OSError, ManagedSceneEvidenceError) as error:
                return {
                    "ready": False,
                    "issues": [str(error)],
                    "current_export": None,
                    "plan_image": None,
                }
        issues: list[str] = []
        try:
            payload = _read_json(path, "facility export")
            if payload.get("schema_version") != "p02-interactive-export-v1":
                issues.append("The current facility export has an unsupported schema.")
            revision = payload.get("source_revision")
            if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
                issues.append("The current facility export revision is malformed.")
            priors = payload.get("camera_mounting_priors")
            prior_ids = (
                tuple(str(item.get("camera_id")) for item in priors if isinstance(item, dict))
                if isinstance(priors, list)
                else ()
            )
            if prior_ids != self.camera_ids:
                issues.append("The facility camera roster differs from this scene.")
            plan = payload.get("plan")
            source_sha256 = plan.get("source_sha256") if isinstance(plan, dict) else None
            width = plan.get("image_width_pixels") if isinstance(plan, dict) else None
            height = plan.get("image_height_pixels") if isinstance(plan, dict) else None
            if (
                not _is_sha256(source_sha256)
                or not _positive_integer(width)
                or not (_positive_integer(height))
            ):
                issues.append("The facility plan identity or rendered dimensions are malformed.")
            facility_frame = payload.get("facility_frame")
            if not isinstance(facility_frame, dict) or not isinstance(
                facility_frame.get("T_world_from_plan_display_pixel"), list
            ):
                issues.append("The facility world-frame transform is missing.")
            display_matches = (
                tuple(
                    sorted(
                        (self.facility_workspace / "display").glob(f"{source_sha256}-page-*.png")
                    )
                )
                if _is_sha256(source_sha256)
                else ()
            )
            if len(display_matches) != 1:
                issues.append("The hash-bound rendered floor-plan image is missing or ambiguous.")
            plan_image = display_matches[0].resolve() if len(display_matches) == 1 else None
        except ManagedSceneEvidenceError as error:
            payload = {}
            revision = None
            plan_image = None
            issues.append(str(error))
        return {
            "ready": not issues,
            "issues": issues,
            "source_revision": revision,
            "current_export": _identity(path),
            "plan_image": None if plan_image is None else _identity(plan_image),
        }

    def capture_status(self) -> dict[str, Any]:
        try:
            current = self.capture_repository.current_bundle()
        except SceneCaptureError as error:
            return {
                "ready": False,
                "issues": [str(error)],
                "current_bundle": None,
                "selection_source": None,
                "missing_camera_ids": list(self.camera_ids),
                "session_count": len(self.capture_repository.list_sessions()),
                "bundle_count": len(self.capture_repository.list_bundle_paths()),
            }
        if current is None:
            return {
                "ready": False,
                "issues": ["Capture every scene camera and select one complete bundle."],
                "current_bundle": None,
                "selection_source": None,
                "missing_camera_ids": list(self.camera_ids),
                "session_count": len(self.capture_repository.list_sessions()),
                "bundle_count": 0,
            }
        path, bundle, selection_source = current
        issues: list[str] = []
        if bundle.scene_id != self.scene_id:
            issues.append("The current capture bundle belongs to another scene.")
        try:
            session = self.capture_repository.read_session(bundle.source_session_id)
        except SceneCaptureError as error:
            session = None
            issues.append(str(error))
        if session is not None:
            roster_ids = tuple(binding.camera_id for binding in session.camera_roster)
            if session.scene_id != self.scene_id or roster_ids != self.camera_ids:
                issues.append("The capture session roster differs from this scene.")
            result_by_camera = {result.camera_id: result for result in session.results}
            for selected in bundle.selected_frames:
                result = result_by_camera.get(selected.camera_id)
                if result is None or result.status is not SceneCaptureStatus.CAPTURED:
                    issues.append(f"{self._label(selected.camera_id)} has no captured source.")
                    continue
                if selected.profile_version != result.profile.profile_version or not any(
                    frame.frame_id == selected.frame_id for frame in result.frames
                ):
                    issues.append(
                        f"{self._label(selected.camera_id)} selected-frame identity is stale."
                    )
                artifact = result.artifact
                if artifact is None:
                    issues.append(
                        f"{self._label(selected.camera_id)} capture artifact is missing."
                    )
                    continue
                artifact_path = (self.capture_repository.root / artifact.relative_path).resolve()
                if not artifact_path.is_relative_to(self.capture_repository.root):
                    issues.append(
                        f"{self._label(selected.camera_id)} capture artifact escaped "
                        "scene storage."
                    )
                elif not artifact_path.is_file() or _sha256(artifact_path) != artifact.sha256:
                    issues.append(
                        f"{self._label(selected.camera_id)} capture artifact is missing "
                        "or changed."
                    )
        selected_ids = tuple(frame.camera_id for frame in bundle.selected_frames)
        missing = tuple(
            camera_id for camera_id in self.camera_ids if camera_id not in selected_ids
        )
        if bundle.status is not SceneBundleStatus.COMPLETE or missing:
            labels = ", ".join(self._label(camera_id) for camera_id in missing)
            issues.append(
                f"The current bundle is incomplete{f'; missing {labels}' if labels else ''}."
            )
        if selected_ids != self.camera_ids and not missing:
            issues.append("The current bundle camera order differs from this scene.")
        return {
            "ready": not issues,
            "issues": list(dict.fromkeys(issues)),
            "current_bundle": {
                **_identity(path),
                "bundle_id": bundle.bundle_id,
                "source_session_id": bundle.source_session_id,
                "status": bundle.status.value,
            },
            "selection_source": selection_source,
            "missing_camera_ids": list(missing),
            "session_count": len(self.capture_repository.list_sessions()),
            "bundle_count": len(self.capture_repository.list_bundle_paths()),
        }

    def artifacts(self) -> tuple[ArtifactReference, ...]:
        artifacts: list[ArtifactReference] = []
        facility = self.facility_status()
        facility_identity = facility.get("current_export")
        if facility.get("ready") and isinstance(facility_identity, dict):
            sha256 = str(facility_identity["sha256"])
            artifacts.append(
                ArtifactReference(
                    f"managed-facility-{sha256[:12]}",
                    "P02",
                    "facility-registration",
                    Path(str(facility_identity["path"])),
                    sha256,
                    "managed scene-local facility export",
                    True,
                )
            )
        capture = self.capture_status()
        capture_identity = capture.get("current_bundle")
        if isinstance(capture_identity, dict):
            sha256 = str(capture_identity["sha256"])
            artifacts.append(
                ArtifactReference(
                    f"managed-capture-{sha256[:12]}",
                    "P03",
                    "capture-bundle",
                    Path(str(capture_identity["path"])),
                    sha256,
                    "managed scene-local raw capture evidence",
                    True,
                )
            )
        return tuple(artifacts)

    def artifact_metadata(self, artifact: ArtifactReference) -> dict[str, Any]:
        if artifact.kind == "facility-registration":
            return {"selectable": True, "managed_scene_input": True}
        if artifact.kind == "capture-bundle":
            capture = self.capture_status()
            return {
                "selectable": bool(capture.get("ready")),
                "managed_scene_input": True,
                "valid_for_downstream": bool(capture.get("ready")),
                "missing_camera_ids": list(capture.get("missing_camera_ids", ())),
            }
        return {}

    def select_artifact(self, artifact: ArtifactReference) -> None:
        """Synchronize a Scene History selection back to the owning input store."""

        if artifact.kind == "capture-bundle":
            self.capture_repository.select_existing_bundle(artifact.path, artifact.sha256)
            return
        if artifact.kind != "facility-registration":
            return
        resolved = artifact.path.resolve()
        exports_root = (self.facility_workspace / "exports").resolve()
        if not resolved.is_relative_to(exports_root) or _sha256(resolved) != artifact.sha256:
            raise ManagedSceneEvidenceError("selected facility export is missing or changed")
        payload = {
            "schema_version": "managed-scene-facility-selection-v1",
            "export_relative_path": resolved.relative_to(self.facility_workspace).as_posix(),
            "export_sha256": artifact.sha256,
        }
        _write_json_atomic(self.facility_selection_path, payload)

    def phase_adapter(self, phase_id: str) -> ManagedSceneEvidencePhaseAdapter:
        if phase_id not in {"P02", "P03"}:
            raise ManagedSceneEvidenceError("managed evidence adapter supports only P02 and P03")
        return ManagedSceneEvidencePhaseAdapter(self, phase_id)

    def _label(self, camera_id: str) -> str:
        return self.camera_names.get(camera_id, camera_id)


class ManagedSceneEvidencePhaseAdapter:
    """Expose managed inputs through the existing phase-inspection contract."""

    def __init__(self, coordinator: ManagedSceneEvidenceCoordinator, phase_id: str) -> None:
        self.coordinator = coordinator
        self.phase_id = phase_id

    def inspect(
        self,
        scene: SceneWorkspace,
        record: PhaseRecord,
        workspace_by_id: Mapping[str, WorkspaceReference],
        artifact_by_id: Mapping[str, ArtifactReference],
    ) -> tuple[str, ...]:
        del scene, record, workspace_by_id, artifact_by_id
        status = (
            self.coordinator.facility_status()
            if self.phase_id == "P02"
            else self.coordinator.capture_status()
        )
        return tuple(str(item) for item in status["issues"])

    def resolved_state(self) -> PhaseState:
        return PhaseState.READY

    def resolved_message(self) -> str:
        return (
            "The current facility snapshot is verified."
            if self.phase_id == "P02"
            else "The current complete camera bundle is verified."
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManagedSceneEvidenceError(f"The current {label} is unreadable.") from error
    if not isinstance(value, dict):
        raise ManagedSceneEvidenceError(f"The current {label} must be an object.")
    return value


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
