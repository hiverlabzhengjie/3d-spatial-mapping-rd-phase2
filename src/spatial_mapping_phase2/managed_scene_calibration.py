"""Generic scene-local P04/P05 provisioning for the combined console."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.managed_scene_evidence import ManagedSceneEvidenceCoordinator
from spatial_mapping_phase2.managed_scene_geocalib import (
    ManagedSceneGeoCalibConfig,
    ManagedSceneGeoCalibRunner,
)
from spatial_mapping_phase2.managed_scene_reconstruction import (
    ManagedSceneReconstructionInputBuilder,
)
from spatial_mapping_phase2.p04_calibration_domain import P04CalibrationError
from spatial_mapping_phase2.p04_calibration_service import (
    CapturedCandidate,
    P04CalibrationService,
)
from spatial_mapping_phase2.xr03_calibration_workflow import (
    IntegratedCalibrationWorkflowAdapter,
    SceneCalibrationRepository,
)
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy


class ManagedSceneCalibrationError(ValueError):
    """Raised when a managed scene cannot expose current calibration inputs."""


class ManagedScenePreviewCapturer:
    """Adapt the reloadable managed-scene preview to P04's candidate contract."""

    def __init__(self, capture_service: Any, camera_id: str) -> None:
        self.capture_service = capture_service
        self._camera_id = camera_id

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def capture(self) -> CapturedCandidate:
        try:
            frame = self.capture_service.preview(self.camera_id)
        except Exception as error:
            raise P04CalibrationError(
                f"{self.camera_id} live calibration capture failed ({type(error).__name__})"
            ) from error
        return CapturedCandidate(
            frame.content,
            frame.observed_at_utc,
            frame.source_pts,
            frame.source_time_base if frame.source_pts is not None else None,
            "scene-stream-profile-v1",
        )


class ManagedSceneCalibrationServiceProxy:
    """Keep one mounted P04 app while facility revisions create new workspaces."""

    def __init__(self, coordinator: ManagedSceneCalibrationCoordinator, camera_id: str) -> None:
        self.coordinator = coordinator
        self.camera_id = camera_id

    def __getattr__(self, name: str) -> Any:
        try:
            service = self.coordinator.current_service(self.camera_id)
        except ManagedSceneCalibrationError as error:
            raise P04CalibrationError(str(error)) from error
        return getattr(service, name)

    def has_state(self) -> bool:
        try:
            return self.coordinator.current_service(self.camera_id).has_state()
        except ManagedSceneCalibrationError:
            return False


class ManagedSceneCalibrationCoordinator:
    """Bind the active facility export to isolated calibration generations."""

    def __init__(
        self,
        evidence: ManagedSceneEvidenceCoordinator,
        capture_service: Any,
        project_id: str,
        scene_id: str,
        camera_ids: Sequence[str],
        root: Path,
        geocalib_config: ManagedSceneGeoCalibConfig,
    ) -> None:
        self.evidence = evidence
        self.capture_service = capture_service
        self.project_id = project_id
        self.scene_id = scene_id
        self.camera_ids = tuple(camera_ids)
        self.root = root.resolve()
        self.geocalib_config = geocalib_config
        self._services: dict[tuple[str, str], P04CalibrationService] = {}
        self._delegates: dict[str, IntegratedCalibrationWorkflowAdapter] = {}
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ManagedSceneCalibrationError("scene calibration requires a unique camera roster")

    def proxy(self, camera_id: str) -> ManagedSceneCalibrationServiceProxy:
        self._require_camera(camera_id)
        return ManagedSceneCalibrationServiceProxy(self, camera_id)

    def capturer(self, camera_id: str) -> ManagedScenePreviewCapturer:
        self._require_camera(camera_id)
        return ManagedScenePreviewCapturer(self.capture_service, camera_id)

    def current_service(self, camera_id: str) -> P04CalibrationService:
        self._require_camera(camera_id)
        generation, facility_path, plan_path = self._current_facility()
        key = (generation, camera_id)
        service = self._services.get(key)
        if service is None:
            service = P04CalibrationService(
                self.root
                / "g"
                / generation
                / "c"
                / hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:12]
            )
            if not service.has_state():
                service.initialize(facility_path, plan_path, camera_id)
            else:
                state = service.load_state()
                if state.camera_id != camera_id:
                    raise ManagedSceneCalibrationError(
                        f"{camera_id} calibration workspace has a different camera identity"
                    )
            self._services[key] = service
        return service

    def current_delegate(self) -> IntegratedCalibrationWorkflowAdapter:
        generation, facility_path, _ = self._current_facility()
        delegate = self._delegates.get(generation)
        if delegate is not None:
            return delegate
        services = {camera_id: self.current_service(camera_id) for camera_id in self.camera_ids}
        generation_root = self.root / "g" / generation
        delegate = IntegratedCalibrationWorkflowAdapter(
            SceneCalibrationRepository(
                generation_root / "calibration.sqlite3",
                self.project_id,
                self.scene_id,
                self.camera_ids,
            ),
            services,
            generation_root / "intrinsics" / "current-evidence.json",
            facility_path,
            generation_root / "pose-runs",
        )
        self._delegates[generation] = delegate
        return delegate

    def prepare_intrinsics(self) -> Path:
        generation, _, _ = self._current_facility()
        services = {camera_id: self.current_service(camera_id) for camera_id in self.camera_ids}
        runner = ManagedSceneGeoCalibRunner(
            self.geocalib_config,
            self.root / "g" / generation / "intrinsics",
        )
        return runner.prepare(self.camera_ids, services)

    def history(self) -> dict[str, Any]:
        try:
            generation, _, _ = self._current_facility()
            history = self.current_delegate().history()
        except ManagedSceneCalibrationError:
            generation = None
            history = {
                "schema_version": "xr03-calibration-history-v1",
                "intrinsic_batches": [],
                "attempts": [],
                "decisions": [],
            }
        return {
            **history,
            "managed_scene_generation": generation,
            "retained_generation_count": len(tuple((self.root / "g").glob("*"))),
        }

    def pending_status(self, reason: str) -> dict[str, Any]:
        return {
            "schema_version": "xr03-calibration-workflow-status-v1",
            "intrinsics_ready": False,
            "intrinsic_batch": None,
            "cameras": [
                {
                    "camera_id": camera_id,
                    "ready": False,
                    "readiness": "pending",
                    "warning": None,
                    "input": {
                        "camera_id": camera_id,
                        "calibrate_ready": False,
                        "reason": reason,
                        "solve_count": 0,
                        "d034_validation_count": 0,
                        "current_export_ready": False,
                    },
                    "assignment": None,
                    "attempt": None,
                    "decision": None,
                    "calibrate_enabled": False,
                }
                for camera_id in self.camera_ids
            ],
            "all_cameras_ready": False,
            "issues": [reason],
            "warnings": [],
        }

    def _current_facility(self) -> tuple[str, Path, Path]:
        status = self.evidence.facility_status()
        if not status.get("ready"):
            issues = status.get("issues")
            reason = (
                str(issues[0]) if isinstance(issues, list) and issues else "Facility not ready"
            )
            raise ManagedSceneCalibrationError(reason)
        export = status.get("current_export")
        plan = status.get("plan_image")
        if not isinstance(export, dict) or not isinstance(plan, dict):
            raise ManagedSceneCalibrationError("Current facility evidence is incomplete")
        sha256 = str(export.get("sha256", ""))
        if len(sha256) != 64:
            raise ManagedSceneCalibrationError("Current facility evidence identity is malformed")
        return f"f-{sha256[:12]}", Path(str(export["path"])), Path(str(plan["path"]))

    def _require_camera(self, camera_id: str) -> None:
        if camera_id not in self.camera_ids:
            raise ManagedSceneCalibrationError("camera is outside the managed-scene roster")


class ManagedSceneCalibrationAdapter:
    """Present the existing integrated calibration API over a dynamic scene generation."""

    def __init__(
        self,
        coordinator: ManagedSceneCalibrationCoordinator,
        reconstruction_inputs: ManagedSceneReconstructionInputBuilder | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.reconstruction_inputs = reconstruction_inputs

    def history(self) -> dict[str, Any]:
        return self.coordinator.history()

    def status(self, policy: SceneCameraPolicy) -> dict[str, Any]:
        try:
            return self.coordinator.current_delegate().status(policy)
        except ManagedSceneCalibrationError as error:
            return self.coordinator.pending_status(str(error))

    def determine_intrinsics(self, policy: SceneCameraPolicy) -> dict[str, Any]:
        self.coordinator.prepare_intrinsics()
        return self.coordinator.current_delegate().determine_intrinsics(policy)

    def calibrate_camera(self, camera_id: str, policy: SceneCameraPolicy) -> dict[str, Any]:
        return self.coordinator.current_delegate().calibrate_camera(camera_id, policy)

    def review_camera(
        self, camera_id: str, attempt_sha256: str, policy: SceneCameraPolicy
    ) -> dict[str, Any]:
        return self.coordinator.current_delegate().review_camera(camera_id, attempt_sha256, policy)

    def override_camera(
        self,
        camera_id: str,
        attempt_sha256: str,
        reason: str,
        acknowledged: bool,
        policy: SceneCameraPolicy,
    ) -> dict[str, Any]:
        return self.coordinator.current_delegate().override_camera(
            camera_id, attempt_sha256, reason, acknowledged, policy
        )

    def prepare_reconstruction_inputs(
        self,
        policy: SceneCameraPolicy,
        baseline_directory: Path | None,
        output_directory: Path,
    ) -> Path:
        delegate = self.coordinator.current_delegate()
        if not delegate.status(policy)["all_cameras_ready"]:
            raise ManagedSceneCalibrationError(
                "every enabled camera must be reviewed or explicitly overridden"
            )
        if self.reconstruction_inputs is not None:
            baseline_directory = self.reconstruction_inputs.build(
                policy,
                output_directory.parent.parent
                / "source-reconstruction-inputs"
                / output_directory.name,
            )
        if baseline_directory is None:
            raise ManagedSceneCalibrationError("reconstruction source inputs are not configured")
        return delegate.prepare_reconstruction_inputs(policy, baseline_directory, output_directory)


def facility_generation_sha256(path: Path) -> str:
    """Return the exact facility identity used by tests and diagnostic tooling."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_generation_manifest(root: Path) -> dict[str, Any]:
    """Summarize retained generations without exposing calibration media."""

    generations = sorted(path.name for path in (root / "g").glob("f-*"))
    return {
        "schema_version": "managed-scene-calibration-generations-v1",
        "generations": generations,
        "generation_count": len(generations),
    }
