"""Multi-scene entry layer and scene-scoped ASGI dispatch for the combined console."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from spatial_mapping_phase2.managed_scene_calibration import (
    ManagedSceneCalibrationAdapter,
    ManagedSceneCalibrationCoordinator,
)
from spatial_mapping_phase2.managed_scene_capture import create_scene_capture_app
from spatial_mapping_phase2.managed_scene_downstream import (
    ManagedFreshFrameProvider,
    ManagedSceneLiveConfig,
    ManagedSceneUpdatePipelineAdapter,
    ManagedSceneXR02Worker,
)
from spatial_mapping_phase2.managed_scene_evidence import ManagedSceneEvidenceCoordinator
from spatial_mapping_phase2.managed_scene_geocalib import ManagedSceneGeoCalibConfig
from spatial_mapping_phase2.managed_scene_reconstruction import (
    ManagedSceneReconstructionConfig,
    ManagedSceneReconstructionInputBuilder,
)
from spatial_mapping_phase2.p02_registration_web import create_p02_registration_app
from spatial_mapping_phase2.p04_calibration_web import create_p04_calibration_app
from spatial_mapping_phase2.p08_floor import FloorProcessingConfig
from spatial_mapping_phase2.p08_operator_workflow import (
    FloorPreviewWorkflowAdapter,
    ReconstructionWorkflowAdapter,
)
from spatial_mapping_phase2.p08_workflow import (
    BoundedJobManager,
    FloorWorkflowAdapter,
    P08WorkflowError,
    SafeRerunLauncher,
    SceneWorkspaceRepository,
    WorkflowService,
)
from spatial_mapping_phase2.p08_workflow_web import create_p08_workflow_app
from spatial_mapping_phase2.xr03_scene_management import (
    SceneReadiness,
    SceneRecord,
    SceneRegistry,
)


@dataclass
class SceneRuntime:
    service: WorkflowService
    legacy_apps: Mapping[str, FastAPI]
    close_callback: Callable[[], None] | None = None

    def close(self) -> None:
        if self.close_callback is not None:
            self.close_callback()
        self.service.close()


class SceneRuntimeManager:
    """Lazily owns one isolated service graph per registered scene."""

    def __init__(
        self,
        registry: SceneRegistry,
        default_scene_uuid: str,
        managed_geocalib_config: ManagedSceneGeoCalibConfig | None = None,
        managed_reconstruction_config: ManagedSceneReconstructionConfig | None = None,
        managed_rerun_launcher: SafeRerunLauncher | None = None,
        managed_live_config: ManagedSceneLiveConfig | None = None,
    ) -> None:
        self.registry = registry
        self.default_scene_uuid = default_scene_uuid
        self.managed_geocalib_config = managed_geocalib_config
        self.managed_reconstruction_config = managed_reconstruction_config
        self.managed_rerun_launcher = managed_rerun_launcher
        self.managed_live_config = managed_live_config
        self._next_managed_live_port = (
            None if managed_live_config is None else managed_live_config.port
        )
        self._runtimes: dict[str, SceneRuntime] = {}
        self._apps: dict[str, ASGIApp] = {}
        self._lock = threading.RLock()

    def register(self, scene_uuid: str, runtime: SceneRuntime) -> None:
        self.registry.require(scene_uuid)
        with self._lock:
            if scene_uuid in self._runtimes:
                raise P08WorkflowError("scene runtime is already registered")
            runtime.service.enable_resource_coordination(
                self.registry,
                scene_uuid,
                lambda: self.registry.require(scene_uuid, include_archived=True).display_name,
            )
            self._runtimes[scene_uuid] = runtime

    def runtime(self, scene_uuid: str) -> SceneRuntime:
        record = self.registry.require(scene_uuid)
        with self._lock:
            runtime = self._runtimes.get(scene_uuid)
            if runtime is None:
                runtime = self._build_draft_runtime(record)
                runtime.service.enable_resource_coordination(
                    self.registry,
                    scene_uuid,
                    lambda: self.registry.require(scene_uuid, include_archived=True).display_name,
                )
                self._runtimes[scene_uuid] = runtime
            return runtime

    def app(self, scene_uuid: str) -> ASGIApp:
        with self._lock:
            app = self._apps.get(scene_uuid)
            if app is not None:
                return app
            runtime = self.runtime(scene_uuid)
            app = create_p08_workflow_app(
                runtime.service,
                runtime.legacy_apps,
                public_prefix=f"/scenes/{scene_uuid}",
            )
            self._apps[scene_uuid] = app
            return app

    def close_scene(self, scene_uuid: str) -> None:
        with self._lock:
            self._apps.pop(scene_uuid, None)
            runtime = self._runtimes.pop(scene_uuid, None)
        if runtime is not None:
            runtime.close()

    def close(self) -> None:
        with self._lock:
            runtimes = tuple(self._runtimes.values())
            self._runtimes.clear()
            self._apps.clear()
        for runtime in runtimes:
            runtime.close()

    def _build_draft_runtime(self, record: SceneRecord) -> SceneRuntime:
        """Build isolated managed-scene capture, calibration and reconstruction tools.

        The registered Office runtime remains an explicit legacy graph.  A managed scene instead
        receives a variable-roster capture service bound only to its own secret file and artifact
        root.  The capture service resolves endpoint values lazily, so saving a P02 endpoint does
        not require rebuilding this cached scene runtime.
        """

        repository = SceneWorkspaceRepository(record.workspace_root)
        scene = repository.load()
        endpoint_keys = {
            camera.camera_id: camera.endpoint_environment_key
            for camera in scene.cameras
            if camera.endpoint_environment_key is not None
        }
        if len(endpoint_keys) != len(scene.cameras):
            raise P08WorkflowError("managed scene camera endpoint bindings are incomplete")
        registration_workspace = record.workspace_root.parent / "facility-registration"
        secret_file = record.workspace_root.parent / "secrets.env"
        facility = create_p02_registration_app(
            registration_workspace,
            secret_file,
            camera_endpoint_keys=endpoint_keys,
        )
        capture = create_scene_capture_app(
            endpoint_keys,
            secret_file,
            scene.artifact_root,
            scene.scene_id,
        )
        evidence = ManagedSceneEvidenceCoordinator(
            registration_workspace,
            capture.state.scene_capture_service.repository,
            scene.scene_id,
            tuple(camera.camera_id for camera in scene.cameras if camera.enabled),
            {camera.camera_id: camera.display_name for camera in scene.cameras if camera.enabled},
        )
        calibration_adapter: Any | None = None
        reconstruction_adapter: ReconstructionWorkflowAdapter | None = None
        floor_adapter: FloorWorkflowAdapter | None = None
        floor_preview_adapter: FloorPreviewWorkflowAdapter | None = None
        managed_rerun_launcher = None
        calibration_apps: dict[str, FastAPI] = {}
        if self.managed_geocalib_config is not None:
            calibration = ManagedSceneCalibrationCoordinator(
                evidence,
                capture.state.scene_capture_service,
                scene.project_id,
                scene.scene_id,
                tuple(camera.camera_id for camera in scene.cameras if camera.enabled),
                record.workspace_root.parent / "calibration",
                self.managed_geocalib_config,
            )
            reconstruction_inputs = ManagedSceneReconstructionInputBuilder(
                capture.state.scene_capture_service.repository,
                scene.scene_id,
                tuple(camera.camera_id for camera in scene.cameras if camera.enabled),
            )
            calibration_adapter = ManagedSceneCalibrationAdapter(
                calibration, reconstruction_inputs
            )
            for camera in scene.cameras:
                if not camera.enabled:
                    continue
                camera_id = camera.camera_id
                calibration_apps[f"calibration-{camera_id}"] = create_p04_calibration_app(
                    candidate_capturer=calibration.capturer(camera_id),
                    service=calibration.proxy(camera_id),
                )
        if self.managed_reconstruction_config is not None:
            config = self.managed_reconstruction_config
            reconstruction_adapter = ReconstructionWorkflowAdapter(
                python_executable=config.python_executable,
                repository_root=config.repository_root,
                p06_run_directory=None,
                source_directory=config.source_directory,
                checkpoint_directory=config.checkpoint_directory,
                d041_manifest_path=None,
                output_root=scene.artifact_root / "reconstruction",
                process_resolution=config.process_resolution,
                input_readiness=lambda: _managed_capture_issues(evidence),
            )
            floor_adapter = FloorWorkflowAdapter(
                contract=None,
                config=FloorProcessingConfig(),
                output_root=scene.artifact_root / "floor",
            )
            if self.managed_rerun_launcher is not None:
                managed_rerun_launcher = self.managed_rerun_launcher.with_allowed_root(
                    scene.artifact_root
                )
                floor_preview_adapter = FloorPreviewWorkflowAdapter(
                    python_executable=config.python_executable,
                    repository_root=config.repository_root,
                    floor_contract_path=None,
                )
        service = WorkflowService(
            repository,
            BoundedJobManager(),
            adapters={
                "P02": evidence.phase_adapter("P02"),
                "P03": evidence.phase_adapter("P03"),
            },
            operator_surface_ids=frozenset({"facility", "capture"}),
            scene_evidence_adapter=evidence,
            calibration_adapter=calibration_adapter,
            reconstruction_adapter=reconstruction_adapter,
            floor_adapter=floor_adapter,
            floor_preview_adapter=floor_preview_adapter,
            rerun_launcher=managed_rerun_launcher,
        )
        if (
            reconstruction_adapter is not None
            and floor_adapter is not None
            and floor_preview_adapter is not None
        ):
            service.enable_scene_updates(
                ManagedSceneUpdatePipelineAdapter(
                    repository=repository,
                    frame_provider=ManagedFreshFrameProvider(capture.state.scene_capture_service),
                    camera_ids=tuple(
                        camera.camera_id for camera in scene.cameras if camera.enabled
                    ),
                    input_output_root=scene.artifact_root / "scene-update-inputs",
                    reconstruction=reconstruction_adapter,
                    floor=floor_adapter,
                    floor_preview=floor_preview_adapter,
                    camera_policy_provider=service.camera_policy_status,
                )
            )
        if self.managed_live_config is not None:
            if self._next_managed_live_port is None:
                raise P08WorkflowError("managed Live operations port allocation is unavailable")
            live_config = replace(
                self.managed_live_config,
                port=self._next_managed_live_port,
            )
            self._next_managed_live_port += 1
            service.enable_live_operations(
                ManagedSceneXR02Worker(
                    live_config,
                    repository,
                    endpoint_keys,
                    secret_file,
                    scene.artifact_root / "live",
                )
            )
        return SceneRuntime(
            service,
            {"facility": facility, "capture": capture, **calibration_apps},
            close_callback=capture.state.scene_capture_service.close,
        )


class MultiSceneConsoleApp:
    """Dispatch explicit scene URLs to isolated per-scene FastAPI applications."""

    def __init__(self, manager: SceneRuntimeManager) -> None:
        self.manager = manager
        self.entry_app = _create_entry_app(manager)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.entry_app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        routed = _scene_route(path)
        if routed is None and path.startswith("/api/") and not path.startswith("/api/scenes"):
            # Keep existing local health checks and engineering helpers working. New operator
            # navigation always emits explicit scene-scoped URLs.
            routed = (self.manager.default_scene_uuid, path)
        if routed is None:
            await self.entry_app(scope, receive, send)
            return
        scene_uuid, inner_path = routed
        try:
            scene_app = self.manager.app(scene_uuid)
            if inner_path == "/":
                self.manager.registry.mark_opened(scene_uuid)
        except P08WorkflowError as error:
            response = JSONResponse({"detail": str(error)}, status_code=404)
            await response(scope, receive, send)
            return
        child_scope = dict(scope)
        child_scope["path"] = inner_path
        child_scope["raw_path"] = inner_path.encode("utf-8")
        # Routes are matched against the stripped path. Public URLs are supplied explicitly
        # when the per-scene application is built, so nested Starlette mounts stay stable.
        child_scope["root_path"] = str(scope.get("root_path", ""))
        await scene_app(child_scope, receive, send)


def create_scene_console_app(manager: SceneRuntimeManager) -> MultiSceneConsoleApp:
    return MultiSceneConsoleApp(manager)


def _managed_capture_issues(
    evidence: ManagedSceneEvidenceCoordinator,
) -> tuple[str, ...]:
    status = evidence.capture_status()
    return () if status["ready"] else tuple(str(item) for item in status["issues"])


def _create_entry_app(manager: SceneRuntimeManager) -> FastAPI:
    app = FastAPI(title="Spatial Mapping Scenes", docs_url=None, redoc_url=None)

    @app.exception_handler(P08WorkflowError)
    async def scene_error(_request: Request, error: P08WorkflowError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/scenes", response_class=HTMLResponse)
    async def scene_home() -> HTMLResponse:
        return HTMLResponse(_scene_asset("index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/scene-assets/{asset_name}")
    async def scene_asset(asset_name: str) -> Response:
        types = {
            "app.js": "application/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }
        if asset_name not in types:
            return JSONResponse({"detail": "asset not found"}, status_code=404)
        return Response(
            _scene_asset(asset_name),
            media_type=types[asset_name],
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/{asset_name}")
    async def workflow_asset(asset_name: str) -> Response:
        types = {
            "app.js": "application/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }
        if asset_name not in types:
            return JSONResponse({"detail": "asset not found"}, status_code=404)
        value = (
            files("spatial_mapping_phase2.p08_web")
            .joinpath(asset_name)
            .read_text(encoding="utf-8")
        )
        return Response(value, media_type=types[asset_name], headers={"Cache-Control": "no-store"})

    @app.get("/api/scenes")
    async def scenes(include_archived: bool = False) -> dict[str, Any]:
        return {
            "schema_version": "xr03-scene-list-v1",
            "scenes": [
                record.to_dict()
                for record in manager.registry.list_scenes(include_archived=include_archived)
            ],
            "default_scene_uuid": manager.default_scene_uuid,
            "resource": manager.registry.resource_status(),
        }

    @app.post("/api/scenes")
    async def create_scene(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        display_name = _required_string(payload, "display_name")
        camera_names = payload.get("camera_names")
        if not isinstance(camera_names, list) or not all(
            isinstance(value, str) for value in camera_names
        ):
            raise P08WorkflowError("camera_names must be a list of names")
        record = manager.registry.create_scene(display_name, camera_names)
        return {"scene": record.to_dict(), "open_url": f"/scenes/{record.scene_uuid}/"}

    @app.post("/api/scenes/register")
    async def register_scene(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        from pathlib import Path

        record = manager.registry.register_existing(
            Path(_required_string(payload, "workspace_root")),
            readiness=SceneReadiness.DRAFT,
        )
        return {"scene": record.to_dict(), "open_url": f"/scenes/{record.scene_uuid}/"}

    @app.patch("/api/scenes/{scene_uuid}")
    async def update_scene(scene_uuid: str, request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        expected_revision = _required_integer(payload, "expected_revision")
        if "display_name" in payload:
            record = manager.registry.rename(
                scene_uuid,
                _required_string(payload, "display_name"),
                expected_revision,
            )
        elif "archived" in payload:
            archived = payload["archived"]
            if not isinstance(archived, bool):
                raise P08WorkflowError("archived must be boolean")
            record = manager.registry.set_archived(scene_uuid, archived, expected_revision)
            if archived:
                manager.close_scene(scene_uuid)
        else:
            raise P08WorkflowError("choose a scene name or archive action")
        return {"scene": record.to_dict()}

    @app.get("/api/scenes/{scene_uuid}/delete-impact")
    async def delete_impact(scene_uuid: str) -> dict[str, Any]:
        return manager.registry.delete_impact(scene_uuid)

    @app.post("/api/scenes/{scene_uuid}/delete")
    async def delete_scene(scene_uuid: str, request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        deletion_token = _required_string(payload, "deletion_token")
        delete_files = _required_boolean(payload, "delete_files")
        expected_revision = _required_integer(payload, "expected_revision")
        record = manager.registry.require(scene_uuid, include_archived=True)
        if record.revision != expected_revision:
            raise P08WorkflowError("scene changed; review deletion impact again")
        impact = manager.registry.delete_impact(scene_uuid)
        if impact["deletion_token"] != deletion_token:
            raise P08WorkflowError("scene changed; review deletion impact again")
        if not impact["can_remove_from_list"]:
            raise P08WorkflowError("stop active scene processing before deletion")
        if delete_files and not impact["can_delete_files"]:
            raise P08WorkflowError("this scene's files are protected")
        manager.close_scene(scene_uuid)
        return manager.registry.delete(
            scene_uuid,
            deletion_token=deletion_token,
            delete_files=delete_files,
            expected_revision=expected_revision,
        )

    @app.get("/pages/{page_id}")
    async def legacy_page(page_id: str) -> RedirectResponse:
        return RedirectResponse(
            f"/scenes/{manager.default_scene_uuid}/pages/{page_id}", status_code=307
        )

    return app


def _scene_route(path: str) -> tuple[str, str] | None:
    if path.startswith("/scenes/"):
        remainder = path.removeprefix("/scenes/")
        scene_uuid, separator, tail = remainder.partition("/")
        if not separator or not scene_uuid:
            return None
        return scene_uuid, f"/{tail}"
    if path.startswith("/api/scenes/"):
        remainder = path.removeprefix("/api/scenes/")
        scene_uuid, separator, tail = remainder.partition("/")
        if not separator or not scene_uuid:
            return None
        if tail in {"delete", "delete-impact"}:
            return None
        return scene_uuid, f"/api/{tail}"
    return None


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise P08WorkflowError("request body must be valid JSON") from error
    if not isinstance(payload, dict):
        raise P08WorkflowError("request body must be an object")
    return payload


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise P08WorkflowError(f"{key} must be a non-blank string")
    return value.strip()


def _required_integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise P08WorkflowError(f"{key} must be a positive integer")
    return value


def _required_boolean(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise P08WorkflowError(f"{key} must be boolean")
    return value


def _scene_asset(name: str) -> str:
    return (
        files("spatial_mapping_phase2.xr03_scene_web").joinpath(name).read_text(encoding="utf-8")
    )
