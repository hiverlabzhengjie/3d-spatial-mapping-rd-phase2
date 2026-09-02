"""Human-facing localhost application for the integrated spatial workflow."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any, TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from spatial_mapping_phase2.p08_scene_updates import UpdateMode
from spatial_mapping_phase2.p08_workflow import P08WorkflowError, WorkflowService


class PageDefinition(TypedDict):
    page_id: str
    title: str
    tool_id: str | None


PAGE_DEFINITIONS: tuple[PageDefinition, ...] = (
    {"page_id": "setup", "title": "Project overview", "tool_id": None},
    {"page_id": "artifacts", "title": "Scene history & storage", "tool_id": None},
    {"page_id": "facility", "title": "Facility & cameras", "tool_id": "facility"},
    {"page_id": "capture", "title": "Capture", "tool_id": "capture"},
    {"page_id": "calibration", "title": "Calibration & pose", "tool_id": None},
    {"page_id": "reconstruction", "title": "Static reconstruction", "tool_id": None},
    {"page_id": "floor", "title": "Floor refinement", "tool_id": None},
    {"page_id": "results", "title": "Final review", "tool_id": None},
    {"page_id": "live", "title": "Live operations", "tool_id": None},
    {"page_id": "updates", "title": "Scene updates", "tool_id": None},
)


def create_p08_workflow_app(
    service: WorkflowService,
    legacy_apps: Mapping[str, FastAPI] | None = None,
    *,
    public_prefix: str = "",
) -> FastAPI:
    """Create one localhost shell over shared services and standalone tool adapters."""

    asset_bundle = {
        asset_name: _asset_text(asset_name)
        for asset_name in ("index.html", "app.js", "styles.css")
    }
    configured_legacy = dict(legacy_apps or {})
    for tool_id in configured_legacy:
        if tool_id not in {"facility", "capture"} and not re.fullmatch(
            r"calibration-[a-z0-9][a-z0-9._-]{0,63}", tool_id
        ):
            raise P08WorkflowError("unknown workflow tool adapter")
    app = FastAPI(
        title="Spatial Mapping Workflow",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.workflow_service = service

    @app.exception_handler(P08WorkflowError)
    async def workflow_error(_request: Request, error: P08WorkflowError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/", response_class=HTMLResponse)
    @app.get("/pages/{page_id}", response_class=HTMLResponse)
    async def index(page_id: str = "setup") -> HTMLResponse:
        if page_id not in {str(item["page_id"]) for item in PAGE_DEFINITIONS}:
            return HTMLResponse("Page not found", status_code=404)
        return HTMLResponse(asset_bundle["index.html"], headers={"Cache-Control": "no-store"})

    @app.get("/assets/{asset_name}")
    async def asset(asset_name: str) -> Response:
        content_types = {
            "app.js": "application/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }
        if asset_name not in content_types:
            return JSONResponse(status_code=404, content={"detail": "asset not found"})
        return Response(
            asset_bundle[asset_name],
            media_type=content_types[asset_name],
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/pages")
    async def pages() -> dict[str, Any]:
        calibration_tools = [
            {
                "camera_id": tool_id.removeprefix("calibration-"),
                "tool_url": f"{public_prefix}/tools/{tool_id}/",
            }
            for tool_id in sorted(configured_legacy)
            if tool_id.startswith("calibration-")
        ]
        return {
            "pages": [
                {
                    "page_id": item["page_id"],
                    "title": item["title"],
                    "tool_url": (
                        f"{public_prefix}/tools/{item['tool_id']}/"
                        if item["tool_id"] in configured_legacy
                        else None
                    ),
                    "calibration_tools": (
                        calibration_tools if item["page_id"] == "calibration" else []
                    ),
                }
                for item in PAGE_DEFINITIONS
            ]
        }

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return service.status()

    @app.get("/api/camera-policy")
    async def camera_policy() -> dict[str, Any]:
        return service.camera_policy_status()

    @app.get("/api/calibration")
    async def calibration_status() -> dict[str, Any]:
        return service.calibration_status()

    @app.post("/api/calibration/determine-intrinsics")
    async def determine_intrinsics(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return await run_in_threadpool(
            service.determine_calibration_intrinsics,
            _required_string(payload, "action_id"),
        )

    @app.post("/api/calibration/cameras/{camera_id}/run")
    async def calibrate_camera(camera_id: str, request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.calibrate_camera(_required_string(payload, "action_id"), camera_id)

    @app.post("/api/calibration/cameras/{camera_id}/review")
    async def review_calibration(camera_id: str, request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.review_camera_calibration(
            _required_string(payload, "action_id"),
            camera_id,
            _required_string(payload, "attempt_sha256"),
        )

    @app.post("/api/calibration/cameras/{camera_id}/override")
    async def override_calibration(camera_id: str, request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.override_camera_calibration(
            _required_string(payload, "action_id"),
            camera_id,
            _required_string(payload, "attempt_sha256"),
            _required_string(payload, "reason"),
            _required_boolean(payload, "acknowledged"),
        )

    @app.post("/api/camera-policy/impact")
    async def camera_policy_impact(request: Request) -> dict[str, Any]:
        return service.camera_policy_impact(await _json_object(request))

    @app.post("/api/camera-policy/apply")
    async def apply_camera_policy(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.apply_camera_policy(
            _required_string(payload, "action_id"),
            payload,
            expected_revision=_optional_integer(payload, "expected_revision"),
            confirm_impacts=_required_boolean(payload, "confirm_impacts"),
        )

    @app.post("/api/camera-policy/rollback")
    async def rollback_camera_policy(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        expected_revision = _optional_integer(payload, "expected_revision")
        if expected_revision is None:
            raise P08WorkflowError("expected_revision must be an integer")
        return service.rollback_camera_policy(
            _required_string(payload, "action_id"),
            _required_integer(payload, "target_revision"),
            expected_revision=expected_revision,
            confirm_impacts=_required_boolean(payload, "confirm_impacts"),
        )

    @app.get("/api/artifacts")
    async def artifact_catalog() -> dict[str, Any]:
        return service.artifact_catalog_status()

    @app.get("/api/artifacts/{artifact_id}/impact")
    async def artifact_selection_impact(artifact_id: str) -> dict[str, Any]:
        return service.artifact_selection_impact(artifact_id)

    @app.get("/api/artifacts/{artifact_id}/delete-impact")
    async def artifact_deletion_impact(artifact_id: str) -> dict[str, Any]:
        return service.artifact_deletion_impact(artifact_id)

    @app.post("/api/artifacts/delete-impact")
    async def artifact_batch_deletion_impact(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.artifact_batch_deletion_impact(
            _required_string_list(payload, "artifact_ids")
        )

    @app.post("/api/artifacts/select")
    async def select_artifact(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.select_artifact_version(
            _required_string(payload, "action_id"),
            _required_string(payload, "artifact_id"),
            confirm_impacts=_required_boolean(payload, "confirm_impacts"),
        )

    @app.post("/api/artifacts/verify")
    async def verify_artifact(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.verify_artifact_version(
            _required_string(payload, "action_id"),
            _required_string(payload, "artifact_id"),
        )

    @app.post("/api/artifacts/archive")
    async def archive_artifact(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.archive_artifact_version(
            _required_string(payload, "action_id"),
            _required_string(payload, "artifact_id"),
            archived=_required_boolean(payload, "archived"),
        )

    @app.post("/api/artifacts/delete")
    async def delete_artifact(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.delete_artifact_version(
            _required_string(payload, "action_id"),
            _required_string(payload, "artifact_id"),
            deletion_token=_required_string(payload, "deletion_token"),
        )

    @app.post("/api/artifacts/delete-batch")
    async def delete_artifact_batch(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.delete_artifact_versions(
            _required_string(payload, "action_id"),
            _required_deletion_tokens(payload, "items"),
        )

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        return service.jobs.status(job_id)

    @app.post("/api/jobs/reconstruction")
    async def start_reconstruction(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.start_reconstruction_job(_required_string(payload, "job_id"))

    @app.post("/api/jobs/floor")
    async def start_floor(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.start_floor_job(_required_string(payload, "job_id"))

    @app.post("/api/jobs/floor-preview")
    async def start_floor_preview(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.start_floor_preview_job(
            _required_string(payload, "job_id"),
            _required_string(payload, "floor_job_id"),
        )

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        return service.jobs.cancel(job_id)

    @app.post("/api/rerun/launch")
    async def launch_rerun(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.launch_rerun(
            _required_string(payload, "action_id"),
            _required_string(payload, "artifact_id"),
        )

    @app.post("/api/approve")
    async def approve(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.approve_result(
            _required_string(payload, "action_id"),
            _required_string(payload, "target"),
        )

    @app.post("/api/session/fresh")
    async def start_fresh_session(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.start_fresh_operator_session(
            _required_string(payload, "action_id"),
            _required_string(payload, "session_id"),
        )

    @app.post("/api/scene-updates/configure")
    async def configure_scene_updates(request: Request) -> dict[str, Any]:
        return service.configure_scene_updates(await _json_object(request))

    @app.post("/api/scene-updates/manual")
    async def start_manual_scene_update(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.start_scene_update(
            _required_string(payload, "update_id"), UpdateMode.MANUAL
        )

    @app.post("/api/scene-updates/candidate/open")
    async def open_scene_update_candidate(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.launch_scene_update_candidate(
            _required_string(payload, "action_id"),
            _required_string(payload, "result_id"),
        )

    @app.post("/api/scene-updates/candidate/adopt")
    async def adopt_scene_update_candidate(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.adopt_manual_scene_update(
            _required_string(payload, "action_id"),
            _required_string(payload, "result_id"),
        )

    @app.post("/api/scene-updates/rollback")
    async def rollback_scene_update(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.rollback_scene_update(
            _required_string(payload, "action_id"),
            _required_string(payload, "result_id"),
        )

    @app.post("/api/live-operations/start")
    async def start_live_operations(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.start_live_operations(_required_string(payload, "mode"))

    @app.post("/api/live-operations/stop")
    async def stop_live_operations() -> dict[str, Any]:
        return service.stop_live_operations()

    @app.post("/api/live-operations/action")
    async def live_operations_action(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.live_operations_action(
            _required_string(payload, "action"),
            session_id=_optional_string(payload, "session_id"),
            label=_optional_string(payload, "label"),
            confirmation=_optional_string(payload, "confirmation"),
        )

    @app.post("/api/live-operations/cancel-resume")
    async def cancel_live_auto_resume() -> dict[str, Any]:
        return service.cancel_live_auto_resume()

    @app.post("/api/live-operations/warning/acknowledge")
    async def acknowledge_live_operations_warning(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.acknowledge_live_operations_warning(_required_string(payload, "warning_id"))

    for tool_id, legacy_app in configured_legacy.items():
        prefix = f"/tools/{tool_id}"
        public_tool_prefix = f"{public_prefix}{prefix}"
        app.mount(
            prefix,
            _LegacyPrefixAdapter(legacy_app, public_tool_prefix),
            name=f"tool-{tool_id}",
        )
    return app


class _LegacyPrefixAdapter:
    """Make an absolute-URL localhost tool safe beneath a workflow mount prefix."""

    def __init__(self, app: ASGIApp, prefix: str) -> None:
        self.app = app
        self.prefix = prefix.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start: Message | None = None
        body_parts: list[bytes] = []

        async def capture(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            body_parts.append(message.get("body", b""))
            if message.get("more_body", False):
                return
            if start is None:
                raise RuntimeError("tool response body arrived before response start")
            headers = list(start.get("headers", []))
            content_type = next(
                (value.lower() for key, value in headers if key.lower() == b"content-type"),
                b"",
            )
            body = b"".join(body_parts)
            if any(
                marker in content_type for marker in (b"text/html", b"javascript", b"text/css")
            ):
                body = body.replace(b"/api/", self.prefix + b"/api/")
                body = body.replace(b"/assets/", self.prefix + b"/assets/")
                headers = [
                    (key, value) for key, value in headers if key.lower() != b"content-length"
                ]
                headers.append((b"content-length", str(len(body)).encode("ascii")))
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, capture)


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        value = await request.json()
    except json.JSONDecodeError as error:
        raise P08WorkflowError("request body must be valid JSON") from error
    if not isinstance(value, dict):
        raise P08WorkflowError("request JSON root must be an object")
    return value


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise P08WorkflowError(f"{key} must be a non-blank string")
    return result.strip()


def _required_boolean(value: dict[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise P08WorkflowError(f"{key} must be boolean")
    return result


def _required_integer(value: dict[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 1:
        raise P08WorkflowError(f"{key} must be a positive integer")
    return result


def _optional_integer(value: dict[str, Any], key: str) -> int | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, int) or isinstance(result, bool) or result < 1:
        raise P08WorkflowError(f"{key} must be null or a positive integer")
    return result


def _optional_string(value: dict[str, Any], key: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip():
        raise P08WorkflowError(f"{key} must be null or a non-blank string")
    return result.strip()


def _required_string_list(value: dict[str, Any], key: str) -> tuple[str, ...]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise P08WorkflowError(f"{key} must be a non-empty list")
    if len(result) > 100:
        raise P08WorkflowError(f"{key} cannot contain more than 100 values")
    parsed = tuple(item.strip() for item in result if isinstance(item, str) and item.strip())
    if len(parsed) != len(result):
        raise P08WorkflowError(f"every {key} value must be a non-blank string")
    return parsed


def _required_deletion_tokens(value: dict[str, Any], key: str) -> dict[str, str]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise P08WorkflowError(f"{key} must be a non-empty list")
    if len(result) > 100:
        raise P08WorkflowError(f"{key} cannot contain more than 100 values")
    tokens: dict[str, str] = {}
    for item in result:
        if not isinstance(item, dict):
            raise P08WorkflowError(f"every {key} value must be an object")
        artifact_id = _required_string(item, "artifact_id")
        deletion_token = _required_string(item, "deletion_token")
        if artifact_id in tokens:
            raise P08WorkflowError("artifact_ids in a deletion batch must be unique")
        tokens[artifact_id] = deletion_token
    return tokens


def _asset_text(asset_name: str) -> str:
    return files("spatial_mapping_phase2.p08_web").joinpath(asset_name).read_text(encoding="utf-8")
