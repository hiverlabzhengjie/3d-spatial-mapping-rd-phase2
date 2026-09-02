"""FastAPI surface for the local P02 interactive registration console."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from spatial_mapping_phase2.p01_observability import P01ContractError
from spatial_mapping_phase2.p02_interactive_registration import InteractiveRegistrationError
from spatial_mapping_phase2.p02_registration_service import P02RegistrationService, PlanRenderer


def create_p02_registration_app(
    workspace: Path,
    secret_file: Path,
    renderer: PlanRenderer | None = None,
    *,
    camera_endpoint_keys: dict[str, str] | None = None,
) -> FastAPI:
    """Create a localhost-only registration application with explicit filesystem dependencies."""

    service = P02RegistrationService(
        workspace,
        secret_file,
        renderer,
        camera_endpoint_keys=camera_endpoint_keys,
    )
    app = FastAPI(
        title="P02 Facility Registration",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.registration_service = service

    @app.exception_handler(InteractiveRegistrationError)
    async def handle_registration_error(
        _request: Request, error: InteractiveRegistrationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(P01ContractError)
    async def handle_endpoint_error(_request: Request, _error: P01ContractError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "RTSP endpoint is malformed"})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_asset_text("index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/assets/{asset_name}")
    async def asset(asset_name: str) -> Response:
        allowed = {
            "app.js": "application/javascript; charset=utf-8",
            "styles.css": "text/css; charset=utf-8",
        }
        if asset_name not in allowed:
            return JSONResponse(status_code=404, content={"detail": "asset not found"})
        return Response(
            _asset_text(asset_name),
            media_type=allowed[asset_name],
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        custom_roster = camera_endpoint_keys is not None
        if not service.has_state():
            result: dict[str, Any] = {"has_state": False}
            if custom_roster:
                result["camera_ids"] = list(service.camera_endpoint_keys)
            return result
        result = {"has_state": True, "state": service.state_response()}
        if custom_roster:
            result["camera_ids"] = list(service.camera_endpoint_keys)
        return result

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return service.state_response()

    @app.post("/api/plan")
    async def upload_plan(request: Request) -> dict[str, Any]:
        filename = request.headers.get("x-filename", "")
        content = await request.body()
        state = service.upload_plan(filename, content)
        return {"state": service.state_response(), "source_sha256": state.plan.source_sha256}

    @app.get("/api/plan-image")
    async def plan_image() -> FileResponse:
        return FileResponse(
            service.plan_image_path(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.put("/api/state")
    async def save_state(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        service.save_state(payload)
        return service.state_response()

    @app.put("/api/cameras/{camera_id}/endpoint")
    async def save_endpoint(camera_id: str, request: Request) -> dict[str, bool]:
        payload = await _json_object(request)
        endpoint_url = payload.get("rtsp_url")
        if not isinstance(endpoint_url, str):
            raise InteractiveRegistrationError("rtsp_url must be a string")
        service.save_endpoint(camera_id, endpoint_url)
        return {"configured": True}

    @app.get("/api/cameras/{camera_id}/endpoint")
    async def get_endpoint(camera_id: str) -> dict[str, str | bool]:
        endpoint_url = service.load_endpoint(camera_id)
        return {"configured": endpoint_url is not None, "rtsp_url": endpoint_url or ""}

    @app.post("/api/export")
    async def export_snapshot() -> dict[str, Any]:
        path, payload = service.export_snapshot()
        return {"filename": path.name, "export": payload}

    return app


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise InteractiveRegistrationError("request body must be valid JSON") from error
    if not isinstance(payload, dict):
        raise InteractiveRegistrationError("request JSON root must be an object")
    return payload


def _asset_text(asset_name: str) -> str:
    resource = files("spatial_mapping_phase2.p02_web").joinpath(asset_name)
    return resource.read_text(encoding="utf-8")
