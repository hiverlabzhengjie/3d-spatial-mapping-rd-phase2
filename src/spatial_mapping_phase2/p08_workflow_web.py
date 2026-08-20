"""Integrated multi-page localhost FastAPI surface for the P02-P08 workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any, TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from spatial_mapping_phase2.p08_workflow import P08WorkflowError, WorkflowService


class PageDefinition(TypedDict):
    page_id: str
    title: str
    phase_ids: tuple[str, ...]
    tool_ids: tuple[str, ...]


PAGE_DEFINITIONS: tuple[PageDefinition, ...] = (
    {
        "page_id": "setup",
        "title": "Project & scene",
        "phase_ids": ("P02", "P03", "P04", "P05", "P06", "P07", "P08"),
        "tool_ids": (),
    },
    {
        "page_id": "facility",
        "title": "Facility registration",
        "phase_ids": ("P02",),
        "tool_ids": ("p02",),
    },
    {
        "page_id": "capture",
        "title": "Stream health & capture",
        "phase_ids": ("P03",),
        "tool_ids": ("p03",),
    },
    {
        "page_id": "calibration",
        "title": "Calibration & pose review",
        "phase_ids": ("P04", "P05"),
        "tool_ids": ("p04", "p05"),
    },
    {
        "page_id": "reconstruction",
        "title": "DA3 reconstruction",
        "phase_ids": ("P06",),
        "tool_ids": (),
    },
    {
        "page_id": "geometry",
        "title": "Geometry & floor",
        "phase_ids": ("P07", "P08"),
        "tool_ids": (),
    },
    {
        "page_id": "artifacts",
        "title": "Artifacts & inspection",
        "phase_ids": ("P06", "P07", "P08"),
        "tool_ids": (),
    },
)


def create_p08_workflow_app(
    service: WorkflowService,
    legacy_apps: Mapping[str, FastAPI] | None = None,
) -> FastAPI:
    """Create one localhost shell over the shared workflow service and existing stage apps."""

    configured_legacy = dict(legacy_apps or {})
    allowed_legacy = {"p02", "p03", "p04", "p05"}
    if not set(configured_legacy) <= allowed_legacy:
        raise P08WorkflowError("unknown legacy application adapter")
    app = FastAPI(
        title="Phase 2 Integrated Workflow Console",
        version="1.0.0",
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

    @app.get("/api/pages")
    async def pages() -> dict[str, Any]:
        return {
            "pages": [
                {
                    **item,
                    "phase_ids": list(item["phase_ids"]),
                    "tool_ids": list(item["tool_ids"]),
                    "tools": [
                        {"tool_id": tool_id, "tool_url": f"/tools/{tool_id}/"}
                        for tool_id in item["tool_ids"]
                        if tool_id in configured_legacy
                    ],
                    "tool_url": next(
                        (
                            f"/tools/{tool_id}/"
                            for tool_id in item["tool_ids"]
                            if tool_id in configured_legacy
                        ),
                        None,
                    ),
                }
                for item in PAGE_DEFINITIONS
            ]
        }

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return service.status()

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        return service.jobs.status(job_id)

    @app.post("/api/jobs/floor")
    async def start_floor(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        return service.start_floor_job(_required_string(payload, "job_id"))

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

    for tool_id, legacy_app in configured_legacy.items():
        prefix = f"/tools/{tool_id}"
        app.mount(
            prefix,
            _LegacyPrefixAdapter(legacy_app, prefix),
            name=f"legacy-{tool_id}",
        )
    return app


class _LegacyPrefixAdapter:
    """Make an absolute-URL legacy localhost app safe beneath a P08 mount prefix."""

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
                raise RuntimeError("legacy ASGI response body arrived before response start")
            headers = list(start.get("headers", []))
            content_type = next(
                (
                    value.lower()
                    for key, value in headers
                    if key.lower() == b"content-type"
                ),
                b"",
            )
            body = b"".join(body_parts)
            if any(
                marker in content_type
                for marker in (b"text/html", b"javascript", b"text/css")
            ):
                body = body.replace(b"/api/", self.prefix + b"/api/")
                body = body.replace(b"/assets/", self.prefix + b"/assets/")
                headers = [
                    (key, value)
                    for key, value in headers
                    if key.lower() != b"content-length"
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


def _asset_text(asset_name: str) -> str:
    return (
        files("spatial_mapping_phase2.p08_web")
        .joinpath(asset_name)
        .read_text(encoding="utf-8")
    )
