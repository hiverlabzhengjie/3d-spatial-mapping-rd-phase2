"""Local FastAPI operator surface for managed-scene capture.

The routes intentionally match the established Capture page's P03-shaped
surface, while the implementation uses only the variable-roster managed-scene
capture contracts in :mod:`managed_scene_capture`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from spatial_mapping_phase2.managed_scene_capture import (
    PyAvSceneCaptureAdapter,
    SceneCaptureAdapter,
    SceneCaptureAdapterError,
    SceneCaptureBusyError,
    SceneCaptureError,
    SceneCapturePolicy,
    SceneCaptureRepository,
    SceneCaptureService,
)


def create_scene_capture_app(
    camera_bindings: Mapping[str, str],
    secret_file: Path,
    artifact_root: Path,
    scene_id: str,
    *,
    adapter: SceneCaptureAdapter | None = None,
) -> FastAPI:
    """Build a variable-roster local capture app for one managed scene.

    Endpoint values are intentionally not passed to this factory.  The service
    reloads the supplied per-scene ``secrets.env`` before every preview, health
    check and capture request, so a P02 save is available immediately without a
    server restart.
    """

    from spatial_mapping_phase2.managed_scene_capture import SceneEndpointLoader

    service = SceneCaptureService(
        SceneEndpointLoader(camera_bindings, secret_file),
        adapter or PyAvSceneCaptureAdapter(),
        SceneCaptureRepository(artifact_root),
        scene_id,
    )
    app = FastAPI(title="Managed Scene Capture", version="1.0.0", docs_url=None, redoc_url=None)
    app.state.scene_capture_service = service

    @app.exception_handler(SceneCaptureBusyError)
    async def capture_busy(_request: Request, error: SceneCaptureBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(SceneCaptureError)
    async def capture_error(_request: Request, error: SceneCaptureError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(SceneCaptureAdapterError)
    async def capture_adapter_error(
        _request: Request, _error: SceneCaptureAdapterError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "A live frame could not be loaded from this camera"},
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = files("spatial_mapping_phase2.managed_scene_capture_web").joinpath("index.html")
        return HTMLResponse(
            page.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/cameras")
    async def cameras() -> dict[str, list[str]]:
        return {"cameras": list(service.camera_ids)}

    @app.get("/api/health")
    async def health() -> dict[str, dict[str, object]]:
        return service.health()

    @app.get("/api/sessions")
    async def sessions() -> dict[str, list[str]]:
        return {"sessions": list(service.repository.list_sessions())}

    @app.get("/api/evidence")
    async def evidence() -> dict[str, object]:
        current = service.repository.current_bundle()
        return {
            "schema_version": "managed-scene-capture-evidence-status-v1",
            "session_count": len(service.repository.list_sessions()),
            "bundle_count": len(service.repository.list_bundle_paths()),
            "current_bundle": (
                None
                if current is None
                else {
                    **current[1].to_dict(),
                    "selection_source": current[2],
                }
            ),
        }

    @app.get("/api/cameras/{camera_id}/preview")
    async def preview(camera_id: str) -> Response:
        frame = service.preview(camera_id)
        return Response(
            frame.content,
            media_type=frame.media_type,
            headers={
                "Cache-Control": "no-store",
                "X-Scene-Capture-Evidence": "ephemeral-preview-not-capture-evidence",
                # The established Capture page reads this compatibility header.
                "X-P03-Observed-At": frame.observed_at_utc,
                "X-Scene-Capture-Observed-At": frame.observed_at_utc,
            },
        )

    @app.get("/api/sessions/{session_id}")
    async def session(session_id: str) -> dict[str, object]:
        return service.repository.read_session_payload(session_id)

    @app.post("/api/capture")
    async def capture(request: Request) -> dict[str, object]:
        payload = await _json_object(request)
        session_id = _required_string(payload, "session_id")
        duration = payload.get("duration_seconds", 2.0)
        if not isinstance(duration, int | float) or isinstance(duration, bool):
            raise SceneCaptureError("duration_seconds must be numeric")
        return service.capture_session(
            session_id,
            SceneCapturePolicy(duration_seconds=float(duration)),
        ).to_dict()

    @app.post("/api/sessions/{session_id}/bundles")
    async def bundle(session_id: str, request: Request) -> dict[str, object]:
        payload = await _json_object(request)
        bundle_id = _required_string(payload, "bundle_id")
        target = payload.get("target_monotonic_ns")
        if target is not None and (
            not isinstance(target, int) or isinstance(target, bool) or target < 0
        ):
            raise SceneCaptureError("target_monotonic_ns must be a non-negative integer")
        return service.select_bundle(
            service.repository.read_session(session_id),
            bundle_id,
            target,
        ).to_dict()

    @app.post("/api/cancel")
    async def cancel() -> dict[str, str]:
        return {"state": ("cancellation-requested" if service.cancel() else "no-capture-active")}

    return app


async def _json_object(request: Request) -> dict[str, object]:
    try:
        value = await request.json()
    except json.JSONDecodeError as error:
        raise SceneCaptureError("request body must be valid JSON") from error
    if not isinstance(value, dict):
        raise SceneCaptureError("request JSON root must be an object")
    return value


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SceneCaptureError(f"{field} must be a non-blank string")
    return value.strip()
