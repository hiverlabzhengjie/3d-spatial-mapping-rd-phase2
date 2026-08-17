"""FastAPI surface for the local P04 linked-landmark calibration console."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from spatial_mapping_phase2.p04_calibration_domain import (
    FrameReviewStatus,
    P04CalibrationError,
)
from spatial_mapping_phase2.p04_calibration_service import (
    CandidateCapturer,
    ImageInspector,
    P04CalibrationService,
)


def create_p04_calibration_app(
    workspace: Path,
    inspector: ImageInspector | None = None,
    candidate_capturer: CandidateCapturer | None = None,
) -> FastAPI:
    """Create the localhost P04 app with an explicit local workspace dependency."""

    service = P04CalibrationService(workspace, inspector)
    app = FastAPI(
        title="P04 Calibration Correspondence Console",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.calibration_service = service

    @app.exception_handler(P04CalibrationError)
    async def handle_calibration_error(
        _request: Request, error: P04CalibrationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

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
        return {
            "has_state": service.has_state(),
            "state": service.state_response() if service.has_state() else None,
        }

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return service.state_response()

    @app.get("/api/plan-image")
    async def plan_image() -> FileResponse:
        return FileResponse(service.plan_image_path(), headers={"Cache-Control": "no-store"})

    @app.get("/api/frames/{frame_id}/image")
    async def frame_image(frame_id: str) -> FileResponse:
        return FileResponse(
            service.frame_image_path(frame_id), headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/frames")
    async def add_frame(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        expected = payload.get("expected_sha256")
        if expected is not None and not isinstance(expected, str):
            raise P04CalibrationError("expected_sha256 must be a string or null")
        service.add_frame(
            Path(_required_string(payload, "source_path")),
            _required_string(payload, "frame_id"),
            _required_string(payload, "profile_version"),
            expected,
        )
        return service.state_response()

    @app.post("/api/capture-candidate")
    async def capture_candidate(request: Request) -> dict[str, Any]:
        if candidate_capturer is None:
            raise P04CalibrationError("live Camera 3 capture is not configured")
        payload = await _json_object(request)
        delay = payload.get("delay_seconds")
        if not isinstance(delay, int | float) or isinstance(delay, bool):
            raise P04CalibrationError("delay_seconds must be a number")
        await run_in_threadpool(
            service.capture_candidate, float(delay), candidate_capturer
        )
        return service.state_response()

    @app.put("/api/frames/{frame_id}/review")
    async def review_frame(frame_id: str, request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        note_value = payload.get("note")
        note = None
        if note_value not in {None, ""}:
            if not isinstance(note_value, str):
                raise P04CalibrationError("review note must be a string or null")
            note = note_value.strip() or None
        try:
            status = FrameReviewStatus(_required_string(payload, "status"))
        except ValueError as error:
            raise P04CalibrationError("status must be approved or rejected") from error
        service.review_frame(frame_id, status, note)
        return service.state_response()

    @app.post("/api/landmarks")
    async def add_landmark(request: Request) -> dict[str, Any]:
        service.add_landmark(await _json_object(request))
        return service.state_response()

    @app.delete("/api/landmarks/{landmark_id}")
    async def remove_landmark(landmark_id: str) -> dict[str, Any]:
        service.remove_landmark(landmark_id)
        return service.state_response()

    @app.post("/api/export")
    async def export_snapshot() -> dict[str, Any]:
        path, payload = service.export_snapshot()
        return {"filename": path.name, "export": payload}

    return app


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise P04CalibrationError("request body must be valid JSON") from error
    if not isinstance(payload, dict):
        raise P04CalibrationError("request JSON root must be an object")
    return payload


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise P04CalibrationError(f"{key} must be a non-blank string")
    return value.strip()


def _asset_text(asset_name: str) -> str:
    return (
        files("spatial_mapping_phase2.p04_web")
        .joinpath(asset_name)
        .read_text(encoding="utf-8")
    )
