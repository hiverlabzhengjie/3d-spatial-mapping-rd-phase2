"""Credential-safe localhost console over the shared P03 workflow service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from importlib.resources import files
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from spatial_mapping_phase2.p03_capture_domain import P03ContractError
from spatial_mapping_phase2.p03_capture_service import CapturePolicy, CaptureWorkflowService
from spatial_mapping_phase2.p03_temporal_capture import WarmTemporalCaptureService


def create_p03_capture_app(
    service: CaptureWorkflowService,
    temporal_factory: Callable[[], WarmTemporalCaptureService] | None = None,
) -> FastAPI:
    app = FastAPI(title="P03 Live Capture", version="1.0.0", docs_url=None, redoc_url=None)
    app.state.capture_service = service

    @app.exception_handler(P03ContractError)
    async def contract_error(_request: Request, error: P03ContractError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = files("spatial_mapping_phase2.p03_web").joinpath("index.html")
        return HTMLResponse(
            page.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/health")
    async def health() -> dict[str, dict[str, object]]:
        return service.health(CapturePolicy())

    @app.get("/api/sessions")
    async def sessions() -> dict[str, tuple[str, ...]]:
        return {"sessions": service.repository.list_sessions()}

    @app.get("/api/cameras/{camera_id}/preview")
    async def preview(camera_id: str) -> Response:
        frame = service.preview(camera_id, CapturePolicy(duration_seconds=1.0))
        return Response(
            frame.content,
            media_type=frame.media_type,
            headers={
                "Cache-Control": "no-store",
                "X-P03-Evidence": "ephemeral-preview-not-capture-evidence",
                "X-P03-Observed-At": frame.observed_at_utc,
            },
        )

    @app.get("/api/sessions/{session_id}")
    async def session(session_id: str) -> dict[str, object]:
        return service.repository.read_session_payload(session_id)

    @app.post("/api/capture")
    async def capture(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("session_id"), str):
            raise P03ContractError("capture requires a session_id string")
        duration = payload.get("duration_seconds", 2.0)
        if not isinstance(duration, int | float):
            raise P03ContractError("duration_seconds must be numeric")
        return asdict(
            service.capture_session(
                payload["session_id"], CapturePolicy(duration_seconds=float(duration))
            )
        )

    @app.post("/api/sessions/{session_id}/bundles")
    async def bundle(session_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("bundle_id"), str):
            raise P03ContractError("bundle selection requires a bundle_id string")
        session_manifest = service.repository.read_session(session_id)
        return asdict(service.select_bundle(session_manifest, payload["bundle_id"]))

    @app.post("/api/cancel")
    async def cancel() -> dict[str, str]:
        service.cancel()
        return {"state": "cancellation-requested"}

    @app.post("/api/temporal-capture")
    async def temporal_capture(request: Request) -> dict[str, Any]:
        if temporal_factory is None:
            raise P03ContractError("temporal capture is not configured")
        payload = await request.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("bundle_id"), str):
            raise P03ContractError("temporal capture requires a bundle_id string")
        max_skew_ms = payload.get("max_skew_ms")
        warmup_seconds = payload.get("warmup_seconds", 15.0)
        if (
            not isinstance(max_skew_ms, int | float)
            or isinstance(max_skew_ms, bool)
            or max_skew_ms <= 0
        ):
            raise P03ContractError("max_skew_ms must be positive")
        if (
            not isinstance(warmup_seconds, int | float)
            or isinstance(warmup_seconds, bool)
            or warmup_seconds <= 0
        ):
            raise P03ContractError("warmup_seconds must be positive")
        temporal = temporal_factory()
        temporal.start()
        try:
            if not temporal.wait_until_ready(float(warmup_seconds)):
                return {"authority_status": "unavailable", "workers": temporal.status()}
            return asdict(
                temporal.capture(payload["bundle_id"], int(float(max_skew_ms) * 1_000_000))
            )
        finally:
            temporal.close()

    return app
