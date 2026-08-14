"""Local-only persistence and PDF rendering for the P02 interactive registration console."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    CAMERA_IDS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p02_interactive_registration import (
    InteractiveRegistrationError,
    InteractiveRegistrationState,
    PlanMetadata,
    build_interactive_export,
    empty_registration,
)

MAX_PLAN_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RenderedPlan:
    image_width_pixels: int
    image_height_pixels: int


class PlanRenderer(Protocol):
    def render_first_page(self, source_pdf: Path, destination_png: Path) -> RenderedPlan:
        """Render page one without modifying the source PDF."""


class PyMuPDFPlanRenderer:
    """Render a source PDF through PyMuPDF, imported lazily for actionable startup errors."""

    def __init__(self, zoom: float = 2.0) -> None:
        if zoom <= 0:
            raise InteractiveRegistrationError("PDF render zoom must be positive")
        self._zoom = zoom

    def render_first_page(self, source_pdf: Path, destination_png: Path) -> RenderedPlan:
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError as error:
            raise InteractiveRegistrationError(
                "PyMuPDF is required to render uploaded floor-plan PDFs"
            ) from error
        try:
            document = fitz.open(source_pdf)
        except Exception as error:
            raise InteractiveRegistrationError("uploaded file is not a readable PDF") from error
        try:
            if document.page_count < 1:
                raise InteractiveRegistrationError("uploaded PDF contains no pages")
            page = document.load_page(0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(self._zoom, self._zoom), alpha=False)
            pixmap.save(destination_png)
            return RenderedPlan(pixmap.width, pixmap.height)
        finally:
            document.close()


class P02RegistrationService:
    """Versioned local workspace with credential-separated endpoint configuration."""

    def __init__(
        self,
        workspace: Path,
        secret_file: Path,
        renderer: PlanRenderer | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.secret_file = secret_file.resolve()
        self.renderer = renderer or PyMuPDFPlanRenderer()
        self._lock = threading.RLock()
        self._state_path = self.workspace / "state.json"
        self._sources_dir = self.workspace / "sources"
        self._display_dir = self.workspace / "display"
        self._history_dir = self.workspace / "history"
        self._exports_dir = self.workspace / "exports"
        for directory in (
            self.workspace,
            self._sources_dir,
            self._display_dir,
            self._history_dir,
            self._exports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def has_state(self) -> bool:
        return self._state_path.is_file()

    def load_state(self) -> InteractiveRegistrationState:
        with self._lock:
            if not self._state_path.is_file():
                raise InteractiveRegistrationError("upload a floor-plan PDF to begin")
            return self._read_state(self._state_path)

    def upload_plan(self, filename: str, content: bytes) -> InteractiveRegistrationState:
        """Preserve a hash-bound PDF and initialize a clean editable registration."""

        safe_name = Path(filename).name
        if not safe_name or Path(safe_name).suffix.lower() != ".pdf":
            raise InteractiveRegistrationError("floor plan must have a .pdf filename")
        if not content or len(content) > MAX_PLAN_BYTES:
            raise InteractiveRegistrationError("floor-plan PDF must be between 1 byte and 50 MiB")
        if not content.startswith(b"%PDF-"):
            raise InteractiveRegistrationError("uploaded floor plan does not have a PDF signature")
        source_hash = hashlib.sha256(content).hexdigest()
        source_path = self._sources_dir / f"{source_hash}.pdf"
        display_path = self._display_dir / f"{source_hash}-page-01.png"
        temporary_source = self.workspace / f".{source_hash}.uploading.pdf"
        temporary_display = self.workspace / f".{source_hash}.rendering.png"
        with self._lock:
            temporary_source.write_bytes(content)
            try:
                rendered = self.renderer.render_first_page(temporary_source, temporary_display)
                if not source_path.exists():
                    temporary_source.replace(source_path)
                if not display_path.exists():
                    temporary_display.replace(display_path)
            finally:
                temporary_source.unlink(missing_ok=True)
                temporary_display.unlink(missing_ok=True)
            if self._state_path.exists():
                self._archive_state(self.load_state())
            state = empty_registration(
                PlanMetadata(
                    source_hash,
                    safe_name,
                    1,
                    rendered.image_width_pixels,
                    rendered.image_height_pixels,
                )
            )
            self._write_state(state)
            return state

    def save_state(self, payload: dict[str, Any]) -> InteractiveRegistrationState:
        """Validate and atomically advance an optimistic-concurrency revision."""

        with self._lock:
            current = self.load_state()
            candidate = InteractiveRegistrationState.from_dict(payload)
            if candidate.revision != current.revision:
                raise InteractiveRegistrationError(
                    f"stale revision {candidate.revision}; current revision is {current.revision}"
                )
            if candidate.plan != current.plan:
                raise InteractiveRegistrationError(
                    "plan identity cannot be changed through state save"
                )
            next_state = replace(candidate, revision=current.revision + 1)
            self._archive_state(current)
            self._write_state(next_state)
            return next_state

    def plan_image_path(self) -> Path:
        state = self.load_state()
        path = self._display_dir / f"{state.plan.source_sha256}-page-01.png"
        if not path.is_file():
            raise InteractiveRegistrationError("rendered plan image is missing")
        return path

    def endpoint_configuration(self) -> dict[str, bool]:
        values = self._read_secret_values()
        return {
            camera_id: bool(values.get(CAMERA_ENDPOINT_KEYS[camera_id], "").strip())
            for camera_id in CAMERA_IDS
        }

    def load_endpoint(self, camera_id: str) -> str | None:
        """Return one camera's local secret only to the dedicated localhost endpoint editor."""

        if camera_id not in CAMERA_IDS:
            raise InteractiveRegistrationError("unknown camera_id")
        endpoint = self._read_secret_values().get(CAMERA_ENDPOINT_KEYS[camera_id], "").strip()
        return endpoint or None

    def save_endpoint(self, camera_id: str, endpoint_url: str) -> None:
        """Validate and store one RTSP URL locally without returning or logging its value."""

        if camera_id not in CAMERA_IDS:
            raise InteractiveRegistrationError("unknown camera_id")
        endpoint_key = CAMERA_ENDPOINT_KEYS[camera_id]
        LocalRtspEndpoint(camera_id, endpoint_key, endpoint_url)
        if "\n" in endpoint_url or "\r" in endpoint_url:
            raise InteractiveRegistrationError("RTSP URL cannot contain a line break")
        with self._lock:
            values = self._read_secret_values()
            values[endpoint_key] = endpoint_url
            self.secret_file.parent.mkdir(parents=True, exist_ok=True)
            existing_lines = (
                self.secret_file.read_text(encoding="utf-8").splitlines()
                if self.secret_file.exists()
                else []
            )
            camera_keys = set(CAMERA_ENDPOINT_KEYS.values())
            preserved = [
                line
                for line in existing_lines
                if not any(line.startswith(f"{key}=") for key in camera_keys)
            ]
            preserved.extend(
                f"{key}={values[key]}" for key in CAMERA_ENDPOINT_KEYS.values() if values.get(key)
            )
            self._atomic_write_text(self.secret_file, "\n".join(preserved).rstrip() + "\n")

    def export_snapshot(self) -> tuple[Path, dict[str, Any]]:
        with self._lock:
            state = self.load_state()
            payload = build_interactive_export(state, self.endpoint_configuration())
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            path = self._exports_dir / f"facility-registration-r{state.revision}-{timestamp}.json"
            self._atomic_write_json(path, payload)
            return path, payload

    def state_response(self) -> dict[str, Any]:
        state = self.load_state()
        payload = state.to_dict()
        payload["derived"] = build_derived_summary(state, self.endpoint_configuration())
        return payload

    def _archive_state(self, state: InteractiveRegistrationState) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self._history_dir / f"state-r{state.revision}-{timestamp}.json"
        self._atomic_write_json(path, state.to_dict())

    def _read_state(self, path: Path) -> InteractiveRegistrationState:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InteractiveRegistrationError("saved registration state is unreadable") from error
        if not isinstance(payload, dict):
            raise InteractiveRegistrationError("saved registration state must be an object")
        return InteractiveRegistrationState.from_dict(payload)

    def _write_state(self, state: InteractiveRegistrationState) -> None:
        self._atomic_write_json(self._state_path, state.to_dict())

    def _read_secret_values(self) -> dict[str, str]:
        values = dict(os.environ)
        if not self.secret_file.is_file():
            return values
        for line in self.secret_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
        return values

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        P02RegistrationService._atomic_write_text(
            path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def build_derived_summary(
    state: InteractiveRegistrationState, endpoint_configured: dict[str, bool]
) -> dict[str, Any]:
    """Return live UI calculations without persisting derived coordinates as source state."""

    cameras: dict[str, dict[str, Any]] = {}
    for camera in state.cameras:
        world_xy = None
        if (
            camera.marker is not None
            and state.frame is not None
            and state.pixels_per_metre is not None
        ):
            x_metres, y_metres = state.world_xy_from_pixel(camera.marker)
            world_xy = {"x_metres": x_metres, "y_metres": y_metres}
        cameras[camera.camera_id] = {
            "status": camera.status(endpoint_configured.get(camera.camera_id, False)).value,
            "world_xy": world_xy,
            "endpoint_configured": endpoint_configured.get(camera.camera_id, False),
        }
    return {
        "pixels_per_metre": state.pixels_per_metre,
        "scale_spread_fraction": state.scale_spread_fraction,
        "frame_ready": state.frame is not None and state.pixels_per_metre is not None,
        "cameras": cameras,
    }
