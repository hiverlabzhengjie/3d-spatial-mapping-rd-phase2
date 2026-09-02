"""Integrated-console adapter for the isolated XR02 operator worker.

The spatial workflow and XR02 intentionally use different pinned Python environments.  This
module keeps one browser/API authority in the P08 process while supervising XR02 as an
authenticated localhost-only worker.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class LiveOperationsError(RuntimeError):
    """Raised when the integrated XR02 worker cannot complete an operator action."""


class XR02WorkerClient:
    """Small authenticated JSON client for one loopback XR02 worker."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 5.0,
        action_timeout_seconds: float = 120.0,
    ) -> None:
        if not base_url.startswith("http://127.0.0.1:"):
            raise LiveOperationsError("XR02 worker must use an IPv4 loopback URL")
        if not token:
            raise LiveOperationsError("XR02 worker token is required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.action_timeout_seconds = action_timeout_seconds

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/api/status")

    def start_live(
        self,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/start-live",
            {
                "resumed_from_session_id": resumed_from_session_id,
                "scene_update_id": scene_update_id,
            },
            timeout_seconds=self.action_timeout_seconds,
        )

    def start_recording(self) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/start-recording",
            {},
            timeout_seconds=self.action_timeout_seconds,
        )

    def stop(self, *, reason: str = "operator") -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/stop",
            {"reason": reason},
            timeout_seconds=self.action_timeout_seconds,
        )

    def open_rerun(self) -> dict[str, Any]:
        return self._request("POST", "/api/open-rerun", {})

    def reset_trails(self) -> dict[str, Any]:
        return self._request("POST", "/api/reset-trails", {})

    def export_evidence_snapshot(self) -> dict[str, Any]:
        return self._request("POST", "/api/export", {})

    def view_recording(self, session_id: str) -> dict[str, Any]:
        return self._request("POST", "/api/view-recording", {"session_id": session_id})

    def save_recording(self, session_id: str, label: str) -> dict[str, Any]:
        return self._request(
            "POST", "/api/save-recording", {"session_id": session_id, "label": label}
        )

    def delete_recording(self, session_id: str, confirmation: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/delete-recording",
            {"session_id": session_id, "confirmation": confirmation},
        )

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, object] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(dict(body)).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=payload,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-XR02-Worker-Token": self.token,
            },
        )
        try:
            timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8")).get("detail")
            except (json.JSONDecodeError, AttributeError):
                detail = None
            raise LiveOperationsError(
                str(detail or f"XR02 worker returned HTTP {error.code}")
            ) from error
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            raise LiveOperationsError("XR02 worker is unavailable") from error
        if not isinstance(value, dict):
            raise LiveOperationsError("XR02 worker returned a malformed response")
        return value


class SupervisedXR02Worker(XR02WorkerClient):
    """Launch and stop one API-only XR02 child in its pinned runtime."""

    def __init__(
        self,
        python_executable: Path,
        worker_script: Path | None,
        arguments: Sequence[str],
        *,
        worker_module: str | None = None,
        port: int = 8094,
        startup_timeout_seconds: float = 45.0,
    ) -> None:
        if not python_executable.is_file():
            raise LiveOperationsError("configured XR02 worker runtime is unavailable")
        if port < 1024 or port > 65535:
            raise LiveOperationsError("XR02 worker port must be within 1024..65535")
        token = secrets.token_urlsafe(32)
        environment = os.environ.copy()
        environment["XR02_WORKER_TOKEN"] = token
        command, working_directory = _worker_launch_command(
            python_executable, worker_script, worker_module, port, arguments
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        super().__init__(f"http://127.0.0.1:{port}", token)
        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise LiveOperationsError("XR02 worker exited during startup preflight")
            try:
                status = self.status()
                if status.get("schema") != "xr02.wp4.operator_status.v5":
                    raise LiveOperationsError("XR02 worker status contract is incompatible")
                return
            except LiveOperationsError:
                time.sleep(0.25)
        self.close()
        raise LiveOperationsError("XR02 worker did not become ready before the startup timeout")

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        try:
            status = self.status()
            if status.get("active") is True:
                self.stop(reason="integrated_console_shutdown")
        except LiveOperationsError:
            pass
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def _worker_launch_command(
    python_executable: Path,
    worker_script: Path | None,
    worker_module: str | None,
    port: int,
    arguments: Sequence[str],
) -> tuple[list[str], str | None]:
    if (worker_script is None) == (worker_module is None):
        raise LiveOperationsError("configure exactly one XR02 worker script or module")
    if worker_script is not None:
        if not worker_script.is_file():
            raise LiveOperationsError("configured XR02 worker script is unavailable")
        entrypoint = [str(worker_script.resolve())]
        working_directory: str | None = str(worker_script.resolve().parents[1])
    else:
        assert worker_module is not None
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", worker_module) is None:
            raise LiveOperationsError("configured XR02 worker module is invalid")
        entrypoint = ["-m", worker_module]
        working_directory = None
    return (
        [
            str(python_executable.resolve()),
            *entrypoint,
            "--port",
            str(port),
            "--no-browser",
            "--api-only",
            *arguments,
        ],
        working_directory,
    )
