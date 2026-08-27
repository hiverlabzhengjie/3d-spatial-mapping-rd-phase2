"""Credential-safe MediaMTX lifecycle and health boundary for XR02.

MediaMTX owns transport fan-out only. Decoded-frame freshness, capture generations and tracker
epochs remain downstream responsibilities of :mod:`xr02_rtsp_capture`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    LocalRtspEndpoint,
)
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS

MEDIAMTX_VERSION = "v1.20.1"
MEDIAMTX_WINDOWS_ARCHIVE_SHA256 = (
    "dc970f8e1f3ad58edafcf536bcd1ffe0adcb4390e7ace9377694a9ea2c1ebe53"
)
MEDIAMTX_WINDOWS_EXE_SHA256 = "114e6c0b514813658e10be55f8ab6eab950ae879943272a59b0a51d55930900a"
_RTSP_VALUE = re.compile(r"(?i)rtsps?://[^\s\"']+")
_PATH_BY_CAMERA = {
    camera_id: f"officecam{index:02d}" for index, camera_id in enumerate(CAMERA_IDS, start=1)
}


class MediaMtxGatewayError(RuntimeError):
    """Raised when the selected local media gateway cannot satisfy its contract."""


class MediaMtxGatewayState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MediaMtxGatewayPolicy:
    """Pinned runtime and bounded lifecycle settings for one deployment-host gateway."""

    binary_path: Path
    expected_executable_sha256: str = MEDIAMTX_WINDOWS_EXE_SHA256
    rtsp_port: int = 8554
    api_port: int = 9997
    metrics_port: int = 9998
    startup_timeout_seconds: float = 12.0
    health_poll_seconds: float = 0.5
    restart_initial_seconds: float = 0.5
    restart_maximum_seconds: float = 8.0

    def __post_init__(self) -> None:
        digest = self.expected_executable_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise MediaMtxGatewayError("MediaMTX executable SHA-256 must be lowercase hexadecimal")
        ports = (self.rtsp_port, self.api_port, self.metrics_port)
        if len(set(ports)) != len(ports) or any(not 1024 <= port <= 65535 for port in ports):
            raise MediaMtxGatewayError("MediaMTX ports must be distinct and within 1024..65535")
        if not 1.0 <= self.startup_timeout_seconds <= 60.0:
            raise MediaMtxGatewayError("MediaMTX startup timeout must be within 1..60 seconds")
        if not 0.1 <= self.health_poll_seconds <= 5.0:
            raise MediaMtxGatewayError("MediaMTX health polling must be within 0.1..5 seconds")
        if not 0.1 <= self.restart_initial_seconds <= self.restart_maximum_seconds <= 60.0:
            raise MediaMtxGatewayError("MediaMTX restart backoff is invalid")


@dataclass(frozen=True, slots=True)
class MediaMtxPathHealth:
    camera_id: str
    path_name: str
    state: str
    inbound_bytes: int | None
    inbound_frame_errors: int | None
    reader_count: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "path_name": self.path_name,
            "state": self.state,
            "inbound_bytes": self.inbound_bytes,
            "inbound_frame_errors": self.inbound_frame_errors,
            "reader_count": self.reader_count,
        }


class MediaMtxGateway:
    """Own one persistent upstream pull per camera and credential-free local fan-out."""

    def __init__(
        self,
        upstream_endpoints: tuple[LocalRtspEndpoint, ...],
        policy: MediaMtxGatewayPolicy,
        runtime_directory: Path,
    ) -> None:
        if tuple(endpoint.camera_id for endpoint in upstream_endpoints) != CAMERA_IDS:
            raise MediaMtxGatewayError("MediaMTX requires the exact ordered office camera roster")
        self._upstream_endpoints = upstream_endpoints
        self.policy = policy
        self.runtime_directory = runtime_directory.resolve()
        self.config_path = self.runtime_directory / "mediamtx-runtime.yml"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._watch_thread: threading.Thread | None = None
        self._log_threads: list[threading.Thread] = []
        self._state = MediaMtxGatewayState.STOPPED
        self._generation = 0
        self._restarts = 0
        self._restart_reason: str | None = None
        self._failure_class: str | None = None
        self._started_monotonic_ns: int | None = None
        self._stopped_monotonic_ns: int | None = None
        self._last_health_monotonic_ns: int | None = None
        self._path_health = self._initial_path_health("unknown")
        self._sanitized_log_tail: deque[str] = deque(maxlen=80)
        self._events: deque[dict[str, object]] = deque(maxlen=512)
        self._config_sha256: str | None = None
        self._binary_sha256: str | None = None

    def local_endpoints(self) -> tuple[LocalRtspEndpoint, ...]:
        return tuple(
            LocalRtspEndpoint(
                camera_id,
                CAMERA_ENDPOINT_KEYS[camera_id],
                f"rtsp://127.0.0.1:{self.policy.rtsp_port}/{_PATH_BY_CAMERA[camera_id]}",
            )
            for camera_id in CAMERA_IDS
        )

    def start(self) -> None:
        with self._lock:
            if self._state is not MediaMtxGatewayState.STOPPED:
                raise MediaMtxGatewayError("MediaMTX gateway is not stopped")
            self._state = MediaMtxGatewayState.STARTING
            self._stop.clear()
            self._started_monotonic_ns = time.monotonic_ns()
            self._stopped_monotonic_ns = None
        self._validate_binary()
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        config = _credential_safe_config(self.policy)
        if "rtsp://" in config.lower() or "rtsps://" in config.lower():
            raise MediaMtxGatewayError("MediaMTX runtime config unexpectedly contains an endpoint")
        self.config_path.write_text(config, encoding="utf-8")
        self._config_sha256 = _sha256(self.config_path)
        try:
            self._launch_process()
        except Exception:
            with self._lock:
                self._state = MediaMtxGatewayState.FAILED
            raise
        self._watch_thread = threading.Thread(
            target=self._watch,
            name="xr02-mediamtx-supervisor",
            daemon=True,
        )
        self._watch_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        self._terminate(process)
        watcher = self._watch_thread
        if watcher is not None:
            watcher.join(timeout=self.policy.startup_timeout_seconds + 2.0)
            if watcher.is_alive():
                raise MediaMtxGatewayError("MediaMTX supervisor exceeded bounded shutdown")
        for thread in tuple(self._log_threads):
            thread.join(timeout=1.0)
        with self._lock:
            self._watch_thread = None
            self._process = None
            self._state = MediaMtxGatewayState.STOPPED
            self._stopped_monotonic_ns = time.monotonic_ns()
            self._record_event("gateway_stopped", None)

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema": "xr02.mediamtx.gateway_status.v1",
                "state": self._state.value,
                "version": MEDIAMTX_VERSION,
                "generation": self._generation,
                "restarts": self._restarts,
                "restart_reason": self._restart_reason,
                "failure_class": self._failure_class,
                "rtsp_port": self.policy.rtsp_port,
                "api_port": self.policy.api_port,
                "metrics_port": self.policy.metrics_port,
                "source_on_demand": False,
                "always_available": False,
                "rtsp_transport": "tcp",
                "path_health": [item.as_dict() for item in self._path_health],
                "last_health_monotonic_ns": self._last_health_monotonic_ns,
            }

    def interrupt_for_diagnostic(self) -> None:
        """Terminate the current gateway generation so the bounded supervisor must recover it."""

        with self._lock:
            if self._state is not MediaMtxGatewayState.RUNNING or self._process is None:
                raise MediaMtxGatewayError(
                    "MediaMTX diagnostic interruption requires running state"
                )
            process = self._process
            self._restart_reason = "owner_authorized_diagnostic_interrupt"
            self._record_event("diagnostic_interrupt", self._restart_reason)
        process.terminate()

    def evidence(self) -> dict[str, object]:
        with self._lock:
            result = self.status()
            result.update(
                {
                    "schema": "xr02.mediamtx.gateway_evidence.v1",
                    "binary_path": str(self.policy.binary_path.resolve()),
                    "binary_sha256": self._binary_sha256,
                    "config_path": str(self.config_path),
                    "config_sha256": self._config_sha256,
                    "runtime_secret_references": [
                        CAMERA_ENDPOINT_KEYS[camera_id] for camera_id in CAMERA_IDS
                    ],
                    "credentials_persisted": False,
                    "local_path_by_camera": dict(_PATH_BY_CAMERA),
                    "started_monotonic_ns": self._started_monotonic_ns,
                    "stopped_monotonic_ns": self._stopped_monotonic_ns,
                    "sanitized_log_tail": list(self._sanitized_log_tail),
                    "gateway_events": list(self._events),
                }
            )
        serialized = json.dumps(result, sort_keys=True).lower()
        if "rtsp://" in serialized or "rtsps://" in serialized:
            raise MediaMtxGatewayError("MediaMTX evidence contains an endpoint")
        return result

    def _validate_binary(self) -> None:
        binary = self.policy.binary_path.resolve()
        if not binary.is_file():
            raise MediaMtxGatewayError("pinned MediaMTX executable is missing")
        digest = _sha256(binary)
        if digest != self.policy.expected_executable_sha256:
            raise MediaMtxGatewayError("pinned MediaMTX executable identity changed")
        self._binary_sha256 = digest

    def _launch_process(self) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._state = (
                MediaMtxGatewayState.STARTING
                if generation == 1
                else MediaMtxGatewayState.RESTARTING
            )
            self._record_event("generation_starting", None)
        environment = _child_environment(self._upstream_endpoints)
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [str(self.policy.binary_path.resolve()), str(self.config_path)],
            cwd=str(self.runtime_directory),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )
        with self._lock:
            self._process = process
        if process.stdout is not None:
            thread = threading.Thread(
                target=self._drain_log,
                args=(process.stdout,),
                name=f"xr02-mediamtx-log-g{generation}",
                daemon=True,
            )
            self._log_threads.append(thread)
            thread.start()
        deadline = time.monotonic() + self.policy.startup_timeout_seconds
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                time.sleep(0.05)
                with self._lock:
                    detail = " | ".join(tuple(self._sanitized_log_tail)[-3:])
                raise MediaMtxGatewayError(
                    f"MediaMTX exited during startup with code {return_code}: {detail}"
                )
            if self._api_ready():
                with self._lock:
                    self._state = MediaMtxGatewayState.RUNNING
                    self._failure_class = None
                    self._restart_reason = None
                    self._record_event("generation_ready", None)
                self._refresh_metrics()
                return
            time.sleep(0.1)
        self._terminate(process)
        raise MediaMtxGatewayError("MediaMTX API did not become ready within the startup bound")

    def _watch(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            with self._lock:
                process = self._process
            if process is None:
                return
            return_code = process.poll()
            if return_code is None:
                self._refresh_metrics()
                self._stop.wait(self.policy.health_poll_seconds)
                continue
            if self._stop.is_set():
                return
            with self._lock:
                self._state = MediaMtxGatewayState.RESTARTING
                self._restarts += 1
                if self._restart_reason != "owner_authorized_diagnostic_interrupt":
                    self._restart_reason = f"process_exit_{return_code}"
                self._failure_class = "MediaMtxProcessExit"
                self._path_health = self._initial_path_health("gateway_offline")
                self._record_event("generation_exited", self._restart_reason)
            delay = min(
                self.policy.restart_maximum_seconds,
                self.policy.restart_initial_seconds * (2**consecutive_failures),
            )
            if self._stop.wait(delay):
                return
            try:
                self._launch_process()
            except Exception as error:
                consecutive_failures += 1
                with self._lock:
                    self._failure_class = type(error).__name__
                    self._restart_reason = "restart_failed"
                    self._sanitized_log_tail.append(_sanitize_log_line(str(error)))
                    self._record_event("generation_restart_failed", type(error).__name__)
                continue
            consecutive_failures = 0

    def _api_ready(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.policy.api_port}/v3/paths/list",
                timeout=min(0.5, self.policy.health_poll_seconds),
            ) as response:
                return int(response.status) == 200
        except (OSError, urllib.error.URLError):
            return False

    def _refresh_metrics(self) -> None:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.policy.metrics_port}/metrics?type=paths",
                timeout=min(0.5, self.policy.health_poll_seconds),
            ) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except (OSError, urllib.error.URLError):
            return
        parsed = _parse_path_metrics(payload)
        health = tuple(
            MediaMtxPathHealth(
                camera_id,
                path_name,
                parsed.get(path_name, {}).get("state", "unknown"),
                _optional_int(parsed.get(path_name, {}).get("inbound_bytes")),
                _optional_int(parsed.get(path_name, {}).get("inbound_frame_errors")),
                _optional_int(parsed.get(path_name, {}).get("reader_count")),
            )
            for camera_id, path_name in _PATH_BY_CAMERA.items()
        )
        with self._lock:
            self._path_health = health
            self._last_health_monotonic_ns = time.monotonic_ns()

    def _drain_log(self, stream: TextIO) -> None:
        try:
            for line in stream:
                sanitized = _sanitize_log_line(line.rstrip())
                if sanitized:
                    with self._lock:
                        self._sanitized_log_tail.append(sanitized)
        finally:
            stream.close()

    def _initial_path_health(self, state: str) -> tuple[MediaMtxPathHealth, ...]:
        return tuple(
            MediaMtxPathHealth(camera_id, path_name, state, None, None, None)
            for camera_id, path_name in _PATH_BY_CAMERA.items()
        )

    def _record_event(self, kind: str, reason: str | None) -> None:
        self._events.append(
            {
                "kind": kind,
                "generation": self._generation,
                "reason": reason,
                "monotonic_ns": time.monotonic_ns(),
            }
        )

    @staticmethod
    def _terminate(process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _credential_safe_config(policy: MediaMtxGatewayPolicy) -> str:
    path_lines = "\n".join(
        f"  {path_name}:\n    source: publisher" for path_name in _PATH_BY_CAMERA.values()
    )
    return (
        "logLevel: info\n"
        "logDestinations: [stdout]\n"
        "logStructured: true\n"
        "readTimeout: 5s\n"
        "writeTimeout: 5s\n"
        "api: true\n"
        f"apiAddress: 127.0.0.1:{policy.api_port}\n"
        "metrics: true\n"
        f"metricsAddress: 127.0.0.1:{policy.metrics_port}\n"
        "pprof: false\n"
        "playback: false\n"
        "rtsp: true\n"
        f"rtspAddress: 127.0.0.1:{policy.rtsp_port}\n"
        'rtspEncryption: "no"\n'
        "rtspTransports: [tcp]\n"
        "rtmp: false\n"
        "hls: false\n"
        "webrtc: false\n"
        "srt: false\n"
        "pathDefaults:\n"
        "  sourceOnDemand: false\n"
        "  alwaysAvailable: false\n"
        "  rtspTransport: tcp\n"
        "paths:\n"
        f"{path_lines}\n"
    )


def _child_environment(
    upstream_endpoints: tuple[LocalRtspEndpoint, ...],
) -> dict[str, str]:
    environment = dict(os.environ)
    for key in CAMERA_ENDPOINT_KEYS.values():
        environment.pop(key, None)
    for endpoint in upstream_endpoints:
        path_name = _PATH_BY_CAMERA[endpoint.camera_id].upper()
        environment[f"MTX_PATHS_{path_name}_SOURCE"] = endpoint.for_read_only_adapter()
        environment[f"MTX_PATHS_{path_name}_SOURCEONDEMAND"] = "false"
        environment[f"MTX_PATHS_{path_name}_ALWAYSAVAILABLE"] = "false"
        environment[f"MTX_PATHS_{path_name}_RTSPTRANSPORT"] = "tcp"
    return environment


def _sanitize_log_line(value: str) -> str:
    return _RTSP_VALUE.sub("<redacted-rtsp-endpoint>", value)


def _parse_path_metrics(payload: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        name_match = re.search(r'name="([^"]+)"', line)
        if name_match is None or name_match.group(1) not in _PATH_BY_CAMERA.values():
            continue
        path_name = name_match.group(1)
        fields = result.setdefault(path_name, {})
        value = line.rsplit(" ", 1)[-1]
        if line.startswith("paths{"):
            state_match = re.search(r'state="([^"]+)"', line)
            if state_match is not None:
                fields["state"] = state_match.group(1)
        elif line.startswith("paths_inbound_bytes{"):
            fields["inbound_bytes"] = value
        elif line.startswith("paths_inbound_frames_in_error{"):
            fields["inbound_frame_errors"] = value
        elif line.startswith("paths_readers{"):
            previous = _optional_int(fields.get("reader_count")) or 0
            current = _optional_int(value) or 0
            fields["reader_count"] = str(previous + current)
    return result


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
