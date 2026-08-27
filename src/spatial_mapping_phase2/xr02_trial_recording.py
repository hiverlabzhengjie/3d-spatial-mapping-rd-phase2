"""Optional credential-safe FFmpeg capture of MediaMTX local fan-out streams."""

from __future__ import annotations

import hashlib
import math
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS


class TrialRecordingError(RuntimeError):
    """Raised when opt-in replay capture cannot preserve its bounded contract."""


@dataclass(frozen=True, slots=True)
class TrialRecordingPolicy:
    ffmpeg_binary: Path
    shutdown_timeout_seconds: float = 8.0
    reconnect_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.ffmpeg_binary.resolve().is_file():
            raise TrialRecordingError("FFmpeg executable is missing")
        if (
            not math.isfinite(self.shutdown_timeout_seconds)
            or not 1.0 <= self.shutdown_timeout_seconds <= 30.0
        ):
            raise TrialRecordingError("recording shutdown timeout must be within 1..30 seconds")
        if (
            not math.isfinite(self.reconnect_delay_seconds)
            or not 0.1 <= self.reconnect_delay_seconds <= 10.0
        ):
            raise TrialRecordingError("recording reconnect delay must be within 0.1..10 seconds")


@dataclass(slots=True)
class _CaptureProcess:
    camera_id: str
    generation: int
    output_path: Path
    process: subprocess.Popen[bytes]
    return_code: int | None = None


class MediaMtxTrialRecorder:
    """Record restart-safe encoded segments without persisting endpoint values.

    Inputs must be credential-free loopback RTSP endpoints supplied by MediaMTX.
    Direct camera endpoints are deliberately rejected.
    """

    def __init__(
        self,
        endpoints: tuple[LocalRtspEndpoint, ...],
        policy: TrialRecordingPolicy,
        output_directory: Path,
    ) -> None:
        if tuple(item.camera_id for item in endpoints) != CAMERA_IDS:
            raise TrialRecordingError("trial recording requires the ordered office camera roster")
        for endpoint in endpoints:
            _validate_credential_free_local_endpoint(endpoint.for_read_only_adapter())
        self._endpoints = endpoints
        self.policy = policy
        self.output_directory = output_directory.resolve()
        self._captures: list[_CaptureProcess] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._launch_failures = {camera_id: 0 for camera_id in CAMERA_IDS}
        self._started_monotonic_ns: int | None = None
        self._stopped_monotonic_ns: int | None = None
        self._binary_sha256: str | None = None
        self._version_line: str | None = None
        self._final_capture_artifacts: list[dict[str, object]] | None = None

    def start(self) -> None:
        if self._started_monotonic_ns is not None:
            raise TrialRecordingError("trial recorder is already started")
        binary = self.policy.ffmpeg_binary.resolve()
        self._binary_sha256 = _sha256(binary)
        self._version_line = _ffmpeg_version(binary)
        self.output_directory.mkdir(parents=True, exist_ok=False)
        self._started_monotonic_ns = time.monotonic_ns()
        self._stop.clear()
        for endpoint in self._endpoints:
            thread = threading.Thread(
                target=self._record_loop,
                args=(endpoint,),
                name=f"xr02-trial-recorder-{endpoint.camera_id}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def stop(self) -> None:
        if self._started_monotonic_ns is None or self._stopped_monotonic_ns is not None:
            return
        self._stop.set()
        deadline = time.monotonic() + self.policy.shutdown_timeout_seconds
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        remaining = self._active_captures()
        for capture in remaining:
            if capture.process.poll() is None:
                try:
                    capture.process.terminate()
                except OSError:
                    pass
        _wait_processes(remaining, time.monotonic() + 2.0)
        for capture in remaining:
            if capture.process.poll() is None:
                try:
                    capture.process.kill()
                except OSError:
                    pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        if any(thread.is_alive() for thread in self._threads):
            raise TrialRecordingError("trial-recording supervisors exceeded bounded shutdown")
        self._stopped_monotonic_ns = time.monotonic_ns()

    def status(self) -> dict[str, object]:
        with self._lock:
            captures = tuple(self._captures)
            failures = dict(self._launch_failures)
        return {
            "schema": "xr02.wp4.trial_recording_status.v1",
            "configured": True,
            "active": self._started_monotonic_ns is not None
            and self._stopped_monotonic_ns is None,
            "mode": "supervised credential-free MediaMTX stream-copy segments",
            "segments_started": len(captures),
            "segments_currently_writing": sum(item.process.poll() is None for item in captures),
            "launch_failures_by_camera": failures,
        }

    def evidence(self) -> dict[str, object]:
        return {
            "schema": "xr02.wp4.trial_recording.v1",
            "enabled": True,
            "source": "credential-free MediaMTX loopback RTSP fan-out",
            "container": "matroska",
            "video_mode": "encoded-stream-copy",
            "reconnect_policy": "supervised per-camera segmented restart",
            "audio_recorded": False,
            "credentials_persisted": False,
            "ffmpeg_binary_path": str(self.policy.ffmpeg_binary.resolve()),
            "ffmpeg_binary_sha256": self._binary_sha256,
            "ffmpeg_version": self._version_line,
            "started_monotonic_ns": self._started_monotonic_ns,
            "stopped_monotonic_ns": self._stopped_monotonic_ns,
            "captures": self.capture_artifacts(),
        }

    def capture_artifacts(self) -> list[dict[str, object]]:
        with self._lock:
            if self._final_capture_artifacts is not None:
                return list(self._final_capture_artifacts)
            captures = tuple(self._captures)
        result = [
            {
                "camera_id": item.camera_id,
                "generation": item.generation,
                "return_code": (
                    item.return_code if item.return_code is not None else item.process.poll()
                ),
                "artifact": _identity(item.output_path),
            }
            for item in captures
        ]
        with self._lock:
            if self._stopped_monotonic_ns is not None:
                self._final_capture_artifacts = result
        return list(result)

    def _record_loop(self, endpoint: LocalRtspEndpoint) -> None:
        binary = self.policy.ffmpeg_binary.resolve()
        generation = 0
        while not self._stop.is_set():
            generation += 1
            output = self.output_directory / (f"{endpoint.camera_id}-g{generation:04d}.mkv")
            try:
                process = _spawn_ffmpeg(
                    _record_command(binary, endpoint.for_read_only_adapter(), output)
                )
            except OSError:
                with self._lock:
                    self._launch_failures[endpoint.camera_id] += 1
                if self._stop.wait(self.policy.reconnect_delay_seconds):
                    return
                continue
            capture = _CaptureProcess(endpoint.camera_id, generation, output, process)
            with self._lock:
                self._captures.append(capture)
            while process.poll() is None and not self._stop.wait(0.10):
                pass
            if self._stop.is_set() and process.poll() is None:
                _request_graceful_stop(process)
                try:
                    process.wait(timeout=min(3.0, self.policy.shutdown_timeout_seconds))
                except subprocess.TimeoutExpired:
                    process.terminate()
            if process.poll() is not None:
                capture.return_code = process.returncode
            if process.stdin is not None:
                process.stdin.close()
            if not self._stop.is_set():
                self._stop.wait(self.policy.reconnect_delay_seconds)

    def _active_captures(self) -> list[_CaptureProcess]:
        with self._lock:
            return [item for item in self._captures if item.process.poll() is None]


def _record_command(binary: Path, endpoint: str, output: Path) -> list[str]:
    _validate_credential_free_local_endpoint(endpoint)
    return [
        str(binary),
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        endpoint,
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-map_metadata",
        "-1",
        "-f",
        "matroska",
        "-y",
        str(output),
    ]


def _spawn_ffmpeg(command: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _validate_credential_free_local_endpoint(endpoint: str) -> None:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as error:
        raise TrialRecordingError("trial recording endpoint is malformed") from error
    if (
        parsed.scheme.lower() != "rtsp"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise TrialRecordingError(
            "trial recording accepts only credential-free MediaMTX loopback endpoints"
        )


def _ffmpeg_version(binary: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary), "-version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TrialRecordingError("failed to identify FFmpeg runtime") from error
    line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    if not line or "ffmpeg version" not in line.lower():
        raise TrialRecordingError("FFmpeg runtime identity output is invalid")
    return line


def _wait_processes(captures: list[_CaptureProcess], deadline: float) -> None:
    while time.monotonic() < deadline:
        if all(item.process.poll() is not None for item in captures):
            return
        time.sleep(0.05)


def _request_graceful_stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None or process.stdin is None:
        return
    try:
        process.stdin.write(b"q\n")
        process.stdin.flush()
    except (OSError, ValueError):
        pass


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "present": False}
    return {
        "path": str(resolved),
        "present": True,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
