"""WP4 service construction, operator lifecycle and credential-safe evidence export."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from spatial_mapping_phase2.p01_observability import load_local_rtsp_endpoints
from spatial_mapping_phase2.p09_projection import (
    FrozenProjectionInputs,
    load_frozen_projection_inputs,
)
from spatial_mapping_phase2.xr02_live_domain import AdoptedSceneSelection, resolve_adopted_scene
from spatial_mapping_phase2.xr02_live_pipeline import LiveModelProfile, XR02LivePipeline
from spatial_mapping_phase2.xr02_live_rerun import XR02LiveRerunLogger
from spatial_mapping_phase2.xr02_live_service import XR02LiveService, XR02LiveServiceConfig
from spatial_mapping_phase2.xr02_mediamtx import (
    MediaMtxGateway,
    MediaMtxGatewayPolicy,
)
from spatial_mapping_phase2.xr02_trial_recording import (
    MediaMtxTrialRecorder,
    TrialRecordingPolicy,
)

WP4_PINNED_PYTHON = Path(os.environ.get("XR02_PINNED_PYTHON", sys.executable))
WP4_REQUIRED_RUNTIME: tuple[tuple[str, str, str], ...] = (
    ("av", "av", "16.0.1"),
    ("boxmot", "boxmot", "21.0.0"),
    ("numpy", "numpy", "2.2.6"),
    ("rerun-sdk", "rerun", "0.22.1"),
    ("scipy", "scipy", "1.17.1"),
    ("torch", "torch", "2.4.1+cu124"),
)


@dataclass(frozen=True, slots=True)
class WP4Paths:
    operator_state: Path
    p06: Path
    p07: Path
    p08_floor_manifest: Path
    p08_floor: Path
    detector: Path
    reid: Path
    output_root: Path
    environment_file: Path


@dataclass(frozen=True, slots=True)
class WP4Hashes:
    p06: str
    p07: str
    p08_floor_manifest: str
    p08_floor: str
    detector: str
    reid: str


class XR02WP4Controller:
    """Own one active run; every restart resolves a fresh immutable scene epoch."""

    def __init__(
        self,
        paths: WP4Paths,
        hashes: WP4Hashes,
        config: XR02LiveServiceConfig | None = None,
        media_gateway_policy: MediaMtxGatewayPolicy | None = None,
        trial_recording_policy: TrialRecordingPolicy | None = None,
        direct_camera_diagnostic: bool = False,
    ) -> None:
        self.paths = paths
        self.hashes = hashes
        self.config = config or XR02LiveServiceConfig()
        if media_gateway_policy is None and not direct_camera_diagnostic:
            raise RuntimeError("adopted MediaMTX ingress requires an explicit pinned policy")
        if trial_recording_policy is not None and direct_camera_diagnostic:
            raise RuntimeError("trial recording requires credential-free MediaMTX ingress")
        self.media_gateway_policy = media_gateway_policy
        self.trial_recording_policy = trial_recording_policy
        self.direct_camera_diagnostic = direct_camera_diagnostic
        self._lock = threading.RLock()
        self._service: XR02LiveService | None = None
        self._last_service: XR02LiveService | None = None
        self._gateway: MediaMtxGateway | None = None
        self._last_gateway: MediaMtxGateway | None = None
        self._recorder: MediaMtxTrialRecorder | None = None
        self._last_recorder: MediaMtxTrialRecorder | None = None
        self._output_dir: Path | None = None
        self._last_manifest: Path | None = None

    def start(self) -> dict[str, object]:
        with self._lock:
            if self._service is not None:
                raise RuntimeError("XR02 WP4 is already running")
            output = self.paths.output_root / datetime.now(UTC).strftime(
                "xr02-wp4-live-%Y%m%dT%H%M%S%fZ"
            )
            output.mkdir(parents=True, exist_ok=False)
            scene = self._resolve_scene()
            frozen = FrozenProjectionInputs(
                self.paths.p06,
                self.hashes.p06,
                self.paths.p07,
                self.hashes.p07,
                self.paths.p08_floor_manifest,
                self.hashes.p08_floor_manifest,
                self.paths.p08_floor,
                self.hashes.p08_floor,
            )
            calibrations, floor = load_frozen_projection_inputs(frozen)
            pipeline = XR02LivePipeline(
                scene,
                calibrations,
                floor,
                LiveModelProfile(
                    self.paths.detector,
                    self.hashes.detector,
                    self.paths.reid,
                    self.hashes.reid,
                ),
                output,
                cadence=self.config.cadence_profile,
            )
            logger = XR02LiveRerunLogger(
                output / "xr02-wp4-live.rrd",
                scene,
                calibrations,
                floor,
                Path(sys.executable),
            )
            environment = _environment(self.paths.environment_file)
            upstream_endpoints = load_local_rtsp_endpoints(environment)
            gateway: MediaMtxGateway | None = None
            recorder: MediaMtxTrialRecorder | None = None
            try:
                service_endpoints = upstream_endpoints
                if not self.direct_camera_diagnostic:
                    policy = self.media_gateway_policy
                    if policy is None:
                        raise RuntimeError("MediaMTX policy disappeared after validation")
                    gateway = MediaMtxGateway(
                        upstream_endpoints,
                        policy,
                        output / "media-gateway",
                    )
                    gateway.start()
                    service_endpoints = gateway.local_endpoints()
                if self.trial_recording_policy is not None:
                    recorder = MediaMtxTrialRecorder(
                        service_endpoints,
                        self.trial_recording_policy,
                        output / "trial-video",
                    )
                    recorder.start()
                service = XR02LiveService(
                    service_endpoints,
                    scene,
                    pipeline,
                    logger,
                    self._resolve_scene,
                    self.config,
                )
                service.start()
            except Exception:
                if recorder is not None:
                    recorder.stop()
                logger.close()
                if gateway is not None:
                    gateway.stop()
                raise
            self._service = service
            self._last_service = service
            self._gateway = gateway
            self._last_gateway = gateway
            self._recorder = recorder
            self._last_recorder = recorder
            self._output_dir = output
            self._last_manifest = None
            return self.status()

    def stop(self) -> dict[str, object]:
        with self._lock:
            service = self._service
            gateway = self._gateway
            recorder = self._recorder
            output = self._output_dir
            if service is None or output is None:
                return self.status()
            self._service = None
            self._gateway = None
            self._recorder = None
        service_error: Exception | None = None
        try:
            service.stop()
        except Exception as error:
            service_error = error
        finally:
            try:
                if recorder is not None:
                    recorder.stop()
            finally:
                if gateway is not None:
                    gateway.stop()
        manifest = self._write_manifest(output, service, gateway, recorder)
        with self._lock:
            self._last_manifest = manifest
        if service_error is not None:
            raise service_error
        return self.status()

    def open_rerun(self) -> dict[str, object]:
        with self._lock:
            service = self._service or self._last_service
        if service is None:
            raise RuntimeError("start one WP4 run before opening Rerun")
        service.open_viewer()
        return self.status()

    def reset_trails(self) -> dict[str, object]:
        with self._lock:
            service = self._service
        if service is None:
            raise RuntimeError("trail reset requires an active run")
        service.reset_trails()
        return self.status()

    def export_evidence_snapshot(self) -> dict[str, object]:
        with self._lock:
            service = self._service or self._last_service
            gateway = self._gateway or self._last_gateway
            recorder = self._recorder or self._last_recorder
            output = self._output_dir
        if service is None or output is None:
            raise RuntimeError("no WP4 evidence is available")
        path = output / "evidence-snapshot.json"
        _write_json(
            path,
            {
                "service": service.evidence(),
                "media_ingress": (
                    {
                        "schema": "xr02.mediamtx.direct_diagnostic.v1",
                        "mode": "direct_camera_diagnostic",
                    }
                    if gateway is None
                    else gateway.evidence()
                ),
                "trial_recording": (
                    {"schema": "xr02.wp4.trial_recording.v1", "enabled": False}
                    if recorder is None
                    else recorder.evidence()
                ),
            },
        )
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}

    def status(self) -> dict[str, object]:
        with self._lock:
            service = self._service or self._last_service
            gateway = self._gateway or self._last_gateway
            recorder = self._recorder or self._last_recorder
            active = self._service is not None
            output = self._output_dir
            manifest = self._last_manifest
        return {
            "schema": "xr02.wp4.operator_status.v4",
            "active": active,
            "output_directory": None if output is None else str(output),
            "manifest": None if manifest is None else str(manifest),
            "service": (
                {
                    "state": "stopped",
                    "running": False,
                    "scene_update_available": False,
                    "camera_health": [],
                    "global_tracks": [],
                }
                if service is None
                else service.status()
            ),
            "operator_shell": "project-owned-stdlib-browser",
            "media_ingress": (
                {
                    "schema": "xr02.mediamtx.gateway_status.v1",
                    "state": (
                        "direct_camera_diagnostic"
                        if self.direct_camera_diagnostic
                        else "configured_mediamtx"
                    ),
                    "adopted_default": not self.direct_camera_diagnostic,
                    "path_health": [],
                }
                if gateway is None
                else gateway.status()
            ),
            "trial_recording": (
                {
                    "schema": "xr02.wp4.trial_recording_status.v1",
                    "configured": self.trial_recording_policy is not None,
                    "active": False,
                }
                if recorder is None
                else recorder.status()
            ),
            "trackstudio_role": "not used; optional/removable WP1 evidence retained",
            "rerun_role": "authoritative native 3D archive/live viewer",
        }

    def _resolve_scene(self) -> AdoptedSceneSelection:
        return resolve_adopted_scene(
            self.paths.operator_state,
            self.paths.p06,
            self.paths.p07,
            self.paths.p08_floor_manifest,
        )

    def _write_manifest(
        self,
        output: Path,
        service: XR02LiveService,
        gateway: MediaMtxGateway | None,
        recorder: MediaMtxTrialRecorder | None,
    ) -> Path:
        rrd = output / "xr02-wp4-live.rrd"
        local_journal = output / "wp4-local-observations.jsonl"
        global_journal = output / "wp4-global-association.jsonl"
        evidence = service.evidence()
        manifest: dict[str, object] = {
            "schema": "xr02.wp4.live_manifest.v4",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "scope": "bounded internal R&D multi-camera multi-person live demonstrator",
            "scene": service.scene.as_dict(),
            "runtime_secret_references": [f"PHASE2_RTSP_CAMERA_{index}" for index in range(1, 5)],
            "credentials_persisted": False,
            "client_frames_persisted_separately": recorder is not None,
            "dynamic_da3_invoked": False,
            "trackstudio_used": False,
            "media_ingress": (
                {
                    "schema": "xr02.mediamtx.direct_diagnostic.v1",
                    "mode": "direct_camera_diagnostic",
                    "adopted_default": False,
                    "credentials_persisted": False,
                }
                if gateway is None
                else gateway.evidence()
            ),
            "trial_recording": (
                {"schema": "xr02.wp4.trial_recording.v1", "enabled": False}
                if recorder is None
                else recorder.evidence()
            ),
            "operator_shell": "project-owned-stdlib-browser",
            "service": evidence,
            "artifacts": {
                "rerun": _identity(rrd),
                "local_journal": _identity(local_journal),
                "global_journal": _identity(global_journal),
                "trial_video": (
                    []
                    if recorder is None
                    else [item["artifact"] for item in recorder.capture_artifacts()]
                ),
            },
            "packages": _package_versions(),
            "claims": {
                "standard_idf1_or_hota": False,
                "survey_or_safety": False,
                "biometric_identity": False,
                "production_sla": False,
                "wp5_or_p10_complete": False,
            },
        }
        path = output / "wp4-live-manifest.json"
        _write_json(path, manifest)
        return path


def _environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("av", "boxmot", "numpy", "rerun-sdk", "scipy", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def validate_wp4_runtime() -> dict[str, str]:
    """Fail before serving the operator UI when the pinned live runtime is unavailable."""

    actual_python = Path(sys.executable).resolve()
    expected_python = WP4_PINNED_PYTHON.resolve()
    if actual_python != expected_python:
        raise RuntimeError(
            "XR02 WP4 runtime preflight failed: the operator console must be launched with "
            f"{expected_python}; received {actual_python}"
        )
    actual_versions: dict[str, str] = {}
    failures: list[str] = []
    imported: dict[str, object] = {}
    for distribution, module_name, expected_version in WP4_REQUIRED_RUNTIME:
        try:
            actual_version = version(distribution)
        except PackageNotFoundError:
            failures.append(f"{distribution} is not installed")
            continue
        actual_versions[distribution] = actual_version
        if actual_version != expected_version:
            failures.append(
                f"{distribution} expected {expected_version}, received {actual_version}"
            )
            continue
        try:
            imported[module_name] = import_module(module_name)
        except Exception as error:
            failures.append(f"{distribution} import failed: {type(error).__name__}: {error}")
    torch_module = imported.get("torch")
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            failures.append("torch CUDA is unavailable")
    if failures:
        raise RuntimeError("XR02 WP4 runtime preflight failed: " + "; ".join(failures))
    return actual_versions


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


def _write_json(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    lowered = serialized.lower()
    if "rtsp://" in lowered or "rtsps://" in lowered:
        raise RuntimeError("credential-safe evidence unexpectedly contains an endpoint")
    path.write_text(serialized, encoding="utf-8")


def apply_offline_model_controls() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "1")
    artifact_root = Path(os.environ.get("SPATIAL_MAPPING_ARTIFACT_ROOT", "runtime_data"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(artifact_root / "cache" / "ultralytics"))
