"""WP4 service construction, operator lifecycle and credential-safe evidence export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint, load_local_rtsp_endpoints
from spatial_mapping_phase2.p09_projection import (
    FrozenProjectionInputs,
    load_frozen_projection_inputs,
)
from spatial_mapping_phase2.xr02_association import office_topology
from spatial_mapping_phase2.xr02_compact_telemetry import CompactLiveTelemetryJournal
from spatial_mapping_phase2.xr02_live_domain import AdoptedSceneSelection, resolve_adopted_scene
from spatial_mapping_phase2.xr02_live_pipeline import LiveModelProfile, XR02LivePipeline
from spatial_mapping_phase2.xr02_live_rerun import XR02LiveRerunLogger
from spatial_mapping_phase2.xr02_live_service import XR02LiveService, XR02LiveServiceConfig
from spatial_mapping_phase2.xr02_mediamtx import (
    MediaMtxGateway,
    MediaMtxGatewayPolicy,
)
from spatial_mapping_phase2.xr02_recording_catalog import (
    OperatorRun,
    RecordingCatalogError,
    XR02RecordingCatalog,
    XR02RunMode,
    XR02RunState,
)
from spatial_mapping_phase2.xr02_trial_recording import (
    MediaMtxTrialRecorder,
    TrialRecordingPolicy,
)
from spatial_mapping_phase2.xr03_camera_policy import CameraPolicyRepository, SceneCameraPolicy
from spatial_mapping_phase2.xr03_xr02_policy import topology_from_camera_policy

WP4_REQUIRED_PYTHON = (3, 11)
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
    camera_policy: Path | None = None


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
        recording_free_space_reserve_bytes: int = 5 * 1024**3,
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
        if recording_free_space_reserve_bytes <= 0:
            raise RuntimeError("recording free-space reserve must be positive")
        self.recording_free_space_reserve_bytes = recording_free_space_reserve_bytes
        self.paths.output_root.mkdir(parents=True, exist_ok=True)
        self._catalog = XR02RecordingCatalog(self.paths.output_root / "xr02-recordings.sqlite3")
        self._lock = threading.RLock()
        self._service: XR02LiveService | None = None
        self._last_service: XR02LiveService | None = None
        self._gateway: MediaMtxGateway | None = None
        self._last_gateway: MediaMtxGateway | None = None
        self._recorder: MediaMtxTrialRecorder | None = None
        self._last_recorder: MediaMtxTrialRecorder | None = None
        self._output_dir: Path | None = None
        self._last_manifest: Path | None = None
        self._active_run: OperatorRun | None = None
        self._telemetry: CompactLiveTelemetryJournal | None = None
        self._started_monotonic: float | None = None
        self._viewer_opened = False
        self._disk_guard_stop: threading.Event | None = None
        self._disk_guard_thread: threading.Thread | None = None
        self._storage_guard_message: str | None = None

    def start(self) -> dict[str, object]:
        """Compatibility entrypoint preserving the former full-evidence start."""

        return self.start_recording()

    def start_live(
        self,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, object]:
        return self._start(
            XR02RunMode.LIVE,
            resumed_from_session_id=resumed_from_session_id,
            scene_update_id=scene_update_id,
        )

    def start_recording(self) -> dict[str, object]:
        return self._start(XR02RunMode.RECORDING)

    def _start(
        self,
        mode: XR02RunMode,
        *,
        resumed_from_session_id: str | None = None,
        scene_update_id: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self._service is not None:
                raise RuntimeError("XR02 WP4 is already running")
            if mode is XR02RunMode.RECORDING:
                if self.trial_recording_policy is None:
                    raise RuntimeError(
                        "replayable recording requires the console recording profile"
                    )
                free = shutil.disk_usage(self.paths.output_root).free
                if free < self.recording_free_space_reserve_bytes:
                    raise RuntimeError(
                        "recording start blocked by the configured free-space reserve"
                    )
            if mode is XR02RunMode.RECORDING and (
                resumed_from_session_id is not None or scene_update_id is not None
            ):
                raise RuntimeError("only Live Service may resume after a scene update")
            if (resumed_from_session_id is None) != (scene_update_id is None):
                raise RuntimeError("Live resume requires both prior session and scene-update IDs")
            camera_policy = (
                CameraPolicyRepository.open(self.paths.camera_policy).active(require_overlap=True)
                if self.paths.camera_policy is not None
                else None
            )
            scene = self._resolve_scene(None if camera_policy is None else camera_policy.sha256)
            prefix = "xr02-live" if mode is XR02RunMode.LIVE else "xr02-recording"
            output = self.paths.output_root / datetime.now(UTC).strftime(
                f"{prefix}-%Y%m%dT%H%M%S%fZ"
            )
            run = self._catalog.begin(
                mode,
                output,
                scene_context_sha256=scene.scene.context_sha256,
                scene_binding_sha256=scene.selection_signature_sha256,
                resumed_from_session_id=resumed_from_session_id,
                scene_update_id=scene_update_id,
            )
            try:
                output.mkdir(parents=True, exist_ok=False)
            except Exception as error:
                self._catalog.require_recovery(run.session_id, str(error))
                raise
            logger: XR02LiveRerunLogger | None = None
            telemetry: CompactLiveTelemetryJournal | None = None
            gateway: MediaMtxGateway | None = None
            recorder: MediaMtxTrialRecorder | None = None
            try:
                pipeline, logger, telemetry, upstream_endpoints = self._build_run_components(
                    mode, output, scene, camera_policy
                )
                service_endpoints = upstream_endpoints
                if not self.direct_camera_diagnostic:
                    gateway_policy = self.media_gateway_policy
                    if gateway_policy is None:
                        raise RuntimeError("MediaMTX policy disappeared after validation")
                    gateway = MediaMtxGateway(
                        upstream_endpoints,
                        gateway_policy,
                        output / "media-gateway",
                    )
                    gateway.start()
                    service_endpoints = gateway.local_endpoints()
                if mode is XR02RunMode.RECORDING and self.trial_recording_policy is not None:
                    recorder = MediaMtxTrialRecorder(
                        service_endpoints,
                        self.trial_recording_policy,
                        output / "trial-video",
                    )
                    recorder.start()
                service = XR02LiveService(
                    service_endpoints,
                    pipeline.scene,
                    pipeline,
                    logger,
                    self._resolve_scene,
                    self.config,
                    compact_telemetry=telemetry,
                    retain_full_evidence=mode is XR02RunMode.RECORDING,
                )
                service.start()
                run = self._catalog.mark_running(run.session_id)
            except Exception as error:
                if recorder is not None:
                    recorder.stop()
                if logger is not None:
                    logger.close()
                if telemetry is not None:
                    telemetry.close()
                if gateway is not None:
                    gateway.stop()
                try:
                    self._catalog.require_recovery(run.session_id, str(error))
                except RecordingCatalogError:
                    pass
                raise
            self._service = service
            self._last_service = service
            self._gateway = gateway
            self._last_gateway = gateway
            self._recorder = recorder
            self._last_recorder = recorder
            self._output_dir = output
            self._last_manifest = None
            self._active_run = run
            self._telemetry = telemetry
            self._started_monotonic = monotonic()
            self._viewer_opened = False
            self._storage_guard_message = None
            if mode is XR02RunMode.RECORDING:
                self._start_disk_guard(run.session_id)
            return self.status()

    def _build_run_components(
        self,
        mode: XR02RunMode,
        output: Path,
        scene: AdoptedSceneSelection,
        camera_policy: SceneCameraPolicy | None,
    ) -> tuple[
        XR02LivePipeline,
        XR02LiveRerunLogger,
        CompactLiveTelemetryJournal | None,
        tuple[LocalRtspEndpoint, ...],
    ]:
        frozen = FrozenProjectionInputs(
            self.paths.p06,
            self.hashes.p06,
            self.paths.p07,
            self.hashes.p07,
            Path(scene.p08_floor_manifest.path),
            scene.p08_floor_manifest.sha256,
            Path(scene.floor.path),
            scene.floor.sha256,
        )
        calibrations, floor = load_frozen_projection_inputs(frozen)
        topology = office_topology()
        if camera_policy is not None:
            if camera_policy.camera_ids != tuple(calibrations):
                raise RuntimeError("active camera policy differs from the XR02 calibration roster")
            topology = topology_from_camera_policy(
                camera_policy,
                transition_edges=topology.transition_edges,
            )
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
            topology=topology,
            durable_evidence=mode is XR02RunMode.RECORDING,
        )
        logger: XR02LiveRerunLogger | None = None
        try:
            logger = XR02LiveRerunLogger(
                output / "xr02-wp4-live.rrd",
                scene,
                calibrations,
                floor,
                Path(sys.executable),
                durable_archive=mode is XR02RunMode.RECORDING,
            )
            telemetry = (
                CompactLiveTelemetryJournal(output / "live-telemetry.jsonl")
                if mode is XR02RunMode.LIVE
                else None
            )
            environment = _environment(self.paths.environment_file)
            endpoints = load_local_rtsp_endpoints(environment)
        except Exception:
            if logger is not None:
                logger.close()
            raise
        return pipeline, logger, telemetry, endpoints

    def stop(self, *, reason: str = "operator") -> dict[str, object]:
        normalized_reason = " ".join(reason.split())
        if not normalized_reason or len(normalized_reason) > 80:
            raise RuntimeError("XR02 stop reason must contain 1..80 visible characters")
        with self._lock:
            service = self._service
            gateway = self._gateway
            recorder = self._recorder
            output = self._output_dir
            run = self._active_run
            if service is None or output is None or run is None:
                return self.status()
            run = self._catalog.mark_finalizing(run.session_id)
            disk_guard_stop = self._disk_guard_stop
            if disk_guard_stop is not None:
                disk_guard_stop.set()
            self._disk_guard_stop = None
            self._disk_guard_thread = None
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
        try:
            if service_error is not None:
                raise service_error
            manifest = (
                self._write_manifest(output, service, gateway, recorder)
                if run.mode is XR02RunMode.RECORDING
                else self._write_live_summary(output, service, gateway)
            )
            telemetry_path = (
                output / "live-telemetry.jsonl" if run.mode is XR02RunMode.LIVE else None
            )
            run = self._catalog.finalize(
                run.session_id,
                manifest_path=manifest,
                telemetry_path=telemetry_path,
                byte_count=_directory_bytes(output),
                stop_reason=normalized_reason,
            )
        except Exception as error:
            try:
                run = self._catalog.require_recovery(run.session_id, str(error))
            except RecordingCatalogError:
                pass
            with self._lock:
                self._active_run = None
                self._telemetry = None
                self._started_monotonic = None
            raise
        with self._lock:
            self._last_manifest = manifest
            self._active_run = None
            self._telemetry = None
            self._started_monotonic = None
        return self.status()

    def _start_disk_guard(self, session_id: str) -> None:
        stop = threading.Event()

        def guard() -> None:
            while not stop.wait(5.0):
                free = shutil.disk_usage(self.paths.output_root).free
                if free >= self.recording_free_space_reserve_bytes:
                    continue
                with self._lock:
                    active = self._active_run
                    if active is None or active.session_id != session_id:
                        return
                    self._storage_guard_message = (
                        "Recording stopped automatically at the configured free-space reserve."
                    )
                try:
                    self.stop(reason="recording_storage_reserve")
                except Exception as error:
                    with self._lock:
                        self._storage_guard_message = (
                            "Recording storage guard entered recovery: " + type(error).__name__
                        )
                return

        thread = threading.Thread(
            target=guard,
            name="xr02-recording-storage-guard",
            daemon=True,
        )
        self._disk_guard_stop = stop
        self._disk_guard_thread = thread
        thread.start()

    def open_rerun(self) -> dict[str, object]:
        with self._lock:
            service = self._service or self._last_service
        if service is None:
            raise RuntimeError("start one WP4 run before opening Rerun")
        service.open_viewer()
        with self._lock:
            self._viewer_opened = True
        return self.status()

    def view_recording(self, session_id: str) -> dict[str, object]:
        run = self._catalog.require(session_id)
        if run.mode is not XR02RunMode.RECORDING or run.state not in {
            XR02RunState.AWAITING_DISPOSITION,
            XR02RunState.SAVED,
        }:
            raise RuntimeError("only finalized replayable recordings can be viewed")
        with self._lock:
            service = (
                self._last_service
                if self._output_dir is not None
                and self._output_dir.resolve() == run.run_directory.resolve()
                else None
            )
        if service is not None:
            service.open_viewer()
        else:
            rrd = run.run_directory / "xr02-wp4-live.rrd"
            if not rrd.is_file():
                raise RuntimeError("saved recording is missing its Rerun archive")
            rerun = Path(sys.executable).resolve().with_name("rerun.exe")
            if not rerun.is_file():
                raise RuntimeError("Rerun CLI is missing from the selected runtime")
            subprocess.Popen(
                [str(rerun), str(rrd), "--port", "0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        return self.status()

    def save_recording(self, session_id: str, label: str) -> dict[str, object]:
        self._catalog.save(session_id, label)
        return self.status()

    def delete_recording(self, session_id: str, confirmation: str) -> dict[str, object]:
        if confirmation != f"DELETE {session_id}":
            raise RuntimeError("delete confirmation does not match the exact staged run")
        run = self._catalog.require(session_id)
        if run.state not in {
            XR02RunState.AWAITING_DISPOSITION,
            XR02RunState.RECOVERY_REQUIRED,
        }:
            raise RuntimeError("only a staged or recovery run can be deleted")
        root = self.paths.output_root.resolve()
        target = run.run_directory.resolve()
        if target.parent != root or not target.name.startswith("xr02-"):
            raise RuntimeError("refusing to delete a run outside the XR02 output root")
        with self._lock:
            service = (
                self._last_service
                if self._output_dir is not None and self._output_dir.resolve() == target
                else None
            )
        if service is not None:
            service.close_viewer()
        if target.exists():
            shutil.rmtree(target)
        self._catalog.mark_deleted(session_id)
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
            active_run = self._active_run
            started_monotonic = self._started_monotonic
            viewer_opened = self._viewer_opened
            storage_guard_message = self._storage_guard_message
        service_status = (
            {
                "state": "stopped",
                "running": False,
                "scene_update_available": False,
                "camera_health": [],
                "global_tracks": [],
            }
            if service is None
            else service.status()
        )
        health_value = service_status.get("camera_health", [])
        health = health_value if isinstance(health_value, list) else []
        all_green = bool(health) and all(
            isinstance(item, dict) and item.get("state") == "current" for item in health
        )
        startup_elapsed = (
            0.0 if started_monotonic is None else max(0.0, monotonic() - started_monotonic)
        )
        auto_open_eligible = (
            active and not viewer_opened and (all_green or startup_elapsed >= 30.0)
        )
        blocking = self._catalog.blocking()
        pending = (
            blocking
            if blocking is not None
            and blocking.state
            in {XR02RunState.AWAITING_DISPOSITION, XR02RunState.RECOVERY_REQUIRED}
            else None
        )
        if active_run is not None:
            operator_state = f"{active_run.mode.value}_running"
        elif pending is not None:
            operator_state = pending.state.value
        else:
            operator_state = "ready"
        return {
            "schema": "xr02.wp4.operator_status.v5",
            "active": active,
            "active_mode": None if active_run is None else active_run.mode.value,
            "active_session_id": None if active_run is None else active_run.session_id,
            "operator_state": operator_state,
            "pending_run": None if pending is None else pending.as_dict(),
            "saved_recordings": [item.as_dict() for item in self._catalog.saved()],
            "recent_live_runs": [item.as_dict() for item in self._catalog.recent_live()],
            "recording_available": self.trial_recording_policy is not None,
            "storage": {
                "free_bytes": shutil.disk_usage(self.paths.output_root).free,
                "recording_reserve_bytes": self.recording_free_space_reserve_bytes,
                "guard_message": storage_guard_message,
            },
            "viewer_auto_open": {
                "eligible": auto_open_eligible,
                "opened": viewer_opened,
                "all_cameras_current": all_green,
                "startup_elapsed_seconds": startup_elapsed,
                "timeout_seconds": 30.0,
                "degraded_after_timeout": auto_open_eligible and not all_green,
            },
            "output_directory": None if output is None else str(output),
            "manifest": None if manifest is None else str(manifest),
            "service": service_status,
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

    def _resolve_scene(self, camera_policy_sha256: str | None = None) -> AdoptedSceneSelection:
        return resolve_adopted_scene(
            self.paths.operator_state,
            self.paths.p06,
            self.paths.p07,
            self.paths.p08_floor_manifest,
            camera_policy_sha256,
        )

    def _write_live_summary(
        self,
        output: Path,
        service: XR02LiveService,
        gateway: MediaMtxGateway | None,
    ) -> Path:
        telemetry = output / "live-telemetry.jsonl"
        path = output / "live-run-summary.json"
        _write_json(
            path,
            {
                "schema": "xr02.live_run_summary.v1",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "retention_profile": "compact-anonymous-telemetry-only",
                "scene": service.scene.as_dict(),
                "credentials_persisted": False,
                "client_frames_persisted": False,
                "rerun_archive_persisted": False,
                "decision_journals_persisted": False,
                "appearance_gallery_persisted": False,
                "compact_telemetry": _identity(telemetry),
                "service_summary": service.evidence(),
                "media_ingress": (
                    {
                        "schema": "xr02.mediamtx.direct_diagnostic.v1",
                        "mode": "direct_camera_diagnostic",
                    }
                    if gateway is None
                    else gateway.evidence()
                ),
            },
        )
        return path

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


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


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

    actual_python = sys.version_info[:2]
    if actual_python != WP4_REQUIRED_PYTHON:
        raise RuntimeError(
            "XR02 WP4 runtime preflight failed: Python "
            f"{WP4_REQUIRED_PYTHON[0]}.{WP4_REQUIRED_PYTHON[1]} is required; "
            f"received {actual_python[0]}.{actual_python[1]}"
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


def apply_offline_model_controls(ultralytics_config: Path | None = None) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["YOLO_OFFLINE"] = "1"
    if ultralytics_config is not None:
        os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
