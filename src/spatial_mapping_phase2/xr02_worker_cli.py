"""Launch the isolated XR02 live worker from an explicit deployment manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from spatial_mapping_phase2.xr02_deployment import (
    XR02Deployment,
    XR02DeploymentError,
    load_xr02_deployment,
)

if TYPE_CHECKING:
    from spatial_mapping_phase2.xr02_wp4 import XR02WP4Controller


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        deployment = load_xr02_deployment(arguments.deployment_config)
    except XR02DeploymentError as error:
        parser.error(str(error))
    if deployment.wp2_overlay is not None:
        if not deployment.wp2_overlay.is_dir():
            parser.error(f"configured WP2 overlay is unavailable: {deployment.wp2_overlay}")
        sys.path.insert(0, str(deployment.wp2_overlay))

    from spatial_mapping_phase2.xr02_wp4 import apply_offline_model_controls, validate_wp4_runtime

    apply_offline_model_controls(deployment.ultralytics_config)
    runtime_versions = validate_wp4_runtime()
    print(
        "XR02 pinned runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in runtime_versions.items())
    )
    if arguments.preflight:
        return 0
    from spatial_mapping_phase2.xr02_rtsp_capture import CaptureBackend

    ffmpeg_binary = arguments.ffmpeg_binary
    if arguments.record_trial_video and ffmpeg_binary is None:
        discovered = shutil.which("ffmpeg")
        if discovered is None:
            parser.error("--record-trial-video requires FFmpeg on PATH or --ffmpeg-binary")
        ffmpeg_binary = Path(discovered)
    if not arguments.direct_camera_diagnostic and arguments.mediamtx_binary is None:
        parser.error("adopted ingress requires --mediamtx-binary")
    if (
        arguments.capture_backend == CaptureBackend.GSTREAMER.value
        and arguments.gstreamer_overlay is None
    ):
        parser.error("GStreamer capture requires --gstreamer-overlay")
    controller = _controller(deployment, arguments, ffmpeg_binary)
    if arguments.headless_seconds is not None:
        if arguments.headless_seconds <= 0:
            parser.error("--headless-seconds must be positive")
        if arguments.record_trial_video:
            controller.start_recording()
        else:
            controller.start_live()
        try:
            if arguments.open_rerun:
                controller.open_rerun()
            time.sleep(arguments.headless_seconds)
        finally:
            status = controller.stop()
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    from spatial_mapping_phase2.xr02_operator_web import XR02OperatorServer

    api_token = os.environ.get("XR02_WORKER_TOKEN") if arguments.api_only else None
    if arguments.api_only and not api_token:
        parser.error("--api-only requires XR02_WORKER_TOKEN in the child environment")
    server = XR02OperatorServer(
        controller,
        arguments.port,
        api_token=api_token,
        serve_page=not arguments.api_only,
    )
    print(f"XR02 operator console: {server.url}")
    print("RTSP endpoint values are held in memory and will never be printed.")
    try:
        server.serve_forever(open_browser=not arguments.no_browser)
    except KeyboardInterrupt:
        controller.stop()
    return 0


def _controller(
    deployment: XR02Deployment,
    arguments: argparse.Namespace,
    ffmpeg_binary: Path | None,
) -> XR02WP4Controller:
    from spatial_mapping_phase2.xr02_live_service import XR02LiveServiceConfig
    from spatial_mapping_phase2.xr02_mediamtx import MediaMtxGatewayPolicy
    from spatial_mapping_phase2.xr02_rtsp_capture import CaptureBackend
    from spatial_mapping_phase2.xr02_trial_recording import TrialRecordingPolicy
    from spatial_mapping_phase2.xr02_wp4 import XR02WP4Controller

    if arguments.recording_free_space_reserve_gb <= 0:
        raise RuntimeError("recording free-space reserve must be positive")
    capture_backend = CaptureBackend(arguments.capture_backend)
    return XR02WP4Controller(
        deployment.wp4_paths(arguments.output_root),
        deployment.wp4_hashes(),
        XR02LiveServiceConfig(
            local_tracking_hz=arguments.local_tracking_hz,
            appearance_refresh_hz=arguments.appearance_hz,
            association_hz=arguments.association_hz,
            publication_hz=arguments.publication_hz,
            inference_pending_max_age_ms=arguments.inference_pending_max_age_ms,
            capture_backend=capture_backend,
            gstreamer_overlay_path=(
                str(arguments.gstreamer_overlay.resolve())
                if capture_backend is CaptureBackend.GSTREAMER
                and arguments.gstreamer_overlay is not None
                else None
            ),
        ),
        media_gateway_policy=(
            None
            if arguments.direct_camera_diagnostic
            else MediaMtxGatewayPolicy(binary_path=arguments.mediamtx_binary)
        ),
        trial_recording_policy=(
            TrialRecordingPolicy(ffmpeg_binary)
            if arguments.record_trial_video and ffmpeg_binary is not None
            else None
        ),
        direct_camera_diagnostic=arguments.direct_camera_diagnostic,
        recording_free_space_reserve_bytes=int(
            round(arguments.recording_free_space_reserve_gb * 1024**3)
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate the pinned runtime and deployment manifest without starting a run or server."
        ),
    )
    parser.add_argument("--headless-seconds", type=float)
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Disable the standalone operator page and require the integrated-console token.",
    )
    parser.add_argument("--open-rerun", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--mediamtx-binary", type=Path)
    parser.add_argument(
        "--direct-camera-diagnostic",
        action="store_true",
        help="Explicit rollback/diagnostic only; bypasses adopted MediaMTX ingress.",
    )
    parser.add_argument("--local-tracking-hz", type=float, default=8.0)
    parser.add_argument("--appearance-hz", type=float, default=2.0)
    parser.add_argument("--association-hz", type=float, default=8.0)
    parser.add_argument("--publication-hz", type=float, default=2.0)
    parser.add_argument("--inference-pending-max-age-ms", type=float, default=150.0)
    parser.add_argument("--record-trial-video", action="store_true")
    parser.add_argument("--recording-free-space-reserve-gb", type=float, default=5.0)
    parser.add_argument("--ffmpeg-binary", type=Path)
    parser.add_argument("--capture-backend", choices=("pyav", "gstreamer"), default="pyav")
    parser.add_argument("--gstreamer-overlay", type=Path)
    return parser
