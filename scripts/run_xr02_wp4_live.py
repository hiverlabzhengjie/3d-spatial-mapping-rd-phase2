"""Launch the authorized XR02 WP4 live demonstrator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.environ.get("SPATIAL_MAPPING_ARTIFACT_ROOT", "runtime_data")).resolve()
WP2_OVERLAY = Path(
    os.environ.get("XR02_OPTIONAL_OVERLAY", str(ARTIFACT_ROOT / "runtime" / "xr02-overlay"))
)
sys.path.insert(0, str(REPOSITORY / "src"))
if WP2_OVERLAY.is_dir():
    sys.path.insert(0, str(WP2_OVERLAY))

from spatial_mapping_phase2.xr02_live_service import XR02LiveServiceConfig  # noqa: E402
from spatial_mapping_phase2.xr02_mediamtx import MediaMtxGatewayPolicy  # noqa: E402
from spatial_mapping_phase2.xr02_operator_web import XR02OperatorServer  # noqa: E402
from spatial_mapping_phase2.xr02_rtsp_capture import CaptureBackend  # noqa: E402
from spatial_mapping_phase2.xr02_trial_recording import TrialRecordingPolicy  # noqa: E402
from spatial_mapping_phase2.xr02_wp4 import (  # noqa: E402
    WP4Hashes,
    WP4Paths,
    XR02WP4Controller,
    apply_offline_model_controls,
    validate_wp4_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless-seconds", type=float)
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--open-rerun", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARTIFACT_ROOT / "runs",
    )
    parser.add_argument(
        "--mediamtx-binary",
        type=Path,
        default=ARTIFACT_ROOT / "bin" / "mediamtx.exe",
    )
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
    parser.add_argument(
        "--record-trial-video",
        action="store_true",
        help="Retain credential-free MediaMTX stream-copy MKVs for replay diagnostics.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        type=Path,
        help="FFmpeg executable used only when --record-trial-video is enabled.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=tuple(item.value for item in CaptureBackend),
        default=CaptureBackend.PYAV.value,
    )
    parser.add_argument(
        "--gstreamer-overlay",
        type=Path,
        default=ARTIFACT_ROOT / "runtime" / "gstreamer-overlay",
    )
    args = parser.parse_args()
    ffmpeg_binary = args.ffmpeg_binary
    if args.record_trial_video and ffmpeg_binary is None:
        discovered = shutil.which("ffmpeg")
        if discovered is None:
            parser.error("--record-trial-video requires FFmpeg on PATH or --ffmpeg-binary")
        ffmpeg_binary = Path(discovered)
    apply_offline_model_controls()
    runtime_versions = validate_wp4_runtime()
    print(
        "XR02 pinned runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in runtime_versions.items())
    )
    controller = _controller(
        args.output_root,
        args.local_tracking_hz,
        args.appearance_hz,
        args.association_hz,
        args.publication_hz,
        args.inference_pending_max_age_ms,
        CaptureBackend(args.capture_backend),
        args.gstreamer_overlay,
        args.mediamtx_binary,
        args.direct_camera_diagnostic,
        args.record_trial_video,
        ffmpeg_binary,
    )
    if args.headless_seconds is not None:
        if args.headless_seconds <= 0:
            parser.error("--headless-seconds must be positive")
        controller.start()
        try:
            if args.open_rerun:
                controller.open_rerun()
            time.sleep(args.headless_seconds)
        finally:
            status = controller.stop()
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    server = XR02OperatorServer(controller, args.port)
    print(f"XR02 operator console: {server.url}")
    print("RTSP endpoint values are held in memory and will never be printed.")
    try:
        server.serve_forever(open_browser=not args.no_browser)
    except KeyboardInterrupt:
        controller.stop()
    return 0


def _controller(
    output_root: Path,
    local_tracking_hz: float,
    appearance_hz: float,
    association_hz: float,
    publication_hz: float,
    inference_pending_max_age_ms: float,
    capture_backend: CaptureBackend = CaptureBackend.PYAV,
    gstreamer_overlay: Path | None = None,
    mediamtx_binary: Path = ARTIFACT_ROOT / "bin" / "mediamtx.exe",
    direct_camera_diagnostic: bool = False,
    record_trial_video: bool = False,
    ffmpeg_binary: Path | None = None,
) -> XR02WP4Controller:
    if record_trial_video and ffmpeg_binary is None:
        raise RuntimeError("trial-video recording requires an explicit FFmpeg executable")
    paths = WP4Paths(
        operator_state=ARTIFACT_ROOT / "inputs" / "operator-state.json",
        p06=ARTIFACT_ROOT / "inputs" / "p06-input-manifest.json",
        p07=ARTIFACT_ROOT / "inputs" / "p07-frustum-preview-manifest.json",
        p08_floor_manifest=ARTIFACT_ROOT / "inputs" / "p08-floor-completion-manifest.json",
        p08_floor=ARTIFACT_ROOT / "inputs" / "authoritative_floor_plane.npz",
        detector=ARTIFACT_ROOT / "model_weights" / "yolo11n.pt",
        reid=ARTIFACT_ROOT / "model_weights" / "osnet_x0_25_msmt17.pt",
        output_root=output_root,
        environment_file=REPOSITORY / ".env",
    )
    hashes = WP4Hashes(
        p06="d3c1bfd314a865270a9d9352efd33cd61a470bdc8829a084d5a0f2996f4aa8e4",
        p07="df883eedae46f48aab9c84a86b8e2398fa44b37c2664cac4792d21e5a7d8ef51",
        p08_floor_manifest=("1462f65068156b4ffe611fd705b7ae62468fe8b665cd5eebae8ed96132adc399"),
        p08_floor="1079e8573938c19bd668a73c3bb7706c684fd661a36c95db85eb64592cb25eb0",
        detector="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        reid="6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99",
    )
    return XR02WP4Controller(
        paths,
        hashes,
        XR02LiveServiceConfig(
            local_tracking_hz=local_tracking_hz,
            appearance_refresh_hz=appearance_hz,
            association_hz=association_hz,
            publication_hz=publication_hz,
            inference_pending_max_age_ms=inference_pending_max_age_ms,
            capture_backend=capture_backend,
            gstreamer_overlay_path=(
                str(gstreamer_overlay.resolve())
                if capture_backend is CaptureBackend.GSTREAMER and gstreamer_overlay is not None
                else None
            ),
        ),
        media_gateway_policy=(
            None
            if direct_camera_diagnostic
            else MediaMtxGatewayPolicy(binary_path=mediamtx_binary)
        ),
        trial_recording_policy=(
            TrialRecordingPolicy(ffmpeg_binary)
            if record_trial_video and ffmpeg_binary is not None
            else None
        ),
        direct_camera_diagnostic=direct_camera_diagnostic,
    )


if __name__ == "__main__":
    raise SystemExit(main())
