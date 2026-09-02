"""Standalone few-button and headless launch surface for the P09 live demonstrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

from spatial_mapping_phase2.p01_observability import load_local_rtsp_endpoints
from spatial_mapping_phase2.p09_fusion import AnonymousWorldTracker, FusionConfig
from spatial_mapping_phase2.p09_live_service import LiveServiceConfig, P09LiveService
from spatial_mapping_phase2.p09_pipeline import P09TrackingPipeline
from spatial_mapping_phase2.p09_projection import (
    FrozenProjectionInputs,
    load_frozen_projection_inputs,
)
from spatial_mapping_phase2.p09_rerun import P09RerunLogger
from spatial_mapping_phase2.p09_yolo11 import Yolo11CudaDetector, Yolo11ModelSpec

ARTIFACT_ROOT = Path(os.environ.get("SPATIAL_MAPPING_ARTIFACT_ROOT", "runtime_data")).resolve()
os.environ.setdefault("YOLO_CONFIG_DIR", str(ARTIFACT_ROOT / "cache" / "ultralytics"))

P06 = ARTIFACT_ROOT / "inputs" / "p06-input-manifest.json"
P07 = ARTIFACT_ROOT / "inputs" / "p07-frustum-preview-manifest.json"
P08 = ARTIFACT_ROOT / "inputs" / "p08-floor-completion-manifest.json"
P08_PLANE = P08.parent / "authoritative_floor_plane.npz"
MODEL = ARTIFACT_ROOT / "model_weights" / "yolo11n.pt"
GEOMETRY = ARTIFACT_ROOT / "inputs" / "working-facility-geometry.npz"
HASHES = {
    "p06": os.environ.get("P09_P06_SHA256", "0" * 64),
    "p07": os.environ.get("P09_P07_SHA256", "0" * 64),
    "p08": os.environ.get("P09_P08_MANIFEST_SHA256", "0" * 64),
    "p08_plane": os.environ.get("P09_P08_PLANE_SHA256", "0" * 64),
    "geometry": os.environ.get("P09_GEOMETRY_SHA256", "0" * 64),
    "model": os.environ.get("P09_MODEL_SHA256", "0" * 64),
}


def _environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _build_service(
    output_dir: Path, environment: dict[str, str]
) -> tuple[P09LiveService, Yolo11CudaDetector]:
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs = FrozenProjectionInputs(
        P06, HASHES["p06"], P07, HASHES["p07"], P08, HASHES["p08"], P08_PLANE, HASHES["p08_plane"]
    )
    calibrations, floor = load_frozen_projection_inputs(inputs)
    if hashlib.sha256(GEOMETRY.read_bytes()).hexdigest() != HASHES["geometry"]:
        raise RuntimeError("frozen P07 geometry identity changed")
    detector = Yolo11CudaDetector(Yolo11ModelSpec(MODEL, HASHES["model"]))
    tracker = AnonymousWorldTracker(
        FusionConfig(
            maximum_observation_age_ms=750.0,
            maximum_cross_camera_skew_ms=300.0,
            spatial_gate_metres=1.25,
            maximum_speed_metres_per_second=3.0,
            motion_gate_slack_metres=0.75,
        )
    )
    pipeline = P09TrackingPipeline(calibrations, floor, detector, tracker)
    logger = P09RerunLogger(
        output_dir / "p09-live-tracking.rrd",
        calibrations,
        floor,
        P06,
        GEOMETRY,
        Path(sys.executable),
    )
    return (
        P09LiveService(
            load_local_rtsp_endpoints(environment),
            pipeline,
            logger,
            LiveServiceConfig(inference_hz=2.0, maximum_frame_age_ms=750.0),
        ),
        detector,
    )


def _write_manifest(
    output_dir: Path, service: P09LiveService, detector: Yolo11CudaDetector
) -> Path:
    rrd = output_dir / "p09-live-tracking.rrd"
    manifest: dict[str, Any] = {
        "schema_version": "p09-live-demo-v3-yolo11-live-rerun",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "anonymous_single_person_only": True,
        "dynamic_DA3_invoked": False,
        "credentials_persisted": False,
        "client_frames_persisted_separately": False,
        "detector": {
            "family": "YOLO11n",
            "backend": "ultralytics-pytorch-cuda",
            "confidence_threshold": detector.spec.confidence_threshold,
            "nms_iou_threshold": detector.spec.nms_iou_threshold,
            "cpu_fallback_allowed": False,
            "license": "AGPL-3.0",
            "open_source_release_commitment_confirmed": True,
            "cuda": asdict(detector.cuda_evidence()),
        },
        "frozen_sha256": HASHES,
        "runtime": service.evidence(),
        "rerun": {
            "path": str(rrd),
            "byte_count": rrd.stat().st_size,
            "sha256": hashlib.sha256(rrd.read_bytes()).hexdigest(),
        },
    }
    path = output_dir / "live-demo-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _headless(output_dir: Path, duration_seconds: float, open_viewer: bool) -> int:
    service, detector = _build_service(output_dir, _environment(Path(".env")))
    service.start()
    if open_viewer:
        service.open_viewer()
    try:
        sleep(duration_seconds)
    finally:
        service.stop()
    manifest = _write_manifest(output_dir, service, detector)
    print(json.dumps({"manifest": str(manifest), "status": service.status()}, indent=2))
    return 0


def _gui(output_root: Path) -> int:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("P09 Anonymous Person XY Tracker")
    root.geometry("760x500")
    root.configure(bg="#15181c")
    service: P09LiveService | None = None
    last_service: P09LiveService | None = None
    detector: Yolo11CudaDetector | None = None
    output_dir: Path | None = None
    state_var = tk.StringVar(value="Stopped")
    detail_var = tk.StringVar(value="Press Start live tracking. No identity is retained.")

    def start() -> None:
        nonlocal service, last_service, detector, output_dir
        if service is not None:
            messagebox.showinfo("P09", "Live tracking is already running.")
            return
        suffix = datetime.now(UTC).strftime("p09-live-%Y%m%dT%H%M%SZ")
        output_dir = output_root / suffix
        try:
            service, detector = _build_service(output_dir, _environment(Path(".env")))
            service.start()
            service.open_viewer()
            last_service = service
        except Exception as error:
            service = None
            detector = None
            state_var.set("Start failed")
            detail_var.set(type(error).__name__)
            messagebox.showerror("P09 start failed", str(error))
            return
        state_var.set("Starting four live sources…")

    def stop() -> None:
        nonlocal service, last_service, detector
        if service is None:
            return
        active = service
        service = None
        last_service = active
        try:
            active.stop()
            if output_dir is not None and detector is not None:
                manifest = _write_manifest(output_dir, active, detector)
                detail_var.set(f"Stopped cleanly. Evidence: {manifest}")
            detector = None
            state_var.set("Stopped")
        except Exception as error:
            state_var.set("Stop failed")
            detail_var.set(type(error).__name__)

    def open_rerun() -> None:
        viewer_service = service if service is not None else last_service
        if viewer_service is None:
            messagebox.showinfo("P09", "Start live tracking once to create a recording.")
            return
        try:
            viewer_service.open_viewer()
        except Exception as error:
            messagebox.showerror("P09 Rerun launch failed", str(error))

    def reset_trail() -> None:
        if service is not None:
            service.reset_trail()

    def refresh() -> None:
        if service is not None:
            status = service.status()
            state_var.set(str(status["global_state"]))
            decoders = status["decoders"]
            detail_var.set(
                "\n".join(
                    [str(status["global_reason"])]
                    + [
                        f"{decoder['camera_id']}: frames={decoder['decoded_frames']} "
                        f"reconnects={decoder['reconnects']} "
                        f"error={decoder['failure_class'] or 'none'}"
                        for decoder in decoders
                    ]
                    + [f"busy drops={status['worker']['busy_dropped_ticks']}"]
                )
            )
        root.after(500, refresh)

    def close_window() -> None:
        stop()
        root.destroy()

    tk.Label(
        root,
        text="P09 LIVE ANONYMOUS XY",
        fg="#8fe3ff",
        bg="#15181c",
        font=("Segoe UI", 22, "bold"),
    ).pack(pady=(24, 6))
    tk.Label(
        root, textvariable=state_var, fg="#8affab", bg="#15181c", font=("Segoe UI", 18, "bold")
    ).pack(pady=8)
    buttons = tk.Frame(root, bg="#15181c")
    buttons.pack(pady=16)
    for label, command in (
        ("Start live tracking", start),
        ("Stop", stop),
        ("Open / Reopen Rerun", open_rerun),
        ("Reset trail", reset_trail),
    ):
        tk.Button(
            buttons, text=label, command=command, width=20, height=2, font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT, padx=5)
    tk.Label(
        root,
        textvariable=detail_var,
        justify=tk.LEFT,
        anchor="nw",
        fg="#d8dde5",
        bg="#22262c",
        font=("Consolas", 10),
        padx=16,
        pady=12,
    ).pack(fill=tk.BOTH, expand=True, padx=24, pady=(8, 24))
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(500, refresh)
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--open-viewer", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=ARTIFACT_ROOT / "runs")
    args = parser.parse_args()
    if args.headless:
        if args.output_dir is None or not 1.0 <= args.duration_seconds <= 3600.0:
            parser.error("headless mode requires --output-dir and duration within 1..3600 seconds")
        return _headless(args.output_dir.resolve(), args.duration_seconds, args.open_viewer)
    return _gui(args.output_root.resolve())


if __name__ == "__main__":
    sys.exit(main())
