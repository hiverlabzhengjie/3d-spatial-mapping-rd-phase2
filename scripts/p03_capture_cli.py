"""Thin CLI over the shared P03 workflow service."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from spatial_mapping_phase2.p01_observability import load_local_rtsp_endpoints
from spatial_mapping_phase2.p03_capture_service import (
    CapturePolicy,
    CaptureRepository,
    CaptureWorkflowService,
)
from spatial_mapping_phase2.p03_pyav_adapter import PyAvCaptureAdapter
from spatial_mapping_phase2.p03_temporal_capture import (
    TemporalBundleRepository,
    WarmTemporalCaptureService,
)


def _environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def build_service(environment: dict[str, str]) -> CaptureWorkflowService:
    root = Path(environment["PHASE2_ARTIFACT_ROOT"])
    return CaptureWorkflowService(
        load_local_rtsp_endpoints(environment),
        PyAvCaptureAdapter(),
        CaptureRepository(root),
        "spatial-mapping-phase2-p03-v1",
    )


def build_temporal_service(environment: dict[str, str]) -> WarmTemporalCaptureService:
    root = Path(environment["PHASE2_ARTIFACT_ROOT"])
    return WarmTemporalCaptureService(
        load_local_rtsp_endpoints(environment),
        PyAvCaptureAdapter(),
        TemporalBundleRepository(root),
        CapturePolicy(duration_seconds=2.0, read_timeout_seconds=5.0),
        "spatial-mapping-phase2-p03-temporal-v1",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Credential-safe P03 capture workflow")
    parser.add_argument(
        "command",
        choices=("health", "preview", "capture", "sessions", "select", "sync-capture"),
    )
    parser.add_argument("--session-id")
    parser.add_argument("--camera-id")
    parser.add_argument("--bundle-id", default="selected-default")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--max-skew-ms", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    environment = _environment(Path(".env"))
    if args.command == "sync-capture":
        if not args.bundle_id or args.bundle_id == "selected-default":
            parser.error("sync-capture requires --bundle-id")
        if args.max_skew_ms is None or args.max_skew_ms <= 0:
            parser.error("sync-capture requires positive --max-skew-ms")
        temporal = build_temporal_service(environment)
        temporal.start()
        try:
            if not temporal.wait_until_ready(args.warmup_seconds):
                print(
                    json.dumps(
                        {"authority_status": "unavailable", "workers": temporal.status()},
                        indent=2,
                    )
                )
                return 2
            manifest = temporal.capture(args.bundle_id, int(args.max_skew_ms * 1_000_000))
            print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
            return 0 if manifest.authority_status.value.startswith("authoritative") else 3
        finally:
            temporal.close()
    service = build_service(environment)
    try:
        if args.command == "health":
            print(json.dumps(service.health(CapturePolicy()), indent=2, sort_keys=True))
        elif args.command == "preview":
            if not args.camera_id:
                parser.error("preview requires --camera-id")
            frame = service.preview(args.camera_id, CapturePolicy(duration_seconds=1.0))
            print(
                json.dumps(
                    {
                        "camera_id": frame.camera_id,
                        "media_type": frame.media_type,
                        "byte_count": len(frame.content),
                        "observed_at_utc": frame.observed_at_utc,
                        "source_pts": frame.source_pts,
                        "source_time_base": frame.source_time_base,
                        "evidence_status": "ephemeral-preview-not-capture-evidence",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "capture":
            if not args.session_id:
                parser.error("capture requires --session-id")
            manifest = service.capture_session(
                args.session_id, CapturePolicy(duration_seconds=args.duration)
            )
            print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
        elif args.command == "sessions":
            print(json.dumps({"sessions": service.repository.list_sessions()}, indent=2))
        else:
            if not args.session_id:
                parser.error("select requires --session-id")
            session = service.repository.read_session(args.session_id)
            bundle = service.select_bundle(session, args.bundle_id)
            print(json.dumps(asdict(bundle), indent=2, sort_keys=True))
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
