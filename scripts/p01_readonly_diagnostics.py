"""Run bounded P01 RTSP preflight and diagnostic capture without revealing endpoint values."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import av

from spatial_mapping_phase2.artifact_layout import ArtifactLayout
from spatial_mapping_phase2.p01_observability import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    STREAM_PROFILE_SCHEMA_VERSION,
    CapturedDiagnosticArtifact,
    DiagnosticCaptureManifest,
    DiagnosticCaptureRequest,
    StreamProbeResult,
    StreamProfile,
    load_local_rtsp_endpoints,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_local_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(endpoint_url: str, timeout_seconds: float) -> StreamProbeResult:
    observed_at = _utc_now()
    started = monotonic()
    with av.open(
        endpoint_url,
        mode="r",
        options={"rtsp_transport": "tcp"},
        timeout=(timeout_seconds, timeout_seconds),
    ) as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            raise RuntimeError("no video stream")
        decoded = 0
        first_pts: int | None = None
        last_pts: int | None = None
        keyframes = 0
        for frame in container.decode(stream):
            decoded += 1
            keyframes += int(frame.key_frame)
            if frame.pts is not None:
                first_pts = frame.pts if first_pts is None else first_pts
                last_pts = frame.pts
            if decoded >= 25 or monotonic() - started >= timeout_seconds:
                break
        observed_fps: float | None = None
        if (
            decoded >= 2
            and first_pts is not None
            and last_pts is not None
            and last_pts > first_pts
        ):
            elapsed_seconds = float((last_pts - first_pts) * stream.time_base)
            if elapsed_seconds > 0:
                observed_fps = (decoded - 1) / elapsed_seconds
        rotation = stream.metadata.get("rotate")
        return StreamProbeResult(
            observed_at=observed_at,
            width_pixels=stream.codec_context.width,
            height_pixels=stream.codec_context.height,
            codec=stream.codec_context.name,
            nominal_fps=float(stream.average_rate) if stream.average_rate else None,
            observed_fps=observed_fps,
            time_base=str(stream.time_base) if stream.time_base else None,
            rotation_degrees=int(rotation) if rotation and rotation.isdigit() else None,
            crop_description="not automatically observable in bounded preflight",
            overlay_description="requires visual inspection of retained diagnostic frames",
            dewarping_indicator="not automatically observable in bounded preflight",
            keyframe_behavior=f"{keyframes} decoded keyframe(s) in {decoded} frame(s)",
            stability_note=f"decoded {decoded} frame(s) in {monotonic() - started:.2f} seconds",
        )


def _capture(
    endpoint_url: str,
    request: DiagnosticCaptureRequest,
    output_path: Path,
    relative_artifact_path: str,
) -> CapturedDiagnosticArtifact:
    started_at = _utc_now()
    started = monotonic()
    packets = 0
    first_pts: int | None = None
    last_pts: int | None = None
    with av.open(
        endpoint_url,
        mode="r",
        options={"rtsp_transport": "tcp"},
        timeout=(request.connect_timeout_seconds, request.connect_timeout_seconds),
    ) as input_container:
        input_stream = next(iter(input_container.streams.video), None)
        if input_stream is None:
            raise RuntimeError("no video stream")
        with av.open(str(output_path), mode="w") as output_container:
            output_stream = output_container.add_stream_from_template(input_stream)
            for packet in input_container.demux(input_stream):
                if monotonic() - started >= request.duration_seconds:
                    break
                if packet.dts is None:
                    continue
                if packet.pts is not None:
                    first_pts = packet.pts if first_pts is None else first_pts
                    last_pts = packet.pts
                packet.stream = output_stream
                output_container.mux(packet)
                packets += 1
    if packets == 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("diagnostic capture contains no media packets")
    time_base = input_stream.time_base
    source_start = float(first_pts * time_base) if first_pts is not None and time_base else None
    source_end = float(last_pts * time_base) if last_pts is not None and time_base else None
    return CapturedDiagnosticArtifact(
        relative_artifact_path=relative_artifact_path,
        sha256=_sha256(output_path),
        source_pts_start_seconds=source_start,
        source_pts_end_seconds=source_end,
        acquisition_started_at=started_at,
        acquisition_finished_at=_utc_now(),
    )


def _capture_representative_frame(
    endpoint_url: str,
    request: DiagnosticCaptureRequest,
    output_path: Path,
    relative_artifact_path: str,
) -> CapturedDiagnosticArtifact:
    """Retain one source-decoded frame when packet remuxing is unsupported."""

    started_at = _utc_now()
    with av.open(
        endpoint_url,
        mode="r",
        options={"rtsp_transport": "tcp"},
        timeout=(request.connect_timeout_seconds, request.connect_timeout_seconds),
    ) as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            raise RuntimeError("no video stream")
        frame = next(iter(container.decode(stream)), None)
        if frame is None:
            raise RuntimeError("no decodable video frame")
        frame.to_image().save(output_path, format="JPEG", quality=95)
        source_pts = float(frame.pts * stream.time_base) if frame.pts is not None else None
    return CapturedDiagnosticArtifact(
        relative_artifact_path=relative_artifact_path,
        sha256=_sha256(output_path),
        source_pts_start_seconds=source_pts,
        source_pts_end_seconds=source_pts,
        acquisition_started_at=started_at,
        acquisition_finished_at=_utc_now(),
    )


def main() -> int:
    environment = _read_local_environment(Path(".env"))
    layout = ArtifactLayout.from_root(environment["PHASE2_ARTIFACT_ROOT"])
    layout.validate_complete()
    endpoints = load_local_rtsp_endpoints(environment)
    requested_camera_ids = set(sys.argv[1:])
    if requested_camera_ids:
        endpoints = tuple(
            endpoint for endpoint in endpoints if endpoint.camera_id in requested_camera_ids
        )
        if not endpoints or len(endpoints) != len(requested_camera_ids):
            raise ValueError("requested camera IDs must be configured fixed office camera IDs")
    request = DiagnosticCaptureRequest(duration_seconds=10.0, connect_timeout_seconds=5.0)
    for endpoint in endpoints:
        camera_root = layout.path_for("captures") / "p01" / endpoint.camera_id
        camera_root.mkdir(parents=True, exist_ok=True)
        capture_id = f"p01-{endpoint.camera_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        try:
            observation = _probe(endpoint.for_read_only_adapter(), request.connect_timeout_seconds)
            profile = StreamProfile(
                schema_version=STREAM_PROFILE_SCHEMA_VERSION,
                camera_id=endpoint.camera_id,
                profile_version="stream-profile-v1",
                endpoint_environment_key=endpoint.environment_key,
                observation=observation,
            )
            relative_path = f"captures/p01/{endpoint.camera_id}/{capture_id}.mp4"
            capture_mode = "packet-preserving MP4 remux"
            remux_failure_class: str | None = None
            try:
                artifact = _capture(
                    endpoint.for_read_only_adapter(),
                    request,
                    layout.root / relative_path,
                    relative_path,
                )
            except Exception as error:
                remux_failure_class = type(error).__name__
                relative_path = f"captures/p01/{endpoint.camera_id}/{capture_id}.jpg"
                artifact = _capture_representative_frame(
                    endpoint.for_read_only_adapter(),
                    request,
                    layout.root / relative_path,
                    relative_path,
                )
                capture_mode = "representative JPEG fallback after MP4 remux failure"
            manifest = DiagnosticCaptureManifest(
                schema_version=CAPTURE_MANIFEST_SCHEMA_VERSION,
                capture_id=capture_id,
                camera_id=endpoint.camera_id,
                stream_profile_version=profile.profile_version,
                request=request,
                artifact=artifact,
            )
            (camera_root / f"{capture_id}.json").write_text(
                json.dumps(
                    {
                        "stream_profile": asdict(profile),
                        "capture": asdict(manifest),
                        "capture_mode": capture_mode,
                        "remux_failure_class": remux_failure_class,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"{endpoint.camera_id}: captured; {observation.width_pixels}x"
                f"{observation.height_pixels}; codec={observation.codec}; "
                f"observed_fps={observation.observed_fps}; mode={capture_mode}; "
                f"sha256={artifact.sha256}"
            )
        except Exception as error:
            print(f"{endpoint.camera_id}: failed; error_class={type(error).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
