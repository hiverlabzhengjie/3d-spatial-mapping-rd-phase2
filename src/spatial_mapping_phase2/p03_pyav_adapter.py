"""Bounded, read-only PyAV RTSP adapter for P03 evidence capture."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import monotonic, monotonic_ns
from typing import Any

from spatial_mapping_phase2.p01_observability import LocalRtspEndpoint
from spatial_mapping_phase2.p03_capture_domain import (
    PROFILE_SCHEMA,
    CaptureArtifact,
    RationalTimeBase,
    SourceFrame,
    StorageMode,
    StreamProfileIdentity,
    sha256_file,
)
from spatial_mapping_phase2.p03_capture_service import (
    CaptureAdapterError,
    CaptureCancelledError,
    CapturePolicy,
    ConnectTimeoutError,
    PreviewFrame,
    ReadTimeoutError,
)
from spatial_mapping_phase2.p03_temporal_capture import BufferedFrame


class PyAvCaptureAdapter:
    """Open each source only for a bounded operation; never changes recorder settings."""

    def __init__(self, profile_versions: dict[str, str] | None = None) -> None:
        self.profile_versions = profile_versions or {}
        self._closed = threading.Event()

    def profile(self, endpoint: LocalRtspEndpoint, policy: CapturePolicy) -> StreamProfileIdentity:
        av = _av()
        try:
            with av.open(
                endpoint.for_read_only_adapter(),
                mode="r",
                options={"rtsp_transport": "tcp"},
                timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
            ) as container:
                stream = next(iter(container.streams.video), None)
                if stream is None:
                    raise CaptureAdapterError("source has no video stream")
                time_base = stream.time_base
                if time_base is None:
                    raise CaptureAdapterError("source video time base is unavailable")
                rotation = stream.metadata.get("rotate")
                return StreamProfileIdentity(
                    PROFILE_SCHEMA,
                    endpoint.camera_id,
                    self.profile_versions.get(endpoint.camera_id, "stream-profile-v1"),
                    endpoint.environment_key,
                    f"local-{endpoint.environment_key.lower()}",
                    stream.codec_context.width,
                    stream.codec_context.height,
                    stream.codec_context.name,
                    RationalTimeBase(time_base.numerator, time_base.denominator),
                    None,
                    int(rotation) if rotation and rotation.isdigit() else None,
                    _utc_now(),
                )
        except CaptureAdapterError:
            raise
        except Exception as error:
            raise ConnectTimeoutError("bounded RTSP profile connection failed") from error

    def capture(
        self,
        endpoint: LocalRtspEndpoint,
        profile: StreamProfileIdentity,
        output_path: Path,
        relative_path: str,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SourceFrame, ...], CaptureArtifact]:
        try:
            return self._remux(endpoint, profile, output_path, relative_path, policy, cancel)
        except (CaptureCancelledError, ReadTimeoutError):
            raise
        except Exception as remux_error:
            fallback_path = output_path.with_suffix(".jpg")
            fallback_relative = str(Path(relative_path).with_suffix(".jpg")).replace("\\", "/")
            return self._decoded_fallback(
                endpoint,
                profile,
                fallback_path,
                fallback_relative,
                policy,
                cancel,
                type(remux_error).__name__,
            )

    def preview(
        self,
        endpoint: LocalRtspEndpoint,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> PreviewFrame:
        """Decode one JPEG entirely in memory; never create a capture artifact."""

        av = _av()
        try:
            with av.open(
                endpoint.for_read_only_adapter(),
                mode="r",
                options={"rtsp_transport": "tcp"},
                timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
            ) as source:
                stream = next(iter(source.streams.video), None)
                if stream is None:
                    raise CaptureAdapterError("source has no video stream")
                if cancel.is_set() or self._closed.is_set():
                    raise CaptureCancelledError("preview cancelled")
                frame = next(iter(source.decode(stream)), None)
                if frame is None:
                    raise ReadTimeoutError("source produced no preview frame")
                destination = BytesIO()
                frame.to_image().save(destination, format="JPEG", quality=80)
                time_base = str(stream.time_base) if stream.time_base is not None else None
                return PreviewFrame(
                    endpoint.camera_id,
                    "image/jpeg",
                    destination.getvalue(),
                    _utc_now(),
                    frame.pts,
                    time_base,
                )
        except CaptureAdapterError:
            raise
        except Exception as error:
            raise ReadTimeoutError("bounded RTSP preview failed") from error

    def frames(
        self,
        endpoint: LocalRtspEndpoint,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> Iterable[BufferedFrame]:
        """Yield JPEG frames from one warm RTSP connection for temporal selection."""

        av = _av()
        with av.open(
            endpoint.for_read_only_adapter(),
            mode="r",
            options={"rtsp_transport": "tcp"},
            timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
        ) as source:
            stream = next(iter(source.streams.video), None)
            if stream is None:
                raise CaptureAdapterError("source has no video stream")
            time_base = str(stream.time_base) if stream.time_base is not None else None
            sequence = 0
            for frame in source.decode(stream):
                if cancel.is_set() or self._closed.is_set():
                    return
                acquired = monotonic_ns()
                destination = BytesIO()
                frame.to_image().save(destination, format="JPEG", quality=90)
                yield BufferedFrame(
                    endpoint.camera_id,
                    f"{endpoint.camera_id}-warm-{sequence:09d}",
                    self.profile_versions.get(endpoint.camera_id, "stream-profile-v1"),
                    acquired,
                    _utc_now(),
                    frame.pts,
                    time_base if frame.pts is not None else None,
                    destination.getvalue(),
                )
                sequence += 1

    def _remux(
        self,
        endpoint: LocalRtspEndpoint,
        profile: StreamProfileIdentity,
        output_path: Path,
        relative_path: str,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> tuple[tuple[SourceFrame, ...], CaptureArtifact]:
        av = _av()
        frames: list[SourceFrame] = []
        with av.open(
            endpoint.for_read_only_adapter(),
            mode="r",
            options={"rtsp_transport": "tcp"},
            timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
        ) as source:
            stream = next(iter(source.streams.video), None)
            if stream is None:
                raise CaptureAdapterError("source has no video stream")
            # PyAV already enforces the bounded connect/read timeouts.  Begin the
            # evidence window after opening the stream so slow connection scheduling
            # cannot consume or invalidate the requested capture duration.
            started = monotonic()
            with av.open(str(output_path), mode="w") as destination:
                output_stream = destination.add_stream_from_template(stream)
                for packet in source.demux(stream):
                    if cancel.is_set() or self._closed.is_set():
                        raise CaptureCancelledError("capture cancelled")
                    elapsed = monotonic() - started
                    if elapsed >= policy.duration_seconds:
                        break
                    if packet.dts is None:
                        continue
                    acquisition = monotonic_ns()
                    frame_id = f"{endpoint.camera_id}-packet-{len(frames):06d}"
                    frames.append(
                        SourceFrame(
                            frame_id,
                            endpoint.camera_id,
                            profile.profile_version,
                            packet.pts,
                            profile.time_base if packet.pts is not None else None,
                            acquisition,
                            _utc_now(),
                        )
                    )
                    packet.stream = output_stream
                    destination.mux(packet)
                    if len(frames) > policy.queue_capacity:
                        raise CaptureAdapterError("bounded capture queue capacity exceeded")
        if not frames or not output_path.is_file() or output_path.stat().st_size == 0:
            raise CaptureAdapterError("packet-preserving capture produced no media")
        return tuple(frames), CaptureArtifact(
            relative_path,
            sha256_file(output_path),
            output_path.stat().st_size,
            StorageMode.PACKET_PRESERVING_MP4,
            False,
            None,
        )

    def _decoded_fallback(
        self,
        endpoint: LocalRtspEndpoint,
        profile: StreamProfileIdentity,
        output_path: Path,
        relative_path: str,
        policy: CapturePolicy,
        cancel: threading.Event,
        reason: str,
    ) -> tuple[tuple[SourceFrame, ...], CaptureArtifact]:
        av = _av()
        with av.open(
            endpoint.for_read_only_adapter(),
            mode="r",
            options={"rtsp_transport": "tcp"},
            timeout=(policy.connect_timeout_seconds, policy.read_timeout_seconds),
        ) as source:
            stream = next(iter(source.streams.video), None)
            if stream is None:
                raise CaptureAdapterError("source has no video stream")
            if cancel.is_set() or self._closed.is_set():
                raise CaptureCancelledError("capture cancelled")
            frame = next(iter(source.decode(stream)), None)
            if frame is None:
                raise ReadTimeoutError("source produced no decoded fallback frame")
            frame.to_image().save(output_path, format="JPEG", quality=95)
            acquisition = monotonic_ns()
            source_frame = SourceFrame(
                f"{endpoint.camera_id}-fallback-000000",
                endpoint.camera_id,
                profile.profile_version,
                frame.pts,
                profile.time_base if frame.pts is not None else None,
                acquisition,
                _utc_now(),
            )
        artifact = CaptureArtifact(
            relative_path,
            sha256_file(output_path),
            output_path.stat().st_size,
            StorageMode.DECODED_FRAME_FALLBACK,
            False,
            reason,
        )
        return (source_frame,), artifact

    def close(self) -> None:
        self._closed.set()


def _av() -> Any:
    try:
        import av
    except ImportError as error:
        raise CaptureAdapterError("PyAV runtime dependency is unavailable") from error
    return av


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
