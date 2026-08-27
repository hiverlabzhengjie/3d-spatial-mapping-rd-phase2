"""Rerun adapter for P09 static facility context and bounded live tracking ticks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np

from spatial_mapping_phase2.p09_pipeline import PipelineTickResult
from spatial_mapping_phase2.p09_projection import CameraProjectionCalibration, FloorRectangle
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS, TrackingState
from spatial_mapping_phase2.rerun_camera_visualization import (
    RerunCameraFrustum,
    log_camera_frustum,
)


class P09RerunError(RuntimeError):
    """Raised when the live recording cannot preserve its display contract."""


class _SegmentedTrail:
    """Bounded polyline history that never joins separated tracking periods."""

    def __init__(self, point_limit: int) -> None:
        if point_limit <= 1:
            raise P09RerunError("trail limit must exceed one point")
        self._point_limit = point_limit
        self._segments: list[list[list[float]]] = []
        self._break_pending = True

    def append(self, xy_metres: tuple[float, float]) -> None:
        point = [xy_metres[0], xy_metres[1], 0.06]
        if self._break_pending or not self._segments:
            self._segments.append([point])
            self._break_pending = False
        else:
            self._segments[-1].append(point)
        while self.point_count > self._point_limit:
            self._segments[0].pop(0)
            if not self._segments[0]:
                self._segments.pop(0)

    def break_segment(self) -> None:
        self._break_pending = True

    def clear(self) -> None:
        self._segments.clear()
        self._break_pending = True

    @property
    def point_count(self) -> int:
        return sum(len(segment) for segment in self._segments)

    @property
    def line_strips(self) -> list[list[list[float]]]:
        return [segment for segment in self._segments if len(segment) >= 2]


class P09RerunLogger:
    """Thread-safe archive plus optional low-latency live Rerun stream."""

    def __init__(
        self,
        recording_path: Path,
        calibrations: dict[str, CameraProjectionCalibration],
        floor: FloorRectangle,
        p06_input_manifest: Path,
        selected_geometry_path: Path,
        python_executable: Path,
        trail_limit: int = 80,
    ) -> None:
        self.recording_path = recording_path.resolve()
        self._python_executable = python_executable.resolve()
        self._rerun_executable = self._python_executable.with_name("rerun.exe")
        if not self._rerun_executable.is_file():
            raise P09RerunError("Rerun CLI is missing from the selected P09 runtime")
        self._calibrations = calibrations
        self._floor = floor
        self._trail = _SegmentedTrail(trail_limit)
        self._tick_index = 0
        self._lock = threading.Lock()
        self._viewer_process: subprocess.Popen[bytes] | None = None
        self._viewer_port: int | None = None
        self._live_recording: Any | None = None
        self._live_connection_count = 0
        self._live_tick_count = 0
        self._live_connected_before_close = False
        self._closed = False
        self._p06_input_manifest = p06_input_manifest.resolve()
        self._selected_geometry_path = selected_geometry_path.resolve()

        import rerun as rr

        self._rr = rr
        self._archive_recording: Any = rr.new_recording(
            "p09-live-anonymous-person-xy",
            recording_id=self.recording_path.stem,
        )
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)
        self._archive_recording.save(str(self.recording_path))
        self._log_static_context((self._archive_recording,))
        self._send_blueprint((self._archive_recording,))

    def _log_static_context(self, recordings: tuple[Any, ...]) -> None:
        rr = self._rr
        with np.load(self._selected_geometry_path, allow_pickle=False) as archive:
            points = np.asarray(archive["points"], dtype=np.float64)
            colors = np.asarray(archive["colors_rgb"], dtype=np.uint8)
        _log_to(
            recordings,
            "p09/world/p07_v2_cloud_actual_rgb",
            rr.Points3D(points, colors=colors, radii=0.018),
            static=True,
        )
        x0, y0 = self._floor.minimum_xy_metres
        x1, y1 = self._floor.maximum_xy_metres
        floor_outline = np.asarray(
            [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]]
        )
        _log_to(
            recordings,
            "p09/world/p08_floor_z0",
            rr.LineStrips3D([floor_outline], colors=[[210, 210, 210]], radii=0.02),
            static=True,
        )
        _log_to(
            recordings,
            "p09/world/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=[[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
                colors=[[255, 64, 64], [64, 220, 96], [64, 128, 255]],
            ),
            static=True,
        )
        p06 = json.loads(self._p06_input_manifest.read_text(encoding="utf-8"))
        records = {record["camera_id"]: record for record in p06["cameras"]}
        for camera_id in CAMERA_IDS:
            image = cv2.imread(records[camera_id]["pinhole_derivative"]["path"], cv2.IMREAD_COLOR)
            if image is None:
                raise P09RerunError(f"cached context frame unavailable for {camera_id}")
            image = cv2.resize(image, (504, 280), interpolation=cv2.INTER_AREA)
            calibration = self._calibrations[camera_id]
            frustum = RerunCameraFrustum(
                camera_id,
                calibration.T_world_from_camera,
                calibration.K_processed,
                np.ascontiguousarray(image[..., ::-1]),
                image_plane_distance_metres=0.75,
                axis_length_metres=0.55,
            )
            log_camera_frustum(
                _RecordingBoundRerun(rr, recordings),
                "p09/world/cameras",
                "p09/world/camera_labels",
                frustum,
            )

    def _send_blueprint(self, recordings: tuple[Any, ...]) -> None:
        import rerun.blueprint as rrb

        camera_views = [
            rrb.Spatial2DView(
                origin=f"/p09/live/{camera_id}", name=camera_id.replace("office-", "")
            )
            for camera_id in CAMERA_IDS
        ]
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/", contents="p09/world/**", name="Facility XY"),
                rrb.Vertical(
                    rrb.Horizontal(camera_views[0], camera_views[1]),
                    rrb.Horizontal(camera_views[2], camera_views[3]),
                    row_shares=[1, 1],
                ),
                column_shares=[2, 3],
            ),
            rrb.TextDocumentView(origin="/p09/status", name="Tracking state"),
            auto_layout=False,
            auto_views=False,
        )
        for recording in recordings:
            recording.send_blueprint(blueprint)

    def _active_recordings(self) -> tuple[Any, ...]:
        if self._live_recording is None:
            return (self._archive_recording,)
        return self._archive_recording, self._live_recording

    def _log(self, path: str, entity: Any, *, static: bool = False) -> None:
        _log_to(self._active_recordings(), path, entity, static=static)

    def log_tick(self, result: PipelineTickResult) -> None:
        with self._lock:
            live_stream_active = self._live_recording is not None
            self._tick_index += 1
            for recording in self._active_recordings():
                recording.set_time_sequence("tick", self._tick_index)
                recording.set_time_nanos("host_monotonic", result.tick_monotonic_ns)
            active_camera_ids = {camera.camera_id for camera in result.camera_results}
            for camera in result.camera_results:
                path = f"p09/live/{camera.camera_id}"
                self._log(path, self._rr.Image(camera.calibrated_frame_bgr[..., ::-1]))
                boxes = [detection.bbox_xyxy for detection in camera.detections]
                if boxes:
                    labels = [
                        f"person {detection.confidence:.2f} | {detection.footpoint_kind.value}"
                        for detection in camera.detections
                    ]
                    self._log(
                        f"{path}/detections",
                        self._rr.Boxes2D(
                            array=boxes,
                            array_format=self._rr.Box2DFormat.XYXY,
                            labels=labels,
                            colors=[[255, 214, 64]] * len(boxes),
                        ),
                    )
                    self._log(
                        f"{path}/footpoints",
                        self._rr.Points2D(
                            [detection.image_point_uv for detection in camera.detections],
                            colors=[[255, 70, 70]] * len(boxes),
                            radii=5.0,
                        ),
                    )
                else:
                    self._log(f"{path}/detections", self._rr.Clear(recursive=True))
                    self._log(f"{path}/footpoints", self._rr.Clear(recursive=True))
                candidates = [
                    projection.observation
                    for projection in camera.projections
                    if projection.observation
                ]
                if candidates:
                    self._log(
                        f"p09/world/candidates/{camera.camera_id}",
                        self._rr.Points3D(
                            [
                                [candidate.xy_metres[0], candidate.xy_metres[1], 0.04]
                                for candidate in candidates
                            ],
                            colors=[[72, 190, 255]] * len(candidates),
                            labels=[camera.camera_id] * len(candidates),
                            radii=0.10,
                        ),
                    )
                else:
                    self._log(
                        f"p09/world/candidates/{camera.camera_id}", self._rr.Clear(recursive=True)
                    )
            for camera_id in set(CAMERA_IDS) - active_camera_ids:
                self._log(f"p09/live/{camera_id}", self._rr.Clear(recursive=True))
                self._log(f"p09/world/candidates/{camera_id}", self._rr.Clear(recursive=True))

            tracked = result.tracking.current_xy_metres
            if tracked is not None:
                self._trail.append(tracked)
                color = (
                    [80, 240, 120]
                    if result.tracking.state is TrackingState.TRACKED_FUSED
                    else [255, 196, 64]
                )
                self._log(
                    "p09/world/current_anonymous_xy",
                    self._rr.Points3D(
                        [[tracked[0], tracked[1], 0.08]], colors=[color], radii=0.18
                    ),
                )
                line_strips = self._trail.line_strips
                if line_strips:
                    self._log(
                        "p09/world/recent_trail",
                        self._rr.LineStrips3D(line_strips, colors=[[80, 240, 120]], radii=0.045),
                    )
            else:
                self._trail.break_segment()
                self._log("p09/world/current_anonymous_xy", self._rr.Clear(recursive=True))
            self._log(
                "p09/status",
                self._rr.TextDocument(_status_markdown(result), media_type="text/markdown"),
            )
            self._log(
                "p09/telemetry/processing_latency_ms",
                self._rr.Scalar(result.processing_latency_ms),
            )
            if result.cross_camera_skew_ms is not None:
                self._log(
                    "p09/telemetry/cross_camera_skew_ms",
                    self._rr.Scalar(result.cross_camera_skew_ms),
                )
            if live_stream_active:
                self._live_tick_count += 1

    def reset_trail(self) -> None:
        with self._lock:
            self._trail.clear()
            self._log("p09/world/recent_trail", self._rr.Clear(recursive=True))

    def open_viewer(self) -> None:
        with self._lock:
            if self._closed:
                self._open_archive_viewer()
                return
            if self._viewer_port is not None and _listener_is_open(self._viewer_port):
                return
        port = _reserve_local_port()
        previous_path = os.environ.get("PATH")
        os.environ["PATH"] = os.pathsep.join(
            part for part in (str(self._rerun_executable.parent), previous_path) if part
        )
        try:
            self._rr.spawn(
                port=port,
                connect=False,
                hide_welcome_screen=True,
                recording=self._archive_recording,
            )
        except Exception as error:
            raise P09RerunError("failed to launch the Rerun desktop viewer") from error
        finally:
            if previous_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous_path
        if not _wait_for_listener(port, timeout_seconds=10.0):
            raise P09RerunError("Rerun desktop viewer did not start its live listener")
        with self._lock:
            if self._closed:
                self._open_archive_viewer()
                return
            if self._live_recording is not None:
                self._live_recording.disconnect()
            live_recording = self._rr.new_recording(
                "p09-live-anonymous-person-xy",
                recording_id=self.recording_path.stem,
            )
            self._rr.connect_tcp(f"127.0.0.1:{port}", recording=live_recording)
            self._live_recording = live_recording
            self._viewer_port = port
            self._live_connection_count += 1
            self._log_static_context((live_recording,))
            self._send_blueprint((live_recording,))
            live_recording.flush()

    def _open_archive_viewer(self) -> None:
        command = [str(self._rerun_executable), str(self.recording_path), "--port", "0"]
        try:
            self._viewer_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            raise P09RerunError("failed to launch the saved Rerun recording") from error

    def close(self) -> None:
        with self._lock:
            self._archive_recording.flush()
            self._archive_recording.disconnect()
            if self._live_recording is not None:
                self._live_connected_before_close = True
                self._live_recording.flush()
                self._live_recording.disconnect()
                self._live_recording = None
            self._closed = True

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sink_mode": "independent-archive-and-live-tcp",
                "active_viewer_launch_mode": "rerun-sdk-spawn",
                "archive_path": str(self.recording_path),
                "archive_tick_count": self._tick_index,
                "live_connection_count": self._live_connection_count,
                "live_tick_count": self._live_tick_count,
                "live_connected_before_close": self._live_connected_before_close,
                "live_connected_now": self._live_recording is not None,
                "last_live_port": self._viewer_port,
                "closed": self._closed,
            }


class _RecordingBoundRerun:
    """Minimal proxy making the shared frustum helper use every selected recording."""

    def __init__(self, rr: Any, recordings: tuple[Any, ...]) -> None:
        self._rr = rr
        self._recordings = recordings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rr, name)

    def log(self, path: str, entity: Any, *, static: bool = False) -> None:
        _log_to(self._recordings, path, entity, static=static)


def _log_to(recordings: tuple[Any, ...], path: str, entity: Any, *, static: bool = False) -> None:
    for recording in recordings:
        recording.log(path, entity, static=static)


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    return port


def _listener_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_listener(port: int, timeout_seconds: float) -> bool:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        if _listener_is_open(port):
            return True
        sleep(0.05)
    return False


def _status_markdown(result: PipelineTickResult) -> str:
    state = result.tracking.state.value
    current = (
        "none"
        if result.tracking.current_xy_metres is None
        else (
            f"({result.tracking.current_xy_metres[0]:.2f}, "
            f"{result.tracking.current_xy_metres[1]:.2f}) m"
        )
    )
    skew = (
        "n/a" if result.cross_camera_skew_ms is None else f"{result.cross_camera_skew_ms:.1f} ms"
    )
    return "\n".join(
        (
            f"# {state}",
            f"Current XY: **{current}**",
            f"Reason: {result.tracking.reason}",
            f"Contributors: {', '.join(result.tracking.contributing_camera_ids) or 'none'}",
            f"Stale: {', '.join(result.stale_camera_ids) or 'none'}",
            f"Missing: {', '.join(result.missing_camera_ids) or 'none'}",
            f"Cross-camera skew: {skew}",
            f"Processing: {result.processing_latency_ms:.1f} ms",
        )
    )
