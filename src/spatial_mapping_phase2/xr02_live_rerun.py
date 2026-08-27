"""Rerun archive/live publication for XR02 WP4 multi-person global tracks."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np

from spatial_mapping_phase2.p09_projection import CameraProjectionCalibration, FloorRectangle
from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS
from spatial_mapping_phase2.rerun_camera_visualization import (
    RerunCameraFrustum,
    log_camera_frustum,
)
from spatial_mapping_phase2.xr02_global_domain import (
    GlobalTrackSnapshot,
    GlobalTrackState,
    MemberAssignment,
)
from spatial_mapping_phase2.xr02_live_domain import AdoptedSceneSelection, XR02LiveContractError
from spatial_mapping_phase2.xr02_live_pipeline import LiveAssociationTick


class XR02LiveRerunError(RuntimeError):
    """Raised when WP4 cannot preserve its archive/live Rerun contract."""


@dataclass(slots=True)
class _TrailState:
    segment_index: int = 0
    points: list[list[float]] = field(default_factory=list)
    last_observed_monotonic_ns: int | None = None
    last_camera_ids: tuple[str, ...] = ()
    break_pending: bool = False


class XR02LiveRerunLogger:
    def __init__(
        self,
        recording_path: Path,
        scene: AdoptedSceneSelection,
        calibrations: dict[str, CameraProjectionCalibration],
        floor: FloorRectangle,
        python_executable: Path,
        *,
        trail_limit: int = 120,
        image_every_n_ticks: int = 1,
    ) -> None:
        if trail_limit <= 1 or image_every_n_ticks <= 0:
            raise XR02LiveContractError("Rerun trail/image cadence configuration is invalid")
        self.recording_path = recording_path.resolve()
        self._scene = scene
        self._calibrations = dict(calibrations)
        self._floor = floor
        self._trail_limit = trail_limit
        self._image_every_n_ticks = image_every_n_ticks
        self._trails: dict[str, _TrailState] = {}
        self._previous_track_states: dict[str, GlobalTrackState] = {}
        self._previous_track_cameras: dict[str, tuple[str, ...]] = {}
        self._previous_duplicate_ids: frozenset[str] = frozenset()
        self._previous_ambiguous_local_ids: frozenset[str] = frozenset()
        self._last_camera_dimensions: dict[str, tuple[int, int]] = {}
        self._diagnostic_events: deque[str] = deque(maxlen=24)
        self._tick_count = 0
        self._lock = threading.Lock()
        self._live_recording: Any | None = None
        self._live_connections = 0
        self._live_ticks = 0
        self._viewer_port: int | None = None
        self._viewer_process: subprocess.Popen[bytes] | None = None
        self._closed = False
        self._rerun_executable = python_executable.resolve().with_name("rerun.exe")
        if not self._rerun_executable.is_file():
            raise XR02LiveRerunError("Rerun CLI is missing from the selected runtime")
        import rerun as rr

        self._rr = rr
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)
        self._archive: Any = rr.new_recording(
            "xr02-wp4-live-global-tracking",
            recording_id=self.recording_path.stem,
        )
        self._archive.save(str(self.recording_path))
        self._log_static((self._archive,))
        self._send_blueprint((self._archive,))

    def log_tick(self, tick: LiveAssociationTick) -> None:
        with self._lock:
            self._tick_count += 1
            recordings = self._active_recordings()
            for recording in recordings:
                recording.set_time_sequence("association_tick", tick.tick_index)
                recording.set_time_nanos("host_monotonic", tick.admitted_monotonic_ns)
            assignments = {item.observation_id: item for item in tick.association.assignments}
            assignments_by_local_track = {
                item.local_track_stable_id: item for item in tick.association.assignments
            }
            track_by_id = {item.global_track_id: item for item in tick.association.tracks}
            active_cameras = {item.camera_id for item in tick.camera_results}
            for camera in tick.camera_results:
                root = f"xr02/live/cameras/{camera.camera_id}"
                height, width = camera.calibrated_bgr.shape[:2]
                self._last_camera_dimensions[camera.camera_id] = (width, height)
                if tick.tick_index % self._image_every_n_ticks == 0:
                    self._log(root, self._rr.Image(camera.calibrated_bgr[..., ::-1]))
                self._clear(f"{root}/feed_status")
                boxes = [observation.bbox_xyxy for observation in camera.observations]
                if boxes:
                    labels: list[str] = []
                    colors: list[list[int]] = []
                    for observation in camera.observations:
                        observation_id = (
                            f"{observation.frame.frame_id}.d{observation.detection_index}"
                        )
                        assignment = assignments.get(
                            observation_id
                        ) or assignments_by_local_track.get(observation.track.stable_id)
                        label, color = _camera_observation_presentation(
                            observation.track.local_track_id,
                            observation.confidence,
                            observation.embedding_status.value,
                            assignment,
                            track_by_id,
                        )
                        labels.append(label)
                        colors.append(color)
                    self._log(
                        f"{root}/people",
                        self._rr.Boxes2D(
                            array=boxes,
                            array_format=self._rr.Box2DFormat.XYXY,
                            labels=labels,
                            colors=colors,
                        ),
                    )
                    self._log(
                        f"{root}/footpoints",
                        self._rr.Points2D(
                            [item.footpoint_uv for item in camera.observations],
                            colors=colors,
                            labels=labels,
                            radii=5.0,
                        ),
                    )
                else:
                    self._clear(f"{root}/people")
                    self._clear(f"{root}/footpoints")
            for camera_id in set(CAMERA_IDS) - active_cameras:
                root = f"xr02/live/cameras/{camera_id}"
                self._clear(f"{root}/people")
                self._clear(f"{root}/footpoints")
                dimensions = self._last_camera_dimensions.get(camera_id)
                if dimensions is None:
                    continue
                width, height = dimensions
                status, color = _camera_feed_status(camera_id, tick)
                # Preserve the last good camera image. Only dynamic person overlays
                # are cleared; this labelled border makes staleness explicit.
                self._log(
                    f"{root}/feed_status",
                    self._rr.Boxes2D(
                        array=[[0.0, 0.0, float(width - 1), float(height - 1)]],
                        array_format=self._rr.Box2DFormat.XYXY,
                        labels=[status],
                        colors=[color],
                    ),
                )

            observed_track_ids = {
                assignment.global_track_id
                for assignment in tick.association.assignments
                if assignment.global_track_id is not None
                and assignment.state is not GlobalTrackState.DUPLICATE
            }
            self._update_diagnostic_events(tick, observed_track_ids)
            current_ids: set[str] = set()
            for track in tick.association.tracks:
                path_id = track.global_track_id.replace(":", "_")
                root = f"xr02/world/global_tracks/{path_id}"
                if track.state is GlobalTrackState.ENDED:
                    self._clear(root)
                    self._trails.pop(track.global_track_id, None)
                    continue
                point = [*track.last_world_xy_metres, 0.08]
                observed_now = track.global_track_id in observed_track_ids
                identity_color = _identity_color(track.global_track_id)
                state_color = _state_color(track.state, observed_now)
                freshness = "OBSERVED NOW" if observed_now else "NOT OBSERVED"
                camera_label = ", ".join(_short_camera(item) for item in track.camera_ids)
                label = (
                    f"{_short_global_id(track.global_track_id)} | "
                    f"{track.state.value.upper()} | {freshness} | "
                    f"{camera_label} | hits {track.hit_count}"
                )
                self._log(
                    f"{root}/state_halo",
                    self._rr.Points3D(
                        [[point[0], point[1], 0.075]],
                        colors=[state_color],
                        radii=0.24 if observed_now else 0.18,
                    ),
                )
                self._log(
                    f"{root}/position",
                    self._rr.Points3D(
                        [[point[0], point[1], 0.10]],
                        labels=[label],
                        colors=[identity_color],
                        radii=0.15 if observed_now else 0.10,
                    ),
                )
                self._log(
                    f"{root}/state",
                    self._rr.TextDocument(
                        f"{track.global_track_id}: **{track.state.value}**",
                        media_type="text/markdown",
                    ),
                )
                trail_state = self._trails.setdefault(track.global_track_id, _TrailState())
                if observed_now:
                    break_reason = _trail_break_reason(
                        trail_state,
                        point,
                        track.last_observed_monotonic_ns,
                        track.camera_ids,
                    )
                    if break_reason is not None:
                        trail_state.segment_index += 1
                        trail_state.points = []
                    trail_state.points.append(point)
                    del trail_state.points[: -self._trail_limit]
                    trail_state.last_observed_monotonic_ns = track.last_observed_monotonic_ns
                    trail_state.last_camera_ids = track.camera_ids
                    trail_state.break_pending = False
                    if len(trail_state.points) >= 2:
                        self._log(
                            f"{root}/trail_segments/s{trail_state.segment_index:04d}",
                            self._rr.LineStrips3D(
                                [trail_state.points],
                                colors=[identity_color],
                                radii=0.045,
                            ),
                        )
                else:
                    trail_state.break_pending = True
                current_ids.add(track.global_track_id)
            for global_id in set(self._trails) - current_ids:
                path_id = global_id.replace(":", "_")
                self._clear(f"xr02/world/global_tracks/{path_id}/position")

            self._log_diagnostics(tick, observed_track_ids)
            for name, value in (
                ("processing_latency_ms", tick.processing_latency_ms),
                ("rectification_ms", tick.rectification_ms),
                ("detector_ms", tick.detector_ms),
                ("reid_ms", tick.reid_ms),
                ("local_tracker_ms", tick.local_tracker_ms),
                ("association_ms", tick.association_ms),
                ("evidence/appearance_unavailable", tick.appearance_unavailable_count),
                ("evidence/appearance_fresh", tick.appearance_fresh_count),
                ("evidence/appearance_persisted", tick.appearance_persisted_count),
                ("evidence/flush_due", int(tick.evidence_flush_due)),
                (
                    "evidence/pending_association_observations",
                    tick.pending_association_observations,
                ),
                ("cadence/association_updated", int(tick.association_updated)),
            ):
                self._log(f"xr02/telemetry/{name}", self._rr.Scalar(value))
            assignment_counts = _assignment_counts(tick)
            track_counts = _track_state_counts(tick)
            for state in GlobalTrackState:
                self._log(
                    f"xr02/telemetry/track_states/{state.value}",
                    self._rr.Scalar(track_counts.get(state.value, 0)),
                )
            for state in (GlobalTrackState.AMBIGUOUS, GlobalTrackState.DUPLICATE):
                self._log(
                    f"xr02/telemetry/assignment_evidence/{state.value}",
                    self._rr.Scalar(assignment_counts.get(state.value, 0)),
                )
            if tick.cross_camera_skew_ms is not None:
                self._log(
                    "xr02/telemetry/cross_camera_skew_ms",
                    self._rr.Scalar(tick.cross_camera_skew_ms),
                )
            if self._live_recording is not None:
                self._live_ticks += 1

    def _update_diagnostic_events(
        self,
        tick: LiveAssociationTick,
        observed_track_ids: set[str],
    ) -> None:
        current_ids: set[str] = set()
        for track in tick.association.tracks:
            global_id = track.global_track_id
            current_ids.add(global_id)
            previous_state = self._previous_track_states.get(global_id)
            previous_cameras = self._previous_track_cameras.get(global_id)
            observed_now = global_id in observed_track_ids
            short_id = _short_global_id(global_id)
            cameras = ", ".join(_short_camera(item) for item in track.camera_ids)
            if previous_state is None:
                self._add_event(
                    tick.tick_index,
                    f"NEW {short_id} · {track.state.value.upper()} · {cameras}",
                )
            elif (
                observed_now
                and previous_state
                in {
                    GlobalTrackState.OCCLUDED,
                    GlobalTrackState.LOST,
                    GlobalTrackState.DORMANT,
                }
                and track.state in {GlobalTrackState.TENTATIVE, GlobalTrackState.CONFIRMED}
            ):
                self._add_event(
                    tick.tick_index,
                    f"REACQUIRED {short_id} · {previous_state.value.upper()} → "
                    f"{track.state.value.upper()} · {cameras}",
                )
            elif previous_state is not track.state:
                self._add_event(
                    tick.tick_index,
                    f"STATE {short_id} · {previous_state.value.upper()} → "
                    f"{track.state.value.upper()}",
                )
            if (
                previous_cameras is not None
                and previous_cameras != track.camera_ids
                and observed_now
            ):
                before = ", ".join(_short_camera(item) for item in previous_cameras)
                self._add_event(
                    tick.tick_index,
                    f"CAMERA CHANGE {short_id} · {before} → {cameras}",
                )
            self._previous_track_states[global_id] = track.state
            self._previous_track_cameras[global_id] = track.camera_ids

        duplicate_ids = frozenset(
            item.global_track_id
            for item in tick.association.assignments
            if item.state is GlobalTrackState.DUPLICATE and item.global_track_id is not None
        )
        for global_id in sorted(duplicate_ids - self._previous_duplicate_ids):
            duplicate_camera_ids = sorted(
                {
                    item.camera_id
                    for item in tick.association.assignments
                    if item.global_track_id == global_id
                }
            )
            self._add_event(
                tick.tick_index,
                f"OVERLAP DEDUP {_short_global_id(global_id)} · "
                f"{', '.join(_short_camera(item) for item in duplicate_camera_ids)}",
            )
        self._previous_duplicate_ids = duplicate_ids

        ambiguous_ids = frozenset(
            item.local_track_stable_id
            for item in tick.association.assignments
            if item.state is GlobalTrackState.AMBIGUOUS
        )
        newly_ambiguous = ambiguous_ids - self._previous_ambiguous_local_ids
        if newly_ambiguous:
            ambiguous_camera_ids = sorted(
                {
                    item.camera_id
                    for item in tick.association.assignments
                    if item.local_track_stable_id in newly_ambiguous
                }
            )
            self._add_event(
                tick.tick_index,
                f"AMBIGUOUS EVIDENCE ×{len(newly_ambiguous)} · "
                f"{', '.join(_short_camera(item) for item in ambiguous_camera_ids)} · "
                "no forced ID",
            )
        self._previous_ambiguous_local_ids = ambiguous_ids

        for global_id in set(self._previous_track_states) - current_ids:
            self._previous_track_states.pop(global_id, None)
            self._previous_track_cameras.pop(global_id, None)

    def _add_event(self, tick_index: int, detail: str) -> None:
        self._diagnostic_events.appendleft(f"tick {tick_index:04d} · {detail}")

    def _log_diagnostics(
        self,
        tick: LiveAssociationTick,
        observed_track_ids: set[str],
    ) -> None:
        self._log(
            "xr02/diagnostics/active_tracks",
            self._rr.TextDocument(
                _active_tracks_markdown(tick, observed_track_ids),
                media_type="text/markdown",
            ),
        )
        self._log(
            "xr02/diagnostics/events",
            self._rr.TextDocument(
                _events_markdown(self._diagnostic_events),
                media_type="text/markdown",
            ),
        )
        self._log(
            "xr02/status/live",
            self._rr.TextDocument(_status_markdown(tick), media_type="text/markdown"),
        )

    def log_service_state(self, state: str, detail: str) -> None:
        with self._lock:
            self._log(
                "xr02/status/service",
                self._rr.TextDocument(f"# {state}\n\n{detail}", media_type="text/markdown"),
            )

    def reset_trails(self) -> None:
        with self._lock:
            for global_id in self._trails:
                path_id = global_id.replace(":", "_")
                self._clear(f"xr02/world/global_tracks/{path_id}/trail")
                self._clear(f"xr02/world/global_tracks/{path_id}/trail_segments")
            self._trails.clear()

    def open_viewer(self) -> None:
        with self._lock:
            if self._closed:
                self._open_archive_viewer()
                return
            if self._viewer_port is not None and _listener_open(self._viewer_port):
                return
        port = _reserve_port()
        previous_path = os.environ.get("PATH")
        os.environ["PATH"] = os.pathsep.join(
            item for item in (str(self._rerun_executable.parent), previous_path) if item
        )
        try:
            self._rr.spawn(
                port=port,
                connect=False,
                hide_welcome_screen=True,
                recording=self._archive,
            )
        except Exception as error:
            raise XR02LiveRerunError("failed to launch native Rerun") from error
        finally:
            if previous_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous_path
        if not _wait_listener(port, 10.0):
            raise XR02LiveRerunError("native Rerun live listener did not start")
        with self._lock:
            live = self._rr.new_recording(
                "xr02-wp4-live-global-tracking",
                recording_id=self.recording_path.stem,
            )
            self._rr.connect_tcp(f"127.0.0.1:{port}", recording=live)
            self._live_recording = live
            self._viewer_port = port
            self._live_connections += 1
            self._log_static((live,))
            self._send_blueprint((live,))
            live.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._archive.flush()
            self._archive.disconnect()
            if self._live_recording is not None:
                self._live_recording.flush()
                self._live_recording.disconnect()
                self._live_recording = None
            self._closed = True

    def evidence(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema": "xr02.wp4.rerun_stream.v1",
                "archive_path": str(self.recording_path),
                "archive_tick_count": self._tick_count,
                "live_connection_count": self._live_connections,
                "live_tick_count": self._live_ticks,
                "last_live_port": self._viewer_port,
                "image_every_n_association_ticks": self._image_every_n_ticks,
                "scene_context_sha256": self._scene.scene.context_sha256,
                "static_rerun_source_sha256": self._scene.static_rerun.sha256,
                "presentation_profile": "xr02-wp4-operator-people-summary-v4-stale-hold",
                "identity_color_policy": "stable-global-id-palette-v1",
                "default_visible_panels": [
                    "facility_3d",
                    "four_live_cameras",
                    "people_in_scene",
                ],
                "closed": self._closed,
            }

    def _log_static(self, recordings: tuple[Any, ...]) -> None:
        rr = self._rr
        with np.load(self._scene.geometry.path, allow_pickle=False) as archive:
            points = np.asarray(archive["points"], dtype=np.float64)
            colors = np.asarray(archive["colors_rgb"], dtype=np.uint8)
        _log_to(
            recordings,
            "xr02/world/static/adopted_cloud",
            rr.Points3D(points, colors=colors, radii=0.018),
            static=True,
        )
        x0, y0 = self._floor.minimum_xy_metres
        x1, y1 = self._floor.maximum_xy_metres
        outline = [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]]
        _log_to(
            recordings,
            "xr02/world/static/floor_z0",
            rr.LineStrips3D([outline], colors=[[190, 190, 190]], radii=0.02),
            static=True,
        )
        _log_to(
            recordings,
            "xr02/world/static/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=[[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]],
                colors=[[255, 64, 64], [64, 220, 96], [64, 128, 255]],
            ),
            static=True,
        )
        p06 = json.loads(Path(self._scene.p06_calibration.path).read_text(encoding="utf-8"))
        records = {item["camera_id"]: item for item in p06["cameras"]}
        for camera_id in CAMERA_IDS:
            image = cv2.imread(records[camera_id]["pinhole_derivative"]["path"], cv2.IMREAD_COLOR)
            if image is None:
                raise XR02LiveRerunError(f"static camera image unavailable: {camera_id}")
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
                _RecordingProxy(rr, recordings),
                "xr02/world/static/cameras",
                "xr02/world/static/camera_labels",
                frustum,
            )
        _log_to(
            recordings,
            "xr02/status/scene_context",
            rr.TextDocument(
                "\n".join(
                    (
                        "# Frozen scene context",
                        f"Scene: `{self._scene.scene.context_sha256}`",
                        f"Geometry: `{self._scene.geometry.sha256}`",
                        f"Floor: `{self._scene.floor.sha256}`",
                        f"Static RRD authority: `{self._scene.static_rerun.sha256}`",
                    )
                ),
                media_type="text/markdown",
            ),
            static=True,
        )
        _log_to(
            recordings,
            "xr02/diagnostics/legend",
            rr.TextDocument(_legend_markdown(), media_type="text/markdown"),
            static=True,
        )

    def _send_blueprint(self, recordings: tuple[Any, ...]) -> None:
        import rerun.blueprint as rrb

        camera_views = [
            rrb.Spatial2DView(origin=f"/xr02/live/cameras/{camera}", name=camera)
            for camera in CAMERA_IDS
        ]
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents="xr02/world/**",
                    name="Live people in facility",
                ),
                rrb.Vertical(
                    rrb.Horizontal(camera_views[0], camera_views[1]),
                    rrb.Horizontal(camera_views[2], camera_views[3]),
                    rrb.TextDocumentView(
                        origin="/xr02/diagnostics/active_tracks",
                        name="People in scene",
                    ),
                    row_shares=[3, 3, 2],
                ),
                column_shares=[3, 4],
            ),
            auto_layout=False,
            auto_views=False,
        )
        for recording in recordings:
            recording.send_blueprint(blueprint)

    def _active_recordings(self) -> tuple[Any, ...]:
        return (
            (self._archive,)
            if self._live_recording is None
            else (self._archive, self._live_recording)
        )

    def _log(self, path: str, entity: Any, *, static: bool = False) -> None:
        _log_to(self._active_recordings(), path, entity, static=static)

    def _clear(self, path: str) -> None:
        self._log(path, self._rr.Clear(recursive=True))

    def _open_archive_viewer(self) -> None:
        try:
            self._viewer_process = subprocess.Popen(
                [str(self._rerun_executable), str(self.recording_path), "--port", "0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            raise XR02LiveRerunError("failed to open saved Rerun archive") from error


class _RecordingProxy:
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


def _trail_break_reason(
    state: _TrailState,
    point: list[float],
    observed_monotonic_ns: int,
    camera_ids: tuple[str, ...],
) -> str | None:
    """Keep identity continuity while refusing to draw a false continuous path."""

    if not state.points or state.last_observed_monotonic_ns is None:
        return None
    if state.break_pending:
        return "not_observed_gap"
    elapsed_seconds = (observed_monotonic_ns - state.last_observed_monotonic_ns) / 1_000_000_000.0
    if elapsed_seconds < 0 or elapsed_seconds > 1.0:
        return "observation_time_gap"
    displacement_metres = math.hypot(
        point[0] - state.points[-1][0], point[1] - state.points[-1][1]
    )
    if displacement_metres > 2.0:
        return "position_discontinuity"
    if camera_ids != state.last_camera_ids and displacement_metres > 1.0:
        return "camera_handoff_discontinuity"
    return None


_IDENTITY_PALETTE: tuple[tuple[int, int, int], ...] = (
    (78, 121, 255),
    (255, 92, 92),
    (60, 205, 145),
    (255, 178, 64),
    (177, 113, 255),
    (54, 196, 224),
    (244, 105, 184),
    (166, 207, 68),
    (255, 128, 60),
    (115, 151, 255),
    (225, 84, 122),
    (54, 181, 112),
    (255, 211, 77),
    (145, 105, 220),
    (36, 158, 190),
    (224, 112, 202),
    (125, 184, 73),
    (234, 135, 48),
    (98, 111, 230),
    (230, 101, 72),
    (46, 190, 175),
    (205, 132, 45),
    (158, 92, 196),
    (68, 147, 210),
)


def _identity_color(global_id: str) -> list[int]:
    try:
        index = int(global_id.rsplit(":", 1)[1]) - 1
    except (IndexError, ValueError):
        index = sum(ord(character) for character in global_id)
    return list(_IDENTITY_PALETTE[index % len(_IDENTITY_PALETTE)])


def _state_color(state: GlobalTrackState, observed_now: bool) -> list[int]:
    if state is GlobalTrackState.DORMANT:
        return [170, 110, 225]
    if state is GlobalTrackState.LOST:
        return [255, 72, 72]
    if state is GlobalTrackState.OCCLUDED or not observed_now:
        return [255, 192, 64]
    if state is GlobalTrackState.TENTATIVE:
        return [245, 220, 90]
    if state is GlobalTrackState.ENDED:
        return [105, 105, 115]
    return [80, 240, 135]


def _camera_feed_status(camera_id: str, tick: LiveAssociationTick) -> tuple[str, list[int]]:
    if camera_id in tick.stale_camera_ids:
        return "STALE — last good processed frame retained", [255, 176, 48]
    if camera_id in tick.missing_camera_ids:
        return "OFFLINE/MISSING — last good processed frame retained", [255, 72, 72]
    return "NO NEW PROCESSED FRAME — last good frame retained", [255, 210, 72]


def _camera_observation_presentation(
    local_track_id: int,
    confidence: float,
    embedding_status: str,
    assignment: MemberAssignment | None,
    track_by_id: dict[str, GlobalTrackSnapshot],
) -> tuple[str, list[int]]:
    local = f"L{local_track_id}"
    if assignment is None:
        return (
            f"NO GLOBAL ID · {local} · appearance {embedding_status} · conf {confidence:.2f}",
            [145, 145, 155],
        )
    if assignment.state is GlobalTrackState.AMBIGUOUS:
        return (
            f"AMBIGUOUS · no global ID · {local} · {assignment.reason}",
            [255, 145, 60],
        )
    global_id = assignment.global_track_id
    if global_id is None:
        return f"NO GLOBAL ID · {local}", [145, 145, 155]
    track = track_by_id.get(global_id)
    state = assignment.state if track is None else track.state
    evidence = (
        "DUPLICATE VIEW" if assignment.state is GlobalTrackState.DUPLICATE else state.value.upper()
    )
    return (
        f"{_short_global_id(global_id)} · {evidence} · {local} · conf {confidence:.2f}",
        _identity_color(global_id),
    )


def _short_global_id(global_id: str) -> str:
    return global_id.replace("g:", "G")


def _short_camera(camera_id: str) -> str:
    return camera_id.replace("office-cam-0", "C")


def _assignment_counts(tick: LiveAssociationTick) -> dict[str, int]:
    counts: dict[str, int] = {}
    for assignment in tick.association.assignments:
        counts[assignment.state.value] = counts.get(assignment.state.value, 0) + 1
    return counts


def _track_state_counts(tick: LiveAssociationTick) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track in tick.association.tracks:
        counts[track.state.value] = counts.get(track.state.value, 0) + 1
    return counts


def _active_tracks_markdown(
    tick: LiveAssociationTick,
    observed_track_ids: set[str],
) -> str:
    visible_count = sum(
        track.global_track_id in observed_track_ids and track.state is not GlobalTrackState.ENDED
        for track in tick.association.tracks
    )
    lines = [
        f"# People currently observed: {visible_count}",
        "",
        "| Color code | Tracking ID | Tracking status | XY position (m) |",
        "|---|---|---|---:|",
    ]
    active = sorted(
        (track for track in tick.association.tracks if track.state is not GlobalTrackState.ENDED),
        key=lambda item: item.global_track_id,
    )
    for track in active:
        color = _identity_color(track.global_track_id)
        color_hex = "#" + "".join(f"{value:02X}" for value in color)
        xy = track.last_world_xy_metres
        observed = track.global_track_id in observed_track_ids
        visibility = "OBSERVED" if observed else "NOT CURRENTLY VISIBLE"
        lines.append(
            f"| **{color_hex}** | **{_short_global_id(track.global_track_id)}** | "
            f"**{track.state.value.upper()} · {visibility}** | "
            f"**({xy[0]:.2f}, {xy[1]:.2f})** |"
        )
    if not active:
        lines.append("| — | — | **NO PERSON TRACKED** | — |")
    elif len(active) != visible_count:
        lines.extend(
            (
                "",
                f"Tracked IDs retained: **{len(active)}**. "
                "Occluded or lost rows are retained tracks and are not included "
                "in the observed-person count.",
            )
        )
    return "\n".join(lines)


def _events_markdown(events: deque[str]) -> str:
    lines = [
        "# Lifecycle and association events",
        "",
        "Newest first. Reacquired means the same global ID returned within the bounded gate.",
        "",
    ]
    lines.extend(f"- {event}" for event in events)
    if not events:
        lines.append("- No lifecycle transition yet.")
    return "\n".join(lines)


def _legend_markdown() -> str:
    return "\n".join(
        (
            "# How to read this diagnostic",
            "",
            "**Inner point, trail and camera box color = anonymous global ID.** "
            "The same color should follow the same person across cameras.",
            "",
            "**Outer halo = lifecycle:** green confirmed/current; yellow tentative; "
            "amber occluded/not observed; red lost; purple dormant; grey ended.",
            "",
            "- **TENTATIVE:** new global ID awaiting confirmation.",
            "- **CONFIRMED:** global ID has sufficient accepted observations.",
            "- **OCCLUDED:** temporarily not observed, still inside the short hold.",
            "- **LOST:** not observed beyond the occlusion hold but still reacquirable.",
            "- **DORMANT:** longer absence; identity gallery retained inside the bounded window.",
            "- **REACQUIRED event:** a lost/occluded ID returned without creating a new ID.",
            "- **AMBIGUOUS:** evidence fits multiple possibilities; no global ID is forced.",
            "- **DUPLICATE VIEW:** another camera sees the same global person in this tick.",
            "- **NO GLOBAL ID:** local detection lacks evidence required by the combined policy.",
            "",
            "These are anonymous R&D associations, not biometric identity "
            "or production assurance.",
        )
    )


def _status_markdown(tick: LiveAssociationTick) -> str:
    counts: dict[str, int] = {}
    for assignment in tick.association.assignments:
        counts[assignment.state.value] = counts.get(assignment.state.value, 0) + 1
    active_tracks = sum(
        track.state is not GlobalTrackState.ENDED for track in tick.association.tracks
    )
    return "\n".join(
        (
            "# XR02 live diagnostic",
            f"Scene: `{tick.association.scene_context_sha256[:16]}…`",
            f"Profile: `{tick.association.profile_id}`",
            f"Active global IDs: **{active_tracks}**",
            f"Appearance unavailable: **{tick.appearance_unavailable_count}**",
            f"Assignment states: `{json.dumps(counts, sort_keys=True)}`",
            f"Stale cameras: `{list(tick.stale_camera_ids)}`",
            f"Missing cameras: `{list(tick.missing_camera_ids)}`",
            f"Processing: **{tick.processing_latency_ms:.1f} ms**",
        )
    )


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _listener_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _wait_listener(port: int, timeout_seconds: float) -> bool:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        if _listener_open(port):
            return True
        sleep(0.05)
    return False
