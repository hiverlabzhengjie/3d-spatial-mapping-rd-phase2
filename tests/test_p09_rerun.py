from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from spatial_mapping_phase2.p09_rerun import P09RerunLogger, _log_to, _SegmentedTrail


def _logger(tmp_path: Path, *, closed: bool) -> P09RerunLogger:
    logger = object.__new__(P09RerunLogger)
    logger.recording_path = tmp_path / "completed.rrd"
    logger._rerun_executable = tmp_path / "rerun.exe"
    logger._viewer_process = None
    logger._viewer_port = None
    logger._live_recording = None
    logger._live_connection_count = 0
    logger._live_tick_count = 0
    logger._live_connected_before_close = False
    logger._tick_index = 0
    logger._archive_recording = Mock()
    logger._rr = Mock()
    logger._closed = closed
    logger._lock = threading.Lock()
    return logger


def test_active_open_spawns_unique_listener_and_connects_live_stream(tmp_path: Path) -> None:
    logger = _logger(tmp_path, closed=False)
    live_recording = Mock()
    logger._rr.new_recording.return_value = live_recording
    original_path = os.environ.get("PATH")
    spawn_path: list[str] = []

    def record_spawn_path(**_: object) -> None:
        spawn_path.append(os.environ["PATH"])

    logger._rr.spawn.side_effect = record_spawn_path

    with (
        patch("spatial_mapping_phase2.p09_rerun._reserve_local_port", return_value=23456),
        patch("spatial_mapping_phase2.p09_rerun._wait_for_listener", return_value=True),
        patch.object(logger, "_log_static_context") as static_context,
        patch.object(logger, "_send_blueprint") as blueprint,
    ):
        logger.open_viewer()

    logger._rr.spawn.assert_called_once_with(
        port=23456,
        connect=False,
        hide_welcome_screen=True,
        recording=logger._archive_recording,
    )
    logger._rr.connect_tcp.assert_called_once_with(
        "127.0.0.1:23456", recording=live_recording
    )
    static_context.assert_called_once_with((live_recording,))
    blueprint.assert_called_once_with((live_recording,))
    live_recording.flush.assert_called_once_with()
    assert spawn_path[0].split(os.pathsep)[0] == str(logger._rerun_executable.parent)
    assert os.environ.get("PATH") == original_path
    assert logger._live_recording is live_recording
    assert logger._viewer_port == 23456


def test_stopped_open_loads_archived_rrd_with_isolated_port(tmp_path: Path) -> None:
    logger = _logger(tmp_path, closed=True)

    with patch("spatial_mapping_phase2.p09_rerun.subprocess.Popen") as popen:
        logger.open_viewer()

    popen.assert_called_once_with(
        [str(logger._rerun_executable), str(logger.recording_path), "--port", "0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def test_log_to_duplicates_one_event_across_live_and_archive() -> None:
    archive, live = Mock(), Mock()
    entity = object()
    _log_to((archive, live), "p09/live/camera", entity, static=True)
    archive.log.assert_called_once_with("p09/live/camera", entity, static=True)
    live.log.assert_called_once_with("p09/live/camera", entity, static=True)


def test_evidence_distinguishes_archive_and_live_delivery(tmp_path: Path) -> None:
    logger = _logger(tmp_path, closed=True)
    logger._tick_index = 12
    logger._live_connection_count = 1
    logger._live_tick_count = 10
    logger._live_connected_before_close = True
    logger._viewer_port = 23456
    assert logger.evidence() == {
        "sink_mode": "independent-archive-and-live-tcp",
        "active_viewer_launch_mode": "rerun-sdk-spawn",
        "archive_path": str(logger.recording_path),
        "archive_tick_count": 12,
        "live_connection_count": 1,
        "live_tick_count": 10,
        "live_connected_before_close": True,
        "live_connected_now": False,
        "last_live_port": 23456,
        "closed": True,
    }


def test_trail_break_prevents_line_across_an_unsupported_gap() -> None:
    trail = _SegmentedTrail(8)
    trail.append((1.0, 2.0))
    trail.append((1.2, 2.1))
    trail.break_segment()
    trail.append((8.0, 9.0))

    assert trail.line_strips == [[[1.0, 2.0, 0.06], [1.2, 2.1, 0.06]]]

    trail.append((8.2, 9.1))
    assert trail.line_strips == [
        [[1.0, 2.0, 0.06], [1.2, 2.1, 0.06]],
        [[8.0, 9.0, 0.06], [8.2, 9.1, 0.06]],
    ]


def test_segmented_trail_enforces_total_point_limit_and_reset() -> None:
    trail = _SegmentedTrail(3)
    trail.append((0.0, 0.0))
    trail.append((1.0, 0.0))
    trail.break_segment()
    trail.append((2.0, 0.0))
    trail.append((3.0, 0.0))

    assert trail.point_count == 3
    assert trail.line_strips == [[[2.0, 0.0, 0.06], [3.0, 0.0, 0.06]]]
    trail.clear()
    assert trail.point_count == 0
    assert trail.line_strips == []
