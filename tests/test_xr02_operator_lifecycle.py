from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spatial_mapping_phase2.xr02_compact_telemetry import (
    CompactLiveTelemetryJournal,
    verify_compact_telemetry,
)
from spatial_mapping_phase2.xr02_global_domain import (
    AssociationTickResult,
    GlobalTrackSnapshot,
    GlobalTrackState,
    MemberAssignment,
)
from spatial_mapping_phase2.xr02_journal import (
    VolatileEmbeddingStore,
    XR02JournalError,
)
from spatial_mapping_phase2.xr02_live_pipeline import (
    CalibratedCameraResult,
    LiveAssociationTick,
)
from spatial_mapping_phase2.xr02_recording_catalog import (
    RecordingCatalogError,
    XR02RecordingCatalog,
    XR02RunMode,
    XR02RunState,
)
from spatial_mapping_phase2.xr02_supervision import CanonicalDetections


def test_compact_live_telemetry_is_rate_bounded_and_contains_only_anonymous_xy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live-telemetry.jsonl"
    journal = CompactLiveTelemetryJournal(path, sample_interval_seconds=1.0)
    first = _tick(0, 1_000_000_000)
    assert journal.record(first)
    assert not journal.record(replace(first, tick_index=1, completed_monotonic_ns=1_500_000_000))
    assert journal.record(replace(first, tick_index=2, completed_monotonic_ns=2_100_000_000))
    journal.close()

    verified = verify_compact_telemetry(path)
    assert verified.samples == 2
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    payload = records[0]["payload"]
    assert payload["observed_people_count"] == 1
    assert payload["active_people_count"] == 1
    assert payload["camera_detection_counts"] == {
        "office-cam-01": 2,
        "office-cam-02": 0,
        "office-cam-03": 0,
        "office-cam-04": 0,
    }
    assert payload["tracks"][0]["world_xy_metres"] == [1.25, 2.5]
    encoded = path.read_text(encoding="utf-8").lower()
    for forbidden in ("bbox", "embedding", "rtsp://", "image"):
        assert forbidden not in encoded
    evidence = journal.evidence()
    assert evidence["samples"] == 2
    assert evidence["contains_images"] is False


def test_volatile_embedding_store_is_bounded_and_writes_no_files(tmp_path: Path) -> None:
    store = VolatileEmbeddingStore("fixture", maximum_vectors=2)
    first = store.put(np.asarray([1.0, 0.0], dtype=np.float32))
    second = store.put(np.asarray([0.0, 1.0], dtype=np.float32))
    third = store.put(np.asarray([1.0, 1.0], dtype=np.float32))
    assert np.allclose(store.load(second), np.asarray([0.0, 1.0], dtype=np.float32))
    assert np.allclose(store.load(third), np.asarray([2**-0.5, 2**-0.5], dtype=np.float32))
    with pytest.raises(XR02JournalError, match="no longer cached"):
        store.load(first)
    assert not tuple(tmp_path.iterdir())


def test_catalog_blocks_on_staged_recording_then_saves_without_renaming(
    tmp_path: Path,
) -> None:
    catalog = XR02RecordingCatalog(tmp_path / "xr02-recordings.sqlite3")
    directory = tmp_path / "xr02-recording-fixture"
    run = catalog.begin(XR02RunMode.RECORDING, directory)
    catalog.mark_running(run.session_id)
    catalog.mark_finalizing(run.session_id)
    staged = catalog.finalize(
        run.session_id,
        manifest_path=directory / "manifest.json",
        telemetry_path=None,
        byte_count=123,
    )
    assert staged.state is XR02RunState.AWAITING_DISPOSITION
    with pytest.raises(RecordingCatalogError, match="resolve"):
        catalog.begin(XR02RunMode.LIVE, tmp_path / "xr02-live-blocked")
    saved = catalog.save(run.session_id, "  Loading   dock  shift  ")
    assert saved.label == "Loading dock shift"
    assert saved.run_directory == directory.resolve()
    assert catalog.saved() == (saved,)

    live = catalog.begin(XR02RunMode.LIVE, tmp_path / "xr02-live-fixture")
    catalog.mark_running(live.session_id)
    catalog.mark_finalizing(live.session_id)
    stopped = catalog.finalize(
        live.session_id,
        manifest_path=tmp_path / "live-summary.json",
        telemetry_path=tmp_path / "live-telemetry.jsonl",
        byte_count=45,
    )
    assert stopped.state is XR02RunState.STOPPED
    assert catalog.recent_live() == (stopped,)


def test_catalog_recovers_interrupted_transition_and_requires_resolution(tmp_path: Path) -> None:
    path = tmp_path / "xr02-recordings.sqlite3"
    catalog = XR02RecordingCatalog(path)
    run = catalog.begin(XR02RunMode.LIVE, tmp_path / "xr02-live-interrupted")
    catalog.mark_running(run.session_id)

    reopened = XR02RecordingCatalog(path)
    recovery = reopened.blocking()
    assert recovery is not None
    assert recovery.state is XR02RunState.RECOVERY_REQUIRED
    assert "console process ended" in (recovery.error_detail or "")
    reopened.mark_deleted(run.session_id)
    assert reopened.blocking() is None


def test_catalog_migrates_old_schema_and_retains_scene_resume_provenance(tmp_path: Path) -> None:
    path = tmp_path / "xr02-recordings.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE operator_runs (
                session_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                state TEXT NOT NULL,
                run_directory TEXT NOT NULL UNIQUE,
                started_at_utc TEXT NOT NULL,
                stopped_at_utc TEXT,
                label TEXT,
                manifest_path TEXT,
                telemetry_path TEXT,
                byte_count INTEGER,
                error_detail TEXT
            )"""
        )

    catalog = XR02RecordingCatalog(path)
    run = catalog.begin(
        XR02RunMode.LIVE,
        tmp_path / "resumed-live",
        scene_context_sha256="a" * 64,
        scene_binding_sha256="b" * 64,
        resumed_from_session_id="xr02-previous",
        scene_update_id="auto-1",
    )
    catalog.mark_running(run.session_id)
    catalog.mark_finalizing(run.session_id)
    finalized = catalog.finalize(
        run.session_id,
        manifest_path=tmp_path / "summary.json",
        telemetry_path=tmp_path / "telemetry.jsonl",
        byte_count=12,
        stop_reason="scheduled_scene_update",
    )

    assert finalized.scene_context_sha256 == "a" * 64
    assert finalized.scene_binding_sha256 == "b" * 64
    assert finalized.resumed_from_session_id == "xr02-previous"
    assert finalized.scene_update_id == "auto-1"
    assert finalized.stop_reason == "scheduled_scene_update"


def _tick(tick_index: int, completed_monotonic_ns: int) -> LiveAssociationTick:
    track = GlobalTrackSnapshot(
        "g:000001",
        GlobalTrackState.CONFIRMED,
        (1.25, 2.5),
        completed_monotonic_ns - 1,
        ("office-cam-01",),
        3,
        "fixture",
    )
    assignment = MemberAssignment(
        "obs-1",
        "local-1",
        "office-cam-01",
        track.global_track_id,
        GlobalTrackState.CONFIRMED,
        "fixture",
    )
    association = AssociationTickResult(
        "a" * 64,
        tick_index,
        completed_monotonic_ns - 1,
        "fixture",
        (assignment,),
        (track,),
        (),
    )
    detections = CanonicalDetections(
        np.asarray([[0, 0, 10, 20], [12, 1, 22, 24]], dtype=np.float32),
        np.asarray([0.9, 0.8], dtype=np.float32),
        np.asarray([0, 0], dtype=np.int32),
    )
    camera = CalibratedCameraResult(
        "office-cam-01",
        "frame-1",
        completed_monotonic_ns - 2,
        np.zeros((24, 24, 3), dtype=np.uint8),
        detections,
        (),
    )
    return LiveAssociationTick(
        tick_index,
        completed_monotonic_ns - 2,
        completed_monotonic_ns,
        (camera,),
        (),
        (),
        None,
        association,
        1.0,
        1.0,
        1.0,
        1.0,
        4.0,
        0,
        0,
        0,
    )
