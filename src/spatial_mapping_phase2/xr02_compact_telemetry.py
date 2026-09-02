"""Low-volume time-associated telemetry retained for every XR02 Live run."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from spatial_mapping_phase2.p09_tracking_domain import CAMERA_IDS
from spatial_mapping_phase2.xr02_global_domain import GlobalTrackState
from spatial_mapping_phase2.xr02_live_pipeline import LiveAssociationTick

_GENESIS = "0" * 64


class CompactTelemetryError(RuntimeError):
    """Raised when compact Live telemetry cannot preserve its append-only contract."""


@dataclass(frozen=True, slots=True)
class CompactTelemetryVerification:
    samples: int
    final_sha256: str


class CompactLiveTelemetryJournal:
    """Stream anonymous count/track/XY samples without retaining image evidence."""

    def __init__(self, path: Path, *, sample_interval_seconds: float = 1.0) -> None:
        if not 0.1 <= sample_interval_seconds <= 60.0:
            raise CompactTelemetryError("sample interval must be within 0.1..60 seconds")
        if path.exists() and path.stat().st_size:
            raise CompactTelemetryError("compact telemetry path must be new for each Live run")
        self.path = path
        self.sample_interval_ns = int(round(sample_interval_seconds * 1_000_000_000))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self._sequence = 0
        self._previous_sha256 = _GENESIS
        self._last_sample_monotonic_ns: int | None = None
        self._closed = False
        self._lock = threading.Lock()

    def record(self, tick: LiveAssociationTick) -> bool:
        """Append at the bounded sampling rate; return whether this tick was retained."""

        with self._lock:
            if self._closed:
                raise CompactTelemetryError("compact telemetry journal is closed")
            if (
                self._last_sample_monotonic_ns is not None
                and tick.completed_monotonic_ns - self._last_sample_monotonic_ns
                < self.sample_interval_ns
            ):
                return False
            observed_ids = {
                assignment.global_track_id
                for assignment in tick.association.assignments
                if assignment.global_track_id is not None
                and assignment.state is not GlobalTrackState.DUPLICATE
            }
            active_tracks = tuple(
                track
                for track in tick.association.tracks
                if track.state is not GlobalTrackState.ENDED
            )
            active_ids = {track.global_track_id for track in active_tracks}
            camera_results = {camera.camera_id: camera for camera in tick.camera_results}
            payload: dict[str, object] = {
                "schema": "xr02.live_compact_sample.v1",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "tick_index": tick.tick_index,
                "admitted_monotonic_ns": tick.admitted_monotonic_ns,
                "completed_monotonic_ns": tick.completed_monotonic_ns,
                "observed_people_count": len(observed_ids & active_ids),
                "active_people_count": len(active_tracks),
                "ambiguous_observation_count": sum(
                    assignment.state is GlobalTrackState.AMBIGUOUS
                    for assignment in tick.association.assignments
                ),
                "camera_detection_counts": {
                    camera_id: (
                        0
                        if camera_id not in camera_results
                        else camera_results[camera_id].detections.count
                    )
                    for camera_id in CAMERA_IDS
                },
                "camera_local_track_counts": {
                    camera_id: (
                        0
                        if camera_id not in camera_results
                        else len(camera_results[camera_id].observations)
                    )
                    for camera_id in CAMERA_IDS
                },
                "stale_camera_ids": list(tick.stale_camera_ids),
                "missing_camera_ids": list(tick.missing_camera_ids),
                "tracks": [
                    {
                        "global_track_id": track.global_track_id,
                        "state": track.state.value,
                        "observed_now": track.global_track_id in observed_ids,
                        "world_xy_metres": [
                            round(track.last_world_xy_metres[0], 4),
                            round(track.last_world_xy_metres[1], 4),
                        ],
                        "camera_ids": list(track.camera_ids),
                        "last_observed_monotonic_ns": track.last_observed_monotonic_ns,
                    }
                    for track in active_tracks
                ],
            }
            core: dict[str, object] = {
                "sequence": self._sequence,
                "previous_sha256": self._previous_sha256,
                "payload": payload,
            }
            record_sha256 = _sha256(core)
            self._handle.write(_canonical_json({**core, "record_sha256": record_sha256}))
            self._handle.write("\n")
            self._handle.flush()
            self._sequence += 1
            self._previous_sha256 = record_sha256
            self._last_sample_monotonic_ns = tick.completed_monotonic_ns
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True

    def evidence(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema": "xr02.live_compact_telemetry.v1",
                "path": str(self.path.resolve()),
                "sample_interval_seconds": self.sample_interval_ns / 1_000_000_000.0,
                "samples": self._sequence,
                "final_sha256": self._previous_sha256,
                "bytes": self.path.stat().st_size if self.path.exists() else 0,
                "closed": self._closed,
                "contains_images": False,
                "contains_embeddings": False,
                "contains_bounding_boxes": False,
                "identity_scope": "anonymous scene-global track IDs",
            }


def verify_compact_telemetry(path: Path) -> CompactTelemetryVerification:
    previous = _GENESIS
    samples = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CompactTelemetryError(
                    f"compact telemetry line {line_number} is invalid JSON"
                ) from error
            if not isinstance(record, dict):
                raise CompactTelemetryError("compact telemetry record is not an object")
            if record.get("sequence") != samples or record.get("previous_sha256") != previous:
                raise CompactTelemetryError("compact telemetry sequence or chain changed")
            claimed = record.get("record_sha256")
            core = {key: value for key, value in record.items() if key != "record_sha256"}
            actual = _sha256(core)
            if claimed != actual:
                raise CompactTelemetryError("compact telemetry content changed")
            previous = actual
            samples += 1
    return CompactTelemetryVerification(samples=samples, final_sha256=previous)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()
