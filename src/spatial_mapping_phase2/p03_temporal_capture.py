"""Warm-stream, local-monotonic temporal bundle capture for the P03 revision."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from time import get_clock_info, monotonic, sleep
from typing import Protocol

from spatial_mapping_phase2.p01_observability import CAMERA_IDS, LocalRtspEndpoint
from spatial_mapping_phase2.p03_capture_domain import P03ContractError
from spatial_mapping_phase2.p03_capture_service import CapturePolicy

TEMPORAL_BUNDLE_SCHEMA = "p03-temporal-bundle-v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class TemporalAuthorityStatus(StrEnum):
    AUTHORITATIVE = "authoritative-local-acquisition-window"
    REJECTED_SKEW = "rejected-outside-acquisition-window"


@dataclass(frozen=True, slots=True, repr=False)
class BufferedFrame:
    camera_id: str
    frame_id: str
    profile_version: str
    acquisition_monotonic_ns: int
    observed_at_utc: str
    source_pts: int | None
    source_time_base: str | None
    jpeg_content: bytes

    def __post_init__(self) -> None:
        if self.camera_id not in CAMERA_IDS:
            raise P03ContractError("buffered frame camera_id is invalid")
        if not _IDENTIFIER.fullmatch(self.frame_id):
            raise P03ContractError("buffered frame_id is invalid")
        if self.acquisition_monotonic_ns < 0:
            raise P03ContractError("frame acquisition time must be non-negative")
        if not self.jpeg_content or len(self.jpeg_content) > 10 * 1024 * 1024:
            raise P03ContractError("buffered JPEG size is invalid")
        if (self.source_pts is None) != (self.source_time_base is None):
            raise P03ContractError("source PTS and time base must be present together")


@dataclass(frozen=True, slots=True)
class TemporalArtifact:
    camera_id: str
    frame_id: str
    relative_path: str
    sha256: str
    byte_count: int
    profile_version: str
    acquisition_monotonic_ns: int
    observed_at_utc: str
    source_pts: int | None
    source_time_base: str | None


@dataclass(frozen=True, slots=True)
class TemporalPairwiseSkew:
    camera_a: str
    camera_b: str
    skew_ns: int


@dataclass(frozen=True, slots=True)
class TemporalBundleManifest:
    schema_version: str
    bundle_id: str
    authority_status: TemporalAuthorityStatus
    clock_domain: str
    max_allowed_skew_ns: int
    clock_resolution_ns: int
    bundle_acquisition_monotonic_ns: int
    bundle_observed_at_utc: str
    timestamp_uncertainty_ns: int
    overall_skew_ns: int
    conservative_overall_skew_ns: int
    pairwise_skew: tuple[TemporalPairwiseSkew, ...]
    artifacts: tuple[TemporalArtifact, ...]
    software_identity: str

    def __post_init__(self) -> None:
        if self.schema_version != TEMPORAL_BUNDLE_SCHEMA:
            raise P03ContractError("unsupported temporal-bundle schema")
        if not _IDENTIFIER.fullmatch(self.bundle_id):
            raise P03ContractError("temporal bundle_id is invalid")
        if self.clock_domain != "host-monotonic-acquisition":
            raise P03ContractError("unsupported temporal clock domain")
        if self.max_allowed_skew_ns <= 0:
            raise P03ContractError("maximum skew must be positive")
        if self.clock_resolution_ns <= 0:
            raise P03ContractError("clock resolution must be positive")
        if self.conservative_overall_skew_ns < self.overall_skew_ns:
            raise P03ContractError("conservative skew cannot be below measured skew")
        if tuple(item.camera_id for item in self.artifacts) != CAMERA_IDS:
            raise P03ContractError("temporal bundle requires all four cameras in fixed order")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


class WarmFrameAdapter(Protocol):
    def frames(
        self,
        endpoint: LocalRtspEndpoint,
        policy: CapturePolicy,
        cancel: threading.Event,
    ) -> Iterable[BufferedFrame]:
        """Yield decoded frames from one warm read-only connection."""


class TemporalBundleRepository:
    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.resolve()
        self.bundle_root = self.root / "captures" / "p03" / "temporal_bundles"
        self.bundle_root.mkdir(parents=True, exist_ok=True)

    def persist(
        self,
        bundle_id: str,
        selected: tuple[BufferedFrame, ...],
        max_skew_ns: int,
        clock_resolution_ns: int,
        software_identity: str,
    ) -> TemporalBundleManifest:
        if not _IDENTIFIER.fullmatch(bundle_id):
            raise P03ContractError("temporal bundle_id is invalid")
        directory = self.bundle_root / bundle_id
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise P03ContractError("temporal bundle identity already exists") from error
        times = [frame.acquisition_monotonic_ns for frame in selected]
        ordered = sorted(
            selected, key=lambda frame: (frame.acquisition_monotonic_ns, frame.frame_id)
        )
        anchor = ordered[(len(ordered) - 1) // 2]
        overall_skew = max(times) - min(times)
        conservative_skew = overall_skew + 2 * clock_resolution_ns
        authority = (
            TemporalAuthorityStatus.AUTHORITATIVE
            if conservative_skew <= max_skew_ns
            else TemporalAuthorityStatus.REJECTED_SKEW
        )
        artifacts: list[TemporalArtifact] = []
        for frame in selected:
            filename = f"{frame.camera_id}.jpg"
            path = directory / filename
            path.write_bytes(frame.jpeg_content)
            artifacts.append(
                TemporalArtifact(
                    frame.camera_id,
                    frame.frame_id,
                    f"captures/p03/temporal_bundles/{bundle_id}/{filename}",
                    hashlib.sha256(frame.jpeg_content).hexdigest(),
                    len(frame.jpeg_content),
                    frame.profile_version,
                    frame.acquisition_monotonic_ns,
                    frame.observed_at_utc,
                    frame.source_pts,
                    frame.source_time_base,
                )
            )
        pairwise = tuple(
            TemporalPairwiseSkew(
                first.camera_id,
                second.camera_id,
                abs(first.acquisition_monotonic_ns - second.acquisition_monotonic_ns),
            )
            for first, second in itertools.combinations(selected, 2)
        )
        manifest = TemporalBundleManifest(
            TEMPORAL_BUNDLE_SCHEMA,
            bundle_id,
            authority,
            "host-monotonic-acquisition",
            max_skew_ns,
            clock_resolution_ns,
            anchor.acquisition_monotonic_ns,
            anchor.observed_at_utc,
            max(abs(value - anchor.acquisition_monotonic_ns) for value in times)
            + clock_resolution_ns,
            overall_skew,
            conservative_skew,
            pairwise,
            tuple(artifacts),
            software_identity,
        )
        (directory / "bundle.json").write_text(manifest.to_json(), encoding="utf-8")
        return manifest


class WarmTemporalCaptureService:
    """Own warm per-camera frame rings and temporal-gated immutable selection."""

    def __init__(
        self,
        endpoints: tuple[LocalRtspEndpoint, ...],
        adapter: WarmFrameAdapter,
        repository: TemporalBundleRepository,
        policy: CapturePolicy,
        software_identity: str,
        ring_capacity: int = 16,
        clock_resolution_ns: int | None = None,
    ) -> None:
        if tuple(endpoint.camera_id for endpoint in endpoints) != CAMERA_IDS:
            raise P03ContractError("warm capture requires four fixed camera endpoints")
        if ring_capacity < 2 or ring_capacity > 256:
            raise P03ContractError("ring capacity must be between 2 and 256")
        self._endpoints = endpoints
        self._adapter = adapter
        self._repository = repository
        self._policy = policy
        self._software_identity = software_identity
        observed_resolution = math.ceil(get_clock_info("monotonic").resolution * 1_000_000_000)
        self._clock_resolution_ns = (
            observed_resolution if clock_resolution_ns is None else clock_resolution_ns
        )
        if self._clock_resolution_ns <= 0:
            raise P03ContractError("clock resolution must be positive")
        self._buffers = {
            camera_id: deque[BufferedFrame](maxlen=ring_capacity) for camera_id in CAMERA_IDS
        }
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._threads: list[threading.Thread] = []
        self._failures: dict[str, str | None] = {camera_id: None for camera_id in CAMERA_IDS}

    def start(self) -> None:
        if self._threads:
            raise P03ContractError("warm capture service is already started")
        for endpoint in self._endpoints:
            thread = threading.Thread(
                target=self._worker,
                args=(endpoint,),
                name=f"p03-warm-{endpoint.camera_id}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def wait_until_ready(self, timeout_seconds: float) -> bool:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise P03ContractError("ready timeout must be finite and positive")
        deadline = monotonic() + timeout_seconds
        while monotonic() < deadline:
            with self._lock:
                if all(self._buffers[camera_id] for camera_id in CAMERA_IDS):
                    return True
            sleep(0.01)
        return False

    def status(self) -> dict[str, dict[str, int | str | None]]:
        with self._lock:
            return {
                camera_id: {
                    "buffered_frames": len(self._buffers[camera_id]),
                    "failure_class": self._failures[camera_id],
                }
                for camera_id in CAMERA_IDS
            }

    def capture(self, bundle_id: str, max_skew_ns: int) -> TemporalBundleManifest:
        if max_skew_ns <= 0:
            raise P03ContractError("maximum skew must be positive")
        with self._lock:
            snapshots = {camera_id: tuple(self._buffers[camera_id]) for camera_id in CAMERA_IDS}
        if any(not snapshots[camera_id] for camera_id in CAMERA_IDS):
            raise P03ContractError("all four warm frame buffers must be ready")
        selected = _closest_combination(snapshots)
        return self._repository.persist(
            bundle_id,
            selected,
            max_skew_ns,
            self._clock_resolution_ns,
            self._software_identity,
        )

    def close(self) -> None:
        self._cancel.set()
        for thread in self._threads:
            thread.join(timeout=self._policy.read_timeout_seconds + 1.0)
        self._threads.clear()

    def _worker(self, endpoint: LocalRtspEndpoint) -> None:
        while not self._cancel.is_set():
            try:
                for frame in self._adapter.frames(endpoint, self._policy, self._cancel):
                    if self._cancel.is_set():
                        return
                    with self._lock:
                        self._buffers[endpoint.camera_id].append(frame)
                        self._failures[endpoint.camera_id] = None
            except Exception as error:
                with self._lock:
                    self._failures[endpoint.camera_id] = type(error).__name__
                if not self._cancel.wait(self._policy.initial_backoff_seconds):
                    continue


def _closest_combination(
    snapshots: dict[str, tuple[BufferedFrame, ...]],
) -> tuple[BufferedFrame, ...]:
    candidates = tuple(
        frame.acquisition_monotonic_ns for frames in snapshots.values() for frame in frames
    )
    choices: list[tuple[int, tuple[str, ...], tuple[BufferedFrame, ...]]] = []
    for target in candidates:
        selected = tuple(
            min(
                snapshots[camera_id],
                key=lambda frame: (abs(frame.acquisition_monotonic_ns - target), frame.frame_id),
            )
            for camera_id in CAMERA_IDS
        )
        times = [frame.acquisition_monotonic_ns for frame in selected]
        choices.append(
            (max(times) - min(times), tuple(frame.frame_id for frame in selected), selected)
        )
    return min(choices, key=lambda item: (item[0], item[1]))[2]


class SyntheticWarmFrameAdapter:
    def __init__(self, timestamps: dict[str, tuple[int, ...]]) -> None:
        self.timestamps = timestamps

    def frames(
        self, endpoint: LocalRtspEndpoint, policy: CapturePolicy, cancel: threading.Event
    ) -> Iterable[BufferedFrame]:
        del policy
        for index, timestamp in enumerate(self.timestamps[endpoint.camera_id]):
            if cancel.is_set():
                return
            yield BufferedFrame(
                endpoint.camera_id,
                f"{endpoint.camera_id}-warm-{index:06d}",
                "stream-profile-v1",
                timestamp,
                "2026-08-14T00:00:00Z",
                index * 3000,
                "1/90000",
                f"jpeg:{endpoint.camera_id}:{index}".encode(),
            )
        while not cancel.wait(0.01):
            pass
