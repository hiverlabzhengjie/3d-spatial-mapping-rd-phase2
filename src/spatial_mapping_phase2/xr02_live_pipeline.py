"""Frozen-model live local tracking, floor projection and WP3 global association."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_live_runtime import LatestFrameSnapshot
from spatial_mapping_phase2.p09_projection import (
    CameraProjectionCalibration,
    FloorRectangle,
)
from spatial_mapping_phase2.xr02_association import (
    AssociationConfig,
    SceneGlobalAssociator,
    office_topology,
)
from spatial_mapping_phase2.xr02_boxmot import (
    CameraLocalTracker,
    fixed_camera_profiles,
    live_cadence_profile,
)
from spatial_mapping_phase2.xr02_cadence import (
    DeterministicMultiRateCadence,
    HigherCadenceProfile,
)
from spatial_mapping_phase2.xr02_global_domain import (
    AssociationObservation,
    AssociationTickResult,
    SignalProfile,
)
from spatial_mapping_phase2.xr02_global_journal import GlobalAssociationJournal
from spatial_mapping_phase2.xr02_journal import EmbeddingStore, ObservationJournal
from spatial_mapping_phase2.xr02_live_domain import AdoptedSceneSelection, XR02LiveContractError
from spatial_mapping_phase2.xr02_local_domain import (
    FrameKey,
    LocalTrackObservation,
    WorldProjectionStatus,
)
from spatial_mapping_phase2.xr02_local_pipeline import (
    CropQualityPolicy,
    EmbeddingCadence,
    LocalObservationAssembler,
    P08ProjectionAdapter,
)
from spatial_mapping_phase2.xr02_rectification import FusedLiveFrameRectifier
from spatial_mapping_phase2.xr02_supervision import (
    CanonicalDetections,
    person_detections_from_boxmot,
)


@dataclass(frozen=True, slots=True)
class LiveModelProfile:
    detector_path: Path
    detector_sha256: str
    reid_path: Path
    reid_sha256: str
    detector_confidence: float = 0.15
    new_track_confidence: float = 0.65
    local_confirmation_hits: int = 2
    profile_id: str = "wp4-yolo11n-c015-new065-osnet025-botsort-continuity-v3"

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.detector_confidence)
            and math.isfinite(self.new_track_confidence)
            and 0.0 < self.detector_confidence <= self.new_track_confidence <= 1.0
        ):
            raise XR02LiveContractError(
                "detector confidence must not exceed new-track confidence within (0, 1]"
            )
        if self.local_confirmation_hits <= 1:
            raise XR02LiveContractError("local confirmation hits must exceed one")
        for path, expected, label in (
            (self.detector_path, self.detector_sha256, "detector"),
            (self.reid_path, self.reid_sha256, "ReID"),
        ):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise XR02LiveContractError(f"{label} model identity changed")


@dataclass(frozen=True, slots=True)
class CalibratedCameraResult:
    camera_id: str
    frame_id: str
    acquisition_monotonic_ns: int
    calibrated_bgr: NDArray[np.uint8]
    detections: CanonicalDetections
    observations: tuple[LocalTrackObservation, ...]


@dataclass(frozen=True, slots=True)
class LiveAssociationTick:
    tick_index: int
    admitted_monotonic_ns: int
    completed_monotonic_ns: int
    camera_results: tuple[CalibratedCameraResult, ...]
    stale_camera_ids: tuple[str, ...]
    missing_camera_ids: tuple[str, ...]
    cross_camera_skew_ms: float | None
    association: AssociationTickResult
    detector_ms: float
    reid_ms: float
    local_tracker_ms: float
    association_ms: float
    processing_latency_ms: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    appearance_unavailable_count: int
    association_updated: bool = True
    association_tick_index: int | None = None
    publication_due: bool = True
    pending_association_observations: int = 0
    appearance_persisted_count: int = 0
    appearance_fresh_count: int = 0
    evidence_flush_due: bool = False
    rectification_ms: float = 0.0

    def summary(self) -> dict[str, object]:
        return {
            "tick_index": self.tick_index,
            "admitted_monotonic_ns": self.admitted_monotonic_ns,
            "completed_monotonic_ns": self.completed_monotonic_ns,
            "processed_camera_ids": [item.camera_id for item in self.camera_results],
            "stale_camera_ids": list(self.stale_camera_ids),
            "missing_camera_ids": list(self.missing_camera_ids),
            "cross_camera_skew_ms": self.cross_camera_skew_ms,
            "assignment_state_counts": _state_counts(self.association),
            "global_track_count": len(self.association.tracks),
            "detector_ms": self.detector_ms,
            "reid_ms": self.reid_ms,
            "local_tracker_ms": self.local_tracker_ms,
            "association_ms": self.association_ms,
            "processing_latency_ms": self.processing_latency_ms,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "appearance_unavailable_count": self.appearance_unavailable_count,
            "association_updated": self.association_updated,
            "association_tick_index": self.association_tick_index,
            "publication_due": self.publication_due,
            "pending_association_observations": self.pending_association_observations,
            "appearance_persisted_count": self.appearance_persisted_count,
            "appearance_fresh_count": self.appearance_fresh_count,
            "evidence_flush_due": self.evidence_flush_due,
            "rectification_ms": self.rectification_ms,
            "decision_signature_sha256": self.association.signature_sha256,
        }


@dataclass(frozen=True, slots=True)
class _CachedAppearance:
    camera_id: str
    captured_monotonic_ns: int
    reference_sha256: str
    vector: tuple[float, ...]


class XR02LivePipeline:
    """Single GPU/association actor; caller guarantees zero queued work."""

    def __init__(
        self,
        scene: AdoptedSceneSelection,
        calibrations: dict[str, CameraProjectionCalibration],
        floor: FloorRectangle,
        model: LiveModelProfile,
        evidence_root: Path,
        cadence: HigherCadenceProfile | None = None,
    ) -> None:
        import torch
        from boxmot import Detector, ReIDModel  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            raise XR02LiveContractError("WP4 requires CUDA; CPU fallback is forbidden")
        self.scene = scene
        self.model = model
        self.cadence = cadence or HigherCadenceProfile()
        self._cadence = DeterministicMultiRateCadence(self.cadence)
        self._torch: Any = torch
        self._detector: Any = Detector(
            model.detector_path,
            device="cuda:0",
            image_size=512,
            confidence=model.detector_confidence,
            classes=[0],
            half=True,
            batch=4,
        )
        self._reid: Any = ReIDModel(model.reid_path, device="cuda:0", half=True)
        profile = live_cadence_profile(
            fixed_camera_profiles()[0],
            local_tracking_hz=self.cadence.local_tracking_hz,
            track_buffer_frames=self.cadence.local_track_buffer_frames,
            new_track_confidence=model.new_track_confidence,
            minimum_confirmation_hits=model.local_confirmation_hits,
        )
        self._profile = profile
        self._trackers = {
            camera_id: CameraLocalTracker(camera_id, profile) for camera_id in calibrations
        }
        self._rectifiers = {
            camera_id: FusedLiveFrameRectifier(calibration)
            for camera_id, calibration in calibrations.items()
        }
        embedding_store = EmbeddingStore(evidence_root, "osnet-x0.25-msmt17")
        self._embedding_store = embedding_store
        self._projection = P08ProjectionAdapter(calibrations, floor)
        self._appearance_quality_policy = CropQualityPolicy()
        self._assemblers = {
            camera_id: LocalObservationAssembler(
                _capture_epoch_profile(profile.profile_id, 0),
                self._projection,
                embedding_store,
                quality_policy=self._appearance_quality_policy,
                cadence=EmbeddingCadence(self.cadence.appearance_every_n_local_frames),
            )
            for camera_id in calibrations
        }
        self._capture_generations = {camera_id: 0 for camera_id in calibrations}
        self._tracker_epochs = {camera_id: 0 for camera_id in calibrations}
        self._local_journal = ObservationJournal(evidence_root / "wp4-local-observations.jsonl")
        self._global_journal = GlobalAssociationJournal(
            evidence_root / "wp4-global-association.jsonl"
        )
        self._associator = SceneGlobalAssociator(
            scene.scene.context_sha256,
            office_topology(),
            SignalProfile.COMBINED,
            AssociationConfig.continuity_live(),
        )
        self._last_frame_ids: dict[str, str] = {}
        self._processed_sequences = {camera_id: 0 for camera_id in calibrations}
        self._tick_index = 0
        self._association_tick_index = 0
        self._latest_association: AssociationTickResult | None = None
        self._pending_association_observations: dict[str, AssociationObservation] = {}
        self._pending_local_journal_observations: list[LocalTrackObservation] = []
        self._pending_global_journal_results: list[AssociationTickResult] = []
        self._appearance_cache: dict[str, _CachedAppearance] = {}
        self._warmed = False

    @property
    def profile_identity(self) -> dict[str, object]:
        return {
            "model_profile_id": self.model.profile_id,
            "detector_confidence": self.model.detector_confidence,
            "new_track_confidence": self.model.new_track_confidence,
            "local_confirmation_hits": self.model.local_confirmation_hits,
            "local_tracker_profile": self._profile.profile_id,
            "local_tracker_kwargs": self._profile.tracker_kwargs,
            "global_profile": self._associator.profile_id,
            "detector_sha256": self.model.detector_sha256,
            "reid_sha256": self.model.reid_sha256,
            "cpu_fallback_allowed": False,
            "cadence": self.cadence.as_dict(),
            "appearance_for_local_tracker": "fresh embedding every admitted local frame",
            "appearance_for_global_association": (
                "fresh quality-gated in-memory embedding every admitted local frame"
            ),
            "appearance_persistence": "bounded 2 Hz durable gallery/evidence micro-batches",
            "rectification_profile": FusedLiveFrameRectifier.profile_id,
            "capture_generations": dict(sorted(self._capture_generations.items())),
            "tracker_epochs": dict(sorted(self._tracker_epochs.items())),
        }

    def begin_camera_epoch(
        self,
        camera_id: str,
        capture_generation: int,
        tracker_epoch: int,
        reason: str,
    ) -> None:
        """Reset one local tracker after a supervised hard capture discontinuity."""

        if camera_id not in self._trackers:
            raise XR02LiveContractError("capture epoch camera is absent from the live scene")
        if capture_generation <= self._capture_generations[camera_id]:
            return
        if tracker_epoch <= self._tracker_epochs[camera_id] or not reason:
            raise XR02LiveContractError("capture epoch must increase with an explicit reason")
        self._trackers[camera_id] = CameraLocalTracker(camera_id, self._profile)
        self._assemblers[camera_id] = LocalObservationAssembler(
            _capture_epoch_profile(self._profile.profile_id, tracker_epoch),
            self._projection,
            self._embedding_store,
            quality_policy=self._appearance_quality_policy,
            cadence=EmbeddingCadence(self.cadence.appearance_every_n_local_frames),
        )
        self._processed_sequences[camera_id] = 0
        self._last_frame_ids.pop(camera_id, None)
        self._capture_generations[camera_id] = capture_generation
        self._tracker_epochs[camera_id] = tracker_epoch
        self._pending_association_observations = {
            key: value
            for key, value in self._pending_association_observations.items()
            if value.camera_id != camera_id
        }
        self._appearance_cache = {
            key: value
            for key, value in self._appearance_cache.items()
            if value.camera_id != camera_id
        }

    def warmup(self) -> None:
        if self._warmed:
            return
        blank: NDArray[np.uint8] = np.zeros((280, 504, 3), dtype=np.uint8)
        self._detector.predict([blank, blank, blank, blank])
        self._torch.cuda.synchronize()
        self._torch.cuda.empty_cache()
        self._torch.cuda.reset_peak_memory_stats()
        self._warmed = True

    def process(self, snapshot: LatestFrameSnapshot) -> LiveAssociationTick:
        if not self._warmed:
            self.warmup()
        started = time.perf_counter()
        cadence = self._cadence.decision(self._tick_index)
        fresh = [
            frame
            for frame in snapshot.frames
            if self._last_frame_ids.get(frame.identity.camera_id) != frame.identity.frame_id
        ]
        before_rectification = time.perf_counter()
        calibrated = [
            self._rectifiers[frame.identity.camera_id].rectify(frame.frame_bgr) for frame in fresh
        ]
        rectification_ms = (time.perf_counter() - before_rectification) * 1000.0
        detector_ms = 0.0
        outputs: list[Any] = []
        if calibrated:
            before = time.perf_counter()
            raw = self._detector.predict(calibrated)
            self._torch.cuda.synchronize()
            detector_ms = (time.perf_counter() - before) * 1000.0
            outputs = raw if isinstance(raw, list) else [raw]
            if len(outputs) != len(fresh):
                raise XR02LiveContractError("detector output count changed")

        camera_results: list[CalibratedCameraResult] = []
        total_reid_ms = 0.0
        total_tracker_ms = 0.0
        appearance_unavailable = 0
        appearance_persisted = 0
        appearance_fresh = 0
        local_observations: list[LocalTrackObservation] = []
        for captured, image, output in zip(fresh, calibrated, outputs, strict=True):
            camera_id = captured.identity.camera_id
            detections = person_detections_from_boxmot(output.dets)
            embeddings: NDArray[np.float32] | None = None
            if detections.count:
                before = time.perf_counter()
                embeddings = np.asarray(
                    self._reid.get_features(detections.xyxy, image), dtype=np.float32
                )
                self._torch.cuda.synchronize()
                total_reid_ms += (time.perf_counter() - before) * 1000.0
                if embeddings.shape[0] != detections.count:
                    raise XR02LiveContractError("ReID output count changed")
            sequence = self._processed_sequences[camera_id]
            before = time.perf_counter()
            tracks = self._trackers[camera_id].update(sequence, image, detections, embeddings)
            self._torch.cuda.synchronize()
            total_tracker_ms += (time.perf_counter() - before) * 1000.0
            frame_key = FrameKey(
                scene=self.scene.scene,
                camera_id=camera_id,
                frame_id=captured.identity.frame_id,
                frame_sequence=sequence,
                acquisition_monotonic_ns=captured.identity.acquisition_monotonic_ns,
                observed_at_utc=captured.identity.observed_at_utc,
                width_pixels=image.shape[1],
                height_pixels=image.shape[0],
            )
            observations = self._assemblers[camera_id].assemble(
                frame_key,
                detections,
                tracks,
                embeddings,
                snapshot.snapshot_monotonic_ns,
            )
            for observation in observations:
                local_observations.append(observation)
                if observation.embedding is not None:
                    appearance_persisted += 1
                fresh_embedding = self._fresh_embedding_evidence(observation, embeddings)
                if fresh_embedding is not None:
                    appearance_fresh += 1
                converted = self._to_global_observation(observation, fresh_embedding)
                if converted is not None:
                    self._pending_association_observations[converted.local_track_stable_id] = (
                        converted
                    )
                    if converted.embedding is None:
                        appearance_unavailable += 1
                elif observation.projection_status is WorldProjectionStatus.VALID:
                    appearance_unavailable += 1
            camera_results.append(
                CalibratedCameraResult(
                    camera_id,
                    captured.identity.frame_id,
                    captured.identity.acquisition_monotonic_ns,
                    image,
                    detections,
                    observations,
                )
            )
            self._last_frame_ids[camera_id] = captured.identity.frame_id
            self._processed_sequences[camera_id] += 1

        self._pending_local_journal_observations.extend(local_observations)

        association_updated = cadence.association_due
        association_ms = 0.0
        if cadence.association_due:
            before = time.perf_counter()
            pending = tuple(
                self._pending_association_observations[key]
                for key in sorted(self._pending_association_observations)
            )
            association = self._associator.process_tick(
                self._association_tick_index,
                snapshot.snapshot_monotonic_ns,
                pending,
            )
            association_ms = (time.perf_counter() - before) * 1000.0
            self._pending_global_journal_results.append(association)
            self._pending_association_observations.clear()
            self._latest_association = association
            association_tick_index: int | None = self._association_tick_index
            self._association_tick_index += 1
        else:
            latest_association = self._latest_association
            if latest_association is None:
                raise XR02LiveContractError("first local tick did not initialize association")
            association = latest_association
            association_tick_index = association.tick_index
        if cadence.appearance_persistence_due:
            self._flush_evidence()
        completed_ns = time.monotonic_ns()
        result = LiveAssociationTick(
            tick_index=self._tick_index,
            admitted_monotonic_ns=snapshot.snapshot_monotonic_ns,
            completed_monotonic_ns=completed_ns,
            camera_results=tuple(camera_results),
            stale_camera_ids=snapshot.stale_camera_ids,
            missing_camera_ids=snapshot.missing_camera_ids,
            cross_camera_skew_ms=snapshot.cross_camera_skew_ms,
            association=association,
            detector_ms=detector_ms,
            reid_ms=total_reid_ms,
            local_tracker_ms=total_tracker_ms,
            association_ms=association_ms,
            processing_latency_ms=(time.perf_counter() - started) * 1000.0,
            peak_cuda_allocated_bytes=int(self._torch.cuda.max_memory_allocated()),
            peak_cuda_reserved_bytes=int(self._torch.cuda.max_memory_reserved()),
            appearance_unavailable_count=appearance_unavailable,
            association_updated=association_updated,
            association_tick_index=association_tick_index,
            publication_due=cadence.publication_due,
            pending_association_observations=len(self._pending_association_observations),
            appearance_persisted_count=appearance_persisted,
            appearance_fresh_count=appearance_fresh,
            evidence_flush_due=cadence.appearance_persistence_due,
            rectification_ms=rectification_ms,
        )
        self._tick_index += 1
        return result

    def _to_global_observation(
        self,
        observation: LocalTrackObservation,
        fresh_embedding: tuple[str, tuple[float, ...]] | None,
    ) -> AssociationObservation | None:
        if (
            observation.projection_status is not WorldProjectionStatus.VALID
            or observation.world_xy_metres is None
        ):
            return None
        embedding: tuple[float, ...] | None = None
        embedding_sha: str | None = None
        if fresh_embedding is not None:
            transient_sha, embedding = fresh_embedding
            embedding_sha = (
                observation.embedding.sha256
                if observation.embedding is not None
                else transient_sha
            )
            self._appearance_cache[observation.track.stable_id] = _CachedAppearance(
                camera_id=observation.frame.camera_id,
                captured_monotonic_ns=observation.frame.acquisition_monotonic_ns,
                reference_sha256=embedding_sha,
                vector=embedding,
            )
        elif observation.embedding is not None:
            embedding_sha = observation.embedding.sha256
            embedding = tuple(
                float(value) for value in self._embedding_store.load(observation.embedding)
            )
            self._appearance_cache[observation.track.stable_id] = _CachedAppearance(
                camera_id=observation.frame.camera_id,
                captured_monotonic_ns=observation.frame.acquisition_monotonic_ns,
                reference_sha256=embedding_sha,
                vector=embedding,
            )
        else:
            cached = self._appearance_cache.get(observation.track.stable_id)
            if cached is not None:
                age_seconds = (
                    observation.frame.acquisition_monotonic_ns - cached.captured_monotonic_ns
                ) / 1_000_000_000.0
                if 0.0 <= age_seconds <= self.cadence.cached_embedding_max_age_seconds:
                    embedding_sha = cached.reference_sha256
                    embedding = cached.vector
                elif age_seconds > self.cadence.cached_embedding_max_age_seconds:
                    self._appearance_cache.pop(observation.track.stable_id, None)
        quality = max(
            1e-6,
            min(1.0, observation.confidence * observation.crop_quality.visible_fraction),
        )
        return AssociationObservation(
            scene_context_sha256=self.scene.scene.context_sha256,
            observation_id=f"{observation.frame.frame_id}.d{observation.detection_index}",
            local_track_stable_id=observation.track.stable_id,
            camera_id=observation.frame.camera_id,
            tracker_profile=observation.track.tracker_profile,
            observed_monotonic_ns=observation.frame.acquisition_monotonic_ns,
            world_xy_metres=observation.world_xy_metres,
            confidence=observation.confidence,
            quality_weight=quality,
            embedding_reference_sha256=embedding_sha,
            embedding=embedding,
            bbox_xyxy=observation.bbox_xyxy,
        )

    def _fresh_embedding_evidence(
        self,
        observation: LocalTrackObservation,
        embeddings: NDArray[np.float32] | None,
    ) -> tuple[str, tuple[float, ...]] | None:
        """Return a qualified in-memory appearance vector without forcing disk I/O."""

        policy = self._appearance_quality_policy
        quality = observation.crop_quality
        if (
            embeddings is None
            or quality.area_pixels < policy.minimum_area_pixels
            or quality.visible_fraction < policy.minimum_visible_fraction
        ):
            return None
        values = np.asarray(embeddings[observation.detection_index], dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            return None
        norm = float(np.linalg.norm(values))
        if norm <= 0:
            return None
        normalized = np.ascontiguousarray(values / norm, dtype=np.float32)
        digest = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
        return digest, tuple(float(value) for value in normalized)

    def _flush_evidence(self) -> None:
        if self._pending_local_journal_observations:
            self._local_journal.append_batch(tuple(self._pending_local_journal_observations))
            self._pending_local_journal_observations.clear()
        if self._pending_global_journal_results:
            self._global_journal.append_batch(tuple(self._pending_global_journal_results))
            self._pending_global_journal_results.clear()

    def finalize_evidence(self) -> None:
        """Flush the final bounded partial micro-batch after the GPU actor stops."""

        self._flush_evidence()


def _state_counts(result: AssociationTickResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for assignment in result.assignments:
        counts[assignment.state.value] = counts.get(assignment.state.value, 0) + 1
    return dict(sorted(counts.items()))


def validate_frequency(value: float) -> float:
    if not math.isfinite(value) or not 0.2 <= value <= 20.0:
        raise XR02LiveContractError("association frequency must be within 0.2..20 Hz")
    return value


def _capture_epoch_profile(base_profile: str, tracker_epoch: int) -> str:
    if tracker_epoch < 0:
        raise XR02LiveContractError("tracker epoch must be non-negative")
    return f"{base_profile}.capture-e{tracker_epoch:06d}"
