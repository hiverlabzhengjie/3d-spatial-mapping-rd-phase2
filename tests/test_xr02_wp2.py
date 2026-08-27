from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_projection import (
    CameraProjectionCalibration,
    FloorRectangle,
    LiveFrameRectifier,
)
from spatial_mapping_phase2.xr02_boxmot import (
    BoxMotProfile,
    CameraLocalTracker,
    XR02BoxMotError,
    fixed_camera_profiles,
    live_cadence_profile,
)
from spatial_mapping_phase2.xr02_capture import SceneFrame, SceneLatestFrameBuffer
from spatial_mapping_phase2.xr02_journal import (
    EmbeddingStore,
    ObservationJournal,
    XR02JournalError,
    verify_journal,
)
from spatial_mapping_phase2.xr02_live_pipeline import XR02LivePipeline
from spatial_mapping_phase2.xr02_local_domain import (
    CameraAvailability,
    CropQuality,
    EmbeddingStatus,
    FootpointSource,
    FrameKey,
    LocalTrackKey,
    LocalTrackObservation,
    SceneContextKey,
    WorldProjectionStatus,
    XR02ContractError,
)
from spatial_mapping_phase2.xr02_local_pipeline import (
    CropQualityPolicy,
    EmbeddingCadence,
    LocalObservationAssembler,
    P08ProjectionAdapter,
    build_scene_context,
    evaluate_crop_quality,
)
from spatial_mapping_phase2.xr02_rectification import FusedLiveFrameRectifier
from spatial_mapping_phase2.xr02_supervision import (
    CanonicalDetections,
    SupervisionAdapter,
    XR02SupervisionError,
    bottom_center_points,
    person_detections_from_boxmot,
)


def _scene() -> SceneContextKey:
    return SceneContextKey("office", "office-epoch-001", "a" * 64, "b" * 64, "c" * 64)


def _frame(sequence: int = 0, acquisition_ns: int = 1_000_000_000) -> FrameKey:
    return FrameKey(
        scene=_scene(),
        camera_id="office-cam-01",
        frame_id=f"frame-{sequence}",
        frame_sequence=sequence,
        acquisition_monotonic_ns=acquisition_ns,
        observed_at_utc="2026-08-24T00:00:00Z",
        width_pixels=504,
        height_pixels=280,
    )


def _detections() -> CanonicalDetections:
    return CanonicalDetections(
        xyxy=np.asarray([[200.0, 50.0, 300.0, 200.0]], dtype=np.float32),
        confidence=np.asarray([0.9], dtype=np.float32),
        class_id=np.asarray([0], dtype=np.int32),
    )


def _calibration() -> CameraProjectionCalibration:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [2.0, 3.0, 4.0]
    return CameraProjectionCalibration(
        camera_id="office-cam-01",
        K_native=np.asarray([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
        simple_radial_k1=-0.2,
        K_processed=np.asarray([[100.0, 0.0, 252.0], [0.0, 100.0, 140.0], [0.0, 0.0, 1.0]]),
        T_world_from_camera=transform,
    )


def _observation(embedding: object | None = None) -> LocalTrackObservation:
    return LocalTrackObservation(
        frame=_frame(),
        track=LocalTrackKey(_scene().context_sha256, "office-cam-01", "botsort-v1", 7),
        detection_index=0,
        confidence=0.9,
        bbox_xyxy=(200.0, 50.0, 300.0, 200.0),
        footpoint_uv=(250.0, 200.0),
        footpoint_source=FootpointSource.BBOX_BOTTOM_CENTER,
        crop_quality=CropQuality(1.0, 15_000.0, 2 / 3),
        embedding_status=EmbeddingStatus.NOT_DUE,
        embedding=None,
        projection_status=WorldProjectionStatus.VALID,
        world_xy_metres=(2.0, 3.0),
        projection_reason="test projection",
    )


def test_scene_context_and_local_track_namespace_are_deterministic() -> None:
    scene = _scene()
    assert scene.context_sha256 == _scene().context_sha256
    key = LocalTrackKey(scene.context_sha256, "office-cam-01", "botsort-v1", 7)
    assert key.stable_id.endswith(":office-cam-01:botsort-v1:7")
    with pytest.raises(XR02ContractError, match="lowercase SHA-256"):
        SceneContextKey("office", "epoch", "A" * 64, "b" * 64, "c" * 64)


def test_live_tracker_profile_binds_frame_settings_to_measured_cadence() -> None:
    source = fixed_camera_profiles()[0]
    derived = live_cadence_profile(
        source,
        local_tracking_hz=10.0,
        track_buffer_frames=15,
    )
    assert source.tracker_kwargs["frame_rate"] == 25
    assert source.tracker_kwargs["track_buffer"] == 30
    assert derived.tracker_kwargs["frame_rate"] == 10
    assert derived.tracker_kwargs["track_buffer"] == 15


def test_fresh_global_appearance_does_not_require_durable_embedding_write() -> None:
    pipeline = object.__new__(XR02LivePipeline)
    pipeline._appearance_quality_policy = CropQualityPolicy()  # noqa: SLF001
    observation = _observation()
    embeddings: NDArray[np.float32] = np.asarray([[3.0, 4.0]], dtype=np.float32)
    evidence = pipeline._fresh_embedding_evidence(observation, embeddings)  # noqa: SLF001
    assert evidence is not None
    digest, vector = evidence
    assert len(digest) == 64
    np.testing.assert_allclose(vector, [0.6, 0.8])


def test_fused_rectification_preserves_processed_pinhole_without_native_intermediate() -> None:
    calibration = CameraProjectionCalibration(
        camera_id="office-cam-01",
        K_native=np.asarray([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
        simple_radial_k1=-0.2,
        K_processed=np.asarray([[262.5, 0.0, 252.0], [0.0, 259.259259, 140.0], [0.0, 0.0, 1.0]]),
        T_world_from_camera=np.eye(4, dtype=np.float64),
    )
    y, x = np.indices((1080, 1920))
    native: NDArray[np.uint8] = np.stack(
        (x / 1919 * 255, y / 1079 * 255, (x + y) / 2998 * 255), axis=-1
    ).astype(np.uint8)
    reference = LiveFrameRectifier(calibration).rectify(native)
    fused = FusedLiveFrameRectifier(calibration).rectify(native)
    assert fused.shape == (280, 504, 3)
    assert fused.flags.c_contiguous
    assert np.max(np.abs(reference.astype(np.int16) - fused.astype(np.int16))) <= 1


def test_scene_context_binds_sorted_authority_identities() -> None:
    first = build_scene_context("office", "epoch", "a" * 64, "b" * 64, {"p07": "2", "p06": "1"})
    second = build_scene_context("office", "epoch", "a" * 64, "b" * 64, {"p06": "1", "p07": "2"})
    assert first == second


def test_latest_frame_buffer_scales_roster_and_reports_current_stale_missing() -> None:
    buffer = SceneLatestFrameBuffer(_scene(), ("office-cam-01", "office-cam-02", "office-cam-03"))
    frame = SceneFrame(_frame(), np.zeros((280, 504, 3), dtype=np.uint8))
    buffer.publish(frame)
    current = buffer.snapshot(1_050_000_000, maximum_age_ms=100.0)
    assert [state.availability for state in current.camera_states] == [
        CameraAvailability.CURRENT,
        CameraAvailability.MISSING,
        CameraAvailability.MISSING,
    ]
    stale = buffer.snapshot(1_200_000_000, maximum_age_ms=100.0)
    assert stale.camera_states[0].availability is CameraAvailability.STALE
    replacement = SceneFrame(_frame(1, 1_300_000_000), np.zeros((280, 504, 3), dtype=np.uint8))
    buffer.publish(replacement)
    assert buffer.snapshot(1_300_000_000, 100.0).camera_states[0].replaced_frames == 1


def test_canonical_detection_and_supervision_boundary_are_replaceable() -> None:
    raw = np.asarray([[1, 2, 5, 8, 0.8, 0], [1, 1, 2, 2, 0.7, 2]], dtype=np.float32)
    canonical = person_detections_from_boxmot(raw)
    assert canonical.count == 1
    np.testing.assert_allclose(bottom_center_points(canonical), [[3.0, 8.0]])

    module = ModuleType("fake_supervision")
    captured: dict[str, object] = {}

    class FakeDetections:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module.Detections = FakeDetections  # type: ignore[attr-defined]
    module.__version__ = "test"  # type: ignore[attr-defined]
    adapter = SupervisionAdapter(module)
    adapter.to_supervision(canonical)
    assert adapter.version == "test"
    assert "xyxy" in captured
    with pytest.raises(XR02SupervisionError, match="positive area"):
        CanonicalDetections(
            np.asarray([[1, 1, 1, 2]], dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
        )


class _FakeTrackResults:
    xyxy = np.asarray([[200.0, 50.0, 300.0, 200.0]], dtype=np.float32)
    id = np.asarray([7], dtype=np.int64)
    conf = np.asarray([0.9], dtype=np.float32)
    cls = np.asarray([0], dtype=np.int32)
    det_ind = np.asarray([0], dtype=np.int64)


class _FakeTracker:
    def update(
        self,
        dets: NDArray[np.float32],
        img: NDArray[np.uint8],
        embs: NDArray[np.float32] | None = None,
    ) -> object:
        assert dets.shape == (1, 6)
        assert img.shape == (280, 504, 3)
        assert embs is not None and embs.shape == (1, 4)
        return _FakeTrackResults()


def test_boxmot_profiles_disable_cmc_and_adapter_enforces_order() -> None:
    botsort, deepocsort = fixed_camera_profiles()
    assert botsort.tracker_kwargs["use_cmc"] is False
    assert deepocsort.tracker_kwargs["cmc_off"] is True
    with pytest.raises(XR02BoxMotError, match="disable CMC"):
        BoxMotProfile("deep", "deepocsort", {"cmc_off": False})
    tracker = CameraLocalTracker("office-cam-01", botsort, lambda profile: _FakeTracker())
    image: NDArray[np.uint8] = np.zeros((280, 504, 3), dtype=np.uint8)
    embeddings: NDArray[np.float32] = np.ones((1, 4), dtype=np.float32)
    tracks = tracker.update(0, image, _detections(), embeddings)
    assert tracks.local_track_ids.tolist() == [7]
    with pytest.raises(XR02BoxMotError, match="increase strictly"):
        tracker.update(0, image, _detections(), embeddings)


def test_content_addressed_embeddings_deduplicate_and_verify(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path, "osnet-x0.25-msmt17")
    first = store.put(np.asarray([3.0, 4.0], dtype=np.float32))
    second = store.put(np.asarray([3.0, 4.0], dtype=np.float32))
    assert first == second
    np.testing.assert_allclose(store.load(first), [0.6, 0.8])
    path = tmp_path / first.relative_path
    path.write_bytes(b"changed")
    with pytest.raises(XR02JournalError, match="identity changed"):
        store.load(first)


def test_journal_hash_chain_detects_content_and_order_changes(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    writer = ObservationJournal(path)
    writer.append(_observation())
    writer.append(_observation())
    result = verify_journal(path)
    assert result.records == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["payload"]["confidence"] = 0.1
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(XR02JournalError, match="content changed"):
        verify_journal(path)


def test_observation_journal_batch_preserves_per_record_chain(tmp_path: Path) -> None:
    path = tmp_path / "batched-observations.jsonl"
    writer = ObservationJournal(path)
    first = _observation()
    second = replace(
        first,
        frame=_frame(sequence=1, acquisition_ns=1_100_000_000),
    )
    digests = writer.append_batch((first, second))
    assert len(digests) == 2
    verified = verify_journal(path)
    assert verified.records == 2
    assert verified.final_sha256 == digests[-1]


def test_projection_and_observation_assembly_preserve_p08_math(tmp_path: Path) -> None:
    projection = P08ProjectionAdapter(
        {"office-cam-01": _calibration()}, FloorRectangle((0.0, -10.0), (10.0, 10.0))
    )
    tracks = _FakeTrackResults()
    from spatial_mapping_phase2.xr02_boxmot import LocalTrackRows

    rows = LocalTrackRows(tracks.xyxy, tracks.id, tracks.conf, tracks.cls, tracks.det_ind)
    assembler = LocalObservationAssembler(
        "botsort-fixed-v1",
        projection,
        EmbeddingStore(tmp_path, "osnet-x0.25-msmt17"),
    )
    observations = assembler.assemble(
        _frame(),
        _detections(),
        rows,
        np.asarray([[3.0, 4.0]], dtype=np.float32),
        1_100_000_000,
    )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.embedding_status is EmbeddingStatus.AVAILABLE
    assert observation.projection_status is WorldProjectionStatus.VALID
    assert observation.world_xy_metres is not None
    assert observation.world_xy_metres[0] == pytest.approx(1.92, abs=0.01)


def test_embedding_cadence_and_crop_quality_fail_closed(tmp_path: Path) -> None:
    quality = evaluate_crop_quality((-10.0, 10.0, 50.0, 100.0), 504, 280, 1.0)
    assert quality.visible_fraction < 1.0
    assert "left" in quality.clipped_sides
    assert not EmbeddingCadence(2).is_due(1)
    with pytest.raises(XR02ContractError, match="positive"):
        EmbeddingCadence(0)
