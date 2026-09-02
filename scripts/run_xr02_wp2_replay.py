"""Run the authorized XR02 WP2 local-tracking replay and evidence gate.

The command is offline, uses only exact local clips and model weights, and compares camera-local
BoT-SORT with fixed-camera Deep-OC-SORT. It does not perform cross-camera association.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_projection import (
    FrozenProjectionInputs,
    LiveFrameRectifier,
    load_frozen_projection_inputs,
)
from spatial_mapping_phase2.xr02_boxmot import CameraLocalTracker, fixed_camera_profiles
from spatial_mapping_phase2.xr02_journal import EmbeddingStore, ObservationJournal, verify_journal
from spatial_mapping_phase2.xr02_local_domain import (
    FrameKey,
    LocalTrackObservation,
    SceneContextKey,
)
from spatial_mapping_phase2.xr02_local_pipeline import (
    LocalObservationAssembler,
    P08ProjectionAdapter,
    build_scene_context,
)
from spatial_mapping_phase2.xr02_supervision import (
    SupervisionAdapter,
    person_detections_from_boxmot,
)
from spatial_mapping_phase2.xr02_wp1 import identify_file, summarize_ms

P06_SHA256 = os.environ.get("XR02_P06_SHA256", "0" * 64)
P07_SHA256 = os.environ.get("XR02_P07_SHA256", "0" * 64)
P08_MANIFEST_SHA256 = os.environ.get("XR02_P08_MANIFEST_SHA256", "0" * 64)
P08_FLOOR_SHA256 = os.environ.get("XR02_P08_FLOOR_SHA256", "0" * 64)
P08_STATIC_RRD_SHA256 = os.environ.get("XR02_P08_STATIC_RRD_SHA256", "0" * 64)
YOLO_SHA256 = os.environ.get("XR02_YOLO_SHA256", "0" * 64)
OSNET_SHA256 = os.environ.get("XR02_OSNET_SHA256", "0" * 64)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    camera_id: str
    camera_sequence: int
    source_frame_index: int
    image_bgr: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    replay: ReplayFrame
    detector_output: Any
    embeddings: NDArray[np.float32] | None


@dataclass(frozen=True, slots=True)
class VisualTick:
    frame: ReplayFrame
    observations: tuple[LocalTrackObservation, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--reid", required=True, type=Path)
    parser.add_argument("--p06", required=True, type=Path)
    parser.add_argument("--p07", required=True, type=Path)
    parser.add_argument("--p08-manifest", required=True, type=Path)
    parser.add_argument("--p08-floor", required=True, type=Path)
    parser.add_argument("--p08-static-rrd", required=True, type=Path)
    parser.add_argument("--clip", required=True, action="append", type=Path)
    parser.add_argument("--sample-step", type=int, default=12)
    parser.add_argument("--max-frames-per-camera", type=int, default=30)
    return parser.parse_args()


def force_offline() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[name] = "http://127.0.0.1:9"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE"):
        os.environ[name] = "1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"


def read_frames(
    clip_paths: Sequence[Path],
    rectifiers: dict[str, LiveFrameRectifier],
    sample_step: int,
    maximum: int,
) -> list[ReplayFrame]:
    import cv2

    if len(clip_paths) != len(rectifiers):
        raise RuntimeError("cached clip count must match the calibrated camera roster")
    frames: list[ReplayFrame] = []
    for camera_id, clip in zip(rectifiers, clip_paths, strict=True):
        capture = cv2.VideoCapture(str(clip))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open cached clip: {clip}")
        source_index = 0
        retained = 0
        while retained < maximum:
            ok, native = capture.read()
            if not ok:
                break
            if source_index % sample_step == 0:
                processed = rectifiers[camera_id].rectify(native)
                frames.append(ReplayFrame(camera_id, retained, source_index, processed))
                retained += 1
            source_index += 1
        capture.release()
        if retained == 0:
            raise RuntimeError(f"cached clip yielded no frames: {clip}")
    return frames


def prepare_frames(
    frames: Sequence[ReplayFrame], detector_path: Path, reid_path: Path
) -> tuple[list[PreparedFrame], dict[str, object]]:
    import torch
    from boxmot import Detector, ReIDModel  # type: ignore[import-not-found]

    detector = Detector(
        detector_path,
        device="cuda:0",
        image_size=512,
        confidence=0.15,
        classes=[0],
        half=True,
        batch=4,
    )
    reid = ReIDModel(reid_path, device="cuda:0", half=True)
    detector.predict([frame.image_bgr for frame in frames[:4]])
    torch.cuda.synchronize()
    prepared: list[PreparedFrame] = []
    detector_ms: list[float] = []
    reid_ms: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(frames), 4):
        batch = frames[start : start + 4]
        begin = time.perf_counter()
        raw = detector.predict([frame.image_bgr for frame in batch])
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000.0
        outputs = raw if isinstance(raw, list) else [raw]
        if len(outputs) != len(batch):
            raise RuntimeError("detector output count changed")
        detector_ms.extend([elapsed / len(batch)] * len(batch))
        for frame, output in zip(batch, outputs, strict=True):
            detections = person_detections_from_boxmot(output.dets)
            embeddings: NDArray[np.float32] | None = None
            if detections.count:
                begin = time.perf_counter()
                embeddings = np.asarray(
                    reid.get_features(detections.xyxy, frame.image_bgr), dtype=np.float32
                )
                torch.cuda.synchronize()
                reid_ms.append((time.perf_counter() - begin) * 1000.0)
                if embeddings.shape[0] != detections.count:
                    raise RuntimeError("ReID output count changed")
            prepared.append(PreparedFrame(frame, output, embeddings))
    telemetry: dict[str, object] = {
        "detector_ms_per_frame": summarize_ms(detector_ms).to_dict(),
        "reid_ms_per_detection_frame": summarize_ms(reid_ms).to_dict(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return prepared, telemetry


def run_profile(
    *,
    prepared: Sequence[PreparedFrame],
    profile_index: int,
    scene_context: SceneContextKey,
    projection: P08ProjectionAdapter,
    output_root: Path,
    persist: bool,
) -> tuple[dict[str, object], list[VisualTick]]:
    import torch

    profile = fixed_camera_profiles()[profile_index]
    camera_ids = tuple(dict.fromkeys(item.replay.camera_id for item in prepared))
    trackers = {camera_id: CameraLocalTracker(camera_id, profile) for camera_id in camera_ids}
    embedding_store = EmbeddingStore(output_root, "osnet-x0.25-msmt17")
    assembler = LocalObservationAssembler(profile.profile_id, projection, embedding_store)
    journal_path = output_root / f"{profile.profile_id}.jsonl"
    journal = ObservationJournal(journal_path) if persist else None
    signatures: list[object] = []
    visual_ticks: list[VisualTick] = []
    timing_ms: list[float] = []
    projection_counts: Counter[str] = Counter()
    embedding_counts: Counter[str] = Counter()
    ids_by_camera: dict[str, set[int]] = defaultdict(set)
    track_rows = 0
    for item in prepared:
        canonical = person_detections_from_boxmot(item.detector_output.dets)
        begin = time.perf_counter()
        tracks = trackers[item.replay.camera_id].update(
            item.replay.camera_sequence,
            item.replay.image_bgr,
            canonical,
            item.embeddings,
        )
        torch.cuda.synchronize()
        timing_ms.append((time.perf_counter() - begin) * 1000.0)
        acquisition_ns = (
            list(camera_ids).index(item.replay.camera_id) + 1
        ) * 10_000_000_000 + item.replay.source_frame_index * 40_000_000
        frame_key = FrameKey(
            scene=scene_context,
            camera_id=item.replay.camera_id,
            frame_id=f"{item.replay.camera_id}-{item.replay.source_frame_index}",
            frame_sequence=item.replay.camera_sequence,
            acquisition_monotonic_ns=acquisition_ns,
            observed_at_utc="2026-08-24T00:00:00Z",
            width_pixels=item.replay.image_bgr.shape[1],
            height_pixels=item.replay.image_bgr.shape[0],
        )
        observations = assembler.assemble(
            frame_key,
            canonical,
            tracks,
            item.embeddings,
            acquisition_ns + 50_000_000,
        )
        signature_rows: list[object] = []
        for observation in observations:
            track_rows += 1
            ids_by_camera[observation.frame.camera_id].add(observation.track.local_track_id)
            projection_counts[observation.projection_status.value] += 1
            embedding_counts[observation.embedding_status.value] += 1
            signature_rows.append(
                [
                    observation.track.local_track_id,
                    observation.detection_index,
                    [round(value, 3) for value in observation.bbox_xyxy],
                    None
                    if observation.world_xy_metres is None
                    else [round(value, 4) for value in observation.world_xy_metres],
                ]
            )
            if journal is not None:
                journal.append(observation)
        signatures.append([item.replay.camera_id, item.replay.camera_sequence, signature_rows])
        if persist:
            visual_ticks.append(VisualTick(item.replay, observations))
    signature_sha256 = hashlib.sha256(
        json.dumps(signatures, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()
    journal_result = None if journal is None else verify_journal(journal_path)
    return (
        {
            "profile_id": profile.profile_id,
            "tracker_type": profile.tracker_type,
            "tracker_kwargs": profile.tracker_kwargs,
            "frames": len(prepared),
            "track_rows": track_rows,
            "unique_local_ids_by_camera": {
                key: len(value) for key, value in ids_by_camera.items()
            },
            "projection_states": dict(sorted(projection_counts.items())),
            "embedding_states": dict(sorted(embedding_counts.items())),
            "update_ms": summarize_ms(timing_ms).to_dict(),
            "processed_fps": 1000.0 / (sum(timing_ms) / len(timing_ms)),
            "signature_sha256": signature_sha256,
            "journal": (
                None
                if journal_result is None
                else {
                    "path": str(journal_path),
                    "records": journal_result.records,
                    "final_sha256": journal_result.final_sha256,
                }
            ),
        },
        visual_ticks,
    )


def write_rerun(
    path: Path,
    visual_ticks: Sequence[VisualTick],
    floor_bounds: tuple[tuple[float, float], tuple[float, float]],
    static_rrd: Path,
) -> None:
    import rerun as rr

    recording: Any = rr.new_recording("xr02-wp2-camera-local-diagnostics", recording_id=path.stem)
    recording.save(str(path))
    (x0, y0), (x1, y1) = floor_bounds
    recording.log(
        "xr02/world/p08_floor_z0",
        rr.LineStrips3D(
            [[[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]]],
            colors=[[210, 210, 210]],
            radii=0.02,
        ),
        static=True,
    )
    recording.log(
        "xr02/status/static_scene",
        rr.TextDocument(
            "Camera-local WP2 diagnostics. The immutable accepted P08 static-scene RRD is "
            f"preserved separately at `{static_rrd}`; this artifact does not duplicate it.",
            media_type="text/markdown",
        ),
        static=True,
    )
    for tick_index, tick in enumerate(visual_ticks):
        recording.set_time_sequence("tick", tick_index)
        root = f"xr02/cameras/{tick.frame.camera_id}"
        recording.log(root, rr.Image(tick.frame.image_bgr[..., ::-1]))
        if tick.observations:
            recording.log(
                f"{root}/local_tracks",
                rr.Boxes2D(
                    array=[item.bbox_xyxy for item in tick.observations],
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[item.track.stable_id for item in tick.observations],
                    colors=[[255, 196, 64]] * len(tick.observations),
                ),
            )
            recording.log(
                f"{root}/footpoints",
                rr.Points2D(
                    [item.footpoint_uv for item in tick.observations],
                    colors=[[255, 70, 70]] * len(tick.observations),
                    radii=5.0,
                ),
            )
        world = [item for item in tick.observations if item.world_xy_metres is not None]
        world_path = f"xr02/world/local_candidates/{tick.frame.camera_id}"
        if world:
            recording.log(
                world_path,
                rr.Points3D(
                    [_world_point(item) for item in world],
                    labels=[item.track.stable_id for item in world],
                    colors=[[72, 190, 255]] * len(world),
                    radii=0.10,
                ),
            )
    recording.flush(blocking=True)
    recording.disconnect()


def _world_point(observation: LocalTrackObservation) -> list[float]:
    xy = observation.world_xy_metres
    if xy is None:
        raise RuntimeError("Rerun world point requires a valid floor projection")
    return [xy[0], xy[1], 0.06]


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str] = {}
    for name in ("boxmot", "rerun-sdk", "supervision", "torch", "ultralytics"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def main() -> int:
    args = parse_args()
    force_offline()
    if args.output.exists():
        raise RuntimeError("output directory must not already exist")
    args.output.mkdir(parents=True)
    if len(args.clip) != 4:
        raise RuntimeError("office WP2 evidence requires exactly four cached clips")
    assets = {
        "detector": identify_file(args.detector, expected_sha256=YOLO_SHA256).to_dict(),
        "reid": identify_file(args.reid, expected_sha256=OSNET_SHA256).to_dict(),
        "p06": identify_file(args.p06, expected_sha256=P06_SHA256).to_dict(),
        "p07": identify_file(args.p07, expected_sha256=P07_SHA256).to_dict(),
        "p08_manifest": identify_file(
            args.p08_manifest, expected_sha256=P08_MANIFEST_SHA256
        ).to_dict(),
        "p08_floor": identify_file(args.p08_floor, expected_sha256=P08_FLOOR_SHA256).to_dict(),
        "p08_static_rrd": identify_file(
            args.p08_static_rrd, expected_sha256=P08_STATIC_RRD_SHA256
        ).to_dict(),
        "clips": [identify_file(path).to_dict() for path in args.clip],
    }
    frozen = FrozenProjectionInputs(
        args.p06,
        P06_SHA256,
        args.p07,
        P07_SHA256,
        args.p08_manifest,
        P08_MANIFEST_SHA256,
        args.p08_floor,
        P08_FLOOR_SHA256,
    )
    calibrations, floor = load_frozen_projection_inputs(frozen)
    context = build_scene_context(
        "office",
        "office-cached-replay-wp2-v1",
        P08_STATIC_RRD_SHA256,
        P08_FLOOR_SHA256,
        {"p06": P06_SHA256, "p07": P07_SHA256},
    )
    rectifiers = {
        camera_id: LiveFrameRectifier(calibration)
        for camera_id, calibration in calibrations.items()
    }
    frames = read_frames(args.clip, rectifiers, args.sample_step, args.max_frames_per_camera)
    prepared, model_telemetry = prepare_frames(frames, args.detector, args.reid)
    adapter = SupervisionAdapter()
    supervision_counts = [
        len(adapter.to_supervision(person_detections_from_boxmot(item.detector_output.dets)))
        for item in prepared
    ]
    projection = P08ProjectionAdapter(calibrations, floor)
    profile_results: list[dict[str, object]] = []
    bot_visual: list[VisualTick] = []
    for profile_index in (0, 1):
        first, visual = run_profile(
            prepared=prepared,
            profile_index=profile_index,
            scene_context=context,
            projection=projection,
            output_root=args.output,
            persist=True,
        )
        second, _ = run_profile(
            prepared=prepared,
            profile_index=profile_index,
            scene_context=context,
            projection=projection,
            output_root=args.output / "determinism-second-pass",
            persist=False,
        )
        first["deterministic_second_pass"] = (
            first["signature_sha256"] == second["signature_sha256"]
        )
        profile_results.append(first)
        if profile_index == 0:
            bot_visual = visual
    rrd_path = args.output / "xr02-wp2-camera-local-diagnostics.rrd"
    write_rerun(
        rrd_path,
        bot_visual,
        (floor.minimum_xy_metres, floor.maximum_xy_metres),
        args.p08_static_rrd,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "XR02 WP2 camera-local tracking only; no global association",
        "command_argv": sys.argv,
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "scene_context": {**context.as_dict(), "context_sha256": context.context_sha256},
        "assets": assets,
        "sampling": {
            "sample_step": args.sample_step,
            "max_frames_per_camera": args.max_frames_per_camera,
            "frames": len(frames),
        },
        "models": model_telemetry,
        "supervision_gate": {
            "version": adapter.version,
            "frames_converted": len(supervision_counts),
            "detections_converted": sum(supervision_counts),
        },
        "profiles": profile_results,
        "rerun": identify_file(rrd_path).to_dict(),
    }
    manifest = args.output / "wp2-replay-manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "frames": len(frames)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
