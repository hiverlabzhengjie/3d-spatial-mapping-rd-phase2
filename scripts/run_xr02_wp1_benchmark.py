"""Run the D062-authorized XR02 WP1 cached-replay feasibility benchmark.

This script must run inside the isolated BoxMOT worker runtime. It never downloads an asset: all
inputs are local, hash-checked before vendor code is imported, and network endpoints are replaced
with a deliberately unreachable local proxy for the process lifetime.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.xr02_wp1 import batches, identify_file, summarize_ms

YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
OSNET_X025_SHA256 = "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99"
OSNET_X10_SHA256 = "b7d73dc67c016fd044e4027ff856019496392a7aca8fa0ed56d862a1632c1cf2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--reid-primary", required=True, type=Path)
    parser.add_argument("--reid-challenger", required=True, type=Path)
    parser.add_argument("--clip", required=True, action="append", type=Path)
    parser.add_argument("--sample-step", type=int, default=12)
    parser.add_argument("--max-frames-per-camera", type=int, default=16)
    parser.add_argument("--stability-passes", type=int, default=3)
    return parser.parse_args()


def force_offline() -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[name] = "http://127.0.0.1:9"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE"):
        os.environ[name] = "1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"


def read_sampled_frames(
    clip_paths: Sequence[Path], sample_step: int, max_frames_per_camera: int
) -> tuple[list[Any], list[dict[str, Any]]]:
    import cv2  # type: ignore[import-untyped]

    if sample_step <= 0 or max_frames_per_camera <= 0:
        raise ValueError("Sampling controls must be positive")
    frames: list[Any] = []
    records: list[dict[str, Any]] = []
    for camera_index, clip_path in enumerate(clip_paths, start=1):
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open cached replay: {clip_path}")
        source_index = 0
        retained = 0
        while retained < max_frames_per_camera:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % sample_step == 0:
                frames.append(frame)
                records.append(
                    {
                        "camera_index": camera_index,
                        "source_frame_index": source_index,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
                retained += 1
            source_index += 1
        capture.release()
        if retained == 0:
            raise RuntimeError(f"Cached replay produced no sampled frames: {clip_path}")
    return frames, records


def normalize_detection_batch(result: Any) -> list[Any]:
    return result if isinstance(result, list) else [result]


def run_detector_case(
    *,
    frames: Sequence[Any],
    detector_path: Path,
    image_size: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[Any]]:
    import psutil  # type: ignore[import-untyped]
    import torch
    from boxmot import Detector

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    detector = Detector(
        detector_path,
        device="cuda:0",
        image_size=image_size,
        confidence=0.15,
        classes=[0],
        half=True,
        batch=batch_size,
    )
    warm = list(frames[:batch_size])
    detector.predict(warm if batch_size > 1 else warm[0])
    torch.cuda.synchronize()
    timings: list[float] = []
    outputs: list[Any] = []
    started = time.perf_counter()
    cpu_started = process.cpu_times()
    for frame_batch in batches(frames, batch_size):
        begin = time.perf_counter()
        raw = detector.predict(list(frame_batch) if len(frame_batch) > 1 else frame_batch[0])
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - begin) * 1000.0
        timings.extend([elapsed_ms / len(frame_batch)] * len(frame_batch))
        outputs.extend(normalize_detection_batch(raw))
    wall = time.perf_counter() - started
    cpu_ended = process.cpu_times()
    cpu_seconds = (cpu_ended.user + cpu_ended.system) - (cpu_started.user + cpu_started.system)
    detections = sum(int(output.dets.shape[0]) for output in outputs)
    result = {
        "image_size": image_size,
        "batch_size": batch_size,
        "frames": len(frames),
        "detections": detections,
        "timing": summarize_ms(timings).to_dict(),
        "processed_fps": len(frames) / wall,
        "process_cpu_core_equivalents": cpu_seconds / wall,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del detector
    torch.cuda.empty_cache()
    return result, outputs


def collect_reid_work(outputs: Sequence[Any], frames: Sequence[Any]) -> list[tuple[Any, Any]]:
    work: list[tuple[Any, Any]] = []
    for output, frame in zip(outputs, frames, strict=True):
        if output.dets.shape[0]:
            work.append((output.dets[:, :4], frame))
    if not work:
        raise RuntimeError("No real person detections were available for the ReID benchmark")
    return work


def run_reid_case(weight_path: Path, work: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
    import psutil  # type: ignore[import-untyped]
    import torch
    from boxmot import ReIDModel

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    load_started = time.perf_counter()
    model = ReIDModel(weight_path, device="cuda:0", half=True)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    boxes, image = work[0]
    model.get_features(boxes, image)
    torch.cuda.synchronize()
    timings: list[float] = []
    crops = 0
    for boxes, image in work:
        begin = time.perf_counter()
        embeddings = model.get_features(boxes, image)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000.0
        count = int(embeddings.shape[0])
        if count != int(boxes.shape[0]):
            raise RuntimeError("ReID output count does not match the detected crops")
        if not bool(torch.as_tensor(embeddings).isfinite().all()):
            raise RuntimeError("ReID returned a non-finite embedding")
        timings.append(elapsed)
        crops += count
    result = {
        "weight": str(weight_path),
        "load_ms": load_ms,
        "frames_with_detections": len(work),
        "crops": crops,
        "timing_per_detection_frame": summarize_ms(timings).to_dict(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
    }
    del model
    torch.cuda.empty_cache()
    return result


TRACKER_KWARGS: dict[str, dict[str, Any]] = {
    "bytetrack": {
        "min_conf": 0.1,
        "track_thresh": 0.25,
        "match_thresh": 0.8,
        "track_buffer": 30,
        "frame_rate": 25,
    },
    "botsort": {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "use_cmc": False,
        "with_reid": True,
        "frame_rate": 25,
        "min_hits": 1,
    },
    "occluboost": {
        "det_thresh": 0.2,
        "new_track_thresh": 0.25,
        "instant_confirm_thresh": 0.25,
        "confirm_hits": 1,
        "max_age": 30,
        "min_hits": 1,
        "use_cmc": False,
        "with_reid": True,
        "gta_enabled": False,
    },
}


def run_tracker_case(
    *,
    tracker_name: str,
    outputs: Sequence[Any],
    frames: Sequence[Any],
    reid_path: Path,
    stability_passes: int,
) -> dict[str, Any]:
    import psutil  # type: ignore[import-untyped]
    import torch
    from boxmot.trackers.registry import create_tracker

    if stability_passes <= 0:
        raise ValueError("stability_passes must be positive")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    reserved_by_pass: list[int] = []
    all_timings: list[float] = []
    emitted_tracks = 0
    create_timings: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(stability_passes):
        begin = time.perf_counter()
        tracker = create_tracker(
            tracker_name,
            reid_weights=reid_path,
            device="cuda:0",
            half=True,
            per_class=False,
            tracker_kwargs=TRACKER_KWARGS[tracker_name],
        )
        torch.cuda.synchronize()
        create_timings.append((time.perf_counter() - begin) * 1000.0)
        for output, frame in zip(outputs, frames, strict=True):
            begin = time.perf_counter()
            tracks = tracker.update(output, frame)
            torch.cuda.synchronize()
            all_timings.append((time.perf_counter() - begin) * 1000.0)
            emitted_tracks += int(tracks.shape[0])
        del tracker
        torch.cuda.empty_cache()
        reserved_by_pass.append(torch.cuda.memory_reserved())
    return {
        "tracker": tracker_name,
        "appearance_aware": tracker_name in {"botsort", "occluboost"},
        "frames": len(frames) * stability_passes,
        "passes": stability_passes,
        "emitted_track_rows": emitted_tracks,
        "create_timing": summarize_ms(create_timings).to_dict(),
        "update_timing": summarize_ms(all_timings).to_dict(),
        "processed_fps": 1000.0 / statistics.mean(all_timings),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "reserved_after_each_cleanup_bytes": reserved_by_pass,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
    }


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "boxmot",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "psutil",
        "rerun-sdk",
        "torch",
        "torchvision",
        "ultralytics",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def main() -> int:
    args = parse_args()
    force_offline()
    assets = {
        "detector": identify_file(args.detector, expected_sha256=YOLO_SHA256).to_dict(),
        "reid_primary": identify_file(
            args.reid_primary, expected_sha256=OSNET_X025_SHA256
        ).to_dict(),
        "reid_challenger": identify_file(
            args.reid_challenger, expected_sha256=OSNET_X10_SHA256
        ).to_dict(),
        "clips": [identify_file(path).to_dict() for path in args.clip],
    }

    import cv2  # type: ignore[import-untyped]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("WP1 requires an available CUDA device; CPU fallback is prohibited")
    frames, frame_records = read_sampled_frames(
        args.clip, args.sample_step, args.max_frames_per_camera
    )
    detector_matrix: list[dict[str, Any]] = []
    cached_outputs: list[Any] | None = None
    for image_size in (512, 640):
        for batch_size in (1, 4):
            case, outputs = run_detector_case(
                frames=frames,
                detector_path=args.detector,
                image_size=image_size,
                batch_size=batch_size,
            )
            detector_matrix.append(case)
            if image_size == 512 and batch_size == 4:
                cached_outputs = outputs
    if cached_outputs is None:
        raise AssertionError("Selected detector profile did not execute")

    reid_work = collect_reid_work(cached_outputs, frames)
    reid_matrix = [
        run_reid_case(args.reid_primary, reid_work),
        run_reid_case(args.reid_challenger, reid_work),
    ]
    trackers = [
        run_tracker_case(
            tracker_name=name,
            outputs=cached_outputs,
            frames=frames,
            reid_path=args.reid_primary,
            stability_passes=args.stability_passes,
        )
        for name in ("bytetrack", "botsort", "occluboost")
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_argv": sys.argv,
        "offline_controls": {
            name: os.environ[name]
            for name in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "YOLO_OFFLINE",
            )
        },
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "opencv_runtime": cv2.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "assets": assets,
        "sampling": {
            "sample_step": args.sample_step,
            "max_frames_per_camera": args.max_frames_per_camera,
            "total_frames": len(frames),
            "frame_records": frame_records,
        },
        "detector_matrix": detector_matrix,
        "reid_matrix": reid_matrix,
        "tracker_matrix": trackers,
        "tracker_kwargs": TRACKER_KWARGS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": len(frames)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
