"""Benchmark pinned YOLO11n on CUDA using immutable P06 pinhole derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import monotonic_ns
from typing import Any

import cv2
import numpy as np

from spatial_mapping_phase2.p09_projection import (
    FrozenProjectionInputs,
    load_frozen_projection_inputs,
    project_detection_to_floor,
)
from spatial_mapping_phase2.p09_tracking_domain import LiveFrameIdentity
from spatial_mapping_phase2.p09_yolo11 import Yolo11CudaDetector, Yolo11ModelSpec

MODEL_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
P06_SHA256 = "d3c1bfd314a865270a9d9352efd33cd61a470bdc8829a084d5a0f2996f4aa8e4"
P07_SHA256 = "df883eedae46f48aab9c84a86b8e2398fa44b37c2664cac4792d21e5a7d8ef51"
P08_MANIFEST_SHA256 = "1462f65068156b4ffe611fd705b7ae62468fe8b665cd5eebae8ed96132adc399"
P08_PLANE_SHA256 = "1079e8573938c19bd668a73c3bb7706c684fd661a36c95db85eb64592cb25eb0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--p06-input-manifest", type=Path, required=True)
    parser.add_argument("--p07-frustum-manifest", type=Path, required=True)
    parser.add_argument("--p08-floor-manifest", type=Path, required=True)
    parser.add_argument("--p08-floor-plane", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.repetitions <= 100:
        parser.error("--repetitions must be within 2..100")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    projection_inputs = FrozenProjectionInputs(
        args.p06_input_manifest.resolve(),
        P06_SHA256,
        args.p07_frustum_manifest.resolve(),
        P07_SHA256,
        args.p08_floor_manifest.resolve(),
        P08_MANIFEST_SHA256,
        args.p08_floor_plane.resolve(),
        P08_PLANE_SHA256,
    )
    calibrations, floor = load_frozen_projection_inputs(projection_inputs)
    model_spec = Yolo11ModelSpec(args.model.resolve(), MODEL_SHA256)
    detector = Yolo11CudaDetector(model_spec)
    p06 = _json_object(args.p06_input_manifest)
    camera_records = p06.get("cameras")
    if not isinstance(camera_records, list):
        raise ValueError("P06 input manifest camera list is malformed")

    records: list[dict[str, Any]] = []
    for camera_record in camera_records:
        if not isinstance(camera_record, dict):
            raise ValueError("P06 camera record is malformed")
        camera_id = str(camera_record.get("camera_id"))
        pinhole = camera_record.get("pinhole_derivative")
        if not isinstance(pinhole, dict):
            raise ValueError("P06 pinhole record is malformed")
        image_path = Path(str(pinhole.get("path"))).resolve()
        expected_hash = str(pinhole.get("sha256"))
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"P06 pinhole identity changed for {camera_id}")
        native_pinhole = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if native_pinhole is None or native_pinhole.shape != (1080, 1920, 3):
            raise ValueError(f"P06 pinhole derivative is unreadable for {camera_id}")
        calibrated = cv2.resize(native_pinhole, (504, 280), interpolation=cv2.INTER_AREA)
        frame = LiveFrameIdentity(
            camera_id,
            f"{camera_id}-yolo11-gate",
            monotonic_ns(),
            datetime.now(UTC).isoformat(),
            None,
            None,
            504,
            280,
        )
        detector.detect(frame, calibrated)  # Per-camera warm-up is excluded from timings.
        timings: list[dict[str, float]] = []
        selected = None
        for _ in range(args.repetitions):
            result = detector.detect(frame, calibrated)
            selected = result
            timings.append(
                {
                    "preprocessing_ms": result.preprocessing_ms,
                    "inference_ms": result.inference_ms,
                    "postprocessing_ms": result.postprocessing_ms,
                    "total_ms": result.preprocessing_ms
                    + result.inference_ms
                    + result.postprocessing_ms,
                }
            )
        assert selected is not None
        annotated = calibrated.copy()
        detections: list[dict[str, Any]] = []
        for detection in selected.detections:
            projection = project_detection_to_floor(
                detection, calibrations[camera_id], floor, frame.acquisition_monotonic_ns
            )
            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)
            u, v = (int(round(value)) for value in detection.image_point_uv)
            cv2.circle(annotated, (u, v), 4, (0, 0, 255), -1)
            detections.append(
                {
                    "confidence": detection.confidence,
                    "bbox_xyxy": detection.bbox_xyxy,
                    "image_point_uv": detection.image_point_uv,
                    "footpoint_kind": detection.footpoint_kind,
                    "projection_status": projection.status,
                    "projection_reason": projection.reason,
                    "xy_metres": (
                        projection.observation.xy_metres
                        if projection.observation is not None
                        else None
                    ),
                }
            )
        preview_path = output_dir / f"{camera_id}-yolo11-preview.png"
        if not cv2.imwrite(str(preview_path), annotated):
            raise RuntimeError(f"failed to write detector preview for {camera_id}")
        total_values = np.asarray([timing["total_ms"] for timing in timings])
        inference_values = np.asarray([timing["inference_ms"] for timing in timings])
        records.append(
            {
                "camera_id": camera_id,
                "source_pinhole_sha256": expected_hash,
                "detection_count": len(selected.detections),
                "detections": detections,
                "timing_ms": {
                    "repetitions": args.repetitions,
                    "warmup_excluded": True,
                    "total_median": float(np.median(total_values)),
                    "total_p95": float(np.percentile(total_values, 95)),
                    "inference_median": float(np.median(inference_values)),
                    "inference_p95": float(np.percentile(inference_values, 95)),
                },
                "preview": _file_record(preview_path),
            }
        )

    manifest = {
        "schema_version": "p09-yolo11-gpu-gate-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "success": True,
        "dynamic_DA3_invoked": False,
        "cpu_fallback_allowed": False,
        "model": {
            "family": "YOLO11n",
            "format": "PyTorch",
            "input_size": model_spec.input_size,
            "source_release": "Ultralytics assets v8.3.0",
            "source_url": "https://github.com/ultralytics/ultralytics",
            "documentation_url": (
                "https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md"
            ),
            "checkpoint_url": (
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
            ),
            "license": "AGPL-3.0",
            "open_source_release_commitment_confirmed": True,
            "path": str(model_spec.model_path),
            "sha256": model_spec.model_sha256,
            "byte_count": model_spec.model_path.stat().st_size,
            "confidence_threshold": model_spec.confidence_threshold,
            "nms_iou_threshold": model_spec.nms_iou_threshold,
            "person_class_index": model_spec.person_class_index,
        },
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "ultralytics": version("ultralytics"),
            "backend": "ultralytics-pytorch-cuda",
            "cuda": asdict(detector.cuda_evidence()),
        },
        "frozen_inputs": {
            "P06_input_manifest_sha256": P06_SHA256,
            "P07_frustum_manifest_sha256": P07_SHA256,
            "P08_floor_manifest_sha256": P08_MANIFEST_SHA256,
            "P08_floor_plane_sha256": P08_PLANE_SHA256,
        },
        "camera_records": records,
    }
    manifest_path = output_dir / "yolo11-gpu-gate-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"manifest": _file_record(manifest_path), "camera_records": records}, indent=2)
    )
    return 0


def _json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return loaded


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "byte_count": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    sys.exit(main())
