"""Pinned Ultralytics YOLO11 person adapter with a fail-closed CUDA contract."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_detector import DetectorBatchResult, P09DetectorError
from spatial_mapping_phase2.p09_tracking_domain import (
    FootpointKind,
    LiveFrameIdentity,
    PersonDetection,
)

Array = NDArray[Any]


@dataclass(frozen=True, slots=True)
class Yolo11ModelSpec:
    """Immutable model identity and inference policy for accepted P09 use."""

    model_path: Path
    model_sha256: str
    ultralytics_version: str = "8.4.123"
    input_size: int = 640
    person_class_index: int = 0
    confidence_threshold: float = 0.70
    nms_iou_threshold: float = 0.45
    device: str = "cuda:0"
    bottom_clip_margin_pixels: float = 2.0
    minimum_visible_height_pixels: float = 12.0

    def __post_init__(self) -> None:
        if len(self.model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_sha256
        ):
            raise P09DetectorError("model SHA-256 must be lowercase hexadecimal")
        if self.input_size != 640:
            raise P09DetectorError("P09 YOLO11 must use its frozen 640-pixel input")
        if self.person_class_index != 0:
            raise P09DetectorError("P09 detector must retain COCO person class index 0")
        if self.device != "cuda:0":
            raise P09DetectorError("P09 YOLO11 is required to run on cuda:0")
        for label, value in (
            ("confidence", self.confidence_threshold),
            ("NMS IoU", self.nms_iou_threshold),
        ):
            if not math.isfinite(value) or not 0 < value < 1:
                raise P09DetectorError(f"{label} threshold must be within (0, 1)")
        if self.bottom_clip_margin_pixels < 0 or self.minimum_visible_height_pixels <= 0:
            raise P09DetectorError("detector clipping parameters are invalid")

    def verify_model(self) -> None:
        if not self.model_path.is_file():
            raise P09DetectorError("pinned YOLO11 model is missing")
        actual = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if actual != self.model_sha256:
            raise P09DetectorError("pinned YOLO11 model identity changed")


@dataclass(frozen=True, slots=True)
class CudaRuntimeEvidence:
    device: str
    device_name: str
    torch_version: str
    cuda_version: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class Yolo11CudaDetector:
    """Persistent YOLO11 PyTorch runtime that refuses CPU fallback and downloads."""

    def __init__(self, spec: Yolo11ModelSpec) -> None:
        spec.verify_model()
        try:
            installed_version = version("ultralytics")
        except PackageNotFoundError as error:
            raise P09DetectorError("pinned Ultralytics runtime is not installed") from error
        if installed_version != spec.ultralytics_version:
            raise P09DetectorError(
                f"Ultralytics version changed: expected {spec.ultralytics_version}, "
                f"found {installed_version}"
            )
        try:
            import torch
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception as error:
            raise P09DetectorError("failed to import the pinned YOLO11 CUDA runtime") from error
        if not torch.cuda.is_available():
            raise P09DetectorError("CUDA is unavailable; CPU fallback is prohibited")
        try:
            torch.cuda.set_device(spec.device)
            torch.cuda.reset_peak_memory_stats(spec.device)
            model = YOLO(str(spec.model_path), task="detect")
            model.to(spec.device)
            parameter = next(model.model.parameters())
        except Exception as error:
            raise P09DetectorError("failed to initialize pinned YOLO11 on cuda:0") from error
        if parameter.device.type != "cuda" or parameter.device.index != 0:
            raise P09DetectorError("YOLO11 parameters are not resident on cuda:0")
        self.spec = spec
        self._model = model
        self._torch = torch

    def detect(self, frame: LiveFrameIdentity, calibrated_frame_bgr: Array) -> DetectorBatchResult:
        image = np.asarray(calibrated_frame_bgr)
        if image.dtype != np.uint8 or image.shape != (frame.height_pixels, frame.width_pixels, 3):
            raise P09DetectorError("YOLO11 input must match the frame's HxWx3 uint8 BGR contract")
        try:
            results = self._model.predict(
                source=image,
                imgsz=self.spec.input_size,
                conf=self.spec.confidence_threshold,
                iou=self.spec.nms_iou_threshold,
                classes=[self.spec.person_class_index],
                device=self.spec.device,
                verbose=False,
            )
        except Exception as error:
            raise P09DetectorError("YOLO11 CUDA inference failed") from error
        if len(results) != 1:
            raise P09DetectorError("YOLO11 returned an unexpected result count")
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            boxes_xyxy = np.empty((0, 4), dtype=np.float64)
            scores = np.empty((0,), dtype=np.float64)
            class_indices = np.empty((0,), dtype=np.float64)
        else:
            boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            class_indices = boxes.cls.detach().cpu().numpy()
        detections = person_detections_from_yolo11_arrays(
            frame, boxes_xyxy, scores, class_indices, self.spec
        )
        speed = result.speed
        return DetectorBatchResult(
            detections,
            _timing(speed, "preprocess"),
            _timing(speed, "inference"),
            _timing(speed, "postprocess"),
            "ultralytics-pytorch-cuda",
        )

    def cuda_evidence(self) -> CudaRuntimeEvidence:
        cuda = self._torch.cuda
        properties = cuda.get_device_properties(self.spec.device)
        return CudaRuntimeEvidence(
            self.spec.device,
            str(properties.name),
            str(self._torch.__version__),
            str(self._torch.version.cuda),
            int(cuda.memory_allocated(self.spec.device)),
            int(cuda.memory_reserved(self.spec.device)),
            int(cuda.max_memory_allocated(self.spec.device)),
            int(cuda.max_memory_reserved(self.spec.device)),
        )


def person_detections_from_yolo11_arrays(
    frame: LiveFrameIdentity,
    boxes_xyxy: Array,
    scores: Array,
    class_indices: Array,
    spec: Yolo11ModelSpec,
) -> tuple[PersonDetection, ...]:
    """Convert post-NMS Ultralytics arrays into explicit P09 person evidence."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    confidence = np.asarray(scores, dtype=np.float64)
    classes = np.asarray(class_indices, dtype=np.float64)
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 4
        or confidence.shape != (len(boxes),)
        or classes.shape != (len(boxes),)
    ):
        raise P09DetectorError("YOLO11 boxes, scores, and classes are misaligned")
    valid = np.all(np.isfinite(boxes), axis=1)
    valid &= np.isfinite(confidence) & np.isfinite(classes)
    valid &= confidence >= spec.confidence_threshold
    valid &= classes == spec.person_class_index
    boxes = boxes[valid]
    confidence = confidence[valid]
    width, height = frame.width_pixels, frame.height_pixels
    detections: list[PersonDetection] = []
    for box, score in zip(boxes, confidence, strict=True):
        raw_y2 = float(box[3])
        x1 = float(np.clip(box[0], 0.0, width - 1.0))
        y1 = float(np.clip(box[1], 0.0, height - 1.0))
        x2 = float(np.clip(box[2], 0.0, width - 1.0))
        y2 = float(np.clip(box[3], 0.0, height - 1.0))
        if x2 <= x1 or y2 <= y1 or y2 - y1 < spec.minimum_visible_height_pixels:
            continue
        clipped = raw_y2 >= height - spec.bottom_clip_margin_pixels
        detections.append(
            PersonDetection(
                frame=frame,
                detection_index=len(detections),
                confidence=float(np.clip(score, 0.0, 1.0)),
                bbox_xyxy=(x1, y1, x2, y2),
                image_point_uv=((x1 + x2) / 2.0, y2),
                footpoint_kind=(
                    FootpointKind.TORSO_PROXY if clipped else FootpointKind.BBOX_BOTTOM_CENTER
                ),
                clipped_at_image_bottom=clipped,
            )
        )
    return tuple(detections)


def _timing(speed: Any, key: str) -> float:
    if not isinstance(speed, dict):
        raise P09DetectorError("YOLO11 timing evidence is unavailable")
    value = speed.get(key)
    if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise P09DetectorError(f"YOLO11 {key} timing is invalid")
    return float(value)
