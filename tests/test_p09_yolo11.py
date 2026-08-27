from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatial_mapping_phase2.p09_detector import P09DetectorError
from spatial_mapping_phase2.p09_tracking_domain import FootpointKind, LiveFrameIdentity
from spatial_mapping_phase2.p09_yolo11 import (
    Yolo11ModelSpec,
    person_detections_from_yolo11_arrays,
)


def _frame() -> LiveFrameIdentity:
    return LiveFrameIdentity(
        "office-cam-01",
        "frame-1",
        1,
        "2026-08-20T00:00:00Z",
        None,
        None,
        504,
        280,
    )


def _spec() -> Yolo11ModelSpec:
    return Yolo11ModelSpec(Path("missing.pt"), "0" * 64)


def test_person_arrays_preserve_post_nms_order_and_bbox_footpoint() -> None:
    detections = person_detections_from_yolo11_arrays(
        _frame(),
        np.array([[100.0, 20.0, 200.0, 200.0], [300.0, 30.0, 360.0, 210.0]]),
        np.array([0.9, 0.85]),
        np.array([0.0, 0.0]),
        _spec(),
    )
    assert len(detections) == 2
    assert detections[0].detection_index == 0
    assert detections[0].footpoint_kind is FootpointKind.BBOX_BOTTOM_CENTER
    assert detections[0].image_point_uv == pytest.approx((150.0, 200.0))
    assert detections[1].image_point_uv == pytest.approx((330.0, 210.0))


def test_bottom_clipped_detection_is_explicit_torso_proxy() -> None:
    detections = person_detections_from_yolo11_arrays(
        _frame(),
        np.array([[100.0, 30.0, 220.0, 300.0]]),
        np.array([0.9]),
        np.array([0.0]),
        _spec(),
    )
    assert len(detections) == 1
    assert detections[0].footpoint_kind is FootpointKind.TORSO_PROXY
    assert detections[0].clipped_at_image_bottom
    assert detections[0].image_point_uv[1] == 279.0


def test_array_filter_rejects_low_score_nonperson_nonfinite_and_tiny_boxes() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 5.0],
            [20.0, 20.0, 100.0, 100.0],
            [120.0, 20.0, 200.0, 150.0],
            [np.nan, 20.0, 100.0, 100.0],
        ]
    )
    assert not person_detections_from_yolo11_arrays(
        _frame(), boxes, np.array([0.9, 0.1, 0.9, 0.9]), np.array([0.0, 0.0, 2.0, 0.0]), _spec()
    )


def test_default_person_confidence_gate_is_precision_first() -> None:
    assert _spec().confidence_threshold == 0.70
    assert not person_detections_from_yolo11_arrays(
        _frame(),
        np.array([[20.0, 20.0, 100.0, 120.0]]),
        np.array([0.699]),
        np.array([0.0]),
        _spec(),
    )


def test_array_contract_and_model_hash_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(P09DetectorError, match="misaligned"):
        person_detections_from_yolo11_arrays(
            _frame(), np.zeros((2, 4)), np.zeros(1), np.zeros(2), _spec()
        )
    model = tmp_path / "model.pt"
    model.write_bytes(b"not-a-model")
    with pytest.raises(P09DetectorError, match="identity changed"):
        Yolo11ModelSpec(model, "0" * 64).verify_model()


def test_spec_rejects_cpu_fallback_and_changed_input() -> None:
    with pytest.raises(P09DetectorError, match="cuda:0"):
        Yolo11ModelSpec(Path("model.pt"), "0" * 64, device="cpu")
    with pytest.raises(P09DetectorError, match="640"):
        Yolo11ModelSpec(Path("model.pt"), "0" * 64, input_size=512)
