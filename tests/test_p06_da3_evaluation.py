from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_mapping_phase2.p06_da3_evaluation import (
    CAMERA_IDS,
    EvaluationCase,
    P06EvidenceError,
    T_camera_from_world,
    array_sha256,
    camera_intrinsic_matrix,
    confidence_change_diagnostic,
    depth_change_diagnostic,
    file_sha256,
    frozen_case_matrix,
    masked_distribution,
    projection_depth_diagnostic,
    repeatability_comparison,
    resize_boolean_mask,
    validate_complete_case_matrix,
    validate_prediction_arrays,
)


def test_frozen_case_matrix_contains_only_authorized_bounded_cases() -> None:
    cases = frozen_case_matrix()
    assert len(cases) == 7
    assert tuple(case.camera_ids for case in cases[:4]) == tuple(
        (camera_id,) for camera_id in CAMERA_IDS
    )
    assert cases[4].camera_ids == (
        "office-cam-01",
        "office-cam-02",
        "office-cam-03",
    )
    assert cases[5].camera_ids == ("office-cam-01", "office-cam-03")
    assert cases[6].case_id == "posed-diagnostic-cameras-2-3"
    assert cases[6].camera_ids == ("office-cam-03", "office-cam-02")
    assert "weak non-anchoring" in cases[6].authority
    assert all("office-cam-04" not in case.camera_ids for case in cases[4:])
    assert all(case.reference_view_strategy == "first" for case in cases)


def test_case_rejects_camera4_multiview_unknown_duplicate_and_unapproved_pair() -> None:
    with pytest.raises(P06EvidenceError, match="outside the frozen"):
        EvaluationCase("bad", ("office-cam-01", "office-cam-04"), "posed", "none", "first")
    with pytest.raises(P06EvidenceError, match="duplicate"):
        EvaluationCase(
            "bad",
            ("office-cam-01", "office-cam-01"),
            "pose-conditioned-diagnostic",
            "none",
            "first",
        )
    with pytest.raises(P06EvidenceError, match="unknown"):
        EvaluationCase("bad", ("office-cam-09",), "single-view-metric", "none", "first")
    with pytest.raises(P06EvidenceError, match="outside the frozen"):
        EvaluationCase(
            "bad",
            ("office-cam-01", "office-cam-02"),
            "pose-conditioned-diagnostic",
            "none",
            "first",
        )
    with pytest.raises(P06EvidenceError, match="outside the frozen"):
        EvaluationCase(
            "bad",
            ("office-cam-02", "office-cam-03"),
            "pose-conditioned-diagnostic",
            "none",
            "first",
        )


def test_serialized_matrix_must_match_authority_and_order() -> None:
    serialized = [case.to_dict() for case in frozen_case_matrix()]
    assert validate_complete_case_matrix(serialized) == frozen_case_matrix()
    serialized[0]["authority"] = "accepted geometry"
    with pytest.raises(P06EvidenceError, match="differs"):
        validate_complete_case_matrix(serialized)


def test_explicit_transform_inverse_round_trip_and_rejections() -> None:
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    T_world_from_camera = np.eye(4)
    T_world_from_camera[:3, :3] = rotation
    T_world_from_camera[:3, 3] = [2.0, -3.0, 1.5]
    T_camera_from_world_value = T_camera_from_world(T_world_from_camera)
    assert np.allclose(T_camera_from_world_value @ T_world_from_camera, np.eye(4))
    reflected = T_world_from_camera.copy()
    reflected[0, 0] *= -1
    with pytest.raises(P06EvidenceError, match="orthonormal|proper"):
        T_camera_from_world(reflected)


def test_intrinsic_matrix_requires_finite_positive_focals() -> None:
    matrix = camera_intrinsic_matrix(
        {"fx_pixels": 1200, "fy_pixels": 1210, "cx_pixels": 960, "cy_pixels": 540}
    )
    assert np.array_equal(matrix, np.array([[1200, 0, 960], [0, 1210, 540], [0, 0, 1]]))
    with pytest.raises(P06EvidenceError, match="invalid"):
        camera_intrinsic_matrix(
            {"fx_pixels": -1, "fy_pixels": 1210, "cx_pixels": 960, "cy_pixels": 540}
        )


def _prediction(view_count: int = 2) -> dict[str, NDArray[Any]]:
    return {
        "depth": np.ones((view_count, 3, 4), dtype=np.float32),
        "confidence": np.full((view_count, 3, 4), 0.8, dtype=np.float32),
        "extrinsics": np.repeat(np.eye(4, dtype=np.float32)[None], view_count, axis=0),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], view_count, axis=0),
        "processed_images": np.zeros((view_count, 3, 4, 3), dtype=np.uint8),
    }


def test_raw_prediction_contract_success_rejection_and_failure() -> None:
    arrays = _prediction()
    shapes = validate_prediction_arrays(arrays, 2)
    assert shapes["depth"] == [2, 3, 4]
    missing = dict(arrays)
    del missing["confidence"]
    with pytest.raises(P06EvidenceError, match="missing"):
        validate_prediction_arrays(missing, 2)
    arrays["depth"][0, 0, 0] = np.nan
    with pytest.raises(P06EvidenceError, match="non-finite"):
        validate_prediction_arrays(arrays, 2)


def test_hashes_include_array_shape_dtype_and_file_bytes(tmp_path: Path) -> None:
    first = np.array([1, 2, 3, 4], dtype=np.uint8)
    second = first.reshape(2, 2)
    assert array_sha256(first) != array_sha256(second)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"p06")
    assert file_sha256(path) == "f09729d8752693e14e77558672f95787f41a00ac528285bd079d21339c371798"


def test_repeatability_reports_exact_and_numerical_drift_without_tolerance() -> None:
    first = np.array([1.0, 2.0], dtype=np.float32)
    exact = repeatability_comparison(first, first.copy())
    assert exact["bitwise_equal"] is True
    assert exact["acceptance_tolerance"] is None
    second = first.copy()
    second[1] += np.float32(0.25)
    drift = repeatability_comparison(first, second)
    assert drift["bitwise_equal"] is False
    assert drift["maximum_absolute_difference"] == pytest.approx(0.25)
    assert drift["nonzero_element_count"] == 1


def test_mask_resize_distribution_and_depth_change() -> None:
    mask = np.array([[True, False], [False, True]])
    resized = resize_boolean_mask(mask, (4, 4))
    assert resized[:2, :2].all()
    assert not resized[:2, 2:].any()
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    summary = masked_distribution(values, resized)
    assert summary["count"] == 8
    baseline = np.ones((4, 4), dtype=np.float64) * 2
    candidate = baseline * 1.25
    diagnostic = depth_change_diagnostic(baseline, candidate, resized)
    assert diagnostic["candidate_to_baseline_scale_ratio_median"] == pytest.approx(1.25)
    assert diagnostic["absolute_relative_difference_median"] == pytest.approx(0.25)
    confidence = confidence_change_diagnostic(baseline, candidate, resized)
    assert confidence["candidate_minus_baseline_median"] == pytest.approx(0.5)
    assert confidence["candidate_to_baseline_ratio_median"] == pytest.approx(1.25)


def _identity_projection_case() -> tuple[Any, ...]:
    depth = np.full((5, 7), 4.0, dtype=np.float64)
    intrinsic = np.array([[100.0, 0, 3.0], [0, 100.0, 2.0], [0, 0, 1.0]])
    transform = np.eye(4)
    mask = np.ones_like(depth, dtype=bool)
    return depth, intrinsic, transform, mask


def test_projection_depth_diagnostic_exact_synthetic_transform() -> None:
    depth, intrinsic, transform, mask = _identity_projection_case()
    result = projection_depth_diagnostic(
        depth,
        intrinsic,
        transform,
        depth.copy(),
        intrinsic,
        transform,
        mask,
        mask,
        sampling_stride=1,
    )
    assert result["status"] == "measured-diagnostic"
    assert result["compared_count"] == depth.size
    assert result["absolute_relative_depth_median"] == pytest.approx(0.0)
    assert result["fraction_within_0_05"] == pytest.approx(1.0)
    assert "not overlap" in str(result["authority"])


def test_projection_depth_diagnostic_retains_unsupported_zero_overlap() -> None:
    depth, intrinsic, transform, mask = _identity_projection_case()
    target_transform = transform.copy()
    target_transform[0, 3] = 1000.0
    result = projection_depth_diagnostic(
        depth,
        intrinsic,
        transform,
        depth,
        intrinsic,
        target_transform,
        mask,
        mask,
        sampling_stride=1,
    )
    assert result["status"] == "unsupported-no-comparable-projection"
    assert result["acceptance_threshold"] is None
