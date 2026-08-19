"""Evidence-bounded contracts and metrics for P06 DA3 mode evaluation.

This module deliberately contains no DA3 import.  The exact vendor runtime stays behind a thin
script, while case authorization, artifact validation and diagnostic math remain unit-testable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

CAMERA_IDS = tuple(f"office-cam-0{index}" for index in range(1, 5))
SOURCE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
CHECKPOINT_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
CHECKPOINT_NAME = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
CHECKPOINT_SHA256 = "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"
PROCESS_RESOLUTION = 504
REPETITIONS = 2
RAW_FIELDS = ("depth", "confidence", "extrinsics", "intrinsics", "processed_images")


class P06EvidenceError(ValueError):
    """Raised when P06 evidence violates a frozen identity or authority boundary."""


@dataclass(frozen=True)
class EvaluationCase:
    """One authorized DA3 invocation shape."""

    case_id: str
    camera_ids: tuple[str, ...]
    mode: str
    authority: str
    reference_view_strategy: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise P06EvidenceError("case_id must not be empty")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise P06EvidenceError("a P06 case cannot contain duplicate cameras")
        if any(camera_id not in CAMERA_IDS for camera_id in self.camera_ids):
            raise P06EvidenceError("a P06 case contains an unknown camera")
        allowed = _authorized_case_memberships()
        if (self.mode, self.camera_ids) not in allowed:
            raise P06EvidenceError("case membership is outside the frozen P06 matrix")
        if self.mode == "single-view-metric" and self.reference_view_strategy != "first":
            raise P06EvidenceError("single-view cases use the inert first-view strategy")
        if self.mode == "pose-conditioned-diagnostic" and self.reference_view_strategy != "first":
            raise P06EvidenceError("pose-conditioned diagnostics use their authorized first view")
        if "office-cam-04" in self.camera_ids and len(self.camera_ids) != 1:
            raise P06EvidenceError("Camera 4 is single-view only under D036")

    @property
    def uses_input_camera_parameters(self) -> bool:
        """Return whether DA3 receives the provisional extrinsics and intrinsics."""

        return self.mode == "pose-conditioned-diagnostic"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready case record."""

        return {
            "case_id": self.case_id,
            "camera_ids": list(self.camera_ids),
            "mode": self.mode,
            "authority": self.authority,
            "reference_view_strategy": self.reference_view_strategy,
            "uses_input_camera_parameters": self.uses_input_camera_parameters,
        }


def frozen_case_matrix() -> tuple[EvaluationCase, ...]:
    """Return the complete and only P06 inference matrix authorized at initiation."""

    single = tuple(
        EvaluationCase(
            case_id=f"single-{camera_id}",
            camera_ids=(camera_id,),
            mode="single-view-metric",
            authority=(
                "isolated weak/failure metric baseline only"
                if camera_id in {"office-cam-02", "office-cam-04"}
                else "independent metric baseline; no world-registration authority"
            ),
            reference_view_strategy="first",
        )
        for camera_id in CAMERA_IDS
    )
    return single + (
        EvaluationCase(
            case_id="posed-diagnostic-cameras-1-2-3",
            camera_ids=("office-cam-01", "office-cam-02", "office-cam-03"),
            mode="pose-conditioned-diagnostic",
            authority=(
                "D036 diagnostic hypothesis only; Camera 2 is a weak non-anchoring failure "
                "baseline; no connectivity, registration, scale, fusion or geometry authority"
            ),
            reference_view_strategy="first",
        ),
        EvaluationCase(
            case_id="posed-diagnostic-cameras-1-3",
            camera_ids=("office-cam-01", "office-cam-03"),
            mode="pose-conditioned-diagnostic",
            authority=(
                "D036 Camera-2-removal diagnostic only; provisional Camera 1/3 seeds do not "
                "create connectivity, registration, scale, fusion or geometry authority"
            ),
            reference_view_strategy="first",
        ),
        EvaluationCase(
            case_id="posed-diagnostic-cameras-2-3",
            camera_ids=("office-cam-03", "office-cam-02"),
            mode="pose-conditioned-diagnostic",
            authority=(
                "D037 visual-overlap diagnostic only; Camera 3 is the ordered provisional "
                "reference and Camera 2 remains a weak non-anchoring input; no connectivity, "
                "registration, scale, fusion or geometry authority"
            ),
            reference_view_strategy="first",
        ),
    )


def case_from_mapping(value: Mapping[str, Any]) -> EvaluationCase:
    """Parse and validate one serialized case definition."""

    camera_values = value.get("camera_ids")
    if not isinstance(camera_values, list) or not all(
        isinstance(item, str) for item in camera_values
    ):
        raise P06EvidenceError("case camera_ids must be a list of strings")
    return EvaluationCase(
        case_id=str(value.get("case_id", "")),
        camera_ids=tuple(camera_values),
        mode=str(value.get("mode", "")),
        authority=str(value.get("authority", "")),
        reference_view_strategy=str(value.get("reference_view_strategy", "")),
    )


def validate_complete_case_matrix(
    values: Sequence[Mapping[str, Any]],
) -> tuple[EvaluationCase, ...]:
    """Require a serialized matrix to match the frozen matrix exactly and in order."""

    parsed = tuple(case_from_mapping(value) for value in values)
    expected = frozen_case_matrix()
    if parsed != expected:
        raise P06EvidenceError("serialized case matrix differs from the frozen P06 matrix")
    return parsed


def validate_T_world_from_camera(value: Any) -> Array:
    """Validate an explicit homogeneous camera-to-world rigid transform."""

    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise P06EvidenceError("T_world_from_camera must be a finite 4x4 matrix")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-10):
        raise P06EvidenceError("T_world_from_camera must have a homogeneous final row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise P06EvidenceError("T_world_from_camera rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise P06EvidenceError("T_world_from_camera rotation must be proper")
    return transform


def T_camera_from_world(T_world_from_camera: Any) -> Array:
    """Invert a validated camera-to-world transform without ambiguous naming."""

    transform = validate_T_world_from_camera(T_world_from_camera)
    rotation_world_from_camera = transform[:3, :3]
    center_world = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation_world_from_camera.T
    inverse[:3, 3] = -rotation_world_from_camera.T @ center_world
    return inverse


def camera_intrinsic_matrix(value: Mapping[str, Any]) -> Array:
    """Build a finite pinhole matrix from a selected intrinsic record."""

    fields = ("fx_pixels", "fy_pixels", "cx_pixels", "cy_pixels")
    try:
        fx, fy, cx, cy = (float(value[field]) for field in fields)
    except (KeyError, TypeError, ValueError) as error:
        raise P06EvidenceError("intrinsic record is missing finite pinhole fields") from error
    numbers = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.all(np.isfinite(numbers)) or fx <= 0 or fy <= 0:
        raise P06EvidenceError("intrinsic record contains invalid pinhole values")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def validate_prediction_arrays(
    arrays: Mapping[str, Array], expected_view_count: int
) -> dict[str, list[int]]:
    """Validate the complete preserved raw prediction contract."""

    missing = [field for field in RAW_FIELDS if field not in arrays]
    if missing:
        raise P06EvidenceError(f"raw prediction is missing fields: {missing}")
    depth = np.asarray(arrays["depth"])
    confidence = np.asarray(arrays["confidence"])
    extrinsics = np.asarray(arrays["extrinsics"])
    intrinsics = np.asarray(arrays["intrinsics"])
    images = np.asarray(arrays["processed_images"])
    if depth.ndim != 3 or depth.shape[0] != expected_view_count:
        raise P06EvidenceError("depth must have shape (views, height, width)")
    if confidence.shape != depth.shape:
        raise P06EvidenceError("confidence must match depth shape")
    if extrinsics.shape not in {
        (expected_view_count, 3, 4),
        (expected_view_count, 4, 4),
    }:
        raise P06EvidenceError("extrinsics have an unexpected view or matrix shape")
    if intrinsics.shape != (expected_view_count, 3, 3):
        raise P06EvidenceError("intrinsics have an unexpected view or matrix shape")
    if images.shape != (*depth.shape, 3):
        raise P06EvidenceError("processed images must match depth spatial shape")
    for field in ("depth", "confidence", "extrinsics", "intrinsics"):
        if not np.all(np.isfinite(np.asarray(arrays[field]))):
            raise P06EvidenceError(f"raw prediction field is non-finite: {field}")
    if np.any(depth <= 0):
        raise P06EvidenceError("metric depth must be strictly positive")
    return {field: list(np.asarray(arrays[field]).shape) for field in RAW_FIELDS}


def array_sha256(value: Array) -> str:
    """Return a stable identity for an array including dtype and shape."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading large model weights into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repeatability_comparison(first: Array, second: Array) -> dict[str, Any]:
    """Report exact and numerical repeatability without applying an acceptance tolerance."""

    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise P06EvidenceError("repeatability arrays must have identical shape and dtype")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise P06EvidenceError("repeatability arrays must be finite")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "bitwise_equal": bool(np.array_equal(left, right)),
        "first_sha256": array_sha256(left),
        "second_sha256": array_sha256(right),
        "maximum_absolute_difference": float(np.max(difference)),
        "mean_absolute_difference": float(np.mean(difference)),
        "nonzero_element_count": int(np.count_nonzero(difference)),
        "acceptance_tolerance": None,
    }


def resize_boolean_mask(mask: Array, shape: tuple[int, int]) -> Array:
    """Nearest-neighbour resize for evidence masks without requiring OpenCV."""

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise P06EvidenceError("evaluation mask must be two-dimensional")
    height, width = shape
    if height <= 0 or width <= 0:
        raise P06EvidenceError("mask target shape must be positive")
    y_indices = np.minimum(
        (np.arange(height, dtype=np.float64) * source.shape[0] / height).astype(np.int64),
        source.shape[0] - 1,
    )
    x_indices = np.minimum(
        (np.arange(width, dtype=np.float64) * source.shape[1] / width).astype(np.int64),
        source.shape[1] - 1,
    )
    return np.asarray(source[np.ix_(y_indices, x_indices)], dtype=bool)


def masked_distribution(values: Array, mask: Array) -> dict[str, Any]:
    """Summarize finite values inside an explicit evaluation mask."""

    array = np.asarray(values, dtype=np.float64)
    selected_mask = np.asarray(mask, dtype=bool)
    if array.shape != selected_mask.shape:
        raise P06EvidenceError("distribution values and mask must share a shape")
    selected = array[selected_mask & np.isfinite(array)]
    if selected.size == 0:
        raise P06EvidenceError("distribution mask selects no finite values")
    quantiles = np.quantile(selected, [0.05, 0.5, 0.95])
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "minimum": float(np.min(selected)),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "maximum": float(np.max(selected)),
    }


def depth_change_diagnostic(
    baseline_depth: Array,
    candidate_depth: Array,
    evaluation_mask: Array,
) -> dict[str, Any]:
    """Compare metric depth fields at identical processed pixels without claiming truth."""

    baseline = np.asarray(baseline_depth, dtype=np.float64)
    candidate = np.asarray(candidate_depth, dtype=np.float64)
    mask = np.asarray(evaluation_mask, dtype=bool)
    if baseline.shape != candidate.shape or baseline.shape != mask.shape:
        raise P06EvidenceError("depth comparison arrays and mask must share a shape")
    valid = (
        mask
        & np.isfinite(baseline)
        & np.isfinite(candidate)
        & (baseline > 0)
        & (candidate > 0)
    )
    if not np.any(valid):
        raise P06EvidenceError("depth comparison has no valid metric pixels")
    ratio = candidate[valid] / baseline[valid]
    absolute_relative = np.abs(candidate[valid] - baseline[valid]) / baseline[valid]
    log_delta = np.log(candidate[valid]) - np.log(baseline[valid])
    return {
        "pixel_count": int(np.count_nonzero(valid)),
        "candidate_to_baseline_scale_ratio_median": float(np.median(ratio)),
        "absolute_relative_difference_median": float(np.median(absolute_relative)),
        "absolute_relative_difference_p95": float(np.quantile(absolute_relative, 0.95)),
        "log_depth_rmse": float(np.sqrt(np.mean(np.square(log_delta)))),
        "truth_or_accuracy_authority": "none; same-image mode sensitivity only",
    }


def confidence_change_diagnostic(
    baseline_confidence: Array,
    candidate_confidence: Array,
    evaluation_mask: Array,
) -> dict[str, Any]:
    """Compare model confidence fields without assuming a calibrated probability scale."""

    baseline = np.asarray(baseline_confidence, dtype=np.float64)
    candidate = np.asarray(candidate_confidence, dtype=np.float64)
    mask = np.asarray(evaluation_mask, dtype=bool)
    if baseline.shape != candidate.shape or baseline.shape != mask.shape:
        raise P06EvidenceError("confidence comparison arrays and mask must share a shape")
    valid = mask & np.isfinite(baseline) & np.isfinite(candidate)
    if not np.any(valid):
        raise P06EvidenceError("confidence comparison has no valid pixels")
    delta = candidate[valid] - baseline[valid]
    absolute_delta = np.abs(delta)
    positive_baseline = baseline[valid] > 0
    ratio_median = (
        float(np.median(candidate[valid][positive_baseline] / baseline[valid][positive_baseline]))
        if np.any(positive_baseline)
        else None
    )
    return {
        "pixel_count": int(np.count_nonzero(valid)),
        "baseline_median": float(np.median(baseline[valid])),
        "candidate_median": float(np.median(candidate[valid])),
        "candidate_minus_baseline_median": float(np.median(delta)),
        "absolute_difference_median": float(np.median(absolute_delta)),
        "absolute_difference_p95": float(np.quantile(absolute_delta, 0.95)),
        "candidate_to_baseline_ratio_median": ratio_median,
        "calibration_or_accuracy_authority": (
            "none; DA3 confidence is retained as a comparative score, not probability or truth"
        ),
    }


def projection_depth_diagnostic(
    source_depth: Array,
    source_intrinsics: Array,
    T_world_from_source_camera: Array,
    target_depth: Array,
    target_intrinsics: Array,
    T_world_from_target_camera: Array,
    source_mask: Array,
    target_mask: Array,
    *,
    sampling_stride: int = 8,
) -> dict[str, Any]:
    """Measure z-buffered cross-view depth residuals under provisional supplied transforms.

    The result is a model/seed consistency diagnostic.  It does not establish overlap, pose
    accuracy, world scale, connectivity or geometry acceptance.
    """

    if sampling_stride < 1:
        raise P06EvidenceError("projection sampling stride must be positive")
    source = np.asarray(source_depth, dtype=np.float64)
    target = np.asarray(target_depth, dtype=np.float64)
    source_valid_mask = np.asarray(source_mask, dtype=bool)
    target_valid_mask = np.asarray(target_mask, dtype=bool)
    if source.ndim != 2 or target.ndim != 2:
        raise P06EvidenceError("projection depths must be two-dimensional")
    if source.shape != source_valid_mask.shape or target.shape != target_valid_mask.shape:
        raise P06EvidenceError("projection masks must match their depth maps")
    source_K = _validate_intrinsic_matrix(source_intrinsics)
    target_K = _validate_intrinsic_matrix(target_intrinsics)
    source_T = validate_T_world_from_camera(T_world_from_source_camera)
    target_T_camera_from_world = T_camera_from_world(T_world_from_target_camera)

    rows = np.arange(0, source.shape[0], sampling_stride, dtype=np.int64)
    cols = np.arange(0, source.shape[1], sampling_stride, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(cols, rows)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    flat_depth = source[flat_y, flat_x]
    valid = (
        source_valid_mask[flat_y, flat_x]
        & np.isfinite(flat_depth)
        & (flat_depth > 0)
    )
    flat_x = flat_x[valid]
    flat_y = flat_y[valid]
    flat_depth = flat_depth[valid]
    if flat_depth.size == 0:
        return _empty_projection_result("source mask selected no valid sampled depths")

    pixels = np.stack(
        [flat_x.astype(np.float64), flat_y.astype(np.float64), np.ones(flat_x.size)], axis=0
    )
    camera_points = np.linalg.inv(source_K) @ pixels
    camera_points *= flat_depth[None, :]
    world_points = source_T[:3, :3] @ camera_points + source_T[:3, 3:4]
    target_points = (
        target_T_camera_from_world[:3, :3] @ world_points
        + target_T_camera_from_world[:3, 3:4]
    )
    positive = target_points[2] > 0
    target_points = target_points[:, positive]
    if target_points.shape[1] == 0:
        return _empty_projection_result("no sampled source point is in front of target camera")
    projected = target_K @ target_points
    projected_x = np.rint(projected[0] / projected[2]).astype(np.int64)
    projected_y = np.rint(projected[1] / projected[2]).astype(np.int64)
    projected_z = target_points[2]
    in_frame = (
        (projected_x >= 0)
        & (projected_x < target.shape[1])
        & (projected_y >= 0)
        & (projected_y < target.shape[0])
    )
    projected_x = projected_x[in_frame]
    projected_y = projected_y[in_frame]
    projected_z = projected_z[in_frame]
    if projected_z.size == 0:
        return _empty_projection_result("no sampled source projection falls inside target frame")

    # Keep the nearest projected source surface per target pixel.
    linear = projected_y * target.shape[1] + projected_x
    order = np.lexsort((projected_z, linear))
    sorted_linear = linear[order]
    first = np.concatenate(([True], sorted_linear[1:] != sorted_linear[:-1]))
    selected = order[first]
    projected_x = projected_x[selected]
    projected_y = projected_y[selected]
    projected_z = projected_z[selected]
    target_values = target[projected_y, projected_x]
    valid_target = (
        target_valid_mask[projected_y, projected_x]
        & np.isfinite(target_values)
        & (target_values > 0)
    )
    target_values = target_values[valid_target]
    projected_z = projected_z[valid_target]
    if projected_z.size == 0:
        return _empty_projection_result("target mask rejects every projected source sample")
    signed_relative = (projected_z - target_values) / target_values
    absolute_relative = np.abs(signed_relative)
    return {
        "status": "measured-diagnostic",
        "sampled_source_count": int(flat_depth.size),
        "unique_in_frame_projection_count": int(len(selected)),
        "compared_count": int(projected_z.size),
        "compared_fraction_of_sampled_source": float(projected_z.size / flat_depth.size),
        "signed_relative_depth_median": float(np.median(signed_relative)),
        "absolute_relative_depth_median": float(np.median(absolute_relative)),
        "absolute_relative_depth_p95": float(np.quantile(absolute_relative, 0.95)),
        "fraction_within_0_05": float(np.mean(absolute_relative <= 0.05)),
        "fraction_within_0_10": float(np.mean(absolute_relative <= 0.10)),
        "fraction_within_0_20": float(np.mean(absolute_relative <= 0.20)),
        "sampling_stride_pixels": sampling_stride,
        "acceptance_threshold": None,
        "authority": (
            "model/seed consistency only; not overlap, pose, connectivity, scale, fusion, "
            "accuracy or accepted-geometry evidence"
        ),
    }


def _authorized_case_memberships() -> set[tuple[str, tuple[str, ...]]]:
    return {
        *(('single-view-metric', (camera_id,)) for camera_id in CAMERA_IDS),
        (
            "pose-conditioned-diagnostic",
            ("office-cam-01", "office-cam-02", "office-cam-03"),
        ),
        ("pose-conditioned-diagnostic", ("office-cam-01", "office-cam-03")),
        ("pose-conditioned-diagnostic", ("office-cam-03", "office-cam-02")),
    }


def _validate_intrinsic_matrix(value: Array) -> Array:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise P06EvidenceError("intrinsic matrix must be finite 3x3")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.isclose(matrix[2, 2], 1.0):
        raise P06EvidenceError("intrinsic matrix has invalid focal or homogeneous values")
    return matrix


def _empty_projection_result(reason: str) -> dict[str, Any]:
    return {
        "status": "unsupported-no-comparable-projection",
        "reason": reason,
        "acceptance_threshold": None,
        "authority": "none; unsupported diagnostic retained explicitly",
    }
