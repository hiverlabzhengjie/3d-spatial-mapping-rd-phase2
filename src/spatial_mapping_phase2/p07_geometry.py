"""Evidence-bounded camera geometry, filtering and fusion prerequisites for P07.

The module has no filesystem or viewer dependencies.  It keeps camera-space geometry separate
from provisional facility-frame copies and exposes a gate that can reject fusion without ever
performing it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

CAMERA_FRAME = "camera-opencv-x-right-y-down-z-forward"
FACILITY_FRAME = "facility-world-x-plan-left-y-plan-down-z-up"
METRE_UNITS = "metres"
PIXEL_CENTRE_CONVENTION = "integer (u, v) sample coordinates used by the P06 projection contract"
WORKING_GEOMETRY_AUTHORITY = "owner-approved-working-facility-geometry-v1"
WORKING_GEOMETRY_USE = "internal-rd-usability-geometry-not-survey-grade-xyz"
ALL4_DA3_CAMERA_ORDER = tuple(f"office-cam-0{index}" for index in range(1, 5))


class P07GeometryError(ValueError):
    """Raised when geometry or authority violates the frozen P07 contract."""


@dataclass(frozen=True)
class GeometryPatch:
    """One immutable, aligned point-cloud patch and its source-pixel attributes."""

    camera_id: str
    frame_id: str
    units: str
    points: Array
    colors_rgb: Array
    confidence: Array
    evaluation_mask_keep: Array
    pixel_uv: Array
    source_pixel_count: Array

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise P07GeometryError("camera_id must not be empty")
        if self.frame_id not in {CAMERA_FRAME, FACILITY_FRAME}:
            raise P07GeometryError("geometry frame_id is not a frozen P07 frame")
        if self.units != METRE_UNITS:
            raise P07GeometryError("P07 geometry units must be metres")
        points = _immutable_array(self.points, np.float64)
        colors = _immutable_array(self.colors_rgb, np.uint8)
        confidence = _immutable_array(self.confidence, np.float64)
        mask = _immutable_array(self.evaluation_mask_keep, np.bool_)
        pixels = _immutable_array(self.pixel_uv, np.int32)
        source_counts = _immutable_array(self.source_pixel_count, np.int32)
        count = len(points)
        if points.shape != (count, 3) or not np.all(np.isfinite(points)):
            raise P07GeometryError("points must be a finite Nx3 array")
        if colors.shape != (count, 3):
            raise P07GeometryError("colors_rgb must be an aligned Nx3 uint8 array")
        if confidence.shape != (count,) or not np.all(np.isfinite(confidence)):
            raise P07GeometryError("confidence must be an aligned finite vector")
        if mask.shape != (count,):
            raise P07GeometryError("evaluation_mask_keep must be an aligned vector")
        if pixels.shape != (count, 2):
            raise P07GeometryError("pixel_uv must be an aligned Nx2 vector")
        if source_counts.shape != (count,) or np.any(source_counts < 1):
            raise P07GeometryError("source_pixel_count must be positive and aligned")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors_rgb", colors)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evaluation_mask_keep", mask)
        object.__setattr__(self, "pixel_uv", pixels)
        object.__setattr__(self, "source_pixel_count", source_counts)

    @property
    def point_count(self) -> int:
        """Return the number of represented points."""

        return int(len(self.points))


@dataclass(frozen=True)
class FilterResult:
    """One versioned filter result plus exact count statistics."""

    operation: str
    parameters: Mapping[str, Any]
    patch: GeometryPatch
    input_point_count: int
    rejected_point_count: int

    def __post_init__(self) -> None:
        if not self.operation:
            raise P07GeometryError("filter operation must not be empty")
        if self.input_point_count < self.patch.point_count:
            raise P07GeometryError("a P07 filter cannot create points")
        if self.rejected_point_count != self.input_point_count - self.patch.point_count:
            raise P07GeometryError("filter rejection count is inconsistent")

    def statistics(self) -> dict[str, Any]:
        """Return a JSON-ready count record."""

        return {
            "operation": self.operation,
            "parameters": dict(self.parameters),
            "input_point_count": self.input_point_count,
            "output_point_count": self.patch.point_count,
            "rejected_point_count": self.rejected_point_count,
            "retained_fraction": (
                float(self.patch.point_count / self.input_point_count)
                if self.input_point_count
                else None
            ),
        }


@dataclass(frozen=True)
class FusionCandidate:
    """All explicit prerequisites for one proposed pairwise fusion."""

    camera_ids: tuple[str, str]
    strict_camera_statuses: Mapping[str, str]
    authorized_edges: frozenset[tuple[str, str]]
    transform_authorities: Mapping[str, str]
    frame_ids: Mapping[str, str]
    units: Mapping[str, str]
    independent_relative_pose_validated: bool
    structural_validation_passed: bool
    scale_alignment_authorized: bool

    def __post_init__(self) -> None:
        if len(set(self.camera_ids)) != 2:
            raise P07GeometryError("fusion candidate requires two distinct cameras")


@dataclass(frozen=True)
class FusionGateResult:
    """Rejection-first result; this type contains no fused geometry."""

    status: str
    camera_ids: tuple[str, str]
    rejection_reasons: tuple[str, ...]
    fused_artifact_created: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"eligible-for-separately-authorized-fusion", "rejected"}:
            raise P07GeometryError("unknown fusion gate status")
        if self.status == "rejected" and not self.rejection_reasons:
            raise P07GeometryError("rejected fusion gate must report a reason")
        if self.fused_artifact_created:
            raise P07GeometryError("the P07 gate is not allowed to create fused geometry")


@dataclass(frozen=True)
class DiagnosticFusedCandidate:
    """Non-operational deterministic concatenation with source membership preserved."""

    case_id: str
    camera_ids: tuple[str, ...]
    frame_id: str
    units: str
    points: Array
    colors_rgb: Array
    confidence: Array
    source_pixel_count: Array
    source_camera_index: Array
    operational_fusion_authorized: bool = False
    accepted_geometry: bool = False

    def __post_init__(self) -> None:
        if not self.case_id or len(self.camera_ids) < 2:
            raise P07GeometryError("diagnostic fusion requires a case and at least two cameras")
        if tuple(sorted(set(self.camera_ids))) != self.camera_ids:
            raise P07GeometryError("diagnostic fusion camera_ids must be unique and sorted")
        if self.frame_id != FACILITY_FRAME or self.units != METRE_UNITS:
            raise P07GeometryError("diagnostic fusion requires common facility-frame metre inputs")
        if self.operational_fusion_authorized or self.accepted_geometry:
            raise P07GeometryError(
                "D040 diagnostic fusion cannot authorize operational fusion or geometry"
            )
        points = _immutable_array(self.points, np.float64)
        colors = _immutable_array(self.colors_rgb, np.uint8)
        confidence = _immutable_array(self.confidence, np.float64)
        source_counts = _immutable_array(self.source_pixel_count, np.int32)
        membership = _immutable_array(self.source_camera_index, np.int16)
        count = len(points)
        if points.shape != (count, 3) or not np.all(np.isfinite(points)):
            raise P07GeometryError("diagnostic fusion points must be finite Nx3")
        if colors.shape != (count, 3) or confidence.shape != (count,):
            raise P07GeometryError("diagnostic fusion attributes must align with points")
        if source_counts.shape != (count,) or np.any(source_counts < 1):
            raise P07GeometryError("represented-source counts must be positive and aligned")
        if membership.shape != (count,) or np.any(membership < 0):
            raise P07GeometryError("source-camera membership must be aligned and non-negative")
        if count and int(np.max(membership)) >= len(self.camera_ids):
            raise P07GeometryError("source-camera membership index is out of range")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors_rgb", colors)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_pixel_count", source_counts)
        object.__setattr__(self, "source_camera_index", membership)

    @property
    def point_count(self) -> int:
        """Return the deterministic concatenated point count."""

        return int(len(self.points))

    @property
    def represented_source_pixel_count(self) -> int:
        """Return the total number of represented source pixels."""

        return int(np.sum(self.source_pixel_count))


@dataclass(frozen=True)
class WorkingFacilityGeometry:
    """D041 working geometry with reversible per-camera voxel contributions."""

    camera_ids: tuple[str, ...]
    frame_id: str
    units: str
    operation: str
    voxel_size_metres: float | None
    points: Array
    colors_rgb: Array
    confidence: Array
    input_point_count_by_camera: Array
    represented_source_pixel_count_by_camera: Array
    point_sum_by_camera: Array
    color_sum_by_camera: Array
    confidence_sum_by_camera: Array
    authority: str = WORKING_GEOMETRY_AUTHORITY
    intended_use: str = WORKING_GEOMETRY_USE

    def __post_init__(self) -> None:
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise P07GeometryError("working geometry camera order must be non-empty and unique")
        if self.frame_id != FACILITY_FRAME or self.units != METRE_UNITS:
            raise P07GeometryError("working geometry requires the facility metre frame")
        if not self.operation:
            raise P07GeometryError("working geometry operation must not be empty")
        if self.voxel_size_metres is not None and (
            not np.isfinite(self.voxel_size_metres) or self.voxel_size_metres <= 0
        ):
            raise P07GeometryError("working geometry voxel size must be finite and positive")
        if (
            self.authority != WORKING_GEOMETRY_AUTHORITY
            or self.intended_use != WORKING_GEOMETRY_USE
        ):
            raise P07GeometryError("working geometry authority labels must match D041")

        points = _immutable_array(self.points, np.float64)
        colors = _immutable_array(self.colors_rgb, np.uint8)
        confidence = _immutable_array(self.confidence, np.float64)
        input_counts = _immutable_array(self.input_point_count_by_camera, np.int64)
        represented_counts = _immutable_array(
            self.represented_source_pixel_count_by_camera, np.int64
        )
        point_sums = _immutable_array(self.point_sum_by_camera, np.float64)
        color_sums = _immutable_array(self.color_sum_by_camera, np.float64)
        confidence_sums = _immutable_array(self.confidence_sum_by_camera, np.float64)
        point_count = len(points)
        camera_count = len(self.camera_ids)
        matrix_shape = (point_count, camera_count)
        vector_sum_shape = (point_count, camera_count, 3)
        if points.shape != (point_count, 3) or not np.all(np.isfinite(points)):
            raise P07GeometryError("working geometry points must be finite Nx3")
        if colors.shape != (point_count, 3) or confidence.shape != (point_count,):
            raise P07GeometryError("working geometry attributes must align with points")
        if not np.all(np.isfinite(confidence)):
            raise P07GeometryError("working geometry confidence must be finite")
        if input_counts.shape != matrix_shape or np.any(input_counts < 0):
            raise P07GeometryError("working input-point membership counts are malformed")
        if represented_counts.shape != matrix_shape or np.any(represented_counts < 0):
            raise P07GeometryError("working represented-source counts are malformed")
        if point_sums.shape != vector_sum_shape or color_sums.shape != vector_sum_shape:
            raise P07GeometryError("working per-camera vector sums are malformed")
        if confidence_sums.shape != matrix_shape:
            raise P07GeometryError("working per-camera confidence sums are malformed")
        if not all(
            np.all(np.isfinite(value)) for value in (point_sums, color_sums, confidence_sums)
        ):
            raise P07GeometryError("working per-camera contribution sums must be finite")
        total_input_counts = np.sum(input_counts, axis=1)
        if np.any(total_input_counts < 1):
            raise P07GeometryError("every working output point must retain source membership")
        expected_points = np.sum(point_sums, axis=1) / total_input_counts[:, None]
        expected_colors = np.clip(
            np.rint(np.sum(color_sums, axis=1) / total_input_counts[:, None]), 0, 255
        ).astype(np.uint8)
        expected_confidence = np.sum(confidence_sums, axis=1) / total_input_counts
        if not np.allclose(points, expected_points, atol=1e-12, rtol=0):
            raise P07GeometryError("working points disagree with reversible camera contributions")
        if not np.array_equal(colors, expected_colors):
            raise P07GeometryError("working colors disagree with reversible camera contributions")
        if not np.allclose(confidence, expected_confidence, atol=1e-12, rtol=0):
            raise P07GeometryError(
                "working confidence disagrees with reversible camera contributions"
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors_rgb", colors)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "input_point_count_by_camera", input_counts)
        object.__setattr__(self, "represented_source_pixel_count_by_camera", represented_counts)
        object.__setattr__(self, "point_sum_by_camera", point_sums)
        object.__setattr__(self, "color_sum_by_camera", color_sums)
        object.__setattr__(self, "confidence_sum_by_camera", confidence_sums)

    @property
    def point_count(self) -> int:
        return int(len(self.points))

    @property
    def represented_source_pixel_count(self) -> int:
        return int(np.sum(self.represented_source_pixel_count_by_camera))

    @property
    def source_camera_membership(self) -> Array:
        membership = self.input_point_count_by_camera > 0
        membership.setflags(write=False)
        return membership


def concatenate_working_facility_geometry(
    patches: Mapping[str, GeometryPatch], camera_order: Sequence[str]
) -> WorkingFacilityGeometry:
    """Concatenate D041 inputs in the exact declared order with reversible membership."""

    camera_ids = tuple(camera_order)
    if not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise P07GeometryError("working concatenation order must be non-empty and unique")
    if set(patches) != set(camera_ids):
        raise P07GeometryError("working patches must exactly match the declared camera order")
    ordered = [patches[camera_id] for camera_id in camera_ids]
    for camera_id, patch in zip(camera_ids, ordered, strict=True):
        if patch.camera_id != camera_id:
            raise P07GeometryError("working patch key and camera_id disagree")
        if patch.frame_id != FACILITY_FRAME or patch.units != METRE_UNITS:
            raise P07GeometryError("working inputs must share the facility metre frame")
    points = np.concatenate([patch.points for patch in ordered], axis=0)
    point_count = len(points)
    camera_count = len(camera_ids)
    input_counts = np.zeros((point_count, camera_count), dtype=np.int64)
    represented_counts = np.zeros((point_count, camera_count), dtype=np.int64)
    point_sums = np.zeros((point_count, camera_count, 3), dtype=np.float64)
    color_sums = np.zeros((point_count, camera_count, 3), dtype=np.float64)
    confidence_sums = np.zeros((point_count, camera_count), dtype=np.float64)
    start = 0
    for camera_index, patch in enumerate(ordered):
        stop = start + patch.point_count
        input_counts[start:stop, camera_index] = 1
        represented_counts[start:stop, camera_index] = patch.source_pixel_count
        point_sums[start:stop, camera_index] = patch.points
        color_sums[start:stop, camera_index] = patch.colors_rgb
        confidence_sums[start:stop, camera_index] = patch.confidence
        start = stop
    return _working_geometry_from_contributions(
        camera_ids=camera_ids,
        operation="deterministic-concatenation",
        voxel_size_metres=None,
        input_counts=input_counts,
        represented_counts=represented_counts,
        point_sums=point_sums,
        color_sums=color_sums,
        confidence_sums=confidence_sums,
    )


def voxel_filter_working_geometry(
    geometry: WorkingFacilityGeometry, voxel_size_metres: float
) -> WorkingFacilityGeometry:
    """Apply one deterministic ordinary voxel-centroid filter and retain contributions."""

    if geometry.voxel_size_metres is not None:
        raise P07GeometryError("working geometry has already received its one voxel filter")
    if not np.isfinite(voxel_size_metres) or voxel_size_metres <= 0:
        raise P07GeometryError("working voxel size must be finite and positive")
    keys = np.floor(geometry.points / voxel_size_metres).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    group_count = int(np.max(inverse)) + 1
    camera_count = len(geometry.camera_ids)
    input_counts = np.zeros((group_count, camera_count), dtype=np.int64)
    represented_counts = np.zeros((group_count, camera_count), dtype=np.int64)
    point_sums = np.zeros((group_count, camera_count, 3), dtype=np.float64)
    color_sums = np.zeros((group_count, camera_count, 3), dtype=np.float64)
    confidence_sums = np.zeros((group_count, camera_count), dtype=np.float64)
    np.add.at(input_counts, inverse, geometry.input_point_count_by_camera)
    np.add.at(represented_counts, inverse, geometry.represented_source_pixel_count_by_camera)
    np.add.at(point_sums, inverse, geometry.point_sum_by_camera)
    np.add.at(color_sums, inverse, geometry.color_sum_by_camera)
    np.add.at(confidence_sums, inverse, geometry.confidence_sum_by_camera)
    return _working_geometry_from_contributions(
        camera_ids=geometry.camera_ids,
        operation="ordinary-voxel-centroid",
        voxel_size_metres=float(voxel_size_metres),
        input_counts=input_counts,
        represented_counts=represented_counts,
        point_sums=point_sums,
        color_sums=color_sums,
        confidence_sums=confidence_sums,
    )


def remove_camera_from_working_geometry(
    geometry: WorkingFacilityGeometry, camera_id: str
) -> WorkingFacilityGeometry:
    """Remove one camera and exactly recompute surviving mixed-voxel centroids."""

    if camera_id not in geometry.camera_ids:
        raise P07GeometryError(f"camera {camera_id!r} is absent from working geometry")
    if len(geometry.camera_ids) == 1:
        raise P07GeometryError("working geometry cannot remove its final camera")
    camera_index = geometry.camera_ids.index(camera_id)
    camera_ids = tuple(value for value in geometry.camera_ids if value != camera_id)
    input_counts = np.delete(geometry.input_point_count_by_camera, camera_index, axis=1)
    keep = np.sum(input_counts, axis=1) > 0
    return _working_geometry_from_contributions(
        camera_ids=camera_ids,
        operation=f"{geometry.operation}-remove-{camera_id}",
        voxel_size_metres=geometry.voxel_size_metres,
        input_counts=input_counts[keep],
        represented_counts=np.delete(
            geometry.represented_source_pixel_count_by_camera, camera_index, axis=1
        )[keep],
        point_sums=np.delete(geometry.point_sum_by_camera, camera_index, axis=1)[keep],
        color_sums=np.delete(geometry.color_sum_by_camera, camera_index, axis=1)[keep],
        confidence_sums=np.delete(geometry.confidence_sum_by_camera, camera_index, axis=1)[keep],
    )


def _working_geometry_from_contributions(
    *,
    camera_ids: tuple[str, ...],
    operation: str,
    voxel_size_metres: float | None,
    input_counts: Array,
    represented_counts: Array,
    point_sums: Array,
    color_sums: Array,
    confidence_sums: Array,
) -> WorkingFacilityGeometry:
    totals = np.sum(input_counts, axis=1)
    points = np.sum(point_sums, axis=1) / totals[:, None]
    colors = np.clip(np.rint(np.sum(color_sums, axis=1) / totals[:, None]), 0, 255).astype(
        np.uint8
    )
    confidence = np.sum(confidence_sums, axis=1) / totals
    return WorkingFacilityGeometry(
        camera_ids=camera_ids,
        frame_id=FACILITY_FRAME,
        units=METRE_UNITS,
        operation=operation,
        voxel_size_metres=voxel_size_metres,
        points=points,
        colors_rgb=colors,
        confidence=confidence,
        input_point_count_by_camera=input_counts,
        represented_source_pixel_count_by_camera=represented_counts,
        point_sum_by_camera=point_sums,
        color_sum_by_camera=color_sums,
        confidence_sum_by_camera=confidence_sums,
    )


def select_case_view_arrays(
    case_camera_ids: Sequence[str],
    camera_id: str,
    arrays: Mapping[str, Array],
) -> tuple[int, dict[str, Array]]:
    """Select one view without substituting another case's intrinsics or extrinsics."""

    ordered_ids = tuple(case_camera_ids)
    if len(set(ordered_ids)) != len(ordered_ids) or not ordered_ids:
        raise P07GeometryError("case camera ordering must be non-empty and unique")
    try:
        index = ordered_ids.index(camera_id)
    except ValueError as error:
        raise P07GeometryError(f"camera {camera_id!r} is absent from the posed case") from error
    required = {"depth", "confidence", "intrinsics", "processed_images", "extrinsics"}
    if set(arrays) != required:
        raise P07GeometryError("posed raw arrays do not match the frozen required field set")
    selected: dict[str, Array] = {}
    for name in sorted(required):
        value = np.asarray(arrays[name])
        if value.ndim < 1 or value.shape[0] != len(ordered_ids):
            raise P07GeometryError(f"posed field {name!r} does not align with case ordering")
        item = value[index].copy()
        item.setflags(write=False)
        selected[name] = item
    validate_processed_intrinsics(selected["intrinsics"])
    return index, selected


def concatenate_diagnostic_candidate(
    case_id: str, patches: Mapping[str, GeometryPatch]
) -> DiagnosticFusedCandidate:
    """Concatenate facility patches deterministically without merging or inventing points."""

    camera_ids = tuple(sorted(patches))
    if len(camera_ids) < 2:
        raise P07GeometryError("diagnostic fusion requires at least two unmerged inputs")
    ordered = [patches[camera_id] for camera_id in camera_ids]
    for camera_id, patch in zip(camera_ids, ordered, strict=True):
        if patch.camera_id != camera_id:
            raise P07GeometryError("diagnostic fusion patch key and camera_id disagree")
        if patch.frame_id != FACILITY_FRAME or patch.units != METRE_UNITS:
            raise P07GeometryError("diagnostic fusion inputs must share the facility metre frame")
    membership = np.concatenate(
        [np.full(patch.point_count, index, dtype=np.int16) for index, patch in enumerate(ordered)]
    )
    return DiagnosticFusedCandidate(
        case_id=case_id,
        camera_ids=camera_ids,
        frame_id=FACILITY_FRAME,
        units=METRE_UNITS,
        points=np.concatenate([patch.points for patch in ordered], axis=0),
        colors_rgb=np.concatenate([patch.colors_rgb for patch in ordered], axis=0),
        confidence=np.concatenate([patch.confidence for patch in ordered], axis=0),
        source_pixel_count=np.concatenate([patch.source_pixel_count for patch in ordered], axis=0),
        source_camera_index=membership,
    )


def validate_d040_prohibited_operations(
    controls: Mapping[str, bool],
) -> dict[str, bool]:
    """Reject every correction/refinement path prohibited by the D040 amendment."""

    required = {
        "ICP",
        "pose_refinement",
        "camera_movement",
        "scale_or_alignment_correction",
        "surface_completion",
        "invented_points",
        "filter_retuning",
    }
    if set(controls) != required:
        raise P07GeometryError("D040 processing controls do not match the frozen prohibition set")
    enabled = sorted(name for name, value in controls.items() if value)
    if enabled:
        raise P07GeometryError(
            f"D040 prohibits enabled correction/refinement operations: {', '.join(enabled)}"
        )
    return {name: False for name in sorted(required)}


def nearest_surface_diagnostic(
    source: GeometryPatch,
    target: GeometryPatch,
    coverage_thresholds_metres: Sequence[float] = (0.05, 0.10, 0.25, 0.50),
) -> dict[str, Any]:
    """Measure directional nearest-point distances without correspondence or alignment."""

    if source.frame_id != FACILITY_FRAME or target.frame_id != FACILITY_FRAME:
        raise P07GeometryError("nearest-surface diagnostics require facility-frame patches")
    if source.units != METRE_UNITS or target.units != METRE_UNITS:
        raise P07GeometryError("nearest-surface diagnostics require metre units")
    if source.point_count == 0 or target.point_count == 0:
        raise P07GeometryError("nearest-surface diagnostics require non-empty patches")
    thresholds = np.asarray(tuple(coverage_thresholds_metres), dtype=np.float64)
    if (
        thresholds.ndim != 1
        or len(thresholds) == 0
        or np.any(~np.isfinite(thresholds))
        or np.any(thresholds <= 0)
        or np.any(np.diff(thresholds) <= 0)
    ):
        raise P07GeometryError("coverage thresholds must be finite, positive and increasing")
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]

    distances, _ = cKDTree(target.points).query(source.points, k=1, workers=1)
    values = np.asarray(distances, dtype=np.float64)
    return {
        "source_camera_id": source.camera_id,
        "target_camera_id": target.camera_id,
        "source_point_count": source.point_count,
        "target_point_count": target.point_count,
        "distance_metres": {
            "minimum": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
        },
        "coverage_fraction": {
            f"within_{threshold:.2f}_metres": float(np.mean(values <= threshold))
            for threshold in thresholds
        },
        "alignment_or_correspondence_applied": False,
        "acceptance_threshold": None,
    }


def cross_view_nearest_surface_diagnostics(
    patches: Mapping[str, GeometryPatch],
) -> list[dict[str, Any]]:
    """Return both directional diagnostics for every camera pair in stable order."""

    results: list[dict[str, Any]] = []
    for camera_a, camera_b in combinations(sorted(patches), 2):
        results.append(nearest_surface_diagnostic(patches[camera_a], patches[camera_b]))
        results.append(nearest_surface_diagnostic(patches[camera_b], patches[camera_a]))
    return results


def plan_extent_support(patch: GeometryPatch, polygon_xy: Array) -> dict[str, Any]:
    """Report XY support inside the supplied plan-display extent without claiming structure."""

    if patch.frame_id != FACILITY_FRAME:
        raise P07GeometryError("plan-extent support requires a facility-frame patch")
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if polygon.shape == (5, 2) and np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise P07GeometryError("plan extent must be an Nx2 polygon with at least three vertices")
    if not np.all(np.isfinite(polygon)):
        raise P07GeometryError("plan extent must be finite")
    if patch.point_count == 0:
        return {
            "status": "unsupported-empty-patch",
            "point_count": 0,
            "fraction_xy_inside_plan_extent": None,
            "acceptance_threshold": None,
        }
    x = patch.points[:, 0]
    y = patch.points[:, 1]
    inside = np.zeros(patch.point_count, dtype=bool)
    xj, yj = polygon[-1]
    for xi, yi in polygon:
        crosses = (yi > y) != (yj > y)
        denominator = yj - yi
        x_intersection = (xj - xi) * (y - yi) / (
            denominator if abs(denominator) > np.finfo(float).eps else np.finfo(float).eps
        ) + xi
        inside ^= crosses & (x < x_intersection)
        xj, yj = xi, yi
    return {
        "status": "display-extent-support-only",
        "point_count": patch.point_count,
        "inside_point_count": int(np.sum(inside)),
        "fraction_xy_inside_plan_extent": float(np.mean(inside)),
        "reference": "P02 scanned-plan raster affine display extent",
        "vector_structure_support": False,
        "acceptance_threshold": None,
    }


def validate_processed_intrinsics(value: Array) -> Array:
    """Validate and return an immutable exact processed intrinsic matrix."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise P07GeometryError("processed intrinsics must be finite 3x3")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise P07GeometryError("processed focal lengths must be positive")
    if not np.allclose(matrix[2], np.array([0.0, 0.0, 1.0]), atol=1e-12):
        raise P07GeometryError("processed intrinsics have invalid homogeneous row")
    result = matrix.copy()
    result.setflags(write=False)
    return result


def validate_all4_da3_camera_order(camera_ids: Sequence[str]) -> tuple[str, ...]:
    """Require the owner-requested Camera 1/2/3/4 diagnostic order exactly."""

    ordered = tuple(camera_ids)
    if ordered != ALL4_DA3_CAMERA_ORDER:
        raise P07GeometryError("all-four-view DA3 diagnostic requires exact Camera 1/2/3/4 order")
    return ordered


def validate_scene_da3_camera_order(camera_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate a future scene-authoritative DA3 roster without office-specific assumptions."""

    ordered = tuple(camera_ids)
    if not ordered:
        raise P07GeometryError("scene DA3 camera roster must not be empty")
    if len(set(ordered)) != len(ordered):
        raise P07GeometryError("scene DA3 camera roster must contain unique camera IDs")
    if any(not camera_id.strip() for camera_id in ordered):
        raise P07GeometryError("scene DA3 camera IDs must not be blank")
    return ordered


def concatenate_scene_da3_candidate(
    case_id: str,
    patches: Mapping[str, GeometryPatch],
    camera_order: Sequence[str],
) -> WorkingFacilityGeometry:
    """Concatenate the complete scene roster, including a valid one-camera scene.

    The reversible working-geometry representation supports one or many views and preserves the
    declared scene order. Historical P07 diagnostic types and their stricter all-four rules remain
    unchanged.
    """

    ordered = validate_scene_da3_camera_order(camera_order)
    if set(patches) != set(ordered):
        raise P07GeometryError("scene DA3 patches must exactly match the declared camera roster")
    del case_id
    return concatenate_working_facility_geometry(patches, ordered)


def T_world_from_da3_T_camera_from_world(value: Array) -> Array:
    """Invert one DA3 world-to-camera extrinsic into explicit camera-to-world form."""

    extrinsic = np.asarray(value, dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, np.array([0.0, 0.0, 0.0, 1.0])))
    if extrinsic.shape != (4, 4) or not np.all(np.isfinite(extrinsic)):
        raise P07GeometryError("DA3 T_camera_from_world must be finite 3x4 or 4x4")
    if not np.allclose(extrinsic[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise P07GeometryError("DA3 T_camera_from_world has an invalid homogeneous row")
    rotation = extrinsic[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise P07GeometryError("DA3 T_camera_from_world rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise P07GeometryError("DA3 T_camera_from_world rotation is not proper")
    return validate_T_world_from_camera(np.linalg.inv(extrinsic))


def validate_T_world_from_camera(value: Array) -> Array:
    """Validate a proper homogeneous camera-to-facility rigid transform."""

    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise P07GeometryError("T_world_from_camera must be finite 4x4")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise P07GeometryError("T_world_from_camera has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise P07GeometryError("T_world_from_camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise P07GeometryError("T_world_from_camera rotation is not proper")
    result = transform.copy()
    result.setflags(write=False)
    return result


def validate_frozen_working_transforms(
    expected: Mapping[str, Array], current: Mapping[str, Array]
) -> dict[str, Array]:
    """Require exact D041 transform identity; numerical closeness cannot hide pose mutation."""

    if not expected or set(expected) != set(current):
        raise P07GeometryError("frozen working transform camera sets must match exactly")
    validated: dict[str, Array] = {}
    for camera_id in sorted(expected):
        expected_transform = validate_T_world_from_camera(expected[camera_id])
        current_transform = validate_T_world_from_camera(current[camera_id])
        if not np.array_equal(expected_transform, current_transform):
            raise P07GeometryError(f"frozen working transform changed for {camera_id}")
        validated[camera_id] = current_transform
    return validated


def back_project_depth(
    camera_id: str,
    depth_metres: Array,
    processed_intrinsics: Array,
    colors_rgb: Array,
    confidence: Array,
    evaluation_mask_keep: Array,
) -> GeometryPatch:
    """Back-project every finite positive depth before confidence/range/mask filtering.

    Pixel coordinates are the integer sample coordinates already used in P06 diagnostics.  The
    evaluation mask is carried as an attribute so applying it remains a separately versioned
    derivative rather than silently changing raw camera-space geometry.
    """

    depth = np.asarray(depth_metres, dtype=np.float64)
    color = np.asarray(colors_rgb)
    score = np.asarray(confidence, dtype=np.float64)
    mask = np.asarray(evaluation_mask_keep, dtype=bool)
    if depth.ndim != 2:
        raise P07GeometryError("depth must be two-dimensional")
    if color.shape != (*depth.shape, 3):
        raise P07GeometryError("color image must match depth and have three channels")
    if score.shape != depth.shape or mask.shape != depth.shape:
        raise P07GeometryError("confidence and evaluation mask must match depth")
    if color.dtype != np.uint8:
        raise P07GeometryError("processed color image must be uint8 RGB")
    K = validate_processed_intrinsics(processed_intrinsics)
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(score)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (v.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    points = np.column_stack((x, y, z))
    return GeometryPatch(
        camera_id=camera_id,
        frame_id=CAMERA_FRAME,
        units=METRE_UNITS,
        points=points,
        colors_rgb=color[v, u],
        confidence=score[v, u],
        evaluation_mask_keep=mask[v, u],
        pixel_uv=np.column_stack((u, v)),
        source_pixel_count=np.ones(len(points), dtype=np.int32),
    )


def p06_processed_image_to_rgb(processed_image: Array) -> Array:
    """Validate P06/DA3 processed-image RGB without changing channel order.

    DA3's ``_add_processed_images`` denormalizes tensors whose channel order is RGB. P06 preserves
    that returned uint8 array verbatim, so P07 must not apply an OpenCV BGR conversion.
    """

    image = np.asarray(processed_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise P07GeometryError("P06 processed image must be an HxWx3 uint8 RGB array")
    result = np.ascontiguousarray(image).copy()
    result.setflags(write=False)
    return result


def filter_by_confidence(patch: GeometryPatch, minimum_score: float) -> FilterResult:
    """Keep finite model-confidence scores at or above a declared diagnostic threshold."""

    if not np.isfinite(minimum_score):
        raise P07GeometryError("confidence threshold must be finite")
    return _filter_patch(
        patch,
        patch.confidence >= minimum_score,
        "confidence-minimum",
        {"minimum_score": float(minimum_score), "calibrated_probability": False},
    )


def filter_by_range(
    patch: GeometryPatch, minimum_depth_metres: float, maximum_depth_metres: float
) -> FilterResult:
    """Keep points within declared camera-depth bounds."""

    if (
        not np.isfinite(minimum_depth_metres)
        or not np.isfinite(maximum_depth_metres)
        or minimum_depth_metres < 0
        or maximum_depth_metres <= minimum_depth_metres
    ):
        raise P07GeometryError("range bounds must be finite, non-negative and ordered")
    depth = patch.points[:, 2]
    return _filter_patch(
        patch,
        (depth >= minimum_depth_metres) & (depth <= maximum_depth_metres),
        "camera-z-range",
        {
            "minimum_depth_metres": float(minimum_depth_metres),
            "maximum_depth_metres": float(maximum_depth_metres),
        },
    )


def filter_by_evaluation_mask(patch: GeometryPatch) -> FilterResult:
    """Apply the exact resized P06 evaluation/dynamic-exclusion mask."""

    return _filter_patch(
        patch,
        patch.evaluation_mask_keep,
        "p06-evaluation-mask",
        {"unsupported_pixels_remain_absent": True},
    )


def voxel_downsample(patch: GeometryPatch, voxel_size_metres: float) -> FilterResult:
    """Deterministically reduce each occupied voxel to one attribute-averaged point."""

    if not np.isfinite(voxel_size_metres) or voxel_size_metres <= 0:
        raise P07GeometryError("voxel size must be finite and positive")
    if patch.point_count == 0:
        return FilterResult(
            operation="voxel-centroid",
            parameters={"voxel_size_metres": float(voxel_size_metres)},
            patch=patch,
            input_point_count=0,
            rejected_point_count=0,
        )
    keys = np.floor(patch.points / voxel_size_metres).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    group_count = int(np.max(inverse)) + 1
    counts = np.bincount(inverse, minlength=group_count).astype(np.int32)
    points = _group_mean(patch.points, inverse, group_count)
    colors = np.clip(
        np.rint(_group_mean(patch.colors_rgb.astype(np.float64), inverse, group_count)),
        0,
        255,
    ).astype(np.uint8)
    confidence = _group_mean(patch.confidence[:, None], inverse, group_count)[:, 0]
    mask = (
        np.bincount(
            inverse, weights=patch.evaluation_mask_keep.astype(np.float64), minlength=group_count
        )
        == counts
    )
    representative = np.full(group_count, patch.point_count, dtype=np.int64)
    np.minimum.at(representative, inverse, np.arange(patch.point_count))
    source_counts = np.bincount(
        inverse, weights=patch.source_pixel_count, minlength=group_count
    ).astype(np.int32)
    derived = GeometryPatch(
        camera_id=patch.camera_id,
        frame_id=patch.frame_id,
        units=patch.units,
        points=points,
        colors_rgb=colors,
        confidence=confidence,
        evaluation_mask_keep=mask,
        pixel_uv=patch.pixel_uv[representative],
        source_pixel_count=source_counts,
    )
    return FilterResult(
        operation="voxel-centroid",
        parameters={
            "voxel_size_metres": float(voxel_size_metres),
            "attribute_aggregation": "mean; pixel_uv is lowest original aligned index",
        },
        patch=derived,
        input_point_count=patch.point_count,
        rejected_point_count=patch.point_count - derived.point_count,
    )


def statistical_outlier_filter(
    patch: GeometryPatch, neighbour_count: int, standard_deviation_ratio: float
) -> FilterResult:
    """Use Open3D's established statistical outlier operation and retain aligned attributes."""

    if neighbour_count < 2:
        raise P07GeometryError("outlier neighbour_count must be at least two")
    if not np.isfinite(standard_deviation_ratio) or standard_deviation_ratio <= 0:
        raise P07GeometryError("outlier standard_deviation_ratio must be finite and positive")
    if patch.point_count <= neighbour_count:
        raise P07GeometryError("outlier filter has too few points for the requested neighbours")
    try:
        import open3d as o3d  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - selected runtime has the pinned package
        raise P07GeometryError("pinned Open3D runtime is unavailable") from error
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(patch.points.copy())
    _, retained = cloud.remove_statistical_outlier(
        nb_neighbors=neighbour_count,
        std_ratio=standard_deviation_ratio,
        print_progress=False,
    )
    keep = np.zeros(patch.point_count, dtype=bool)
    keep[np.asarray(retained, dtype=np.int64)] = True
    return _filter_patch(
        patch,
        keep,
        "open3d-statistical-outlier",
        {
            "neighbour_count": neighbour_count,
            "standard_deviation_ratio": float(standard_deviation_ratio),
            "library": "open3d",
        },
    )


def transform_to_provisional_facility(
    patch: GeometryPatch, T_world_from_camera: Array
) -> GeometryPatch:
    """Create a facility-frame copy without changing camera-space source geometry."""

    if patch.frame_id != CAMERA_FRAME:
        raise P07GeometryError("only raw/derived camera-frame patches may be transformed")
    transform = validate_T_world_from_camera(T_world_from_camera)
    points = patch.points @ transform[:3, :3].T + transform[:3, 3]
    return GeometryPatch(
        camera_id=patch.camera_id,
        frame_id=FACILITY_FRAME,
        units=patch.units,
        points=points,
        colors_rgb=patch.colors_rgb,
        confidence=patch.confidence,
        evaluation_mask_keep=patch.evaluation_mask_keep,
        pixel_uv=patch.pixel_uv,
        source_pixel_count=patch.source_pixel_count,
    )


def floor_plane_diagnostic(patch: GeometryPatch) -> dict[str, Any]:
    """Report facility Z distributions without pretending every point is a floor sample."""

    if patch.frame_id != FACILITY_FRAME:
        raise P07GeometryError("floor diagnostic requires a facility-frame patch")
    if patch.point_count == 0:
        return {
            "status": "unsupported-empty-patch",
            "reference": "finished floor Z=0 metres",
            "acceptance_threshold": None,
        }
    z = patch.points[:, 2]
    return {
        "status": "distribution-only-no-floor-segmentation",
        "reference": "finished floor Z=0 metres",
        "point_count": patch.point_count,
        "signed_z_metres": {
            "minimum": float(np.min(z)),
            "p01": float(np.quantile(z, 0.01)),
            "p05": float(np.quantile(z, 0.05)),
            "median": float(np.median(z)),
            "p95": float(np.quantile(z, 0.95)),
            "maximum": float(np.max(z)),
        },
        "fraction_below_floor": float(np.mean(z < 0.0)),
        "acceptance_threshold": None,
        "authority": "structural diagnostic only; points are not classified as floor",
    }


def unavailable_structural_diagnostics() -> dict[str, Any]:
    """Describe facility references absent from the selected P02 vector export."""

    unsupported = {
        "status": "unsupported-reference-geometry-absent",
        "residual": None,
        "acceptance_threshold": None,
    }
    return {
        "walls": {
            **unsupported,
            "reason": "selected P02 revision 3 contains a scanned raster, not vector wall planes",
        },
        "columns": {
            **unsupported,
            "reason": (
                "origin names a pillar corner, but the selected export contains no complete "
                "column polygon and rectangular short-side evidence is missing"
            ),
        },
        "ceiling": {
            **unsupported,
            "reason": "no ceiling plane is encoded in the selected P02 revision-3 export",
        },
    }


def evaluate_fusion_gate(candidate: FusionCandidate) -> FusionGateResult:
    """Evaluate all prerequisites and reject if any authority or evidence is absent."""

    reasons: list[str] = []
    camera_a, camera_b = candidate.camera_ids
    for camera_id in candidate.camera_ids:
        status = candidate.strict_camera_statuses.get(camera_id)
        if status != "accepted":
            reasons.append(f"{camera_id} strict registration status is {status!r}, not 'accepted'")
        transform_authority = candidate.transform_authorities.get(camera_id)
        if transform_authority != "accepted-registration":
            reasons.append(
                f"{camera_id} transform authority is {transform_authority!r}, not "
                "'accepted-registration'"
            )
        if candidate.frame_ids.get(camera_id) != FACILITY_FRAME:
            reasons.append(f"{camera_id} patch is not in the explicit facility frame")
        if candidate.units.get(camera_id) != METRE_UNITS:
            reasons.append(f"{camera_id} patch units are not metres")
    edge = tuple(sorted((camera_a, camera_b)))
    normalized_edges = {tuple(sorted(item)) for item in candidate.authorized_edges}
    if edge not in normalized_edges:
        reasons.append(f"no authorized operational connectivity edge exists for {edge}")
    if not candidate.independent_relative_pose_validated:
        reasons.append("independent relative-pose/connectivity validation is absent")
    if not candidate.structural_validation_passed:
        reasons.append("structural/overlap validation has not passed")
    if not candidate.scale_alignment_authorized:
        reasons.append("scale/alignment authority is absent")
    return FusionGateResult(
        status="rejected" if reasons else "eligible-for-separately-authorized-fusion",
        camera_ids=candidate.camera_ids,
        rejection_reasons=tuple(reasons),
    )


def patch_bounds(patch: GeometryPatch) -> dict[str, Any]:
    """Return finite axis-aligned bounds with explicit frame and units."""

    if patch.point_count == 0:
        return {"status": "empty", "frame_id": patch.frame_id, "units": patch.units}
    return {
        "status": "measured-diagnostic",
        "frame_id": patch.frame_id,
        "units": patch.units,
        "minimum_xyz": np.min(patch.points, axis=0).tolist(),
        "maximum_xyz": np.max(patch.points, axis=0).tolist(),
    }


def _filter_patch(
    patch: GeometryPatch,
    keep: Array,
    operation: str,
    parameters: Mapping[str, Any],
) -> FilterResult:
    selected = np.asarray(keep, dtype=bool)
    if selected.shape != (patch.point_count,):
        raise P07GeometryError("filter selection must align with the point patch")
    derived = GeometryPatch(
        camera_id=patch.camera_id,
        frame_id=patch.frame_id,
        units=patch.units,
        points=patch.points[selected],
        colors_rgb=patch.colors_rgb[selected],
        confidence=patch.confidence[selected],
        evaluation_mask_keep=patch.evaluation_mask_keep[selected],
        pixel_uv=patch.pixel_uv[selected],
        source_pixel_count=patch.source_pixel_count[selected],
    )
    return FilterResult(
        operation=operation,
        parameters=parameters,
        patch=derived,
        input_point_count=patch.point_count,
        rejected_point_count=patch.point_count - derived.point_count,
    )


def _immutable_array(value: Array, dtype: Any) -> Array:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _group_mean(values: Array, inverse: Array, group_count: int) -> Array:
    source = np.asarray(values, dtype=np.float64)
    sums = np.zeros((group_count, source.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, source)
    counts = np.bincount(inverse, minlength=group_count).astype(np.float64)
    return sums / counts[:, None]
