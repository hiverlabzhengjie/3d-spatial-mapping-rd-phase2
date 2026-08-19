"""One-off D032 ChArUco diagnostics for board-free intrinsic policy selection.

The board is an independent checkpoint reference only. Candidate intrinsics are immutable during
fixed-profile evaluation; only one board pose is estimated for each accepted image.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, median
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from spatial_mapping_phase2.p04_intrinsic_fleet import CameraIntrinsicEstimate
from spatial_mapping_phase2.p04_pose_domain import CameraIntrinsics

FloatArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]

CALIBRATION_FLAGS = (
    cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
    | cv2.CALIB_FIX_K4
    | cv2.CALIB_FIX_K5
    | cv2.CALIB_FIX_K6
)


class CharucoCheckpointError(ValueError):
    """Raised when checkpoint evidence is malformed or insufficient."""


@dataclass(frozen=True, slots=True)
class CharucoBoardSpec:
    identity: str
    dictionary_name: str
    squares_x: int
    squares_y: int
    square_length_metres: float
    marker_length_metres: float

    def __post_init__(self) -> None:
        if not self.identity.strip():
            raise CharucoCheckpointError("board identity must be non-blank")
        if self.dictionary_name not in {"DICT_5X5_100"}:
            raise CharucoCheckpointError("unsupported or unverified board dictionary")
        if min(self.squares_x, self.squares_y) < 2:
            raise CharucoCheckpointError("board must contain at least two squares per dimension")
        if not 0 < self.marker_length_metres < self.square_length_metres:
            raise CharucoCheckpointError("marker length must be positive and smaller than square")

    @property
    def maximum_charuco_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def create_board(self) -> Any:
        dictionary_id = getattr(cv2.aruco, self.dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_metres,
            self.marker_length_metres,
            dictionary,
        )


@dataclass(frozen=True, slots=True)
class NaturalFrameMetrics:
    corner_count: int
    line_count: int
    occupied_region_count: int
    sharpness_laplacian_variance: float
    exposure_score: float

    def __post_init__(self) -> None:
        values = (
            self.corner_count,
            self.line_count,
            self.occupied_region_count,
            self.sharpness_laplacian_variance,
            self.exposure_score,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise CharucoCheckpointError("natural-frame metrics must be finite")
        if min(self.corner_count, self.line_count, self.occupied_region_count) < 0:
            raise CharucoCheckpointError("natural-frame counts cannot be negative")
        if not 0 <= self.exposure_score <= 1:
            raise CharucoCheckpointError("exposure score must be between zero and one")


@dataclass(frozen=True, slots=True)
class NaturalCameraEvidence:
    camera_id: str
    frame_metrics: tuple[NaturalFrameMetrics, ...]
    within_camera_focal_cv: float
    converged: bool
    finite_output: bool

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or not self.frame_metrics:
            raise CharucoCheckpointError("camera evidence requires identity and frames")
        if not math.isfinite(self.within_camera_focal_cv) or self.within_camera_focal_cv < 0:
            raise CharucoCheckpointError("within-camera focal CV must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NaturalQualityScore:
    camera_id: str
    score: float
    eligible: bool
    components: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CharucoObservation:
    frame_id: str
    image_width_pixels: int
    image_height_pixels: int
    corner_ids: tuple[int, ...]
    image_points: tuple[tuple[float, float], ...]
    coverage_fraction: float
    occupied_region_count: int
    sharpness_laplacian_variance: float
    mean_luma: float
    clipped_fraction: float
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise CharucoCheckpointError("observation frame identity is required")
        if min(self.image_width_pixels, self.image_height_pixels) <= 0:
            raise CharucoCheckpointError("observation image dimensions must be positive")
        if len(self.corner_ids) != len(self.image_points):
            raise CharucoCheckpointError("corner IDs and image points must have equal counts")
        if len(set(self.corner_ids)) != len(self.corner_ids):
            raise CharucoCheckpointError("corner IDs must be unique within an observation")
        if not all(0 <= value <= 1 for value in (self.coverage_fraction, self.clipped_fraction)):
            raise CharucoCheckpointError("coverage and clipping fractions must be bounded")
        if not 0 <= self.mean_luma <= 255:
            raise CharucoCheckpointError("mean luma must be between zero and 255")

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class FixedProfileEvaluation:
    label: str
    per_view_median_errors_pixels: tuple[float, ...]
    per_view_max_errors_pixels: tuple[float, ...]
    all_errors_pixels: tuple[float, ...]
    edge_errors_pixels: tuple[float, ...]
    board_tilt_degrees: tuple[float, ...] = ()
    board_distance_metres: tuple[float, ...] = ()

    @property
    def median_error_pixels(self) -> float:
        return median(self.all_errors_pixels)

    @property
    def maximum_error_pixels(self) -> float:
        return max(self.all_errors_pixels)


@dataclass(frozen=True, slots=True)
class CharucoReference:
    intrinsics: CameraIntrinsics
    rms_error_pixels: float
    per_view_errors_pixels: tuple[float, ...]
    intrinsic_standard_deviations: tuple[float, ...]


def natural_frame_metrics(image: UInt8Array) -> NaturalFrameMetrics:
    """Extract simple natural-image screening evidence, not accuracy evidence."""

    gray = _gray(image)
    corners = cv2.goodFeaturesToTrack(gray, 400, 0.01, 8)
    corner_count = 0 if corners is None else len(corners)
    points = np.empty((0, 2), dtype=np.float64)
    if corners is not None:
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    regions = _occupied_regions(points, gray.shape[1], gray.shape[0])
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=max(20, min(gray.shape) // 20),
        maxLineGap=10,
    )
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    return NaturalFrameMetrics(
        corner_count,
        0 if lines is None else len(lines),
        regions,
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        1.0 - clipped,
    )


def rank_natural_camera_quality(
    evidence: Sequence[NaturalCameraEvidence],
) -> tuple[NaturalQualityScore, ...]:
    """Rank eligible cameras using visible equal-weight normalized screening components."""

    if len(evidence) < 2 or len({item.camera_id for item in evidence}) != len(evidence):
        raise CharucoCheckpointError("quality ranking requires distinct evidence for two cameras")
    aggregates: dict[str, tuple[float, ...]] = {}
    for item in evidence:
        frames = item.frame_metrics
        aggregates[item.camera_id] = (
            median(tuple(float(frame.corner_count) for frame in frames)),
            median(tuple(float(frame.line_count) for frame in frames)),
            median(tuple(float(frame.occupied_region_count) for frame in frames)),
            median(tuple(frame.sharpness_laplacian_variance for frame in frames)),
            median(tuple(frame.exposure_score for frame in frames)),
            -item.within_camera_focal_cv,
        )
    columns = tuple(zip(*aggregates.values(), strict=True))
    normalized_columns = tuple(_minmax(column) for column in columns)
    output: list[NaturalQualityScore] = []
    for row_index, item in enumerate(evidence):
        components = {
            name: normalized_columns[index][row_index]
            for index, name in enumerate(
                ("corners", "lines", "regions", "sharpness", "exposure", "stability")
            )
        }
        eligible = item.converged and item.finite_output
        score = fmean(components.values()) if eligible else 0.0
        output.append(NaturalQualityScore(item.camera_id, score, eligible, components))
    return tuple(sorted(output, key=lambda item: (-item.score, item.camera_id)))


def robust_intrinsic_cluster(
    estimates: Sequence[CameraIntrinsicEstimate], *, robust_z_limit: float = 3.0
) -> tuple[CameraIntrinsicEstimate, ...]:
    """Reject multivariate robust outliers with one equal vote per compatible camera."""

    if robust_z_limit <= 0:
        raise CharucoCheckpointError("robust z limit must be positive")
    if len(estimates) < 3 or len({item.camera_id for item in estimates}) != len(estimates):
        raise CharucoCheckpointError("robust cluster requires at least three distinct cameras")
    compatibility = {
        (
            item.profile_version,
            item.model,
            item.width_pixels,
            item.height_pixels,
            len(item.distortion),
        )
        for item in estimates
    }
    if len(compatibility) != 1:
        raise CharucoCheckpointError("intrinsic cluster inputs are not compatible")
    rows = np.asarray([item.normalized_parameters() for item in estimates], dtype=np.float64)
    centre = np.median(rows, axis=0)
    mad = np.median(np.abs(rows - centre), axis=0)
    scale = 1.4826 * mad
    informative = scale > 1e-12
    if not bool(np.any(informative)):
        return tuple(estimates)
    robust_z = np.max(
        np.abs(rows[:, informative] - centre[informative]) / scale[informative], axis=1
    )
    retained = tuple(
        item
        for item, distance in zip(estimates, robust_z, strict=True)
        if distance <= robust_z_limit
    )
    if len(retained) < 3:
        raise CharucoCheckpointError("robust filtering left fewer than three compatible cameras")
    return retained


def median_cluster_profile(
    estimates: Sequence[CameraIntrinsicEstimate], *, method: str = "robust-cluster-median"
) -> CameraIntrinsics:
    if len(estimates) < 3:
        raise CharucoCheckpointError("cluster profile requires at least three cameras")
    rows = tuple(item.normalized_parameters() for item in estimates)
    width = estimates[0].width_pixels
    height = estimates[0].height_pixels
    pooled = tuple(median(tuple(row[index] for row in rows)) for index in range(len(rows[0])))
    del method
    return CameraIntrinsics(
        estimates[0].model,
        width,
        height,
        pooled[0] * width,
        pooled[1] * height,
        pooled[2] * width,
        pooled[3] * height,
        pooled[4:],
    )


def detect_charuco_observation(
    frame_id: str,
    image: UInt8Array,
    board_spec: CharucoBoardSpec,
    *,
    minimum_corners: int = 8,
    minimum_coverage_fraction: float = 0.01,
    minimum_sharpness: float = 0.0,
    maximum_clipped_fraction: float = 1.0,
) -> CharucoObservation:
    if (
        minimum_corners < 4
        or not 0 <= minimum_coverage_fraction <= 1
        or minimum_sharpness < 0
        or not 0 <= maximum_clipped_fraction <= 1
    ):
        raise CharucoCheckpointError("invalid observation acceptance thresholds")
    gray = _gray(image)
    board = board_spec.create_board()
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    points = np.empty((0, 2), dtype=np.float64)
    ids: tuple[int, ...] = ()
    if charuco_corners is not None and charuco_ids is not None:
        points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        ids = tuple(int(value) for value in np.asarray(charuco_ids).reshape(-1))
    coverage = _convex_hull_coverage(points, gray.shape[1], gray.shape[0])
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    reasons: list[str] = []
    if len(ids) < minimum_corners:
        reasons.append("insufficient-charuco-corners")
    if coverage < minimum_coverage_fraction:
        reasons.append("insufficient-image-coverage")
    if sharpness < minimum_sharpness:
        reasons.append("insufficient-sharpness")
    if clipped > maximum_clipped_fraction:
        reasons.append("excessive-clipping")
    return CharucoObservation(
        frame_id,
        gray.shape[1],
        gray.shape[0],
        ids,
        tuple((float(point[0]), float(point[1])) for point in points),
        coverage,
        _occupied_regions(points, gray.shape[1], gray.shape[0]),
        sharpness,
        float(np.mean(gray)),
        clipped,
        tuple(reasons),
    )


def calibrate_charuco_reference(
    observations: Sequence[CharucoObservation], board_spec: CharucoBoardSpec
) -> CharucoReference:
    accepted = _accepted_observations(observations, minimum_views=6)
    size = _common_image_size(accepted)
    board_points = np.asarray(board_spec.create_board().getChessboardCorners(), dtype=np.float32)
    object_points: list[npt.NDArray[np.float32]] = []
    image_points: list[npt.NDArray[np.float32]] = []
    for item in accepted:
        object_points.append(board_points[list(item.corner_ids)].reshape(-1, 1, 3))
        image_points.append(np.asarray(item.image_points, dtype=np.float32).reshape(-1, 1, 2))
    try:
        result = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            size,
            np.eye(3, dtype=np.float64),
            np.zeros((5, 1), dtype=np.float64),
            flags=CALIBRATION_FLAGS,
        )
    except cv2.error as error:
        raise CharucoCheckpointError("OpenCV ChArUco reference calibration failed") from error
    rms, camera_matrix, distortion, _, _, std_intrinsic, _, per_view = result
    values = np.concatenate(
        (
            np.asarray(camera_matrix, dtype=np.float64).reshape(-1),
            np.asarray(distortion, dtype=np.float64).reshape(-1),
            np.asarray(per_view, dtype=np.float64).reshape(-1),
        )
    )
    if not np.all(np.isfinite(values)):
        raise CharucoCheckpointError("ChArUco reference produced non-finite values")
    intrinsics = CameraIntrinsics(
        "simple_radial",
        size[0],
        size[1],
        float(camera_matrix[0, 0]),
        float(camera_matrix[1, 1]),
        float(camera_matrix[0, 2]),
        float(camera_matrix[1, 2]),
        (float(np.asarray(distortion).reshape(-1)[0]),),
    )
    return CharucoReference(
        intrinsics,
        float(rms),
        tuple(float(value) for value in np.asarray(per_view).reshape(-1)),
        tuple(float(value) for value in np.asarray(std_intrinsic).reshape(-1)),
    )


def evaluate_fixed_intrinsics(
    label: str,
    intrinsics: CameraIntrinsics,
    observations: Sequence[CharucoObservation],
    board_spec: CharucoBoardSpec,
) -> FixedProfileEvaluation:
    accepted = _accepted_observations(observations, minimum_views=1)
    if (intrinsics.width_pixels, intrinsics.height_pixels) != _common_image_size(accepted):
        raise CharucoCheckpointError("candidate and observation image dimensions differ")
    board_points = np.asarray(board_spec.create_board().getChessboardCorners(), dtype=np.float64)
    camera_matrix, distortion = _opencv_intrinsics(intrinsics)
    per_median: list[float] = []
    per_max: list[float] = []
    all_errors: list[float] = []
    edge_errors: list[float] = []
    tilts: list[float] = []
    distances: list[float] = []
    for item in accepted:
        object_points = board_points[list(item.corner_ids)]
        image_points = np.asarray(item.image_points, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not success:
            raise CharucoCheckpointError(f"fixed-intrinsic board pose failed for {item.frame_id}")
        rotation, _ = cv2.Rodrigues(rvec)
        cosine = max(-1.0, min(1.0, abs(float(rotation[2, 2]))))
        tilts.append(math.degrees(math.acos(cosine)))
        distances.append(float(np.linalg.norm(tvec)))
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        if not np.all(np.isfinite(errors)):
            raise CharucoCheckpointError("fixed-intrinsic reprojection produced non-finite errors")
        per_median.append(float(np.median(errors)))
        per_max.append(float(np.max(errors)))
        all_errors.extend(float(value) for value in errors)
        centre = np.array([item.image_width_pixels / 2, item.image_height_pixels / 2])
        radius = np.linalg.norm((image_points - centre) / centre, axis=1)
        edge_errors.extend(float(value) for value in errors[radius >= 0.65])
    if not all_errors:
        raise CharucoCheckpointError("fixed-intrinsic evaluation has no corner errors")
    return FixedProfileEvaluation(
        label,
        tuple(per_median),
        tuple(per_max),
        tuple(all_errors),
        tuple(edge_errors),
        tuple(tilts),
        tuple(distances),
    )


def select_policy(
    candidate_a: FixedProfileEvaluation,
    candidate_b: FixedProfileEvaluation,
    *,
    adequacy_limit_pixels: float,
    material_tie_pixels: float,
) -> str:
    """Apply the deterministic D032 preference and explicit inadequate-evidence fallback."""

    if adequacy_limit_pixels <= 0 or material_tie_pixels < 0:
        raise CharucoCheckpointError("policy thresholds must be non-negative and finite")
    a_error = candidate_a.median_error_pixels
    b_error = candidate_b.median_error_pixels
    if min(a_error, b_error) > adequacy_limit_pixels:
        return "fleet-prior-with-camera-specific-rollback"
    if b_error <= a_error + material_tie_pixels:
        return "candidate-b-robust-cluster"
    if a_error + material_tie_pixels < b_error:
        return "candidate-a-quality-selected"
    return "fleet-prior-with-camera-specific-rollback"


def _accepted_observations(
    observations: Sequence[CharucoObservation], *, minimum_views: int
) -> tuple[CharucoObservation, ...]:
    accepted = tuple(item for item in observations if item.accepted)
    if len(accepted) < minimum_views:
        raise CharucoCheckpointError(
            f"at least {minimum_views} accepted ChArUco views are required"
        )
    return accepted


def _common_image_size(observations: Sequence[CharucoObservation]) -> tuple[int, int]:
    sizes = {(item.image_width_pixels, item.image_height_pixels) for item in observations}
    if len(sizes) != 1:
        raise CharucoCheckpointError("accepted observations must share one image size")
    return next(iter(sizes))


def _opencv_intrinsics(intrinsics: CameraIntrinsics) -> tuple[FloatArray, FloatArray]:
    if intrinsics.model not in {"pinhole", "simple_radial", "radial"}:
        raise CharucoCheckpointError("fixed ChArUco evaluation requires OpenCV radial model")
    matrix = np.array(
        [
            [intrinsics.fx_pixels, 0.0, intrinsics.cx_pixels],
            [0.0, intrinsics.fy_pixels, intrinsics.cy_pixels],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    distortion[: len(intrinsics.distortion)] = intrinsics.distortion
    return matrix, distortion


def _gray(image: UInt8Array) -> UInt8Array:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in {2, 3}:
        raise CharucoCheckpointError("images must be uint8 grayscale or BGR arrays")
    if array.ndim == 2:
        return np.asarray(array, dtype=np.uint8)
    if array.shape[2] != 3:
        raise CharucoCheckpointError("colour images must have three BGR channels")
    return np.asarray(cv2.cvtColor(array, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def _occupied_regions(points: FloatArray, width: int, height: int) -> int:
    if not len(points):
        return 0
    x = np.clip((points[:, 0] / width * 3).astype(int), 0, 2)
    y = np.clip((points[:, 1] / height * 3).astype(int), 0, 2)
    return len(set(zip(x.tolist(), y.tolist(), strict=True)))


def _convex_hull_coverage(points: FloatArray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull) / (width * height))


def _minmax(values: Sequence[float]) -> tuple[float, ...]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return tuple(1.0 for _ in values)
    return tuple((value - low) / (high - low) for value in values)
