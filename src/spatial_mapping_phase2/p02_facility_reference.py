"""Evidence-bound P02 facility-frame and plan-control contracts.

The module deliberately distinguishes metric plan controls from a scan-pixel display transform.
It can record candidates and rejection states, but it cannot manufacture a reviewed facility
network when printed dimensions or independent physical checks are missing.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from spatial_mapping_phase2.p01_observability import CameraOwnerInput

P02_SCHEMA_VERSION = "p02-facility-reference-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FEATURE_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]*")


class P02ContractError(ValueError):
    """Raised when P02 evidence is incomplete, ambiguous, or violates its authority boundary."""


class FrameReviewState(StrEnum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class EvidenceStatus(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    PROVISIONAL = "provisional"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"


class LandmarkRole(StrEnum):
    CANDIDATE = "candidate"
    CONTROL = "control"
    SOLVE = "solve"
    HELD_OUT = "held-out"
    EXCLUDED = "excluded"


class SpotCheckCoverage(StrEnum):
    OVERLAPPING_CAMERA_AREA = "overlapping-camera-area"
    ISOLATED_CAMERA_4_AREA = "isolated-camera-4-area"


@dataclass(frozen=True, slots=True)
class Point3Metres:
    """A world-frame point in metres; coordinates are always `(X, Y, Z)`."""

    x_metres: float
    y_metres: float
    z_metres: float

    def __post_init__(self) -> None:
        _require_finite(self.x_metres, "x_metres")
        _require_finite(self.y_metres, "y_metres")
        _require_finite(self.z_metres, "z_metres")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Immutable, non-secret source identity; display derivatives are never metric authority."""

    source_id: str
    sha256: str
    source_kind: str
    authority_note: str

    def __post_init__(self) -> None:
        _require_id(self.source_id, "source_id")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise P02ContractError("sha256 must be a lowercase SHA-256 digest")
        _require_non_blank(self.source_kind, "source_kind")
        _require_non_blank(self.authority_note, "authority_note")


@dataclass(frozen=True, slots=True)
class FacilityFrameDefinition:
    """A right-handed world frame tied to one exact physical feature, never to a scan pixel."""

    schema_version: str
    frame_id: str
    origin_feature_id: str
    origin_feature_meaning: str
    x_direction_meaning: str
    y_direction_meaning: str
    z_direction_meaning: str
    units: str
    floor_reference: str
    x_axis: tuple[int, int, int]
    y_axis: tuple[int, int, int]
    z_axis: tuple[int, int, int]
    review_state: FrameReviewState
    sources: tuple[SourceIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != P02_SCHEMA_VERSION:
            raise P02ContractError("unsupported facility frame schema_version")
        _require_id(self.frame_id, "frame_id")
        _require_id(self.origin_feature_id, "origin_feature_id")
        for value, name in (
            (self.origin_feature_meaning, "origin_feature_meaning"),
            (self.x_direction_meaning, "x_direction_meaning"),
            (self.y_direction_meaning, "y_direction_meaning"),
            (self.z_direction_meaning, "z_direction_meaning"),
            (self.floor_reference, "floor_reference"),
        ):
            _require_non_blank(value, name)
        if self.units != "metres":
            raise P02ContractError("facility frame units must be metres")
        if self.z_axis != (0, 0, 1):
            raise P02ContractError("facility +Z must be upward `(0, 0, 1)`")
        if _cross(self.x_axis, self.y_axis) != self.z_axis:
            raise P02ContractError("facility axes must be right-handed: +X cross +Y equals +Z")
        if _dot(self.x_axis, self.y_axis) != 0 or _dot(self.x_axis, self.z_axis) != 0:
            raise P02ContractError("facility axes must be orthogonal unit directions")
        if _dot(self.y_axis, self.z_axis) != 0 or not self.sources:
            raise P02ContractError("facility axes require source-bound orthogonal evidence")
        if self.review_state == FrameReviewState.REVIEWED and "corner" not in (
            self.origin_feature_meaning.lower()
        ):
            raise P02ContractError("a reviewed origin must name the exact physical corner")


@dataclass(frozen=True, slots=True)
class DimensionChain:
    """Printed plan dimensions in metres, independently of pixel spacing."""

    chain_id: str
    source: SourceIdentity
    segment_lengths_metres: tuple[float, ...]
    segment_uncertainties_metres: tuple[float, ...]
    declared_total_metres: float
    declared_total_uncertainty_metres: float
    feature_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.chain_id, "chain_id")
        if not self.segment_lengths_metres or len(self.segment_lengths_metres) != len(
            self.segment_uncertainties_metres
        ):
            raise P02ContractError(
                "dimension chain needs one uncertainty per non-empty segment list"
            )
        if len(self.feature_ids) != len(self.segment_lengths_metres) + 1:
            raise P02ContractError("dimension chain needs one more feature than segments")
        for value in self.segment_lengths_metres:
            _require_positive(value, "dimension segment length")
        for value in self.segment_uncertainties_metres:
            _require_non_negative(value, "dimension segment uncertainty")
        _require_positive(self.declared_total_metres, "declared dimension total")
        _require_non_negative(self.declared_total_uncertainty_metres, "declared total uncertainty")
        for feature_id in self.feature_ids:
            _require_id(feature_id, "dimension feature_id")


@dataclass(frozen=True, slots=True)
class DimensionClosure:
    """Reported closure rather than a silently accepted scan/dimension mismatch."""

    chain_id: str
    residual_metres: float
    combined_uncertainty_metres: float
    consistent_within_uncertainty: bool


def assess_dimension_chain(chain: DimensionChain) -> DimensionClosure:
    """Compare a printed total to its printed segments using declared uncertainties."""

    residual = sum(chain.segment_lengths_metres) - chain.declared_total_metres
    combined_uncertainty = combine_independent_uncertainty(
        *chain.segment_uncertainties_metres, chain.declared_total_uncertainty_metres
    )
    return DimensionClosure(
        chain_id=chain.chain_id,
        residual_metres=residual,
        combined_uncertainty_metres=combined_uncertainty,
        consistent_within_uncertainty=abs(residual) <= combined_uncertainty,
    )


@dataclass(frozen=True, slots=True)
class ControlPoint:
    """Permanent structural point derived from a dimension/control network."""

    feature_id: str
    meaning: str
    world: Point3Metres
    horizontal_uncertainty_metres: float | None
    vertical_uncertainty_metres: float | None
    status: EvidenceStatus
    sources: tuple[SourceIdentity, ...]

    def __post_init__(self) -> None:
        _require_id(self.feature_id, "control feature_id")
        _require_non_blank(self.meaning, "control meaning")
        _require_optional_non_negative(
            self.horizontal_uncertainty_metres, "horizontal uncertainty"
        )
        _require_optional_non_negative(self.vertical_uncertainty_metres, "vertical uncertainty")
        if self.status == EvidenceStatus.ACCEPTED and not self.sources:
            raise P02ContractError("accepted control points require source identities")


@dataclass(frozen=True, slots=True)
class RectangularPillarDimensions:
    """Physical dimensions needed before a corner can locate a rectangular pillar centre."""

    long_side_metres: float
    long_side_uncertainty_metres: float
    short_side_metres: float | None
    short_side_uncertainty_metres: float | None
    source: SourceIdentity

    def __post_init__(self) -> None:
        _require_positive(self.long_side_metres, "rectangular pillar long side")
        _require_non_negative(
            self.long_side_uncertainty_metres,
            "rectangular pillar long-side uncertainty",
        )
        if self.short_side_metres is not None:
            _require_positive(self.short_side_metres, "rectangular pillar short side")
        _require_optional_non_negative(
            self.short_side_uncertainty_metres,
            "rectangular pillar short-side uncertainty",
        )
        if (self.short_side_metres is None) != (self.short_side_uncertainty_metres is None):
            raise P02ContractError(
                "rectangular pillar short side and its uncertainty must be supplied together"
            )

    def require_resolved(self) -> tuple[float, float]:
        """Return both sides or reject coordinate derivation from incomplete geometry."""

        if self.short_side_metres is None:
            raise P02ContractError(
                "rectangular pillar short side is required before deriving X coordinates "
                "from the origin or Camera-1 pillar"
            )
        return self.long_side_metres, self.short_side_metres


@dataclass(frozen=True, slots=True)
class IndependentSpotCheck:
    """Physical tape/laser check that independently constrains a plan region."""

    check_id: str
    coverage: SpotCheckCoverage
    start_feature_id: str
    end_feature_id: str
    measured_distance_metres: float
    measurement_uncertainty_metres: float | None
    plan_distance_metres: float
    plan_uncertainty_metres: float | None
    method: str
    source: SourceIdentity
    status: EvidenceStatus

    def __post_init__(self) -> None:
        _require_id(self.check_id, "spot-check id")
        _require_id(self.start_feature_id, "spot-check start_feature_id")
        _require_id(self.end_feature_id, "spot-check end_feature_id")
        if self.start_feature_id == self.end_feature_id:
            raise P02ContractError("spot check must use two distinct permanent features")
        for value, name in (
            (self.measured_distance_metres, "measured distance"),
            (self.plan_distance_metres, "plan distance"),
        ):
            _require_positive(value, name)
        _require_optional_non_negative(
            self.measurement_uncertainty_metres, "measurement uncertainty"
        )
        _require_optional_non_negative(self.plan_uncertainty_metres, "plan uncertainty")
        _require_non_blank(self.method, "spot-check method")

    @property
    def residual_metres(self) -> float:
        return self.measured_distance_metres - self.plan_distance_metres

    @property
    def combined_uncertainty_metres(self) -> float | None:
        if self.measurement_uncertainty_metres is None or self.plan_uncertainty_metres is None:
            return None
        return combine_independent_uncertainty(
            self.measurement_uncertainty_metres, self.plan_uncertainty_metres
        )

    @property
    def consistent_within_uncertainty(self) -> bool | None:
        combined = self.combined_uncertainty_metres
        if combined is None:
            return None
        return abs(self.residual_metres) <= combined


@dataclass(frozen=True, slots=True)
class CeilingHeightCheck:
    """Finished-floor-to-finished-ceiling measurement required by D010."""

    height_metres: float
    uncertainty_metres: float
    location_feature_id: str
    method: str
    source: SourceIdentity
    status: EvidenceStatus

    def __post_init__(self) -> None:
        _require_positive(self.height_metres, "ceiling height")
        _require_non_negative(self.uncertainty_metres, "ceiling-height uncertainty")
        _require_id(self.location_feature_id, "ceiling location_feature_id")
        _require_non_blank(self.method, "ceiling-height method")


def require_d010_checks(
    ceiling: CeilingHeightCheck | None, spot_checks: Iterable[IndependentSpotCheck]
) -> None:
    """Reject incomplete, unreviewed, or inconsistent D010 physical evidence."""

    if ceiling is None or ceiling.status != EvidenceStatus.ACCEPTED:
        raise P02ContractError("D010 requires an accepted finished-floor-to-ceiling height check")
    accepted = tuple(check for check in spot_checks if check.status == EvidenceStatus.ACCEPTED)
    coverage = {
        check.coverage for check in accepted if check.consistent_within_uncertainty is True
    }
    missing = {
        SpotCheckCoverage.OVERLAPPING_CAMERA_AREA,
        SpotCheckCoverage.ISOLATED_CAMERA_4_AREA,
    } - coverage
    if missing:
        labels = ", ".join(sorted(item.value for item in missing))
        raise P02ContractError(
            f"D010 lacks accepted consistent independent spot checks for {labels}"
        )


@dataclass(frozen=True, slots=True)
class DisplayCorrespondence:
    """A metric control matched to a scan pixel solely for display registration."""

    feature_id: str
    world_x_metres: float
    world_y_metres: float
    pixel_u: float
    pixel_v: float

    def __post_init__(self) -> None:
        _require_id(self.feature_id, "display correspondence feature_id")
        for value, name in (
            (self.world_x_metres, "world_x_metres"),
            (self.world_y_metres, "world_y_metres"),
            (self.pixel_u, "pixel_u"),
            (self.pixel_v, "pixel_v"),
        ):
            _require_finite(value, name)


@dataclass(frozen=True, slots=True)
class PlanDisplayTransform:
    """Affine `T_plan_display_pixel_from_world`; never use it as metric authority."""

    transform_name: str
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    rms_residual_pixels: float
    maximum_residual_pixels: float
    correspondence_count: int

    def __post_init__(self) -> None:
        if self.transform_name != "T_plan_display_pixel_from_world":
            raise P02ContractError("display transform must use T_plan_display_pixel_from_world")
        for value, name in (
            (self.a, "a"),
            (self.b, "b"),
            (self.c, "c"),
            (self.d, "d"),
            (self.e, "e"),
            (self.f, "f"),
            (self.rms_residual_pixels, "rms residual"),
            (self.maximum_residual_pixels, "maximum residual"),
        ):
            _require_finite(value, name)
        if self.rms_residual_pixels < 0 or self.maximum_residual_pixels < 0:
            raise P02ContractError("display residuals must be non-negative")
        if self.correspondence_count < 3:
            raise P02ContractError("display transform requires at least three correspondences")
        if abs(self.a * self.e - self.b * self.d) < 1e-12:
            raise P02ContractError("display transform must be invertible")

    def pixel_from_world(self, x_metres: float, y_metres: float) -> tuple[float, float]:
        """Apply display-only `T_plan_display_pixel_from_world`."""

        return (
            self.a * x_metres + self.b * y_metres + self.c,
            self.d * x_metres + self.e * y_metres + self.f,
        )

    def world_from_pixel(self, pixel_u: float, pixel_v: float) -> tuple[float, float]:
        """Apply the named inverse of the display transform, without creating a metric control."""

        determinant = self.a * self.e - self.b * self.d
        u = pixel_u - self.c
        v = pixel_v - self.f
        return ((self.e * u - self.b * v) / determinant, (-self.d * u + self.a * v) / determinant)


def fit_plan_display_transform(
    correspondences: Iterable[DisplayCorrespondence],
) -> PlanDisplayTransform:
    """Least-squares affine display fit with retained residuals and no metric inference."""

    points = tuple(correspondences)
    if len(points) < 3:
        raise P02ContractError("display transform requires at least three correspondences")
    normal = [[0.0] * 3 for _ in range(3)]
    rhs_u = [0.0, 0.0, 0.0]
    rhs_v = [0.0, 0.0, 0.0]
    for point in points:
        row = (point.world_x_metres, point.world_y_metres, 1.0)
        for row_index in range(3):
            rhs_u[row_index] += row[row_index] * point.pixel_u
            rhs_v[row_index] += row[row_index] * point.pixel_v
            for column_index in range(3):
                normal[row_index][column_index] += row[row_index] * row[column_index]
    a, b, c = _solve_3x3(normal, rhs_u)
    d, e, f = _solve_3x3(normal, rhs_v)
    transform = PlanDisplayTransform(
        "T_plan_display_pixel_from_world", a, b, c, d, e, f, 0.0, 0.0, len(points)
    )
    residuals = tuple(
        math.hypot(
            transform.pixel_from_world(point.world_x_metres, point.world_y_metres)[0]
            - point.pixel_u,
            transform.pixel_from_world(point.world_x_metres, point.world_y_metres)[1]
            - point.pixel_v,
        )
        for point in points
    )
    return PlanDisplayTransform(
        transform.transform_name,
        a,
        b,
        c,
        d,
        e,
        f,
        math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        max(residuals),
        len(points),
    )


@dataclass(frozen=True, slots=True)
class StructuralLandmark:
    """Permanent-feature registry separated by role and evidence status."""

    feature_id: str
    meaning: str
    role: LandmarkRole
    status: EvidenceStatus
    permanent: bool
    world: Point3Metres | None
    sources: tuple[SourceIdentity, ...]
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.feature_id, "landmark feature_id")
        _require_non_blank(self.meaning, "landmark meaning")
        if self.permanent is False and self.status in {
            EvidenceStatus.ACCEPTED,
            EvidenceStatus.PROVISIONAL,
        }:
            raise P02ContractError("movable features cannot be accepted or provisional landmarks")
        if self.role in {LandmarkRole.CONTROL, LandmarkRole.SOLVE, LandmarkRole.HELD_OUT} and (
            self.world is None or self.status != EvidenceStatus.ACCEPTED or not self.permanent
        ):
            raise P02ContractError(
                "control, solve, and held-out landmarks need accepted permanent XYZ"
            )
        if self.role == LandmarkRole.EXCLUDED and not self.exclusion_reason:
            raise P02ContractError("excluded landmark requires an exclusion reason")
        if self.status == EvidenceStatus.UNSUPPORTED and self.exclusion_reason is None:
            raise P02ContractError("unsupported landmark requires an explicit reason")


@dataclass(frozen=True, slots=True)
class MountingPointPrior:
    """P02 mounting reference, explicitly distinct from camera optical-centre pose."""

    camera_id: str
    physical_label: str
    reference_feature_id: str
    reference_meaning: str
    c_world_mount_prior: Point3Metres
    plan_control_residual_metres: float | None
    height_uncertainty_metres: float
    unmeasured_lens_to_reference_offset_metres: float | None
    combined_uncertainty_metres: float | None
    sources: tuple[SourceIdentity, ...]
    status: EvidenceStatus

    def __post_init__(self) -> None:
        if not re.fullmatch(r"office-cam-0[1-4]", self.camera_id):
            raise P02ContractError("mounting prior camera_id must be a fixed office camera ID")
        _require_non_blank(self.physical_label, "physical_label")
        _require_id(self.reference_feature_id, "reference_feature_id")
        _require_non_blank(self.reference_meaning, "reference_meaning")
        _require_optional_non_negative(self.plan_control_residual_metres, "plan-control residual")
        _require_non_negative(self.height_uncertainty_metres, "height uncertainty")
        _require_optional_non_negative(
            self.unmeasured_lens_to_reference_offset_metres,
            "lens-to-reference offset",
        )
        _require_optional_non_negative(self.combined_uncertainty_metres, "combined uncertainty")
        components = (
            self.plan_control_residual_metres,
            self.height_uncertainty_metres,
            self.unmeasured_lens_to_reference_offset_metres,
        )
        if None in components:
            if self.combined_uncertainty_metres is not None:
                raise P02ContractError(
                    "combined mounting uncertainty must remain unknown when a component is unknown"
                )
        else:
            known_components = tuple(float(value) for value in components if value is not None)
            expected = combine_independent_uncertainty(*known_components)
            if self.combined_uncertainty_metres is None or not math.isclose(
                self.combined_uncertainty_metres, expected, abs_tol=1e-12
            ):
                raise P02ContractError(
                    "combined mounting uncertainty must preserve its decomposition"
                )
        if self.status != EvidenceStatus.PROVISIONAL:
            raise P02ContractError("C_world_mount_prior must remain provisional in P02")
        if not self.sources:
            raise P02ContractError("mounting prior requires source identities")


def derive_mounting_point_prior(
    camera: CameraOwnerInput,
    anchor: ControlPoint,
    plan_control_residual_metres: float | None,
    lens_to_reference_offset_metres: float | None,
    sources: tuple[SourceIdentity, ...],
) -> MountingPointPrior:
    """Combine reviewed plan XY with P01 owner height without estimating a lens or pose."""

    if camera.mounting is None:
        raise P02ContractError("mounting prior needs an owner mounting-height observation")
    if anchor.status not in {EvidenceStatus.ACCEPTED, EvidenceStatus.PROVISIONAL}:
        raise P02ContractError("mounting prior needs an accepted or provisional plan anchor")
    _require_optional_non_negative(plan_control_residual_metres, "plan-control residual")
    _require_optional_non_negative(lens_to_reference_offset_metres, "lens-to-reference offset")
    mounting = camera.mounting
    combined = None
    if plan_control_residual_metres is not None and lens_to_reference_offset_metres is not None:
        combined = combine_independent_uncertainty(
            plan_control_residual_metres,
            mounting.uncertainty_metres,
            lens_to_reference_offset_metres,
        )
    return MountingPointPrior(
        camera_id=camera.identity.camera_id,
        physical_label=camera.identity.physical_label,
        reference_feature_id=anchor.feature_id,
        reference_meaning=anchor.meaning,
        c_world_mount_prior=Point3Metres(
            anchor.world.x_metres, anchor.world.y_metres, mounting.height_metres
        ),
        plan_control_residual_metres=plan_control_residual_metres,
        height_uncertainty_metres=mounting.uncertainty_metres,
        unmeasured_lens_to_reference_offset_metres=lens_to_reference_offset_metres,
        combined_uncertainty_metres=combined,
        sources=sources,
        status=EvidenceStatus.PROVISIONAL,
    )


def fingerprint_file(path: str | Path) -> str:
    """Return the SHA-256 identity of a preserved local source or derivative."""

    try:
        with Path(path).open("rb") as source_file:
            return hashlib.file_digest(source_file, "sha256").hexdigest()
    except OSError as error:
        raise P02ContractError(
            "source file could not be read for SHA-256 fingerprinting"
        ) from error


def combine_independent_uncertainty(*uncertainties_metres: float) -> float:
    """Root-sum-square independent one-sigma components in metres."""

    for uncertainty in uncertainties_metres:
        _require_non_negative(uncertainty, "uncertainty")
    return math.sqrt(sum(value * value for value in uncertainties_metres))


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float]:
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for pivot_index in range(3):
        pivot_row = max(
            range(pivot_index, 3), key=lambda index: abs(augmented[index][pivot_index])
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise P02ContractError("display correspondences are geometrically degenerate")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, 4):
            augmented[pivot_index][column_index] /= pivot
        for row_index in range(3):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, 4):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]
    return (augmented[0][3], augmented[1][3], augmented[2][3])


def _cross(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return sum(left[index] * right[index] for index in range(3))


def _require_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _FEATURE_ID_PATTERN.fullmatch(value):
        raise P02ContractError(f"{name} must use lowercase hyphenated identifier syntax")


def _require_non_blank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise P02ContractError(f"{name} must be non-blank")


def _require_finite(value: float, name: str) -> None:
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise P02ContractError(f"{name} must be finite")


def _require_non_negative(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0:
        raise P02ContractError(f"{name} must be non-negative")


def _require_optional_non_negative(value: float | None, name: str) -> None:
    if value is not None:
        _require_non_negative(value, name)


def _require_positive(value: float, name: str) -> None:
    _require_non_negative(value, name)
    if value <= 0:
        raise P02ContractError(f"{name} must be positive")
