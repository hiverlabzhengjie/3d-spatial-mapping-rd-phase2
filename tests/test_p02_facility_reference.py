from dataclasses import replace
from pathlib import Path

import pytest

from spatial_mapping_phase2.p01_observability import (
    CAMERA_ENDPOINT_KEYS,
    ApproximatePlanObservation,
    CameraIdentity,
    CameraOwnerInput,
    MountingObservation,
)
from spatial_mapping_phase2.p02_facility_reference import (
    P02_SCHEMA_VERSION,
    CeilingHeightCheck,
    ControlPoint,
    DimensionChain,
    DisplayCorrespondence,
    EvidenceStatus,
    FacilityFrameDefinition,
    FrameReviewState,
    IndependentSpotCheck,
    LandmarkRole,
    P02ContractError,
    Point3Metres,
    RectangularPillarDimensions,
    SourceIdentity,
    SpotCheckCoverage,
    StructuralLandmark,
    assess_dimension_chain,
    combine_independent_uncertainty,
    derive_mounting_point_prior,
    fingerprint_file,
    fit_plan_display_transform,
    require_d010_checks,
)


def _source(source_id: str = "original-plan") -> SourceIdentity:
    return SourceIdentity(
        source_id, "a" * 64, "scanned-plan", "observation; not coordinate authority"
    )


def _frame(state: FrameReviewState = FrameReviewState.REVIEWED) -> FacilityFrameDefinition:
    return FacilityFrameDefinition(
        P02_SCHEMA_VERSION,
        "facility-world-v1",
        "purple-pillar-nw-corner",
        "northwest corner of the owner-marked solid structural pillar",
        "plan-left from the reviewed origin",
        "plan-down from the reviewed origin",
        "upward from finished floor",
        "metres",
        "finished floor is Z=0",
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, 1),
        state,
        (_source(),),
    )


def _anchor() -> ControlPoint:
    return ControlPoint(
        "purple-pillar-nw-corner",
        "northwest corner of the owner-marked solid structural pillar",
        Point3Metres(0.0, 0.0, 0.0),
        0.03,
        0.0,
        EvidenceStatus.ACCEPTED,
        (_source(),),
    )


def _owner_camera() -> CameraOwnerInput:
    return CameraOwnerInput(
        CameraIdentity(
            "office-cam-01",
            "near pantry (red)",
            CAMERA_ENDPOINT_KEYS["office-cam-01"],
            "stream-profile-v1",
        ),
        ApproximatePlanObservation("owner mark on pillar", "not an optical centre"),
        MountingObservation(3.0, 0.1, "owner tape measurement"),
        None,
        None,
        "unknown",
    )


def _spot_check(coverage: SpotCheckCoverage) -> IndependentSpotCheck:
    return IndependentSpotCheck(
        f"{coverage.value}-check",
        coverage,
        "permanent-corner-a",
        "permanent-corner-b",
        4.02,
        0.02,
        4.0,
        0.03,
        "tape between named permanent corners",
        _source("spot-check"),
        EvidenceStatus.ACCEPTED,
    )


def test_candidate_frame_is_metre_based_and_right_handed() -> None:
    frame = _frame()

    assert frame.units == "metres"
    assert frame.z_axis == (0, 0, 1)
    assert frame.floor_reference == "finished floor is Z=0"
    with pytest.raises(P02ContractError, match="right-handed"):
        replace(frame, y_axis=(0, 1, 0))


def test_dimension_chain_reports_closure_and_inconsistency() -> None:
    chain = DimensionChain(
        "top-exterior-chain",
        _source(),
        (4.0, 2.5, 1.5),
        (0.01, 0.01, 0.01),
        8.0,
        0.01,
        ("a", "b", "c", "d"),
    )
    closure = assess_dimension_chain(chain)

    assert closure.residual_metres == pytest.approx(0.0)
    assert closure.consistent_within_uncertainty is True
    assert (
        assess_dimension_chain(
            replace(chain, declared_total_metres=8.8)
        ).consistent_within_uncertainty
        is False
    )


def test_rectangular_pillar_requires_both_sides_before_coordinate_derivation() -> None:
    unresolved = RectangularPillarDimensions(0.480, 0.020, None, None, _source())

    with pytest.raises(P02ContractError, match="short side is required"):
        unresolved.require_resolved()

    resolved = RectangularPillarDimensions(0.480, 0.020, 0.250, 0.020, _source())
    assert resolved.require_resolved() == (0.480, 0.250)
    with pytest.raises(P02ContractError, match="supplied together"):
        RectangularPillarDimensions(0.480, 0.020, 0.250, None, _source())


def test_display_transform_recovers_synthetic_pixel_mapping_and_inverse() -> None:
    correspondences = (
        DisplayCorrespondence("a", 0.0, 0.0, 100.0, 200.0),
        DisplayCorrespondence("b", 2.0, 0.0, 140.0, 210.0),
        DisplayCorrespondence("c", 0.0, 3.0, 85.0, 260.0),
        DisplayCorrespondence("d", 1.0, 1.0, 115.0, 225.0),
    )
    transform = fit_plan_display_transform(correspondences)

    assert transform.transform_name == "T_plan_display_pixel_from_world"
    assert transform.pixel_from_world(2.0, 3.0) == pytest.approx((125.0, 270.0))
    assert transform.world_from_pixel(125.0, 270.0) == pytest.approx((2.0, 3.0))
    assert transform.rms_residual_pixels == pytest.approx(0.0)
    with pytest.raises(P02ContractError, match="degenerate"):
        fit_plan_display_transform(
            (
                DisplayCorrespondence("a", 0, 0, 0, 0),
                DisplayCorrespondence("b", 1, 1, 1, 1),
                DisplayCorrespondence("c", 2, 2, 2, 2),
            )
        )


def test_d010_requires_ceiling_and_two_consistent_area_checks() -> None:
    ceiling = CeilingHeightCheck(
        3.2, 0.03, "meeting-room-corner", "laser", _source(), EvidenceStatus.ACCEPTED
    )
    overlap = _spot_check(SpotCheckCoverage.OVERLAPPING_CAMERA_AREA)
    isolated = _spot_check(SpotCheckCoverage.ISOLATED_CAMERA_4_AREA)

    require_d010_checks(ceiling, (overlap, isolated))
    with pytest.raises(P02ContractError, match="isolated-camera-4-area"):
        require_d010_checks(ceiling, (overlap,))
    with pytest.raises(P02ContractError, match="ceiling"):
        require_d010_checks(None, (overlap, isolated))
    with pytest.raises(P02ContractError, match="spot checks"):
        require_d010_checks(ceiling, (replace(isolated, measured_distance_metres=4.3), overlap))
    with pytest.raises(P02ContractError, match="spot checks"):
        require_d010_checks(
            ceiling,
            (replace(isolated, measurement_uncertainty_metres=None), overlap),
        )


def test_landmark_roles_reject_movable_or_unsupported_solving_features() -> None:
    with pytest.raises(P02ContractError, match="movable"):
        StructuralLandmark(
            "chair",
            "movable chair",
            LandmarkRole.CANDIDATE,
            EvidenceStatus.PROVISIONAL,
            False,
            None,
            (),
        )
    with pytest.raises(P02ContractError, match="accepted permanent XYZ"):
        StructuralLandmark(
            "wall-corner",
            "wall corner",
            LandmarkRole.SOLVE,
            EvidenceStatus.CANDIDATE,
            True,
            None,
            (),
        )
    rejected = StructuralLandmark(
        "screen",
        "movable screen",
        LandmarkRole.EXCLUDED,
        EvidenceStatus.REJECTED,
        False,
        None,
        (),
        "movable active-scene object",
    )
    assert rejected.status == EvidenceStatus.REJECTED


def test_mounting_prior_preserves_decomposed_uncertainty_and_provisional_status() -> None:
    prior = derive_mounting_point_prior(_owner_camera(), _anchor(), 0.03, 0.2, (_source(),))

    assert prior.c_world_mount_prior == Point3Metres(0.0, 0.0, 3.0)
    assert prior.status == EvidenceStatus.PROVISIONAL
    assert prior.combined_uncertainty_metres == pytest.approx(
        combine_independent_uncertainty(0.03, 0.1, 0.2)
    )
    with pytest.raises(P02ContractError, match="accepted or provisional plan anchor"):
        derive_mounting_point_prior(
            _owner_camera(),
            replace(_anchor(), status=EvidenceStatus.CANDIDATE),
            0.03,
            0.2,
            (_source(),),
        )


def test_mounting_prior_preserves_unknown_plan_and_lens_components() -> None:
    provisional_anchor = replace(
        _anchor(),
        status=EvidenceStatus.PROVISIONAL,
        horizontal_uncertainty_metres=None,
    )

    prior = derive_mounting_point_prior(
        _owner_camera(), provisional_anchor, None, None, (_source(),)
    )

    assert prior.plan_control_residual_metres is None
    assert prior.unmeasured_lens_to_reference_offset_metres is None
    assert prior.combined_uncertainty_metres is None


def test_source_hashing_and_provenance_reject_bad_digest(tmp_path: Path) -> None:
    source_file = tmp_path / "source.bin"
    source_file.write_bytes(b"source")
    assert (
        fingerprint_file(source_file)
        == "41cf6794ba4200b839c53531555f0f3998df4cbb01a4d5cb0b94e3ca5e23947d"
    )
    with pytest.raises(P02ContractError, match="SHA-256"):
        SourceIdentity("bad", "not-a-digest", "scan", "not authority")
