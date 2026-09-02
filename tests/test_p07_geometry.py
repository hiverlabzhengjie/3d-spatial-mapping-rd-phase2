from __future__ import annotations

import numpy as np
import pytest

from spatial_mapping_phase2.p07_geometry import (
    ALL4_DA3_CAMERA_ORDER,
    CAMERA_FRAME,
    FACILITY_FRAME,
    FusionCandidate,
    GeometryPatch,
    P07GeometryError,
    T_world_from_da3_T_camera_from_world,
    back_project_depth,
    concatenate_diagnostic_candidate,
    concatenate_scene_da3_candidate,
    concatenate_working_facility_geometry,
    cross_view_nearest_surface_diagnostics,
    evaluate_fusion_gate,
    filter_by_confidence,
    filter_by_evaluation_mask,
    filter_by_range,
    floor_plane_diagnostic,
    p06_processed_image_to_rgb,
    plan_extent_support,
    remove_camera_from_working_geometry,
    select_case_view_arrays,
    statistical_outlier_filter,
    transform_to_provisional_facility,
    unavailable_structural_diagnostics,
    validate_all4_da3_camera_order,
    validate_d040_prohibited_operations,
    validate_frozen_working_transforms,
    validate_scene_da3_camera_order,
    validate_T_world_from_camera,
    voxel_downsample,
    voxel_filter_working_geometry,
)


def _raw_patch() -> GeometryPatch:
    depth = np.array([[2.0, 4.0], [1.0, np.nan]], dtype=np.float64)
    confidence = np.array([[5.0, 0.5], [2.0, 9.0]], dtype=np.float64)
    colors = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    mask = np.array([[True, False], [True, True]])
    K = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    return back_project_depth("office-cam-01", depth, K, colors, confidence, mask)


def test_back_projection_uses_integer_pixel_coordinates_and_preserves_mask() -> None:
    patch = _raw_patch()
    np.testing.assert_allclose(
        patch.points,
        np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 4.0], [0.0, 0.5, 1.0]]),
    )
    np.testing.assert_array_equal(patch.pixel_uv, np.array([[0, 0], [1, 0], [0, 1]]))
    np.testing.assert_array_equal(patch.evaluation_mask_keep, [True, False, True])
    assert patch.frame_id == CAMERA_FRAME
    assert not patch.points.flags.writeable


def test_p06_processed_image_rgb_contract_preserves_red_and_blue_channels() -> None:
    processed_rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    converted = p06_processed_image_to_rgb(processed_rgb)
    np.testing.assert_array_equal(converted, processed_rgb)
    assert not converted.flags.writeable
    with pytest.raises(P07GeometryError, match="HxWx3 uint8 RGB"):
        p06_processed_image_to_rgb(processed_rgb.astype(np.float32))


def test_back_projection_rejects_shape_and_intrinsic_failures() -> None:
    colors = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(P07GeometryError, match="confidence"):
        back_project_depth(
            "camera",
            np.ones((2, 2)),
            np.eye(3),
            colors,
            np.ones((1, 2)),
            np.ones((2, 2), dtype=bool),
        )
    with pytest.raises(P07GeometryError, match="focal"):
        back_project_depth(
            "camera",
            np.ones((2, 2)),
            np.diag([0.0, 1.0, 1.0]),
            colors,
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
        )


def test_filter_chain_reports_success_rejection_counts_and_preserves_raw() -> None:
    raw = _raw_patch()
    raw_copy = raw.points.copy()
    confidence = filter_by_confidence(raw, 1.0)
    distance = filter_by_range(confidence.patch, 0.5, 3.0)
    masked = filter_by_evaluation_mask(distance.patch)
    assert confidence.statistics()["rejected_point_count"] == 1
    assert distance.patch.point_count == 2
    assert masked.patch.point_count == 2
    np.testing.assert_array_equal(raw.points, raw_copy)
    assert not raw.points.flags.writeable


def test_voxel_downsample_aggregates_without_creating_points() -> None:
    patch = GeometryPatch(
        camera_id="office-cam-01",
        frame_id=CAMERA_FRAME,
        units="metres",
        points=np.array([[0.01, 0.01, 1.0], [0.02, 0.02, 1.01], [1.0, 1.0, 1.0]]),
        colors_rgb=np.array([[0, 10, 20], [10, 20, 30], [100, 100, 100]], dtype=np.uint8),
        confidence=np.array([1.0, 3.0, 5.0]),
        evaluation_mask_keep=np.array([True, True, True]),
        pixel_uv=np.array([[0, 0], [1, 0], [2, 0]]),
        source_pixel_count=np.ones(3, dtype=np.int32),
    )
    result = voxel_downsample(patch, 0.1)
    assert result.patch.point_count == 2
    assert result.rejected_point_count == 1
    assert sorted(result.patch.source_pixel_count.tolist()) == [1, 2]


def test_open3d_outlier_filter_removes_far_isolated_point() -> None:
    grid = np.array([[x, y, 1.0] for x in np.linspace(0, 0.1, 5) for y in np.linspace(0, 0.1, 5)])
    points = np.vstack([grid, [10.0, 10.0, 10.0]])
    count = len(points)
    patch = GeometryPatch(
        camera_id="office-cam-01",
        frame_id=CAMERA_FRAME,
        units="metres",
        points=points,
        colors_rgb=np.zeros((count, 3), dtype=np.uint8),
        confidence=np.ones(count),
        evaluation_mask_keep=np.ones(count, dtype=bool),
        pixel_uv=np.column_stack((np.arange(count), np.zeros(count))),
        source_pixel_count=np.ones(count, dtype=np.int32),
    )
    result = statistical_outlier_filter(patch, 5, 1.0)
    assert result.patch.point_count < count
    assert not np.any(np.all(result.patch.points == [10.0, 10.0, 10.0], axis=1))
    with pytest.raises(P07GeometryError, match="too few"):
        statistical_outlier_filter(_raw_patch(), 5, 1.0)


def test_transform_direction_is_camera_to_world_and_source_is_unchanged() -> None:
    raw = _raw_patch()
    source = raw.points.copy()
    T_world_from_camera = np.eye(4)
    T_world_from_camera[:3, 3] = [1.0, 2.0, 3.0]
    world = transform_to_provisional_facility(raw, T_world_from_camera)
    np.testing.assert_allclose(world.points, raw.points + [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(raw.points, source)
    assert world.frame_id == FACILITY_FRAME
    diagnostic = floor_plane_diagnostic(world)
    assert diagnostic["status"] == "distribution-only-no-floor-segmentation"
    assert diagnostic["acceptance_threshold"] is None


def test_transform_validation_rejects_reflection_and_ambiguous_retransform() -> None:
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(P07GeometryError, match="proper"):
        validate_T_world_from_camera(reflection)
    world = transform_to_provisional_facility(_raw_patch(), np.eye(4))
    with pytest.raises(P07GeometryError, match="camera-frame"):
        transform_to_provisional_facility(world, np.eye(4))


def test_structural_gaps_remain_explicit() -> None:
    gaps = unavailable_structural_diagnostics()
    assert gaps["walls"]["residual"] is None
    assert gaps["columns"]["status"] == "unsupported-reference-geometry-absent"
    assert gaps["ceiling"]["acceptance_threshold"] is None


def test_fusion_gate_synthetic_success_and_current_authority_rejection() -> None:
    accepted = FusionCandidate(
        camera_ids=("office-cam-02", "office-cam-03"),
        strict_camera_statuses={"office-cam-02": "accepted", "office-cam-03": "accepted"},
        authorized_edges=frozenset({("office-cam-02", "office-cam-03")}),
        transform_authorities={
            "office-cam-02": "accepted-registration",
            "office-cam-03": "accepted-registration",
        },
        frame_ids={"office-cam-02": FACILITY_FRAME, "office-cam-03": FACILITY_FRAME},
        units={"office-cam-02": "metres", "office-cam-03": "metres"},
        independent_relative_pose_validated=True,
        structural_validation_passed=True,
        scale_alignment_authorized=True,
    )
    eligible = evaluate_fusion_gate(accepted)
    assert eligible.status == "eligible-for-separately-authorized-fusion"
    assert not eligible.fused_artifact_created

    rejected = evaluate_fusion_gate(
        FusionCandidate(
            camera_ids=("office-cam-02", "office-cam-03"),
            strict_camera_statuses={
                "office-cam-02": "rejected",
                "office-cam-03": "rejected",
            },
            authorized_edges=frozenset(),
            transform_authorities={
                "office-cam-02": "provisional-consumed-evidence",
                "office-cam-03": "provisional-consumed-evidence",
            },
            frame_ids={"office-cam-02": FACILITY_FRAME, "office-cam-03": FACILITY_FRAME},
            units={"office-cam-02": "metres", "office-cam-03": "metres"},
            independent_relative_pose_validated=False,
            structural_validation_passed=False,
            scale_alignment_authorized=False,
        )
    )
    assert rejected.status == "rejected"
    assert len(rejected.rejection_reasons) == 8
    assert not rejected.fused_artifact_created


def test_geometry_patch_rejects_misaligned_attributes() -> None:
    with pytest.raises(P07GeometryError, match="colors"):
        GeometryPatch(
            camera_id="office-cam-01",
            frame_id=CAMERA_FRAME,
            units="metres",
            points=np.zeros((2, 3)),
            colors_rgb=np.zeros((1, 3), dtype=np.uint8),
            confidence=np.ones(2),
            evaluation_mask_keep=np.ones(2, dtype=bool),
            pixel_uv=np.zeros((2, 2), dtype=np.int32),
            source_pixel_count=np.ones(2, dtype=np.int32),
        )


def _facility_patch(camera_id: str, offset: float) -> GeometryPatch:
    return GeometryPatch(
        camera_id=camera_id,
        frame_id=FACILITY_FRAME,
        units="metres",
        points=np.array([[offset, 0.0, 0.0], [offset + 0.1, 0.0, 0.0]]),
        colors_rgb=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        confidence=np.array([2.0, 3.0]),
        evaluation_mask_keep=np.ones(2, dtype=bool),
        pixel_uv=np.array([[0, 0], [1, 0]], dtype=np.int32),
        source_pixel_count=np.array([2, 3], dtype=np.int32),
    )


def test_case_view_selection_uses_case_specific_intrinsics_and_order() -> None:
    camera_ids = ("office-cam-03", "office-cam-02")
    arrays = {
        "depth": np.stack((np.full((2, 2), 3.0), np.full((2, 2), 2.0))),
        "confidence": np.ones((2, 2, 2)),
        "intrinsics": np.stack(
            (
                np.array([[30.0, 0.0, 1.0], [0.0, 31.0, 1.0], [0.0, 0.0, 1.0]]),
                np.array([[20.0, 0.0, 1.0], [0.0, 21.0, 1.0], [0.0, 0.0, 1.0]]),
            )
        ),
        "processed_images": np.zeros((2, 2, 2, 3), dtype=np.uint8),
        "extrinsics": np.zeros((2, 3, 4)),
    }
    index, selected = select_case_view_arrays(camera_ids, "office-cam-02", arrays)
    assert index == 1
    assert selected["intrinsics"][0, 0] == 20.0
    assert selected["depth"][0, 0] == 2.0
    assert not selected["intrinsics"].flags.writeable
    with pytest.raises(P07GeometryError, match="absent"):
        select_case_view_arrays(camera_ids, "office-cam-01", arrays)


def test_diagnostic_concatenation_is_deterministic_and_preserves_membership() -> None:
    camera_1 = _facility_patch("office-cam-01", 0.0)
    camera_3 = _facility_patch("office-cam-03", 1.0)
    first = concatenate_diagnostic_candidate(
        "posed-diagnostic-cameras-1-3",
        {"office-cam-03": camera_3, "office-cam-01": camera_1},
    )
    second = concatenate_diagnostic_candidate(
        "posed-diagnostic-cameras-1-3",
        {"office-cam-01": camera_1, "office-cam-03": camera_3},
    )
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.source_camera_index, [0, 0, 1, 1])
    assert first.camera_ids == ("office-cam-01", "office-cam-03")
    assert first.point_count == camera_1.point_count + camera_3.point_count
    assert first.represented_source_pixel_count == 10
    assert not first.operational_fusion_authorized
    assert not first.accepted_geometry


def test_diagnostic_fusion_rejects_nonfacility_inputs_and_stays_separate_from_gate() -> None:
    with pytest.raises(P07GeometryError, match="facility"):
        concatenate_diagnostic_candidate(
            "invalid",
            {
                "office-cam-01": _raw_patch(),
                "office-cam-02": _facility_patch("office-cam-02", 1.0),
            },
        )
    eligible_gate = evaluate_fusion_gate(
        FusionCandidate(
            camera_ids=("office-cam-01", "office-cam-03"),
            strict_camera_statuses={"office-cam-01": "accepted", "office-cam-03": "accepted"},
            authorized_edges=frozenset({("office-cam-01", "office-cam-03")}),
            transform_authorities={
                "office-cam-01": "accepted-registration",
                "office-cam-03": "accepted-registration",
            },
            frame_ids={"office-cam-01": FACILITY_FRAME, "office-cam-03": FACILITY_FRAME},
            units={"office-cam-01": "metres", "office-cam-03": "metres"},
            independent_relative_pose_validated=True,
            structural_validation_passed=True,
            scale_alignment_authorized=True,
        )
    )
    candidate = concatenate_diagnostic_candidate(
        "posed-diagnostic-cameras-1-3",
        {
            "office-cam-01": _facility_patch("office-cam-01", 0.0),
            "office-cam-03": _facility_patch("office-cam-03", 1.0),
        },
    )
    assert eligible_gate.status == "eligible-for-separately-authorized-fusion"
    assert not eligible_gate.fused_artifact_created
    assert candidate.operational_fusion_authorized is False


def test_nearest_surface_and_plan_extent_diagnostics_are_descriptive_only() -> None:
    patches = {
        "office-cam-01": _facility_patch("office-cam-01", 0.0),
        "office-cam-03": _facility_patch("office-cam-03", 0.1),
    }
    diagnostics = cross_view_nearest_surface_diagnostics(patches)
    assert len(diagnostics) == 2
    assert diagnostics[0]["distance_metres"]["median"] <= 0.1
    assert diagnostics[0]["alignment_or_correspondence_applied"] is False
    assert diagnostics[0]["acceptance_threshold"] is None
    support = plan_extent_support(
        patches["office-cam-01"],
        np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
    )
    assert support["fraction_xy_inside_plan_extent"] == 1.0
    assert support["vector_structure_support"] is False


@pytest.mark.parametrize(
    "enabled_operation",
    [
        "ICP",
        "pose_refinement",
        "camera_movement",
        "scale_or_alignment_correction",
        "surface_completion",
        "invented_points",
        "filter_retuning",
    ],
)
def test_d040_prohibited_correction_and_refinement_paths_reject(
    enabled_operation: str,
) -> None:
    controls = {
        "ICP": False,
        "pose_refinement": False,
        "camera_movement": False,
        "scale_or_alignment_correction": False,
        "surface_completion": False,
        "invented_points": False,
        "filter_retuning": False,
    }
    assert validate_d040_prohibited_operations(controls) == controls
    controls[enabled_operation] = True
    with pytest.raises(P07GeometryError, match="prohibits"):
        validate_d040_prohibited_operations(controls)


def test_d041_working_concatenation_is_ordered_deterministic_and_source_preserving() -> None:
    camera_order = tuple(f"office-cam-0{index}" for index in range(1, 5))
    patches = {
        camera_id: _facility_patch(camera_id, float(index))
        for index, camera_id in enumerate(camera_order)
    }
    snapshots = {camera_id: patch.points.copy() for camera_id, patch in patches.items()}
    first = concatenate_working_facility_geometry(
        dict(reversed(tuple(patches.items()))), camera_order
    )
    second = concatenate_working_facility_geometry(patches, camera_order)
    assert first.camera_ids == camera_order
    assert first.point_count == 8
    assert first.represented_source_pixel_count == 20
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.source_camera_membership.sum(axis=1), 1)
    np.testing.assert_array_equal(first.input_point_count_by_camera.sum(axis=0), [2, 2, 2, 2])
    np.testing.assert_array_equal(
        first.represented_source_pixel_count_by_camera.sum(axis=0), [5, 5, 5, 5]
    )
    for camera_id, patch in patches.items():
        np.testing.assert_array_equal(patch.points, snapshots[camera_id])


def test_d041_voxel_merge_preserves_mixed_membership_and_exact_camera_removal() -> None:
    camera_order = tuple(f"office-cam-0{index}" for index in range(1, 5))
    patches = {
        "office-cam-01": _facility_patch("office-cam-01", 0.00),
        "office-cam-02": _facility_patch("office-cam-02", 0.02),
        "office-cam-03": _facility_patch("office-cam-03", 1.00),
        "office-cam-04": _facility_patch("office-cam-04", 2.00),
    }
    pre_voxel = concatenate_working_facility_geometry(patches, camera_order)
    merged = voxel_filter_working_geometry(pre_voxel, 0.20)
    repeated = voxel_filter_working_geometry(
        concatenate_working_facility_geometry(
            dict(reversed(tuple(patches.items()))), camera_order
        ),
        0.20,
    )
    np.testing.assert_array_equal(merged.points, repeated.points)
    np.testing.assert_array_equal(
        merged.represented_source_pixel_count_by_camera,
        repeated.represented_source_pixel_count_by_camera,
    )
    assert np.any(merged.source_camera_membership.sum(axis=1) > 1)
    assert merged.represented_source_pixel_count == pre_voxel.represented_source_pixel_count

    without_camera_4 = remove_camera_from_working_geometry(merged, "office-cam-04")
    expected_without_camera_4 = voxel_filter_working_geometry(
        concatenate_working_facility_geometry(
            {camera_id: patches[camera_id] for camera_id in camera_order[:3]},
            camera_order[:3],
        ),
        0.20,
    )
    assert without_camera_4.camera_ids == camera_order[:3]
    np.testing.assert_allclose(without_camera_4.points, expected_without_camera_4.points)
    np.testing.assert_array_equal(
        without_camera_4.input_point_count_by_camera,
        expected_without_camera_4.input_point_count_by_camera,
    )
    with pytest.raises(P07GeometryError, match="already received"):
        voxel_filter_working_geometry(merged, 0.20)


def test_d041_frozen_transform_validation_rejects_any_pose_change() -> None:
    expected = {
        "office-cam-01": np.eye(4),
        "office-cam-02": np.array(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }
    validated = validate_frozen_working_transforms(expected, expected)
    assert set(validated) == set(expected)
    assert all(not transform.flags.writeable for transform in validated.values())
    changed = {camera_id: transform.copy() for camera_id, transform in expected.items()}
    changed["office-cam-02"][0, 3] += 1e-9
    with pytest.raises(P07GeometryError, match="changed"):
        validate_frozen_working_transforms(expected, changed)


def test_all4_da3_order_and_extrinsic_direction_contract() -> None:
    assert validate_all4_da3_camera_order(ALL4_DA3_CAMERA_ORDER) == ALL4_DA3_CAMERA_ORDER
    with pytest.raises(P07GeometryError, match="exact Camera 1/2/3/4 order"):
        validate_all4_da3_camera_order(tuple(reversed(ALL4_DA3_CAMERA_ORDER)))

    T_world_from_camera = np.eye(4)
    T_world_from_camera[:3, 3] = [2.0, -3.0, 4.0]
    T_camera_from_world = np.linalg.inv(T_world_from_camera)
    recovered = T_world_from_da3_T_camera_from_world(T_camera_from_world[:3])
    np.testing.assert_allclose(recovered, T_world_from_camera, atol=1e-12)
    assert not recovered.flags.writeable

    malformed = T_camera_from_world.copy()
    malformed[0, 0] = 2.0
    with pytest.raises(P07GeometryError, match="not orthonormal"):
        T_world_from_da3_T_camera_from_world(malformed)


def test_scene_da3_roster_supports_one_or_many_cameras_in_declared_order() -> None:
    single_order = ("camera-z",)
    assert validate_scene_da3_camera_order(single_order) == single_order
    single = concatenate_scene_da3_candidate(
        "single", {"camera-z": _facility_patch("camera-z", 0.0)}, single_order
    )
    assert single.camera_ids == single_order
    assert single.point_count == 2

    multi_order = ("camera-z", "camera-a")
    multi = concatenate_scene_da3_candidate(
        "multi",
        {
            "camera-z": _facility_patch("camera-z", 0.0),
            "camera-a": _facility_patch("camera-a", 0.1),
        },
        multi_order,
    )
    assert multi.camera_ids == multi_order
    assert multi.point_count == 4
    with pytest.raises(P07GeometryError, match="unique"):
        validate_scene_da3_camera_order(("camera-a", "camera-a"))
