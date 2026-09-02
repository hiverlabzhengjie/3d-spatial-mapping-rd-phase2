from __future__ import annotations

import pytest

from spatial_mapping_phase2.xr03_da3_policy import SceneDa3Cohort, SceneDa3PolicyError


def test_multi_camera_cohort_is_one_ordered_joint_invocation() -> None:
    cohort = SceneDa3Cohort.build(("camera-z", "camera-a"), "a" * 64)

    assert cohort.inference_mode == "joint-pose-conditioned-multi-view"
    assert cohort.cli_arguments() == (
        "--camera-id",
        "camera-z",
        "--camera-id",
        "camera-a",
        "--camera-policy-sha256",
        "a" * 64,
    )
    assert cohort.to_dict()["view_overlap_influence"] == "none"


def test_single_camera_route_and_invalid_roster() -> None:
    assert SceneDa3Cohort.build(("camera-a",), "b" * 64).inference_mode == (
        "pose-conditioned-single-view"
    )
    with pytest.raises(SceneDa3PolicyError, match="unique"):
        SceneDa3Cohort.build(("camera-a", "camera-a"), "b" * 64)
