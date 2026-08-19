from typing import Any

import pytest

from spatial_mapping_phase2.p05_pose_candidates import (
    D033_INTRINSICS,
    build_camera2_role_swap_assignments,
    build_intrinsic_policy_candidates,
)


def _fleet_manifest() -> dict[str, Any]:
    def intrinsics(focal: float, k1: float) -> dict[str, Any]:
        return {
            "model": "simple_radial",
            "width_pixels": 1920,
            "height_pixels": 1080,
            "fx_pixels": focal,
            "fy_pixels": focal,
            "cx_pixels": 960.0,
            "cy_pixels": 540.0,
            "distortion": [k1],
        }

    return {
        "decision_authority": "D027",
        "models": [
            {
                "camera_model": "simple_radial",
                "per_camera_estimates": [
                    {
                        "camera_id": camera_id,
                        "fx_pixels": focal,
                        "fy_pixels": focal,
                        "cx_pixels": 960.0,
                        "cy_pixels": 540.0,
                        "distortion": [k1],
                        "manifest_sha256": camera_id[-2:] * 32,
                    }
                    for camera_id, focal, k1 in (
                        ("office-cam-01", 1274.0, -0.265),
                        ("office-cam-02", 1401.0, -0.279),
                        ("office-cam-04", 1410.0, -0.281),
                    )
                ],
                "camera3_pose_evaluations": [
                    {
                        "label": "leave-camera3-out:equal-camera-arithmetic-mean",
                        "intrinsics": intrinsics(1362.0, -0.275),
                    },
                    {
                        "label": "leave-camera3-out:equal-camera-componentwise-huber",
                        "intrinsics": intrinsics(1397.0, -0.277),
                    },
                ],
            }
        ],
    }


@pytest.mark.parametrize("camera_id", ["office-cam-01", "office-cam-02", "office-cam-04"])
def test_d033_policy_retains_default_independent_and_fleet_candidates(
    camera_id: str,
) -> None:
    candidates = build_intrinsic_policy_candidates(_fleet_manifest(), camera_id)

    assert len(candidates) == 4
    assert candidates[0].label == "d033-candidate-b"
    assert candidates[0].intrinsics == D033_INTRINSICS
    assert candidates[1].label == "frozen-independent-geocalib"
    assert candidates[2].label == "d027-fleet-arithmetic"
    assert candidates[3].label == (
        "d029-huber" if camera_id == "office-cam-01" else "d027-fleet-huber"
    )


def test_policy_rejects_camera_three_and_non_d027_manifest() -> None:
    with pytest.raises(ValueError, match="Cameras 1, 2 and 4"):
        build_intrinsic_policy_candidates(_fleet_manifest(), "office-cam-03")
    manifest = _fleet_manifest()
    manifest["decision_authority"] = "D999"
    with pytest.raises(ValueError, match="D027"):
        build_intrinsic_policy_candidates(manifest, "office-cam-01")


def test_camera2_role_swaps_promote_both_and_demote_every_solve_pair() -> None:
    landmarks = [
        {"landmark_id": f"c2p{index}", "role": "solve" if index <= 6 else "held-out"}
        for index in range(1, 9)
    ]

    assignments = build_camera2_role_swap_assignments(landmarks)

    assert len(assignments) == 15
    assert {assignment.promoted_solve_ids for assignment in assignments} == {("c2p7", "c2p8")}
    assert {
        assignment.demoted_validation_ids for assignment in assignments
    } == {
        (f"c2p{first}", f"c2p{second}")
        for first in range(1, 7)
        for second in range(first + 1, 7)
    }
    assert all(len(assignment.solve_items) == 6 for assignment in assignments)
    assert all(len(assignment.validation_items) == 2 for assignment in assignments)


def test_camera2_role_swaps_require_exact_original_role_counts() -> None:
    with pytest.raises(ValueError, match="six solve and two held-out"):
        build_camera2_role_swap_assignments(
            [{"landmark_id": "c2p1", "role": "solve"}]
        )
