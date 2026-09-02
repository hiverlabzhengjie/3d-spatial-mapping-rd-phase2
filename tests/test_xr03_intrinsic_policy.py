from __future__ import annotations

import pytest

from spatial_mapping_phase2.p04_intrinsic_fleet import CameraIntrinsicEstimate
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy
from spatial_mapping_phase2.xr03_intrinsic_policy import (
    GroupedIntrinsicPolicyError,
    build_grouped_intrinsic_candidates,
)


def _estimate(camera_id: str, focal: float) -> CameraIntrinsicEstimate:
    return CameraIntrinsicEstimate(
        camera_id=camera_id,
        profile_version="profile-v1",
        model="opencv-pinhole-radtan",
        width_pixels=1920,
        height_pixels=1080,
        fx_pixels=focal,
        fy_pixels=focal,
        cx_pixels=960.0,
        cy_pixels=540.0,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        within_camera_focal_cv=0.01,
    )


def _policy(groups: list[dict[str, object]]) -> SceneCameraPolicy:
    cameras = tuple(f"camera-{index}" for index in range(1, 6))
    pairs = [
        {"camera_id_a": left, "camera_id_b": right, "verdict": "unreviewed"}
        for index, left in enumerate(cameras)
        for right in cameras[index + 1 :]
    ]
    return SceneCameraPolicy.build("project-a", "scene-a", cameras, groups, pairs)


def test_candidates_use_only_same_lens_group_peers() -> None:
    policy = _policy(
        [
            {
                "group_id": "lens-a",
                "lens_model": "A",
                "camera_ids": ["camera-1", "camera-2", "camera-3", "camera-4"],
            },
            {"group_id": "lens-b", "lens_model": "B", "camera_ids": ["camera-5"]},
        ]
    )
    estimates = [
        _estimate(camera_id, 1000.0 + index) for index, camera_id in enumerate(policy.camera_ids)
    ]

    result = build_grouped_intrinsic_candidates(estimates, policy)

    target = result["groups"][0]["targets"][0]
    assert target["status"] == "candidates-ready"
    assert target["candidates"][0]["included_camera_ids"] == [
        "camera-2",
        "camera-3",
        "camera-4",
    ]
    assert "camera-5" not in target["candidates"][0]["included_camera_ids"]
    assert result["groups"][1]["targets"][0]["status"] == ("insufficient-or-incompatible")


def test_incomplete_grouping_or_estimate_roster_is_refused() -> None:
    incomplete = _policy(
        [
            {
                "group_id": "lens-a",
                "lens_model": "A",
                "camera_ids": ["camera-1", "camera-2", "camera-3", "camera-4"],
            }
        ]
    )
    estimates = [_estimate(camera_id, 1000.0) for camera_id in incomplete.camera_ids]
    with pytest.raises(GroupedIntrinsicPolicyError, match="cover every enabled camera"):
        build_grouped_intrinsic_candidates(estimates, incomplete)

    complete = _policy(
        [
            {
                "group_id": "lens-a",
                "lens_model": "A",
                "camera_ids": list(incomplete.camera_ids),
            }
        ]
    )
    with pytest.raises(GroupedIntrinsicPolicyError, match="do not match"):
        build_grouped_intrinsic_candidates(estimates[:-1], complete)
