from __future__ import annotations

import pytest

from spatial_mapping_phase2.xr02_local_domain import SceneContextKey
from spatial_mapping_phase2.xr02_local_pipeline import build_scene_context
from spatial_mapping_phase2.xr03_camera_policy import CameraPolicyError, SceneCameraPolicy
from spatial_mapping_phase2.xr03_xr02_policy import topology_from_camera_policy


def _policy(last_verdict: str = "no_overlap") -> SceneCameraPolicy:
    cameras = ("camera-a", "camera-b", "camera-c")
    return SceneCameraPolicy.build(
        "project-a",
        "scene-a",
        cameras,
        [
            {
                "group_id": "lens-a",
                "lens_model": "A",
                "camera_ids": list(cameras),
            }
        ],
        [
            {"camera_id_a": "camera-a", "camera_id_b": "camera-b", "verdict": "overlap"},
            {"camera_id_a": "camera-a", "camera_id_b": "camera-c", "verdict": last_verdict},
            {"camera_id_a": "camera-b", "camera_id_b": "camera-c", "verdict": "overlap"},
        ],
    )


def test_only_explicit_overlap_edges_enter_xr02_topology() -> None:
    transitions = (("camera-a", "camera-c"),)
    topology = topology_from_camera_policy(_policy(), transition_edges=transitions)

    assert topology.overlaps("camera-a", "camera-b") is True
    assert topology.overlaps("camera-b", "camera-c") is True
    assert topology.overlaps("camera-a", "camera-c") is False
    assert topology.transition_edges == transitions
    assert topology.can_transition("camera-a", "camera-c") is True
    assert topology.can_transition("camera-a", "camera-b") is False


def test_incomplete_overlap_review_is_refused() -> None:
    with pytest.raises(CameraPolicyError, match="must be complete"):
        topology_from_camera_policy(_policy("unreviewed"), transition_edges=())


def test_camera_policy_hash_creates_a_distinct_scene_context() -> None:
    def context(camera_policy_sha256: str | None = None) -> SceneContextKey:
        return build_scene_context(
            scene_id="scene-a",
            scene_epoch_id="epoch-a",
            geometry_sha256="1" * 64,
            floor_sha256="2" * 64,
            calibration_authority={"p06": "3" * 64},
            camera_policy_sha256=camera_policy_sha256,
        )

    legacy = context()
    first = context("4" * 64)
    second = context("5" * 64)

    assert "camera_policy_sha256" not in legacy.as_dict()
    assert first.as_dict()["camera_policy_sha256"] == "4" * 64
    assert first.context_sha256 != legacy.context_sha256
    assert first.context_sha256 != second.context_sha256
