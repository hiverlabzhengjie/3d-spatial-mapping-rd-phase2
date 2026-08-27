from __future__ import annotations

import pytest

from spatial_mapping_phase2.xr02_scale_evaluation import (
    ScaleEvaluationConfig,
    run_scale_evaluation,
)


def test_scale_config_requires_post_churn_recovery() -> None:
    with pytest.raises(ValueError, match="post-churn"):
        ScaleEvaluationConfig(ticks=80, churn_start_tick=72, churn_duration_ticks=8)


def test_scene_partition_scale_is_deterministic_and_collision_free() -> None:
    config = ScaleEvaluationConfig(
        scene_count=2,
        cameras_per_scene=3,
        people_per_scene=6,
        ticks=90,
        camera_handoff_every_ticks=20,
        churn_start_tick=40,
        churn_duration_ticks=5,
    )
    first = run_scale_evaluation(config)
    second = run_scale_evaluation(config)
    first_results = first["results"]
    second_results = second["results"]
    assert isinstance(first_results, dict)
    assert isinstance(second_results, dict)
    assert first["deterministic_signature_sha256"] == second["deterministic_signature_sha256"]
    assert first_results["cross_scene_namespace_collisions"] == 0
    assert first_results["same_camera_global_collisions"] == 0
    assert first_results["media_route_count"] == 6
    assert first_results["unique_media_route_count"] == 6
    assert first_results["final_track_counts_by_scene"] == {"0": 6, "1": 6}
    assert second_results["final_track_counts_by_scene"] == {"0": 6, "1": 6}
    failure_contracts = first["failure_contracts"]
    assert isinstance(failure_contracts, dict)
    assert all(failure_contracts.values())
