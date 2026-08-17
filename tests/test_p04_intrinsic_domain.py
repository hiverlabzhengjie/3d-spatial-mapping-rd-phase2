from __future__ import annotations

import pytest

from spatial_mapping_phase2.p04_intrinsic_domain import (
    IntrinsicEvidenceError,
    looks_like_geocalib_initialization,
    summarize_intrinsic_candidate,
)


def test_intrinsic_stability_summary_reports_focal_distortion_and_gravity_spread() -> None:
    individual: list[list[float]] = [
        [1920, 1080, 2000, 2000, 960, 540, -0.10, 0],
        [1920, 1080, 2100, 2100, 960, 540, -0.12, 0],
        [1920, 1080, 2050, 2050, 960, 540, -0.11, 0],
    ]
    shared: list[list[float]] = [[1920, 1080, 2050, 2050, 960, 540, -0.11, 0]] * 3
    gravity: list[list[float]] = [
        [0, -1, 0],
        [0.01, -0.99995, 0],
        [0, -0.99985, 0.01745],
    ]

    summary = summarize_intrinsic_candidate(individual, shared, gravity, 1)

    assert summary.frame_count == 3
    assert summary.focal_mean_pixels == pytest.approx(2050)
    assert summary.focal_range_pixels == pytest.approx(100)
    assert summary.shared_focal_pixels == pytest.approx(2050)
    assert summary.distortion_mean == pytest.approx((-0.11,))
    assert 1.1 < summary.max_gravity_separation_degrees < 1.2


def test_intrinsic_summary_rejects_mixed_sizes_and_non_shared_result() -> None:
    gravity = [[0, -1, 0], [0, -1, 0]]
    with pytest.raises(IntrinsicEvidenceError, match="one image size"):
        summarize_intrinsic_candidate(
            [[1920, 1080, 2000, 2000, 960, 540, 0, 0], [1280, 720, 1300, 1300, 640, 360, 0, 0]],
            [[1920, 1080, 2000, 2000, 960, 540, 0, 0], [1280, 720, 1300, 1300, 640, 360, 0, 0]],
            gravity,
        )
    with pytest.raises(IntrinsicEvidenceError, match="does not share"):
        summarize_intrinsic_candidate(
            [[1920, 1080, 2000, 2000, 960, 540, 0, 0]] * 2,
            [
                [1920, 1080, 2000, 2000, 960, 540, 0, 0],
                [1920, 1080, 2001, 2001, 960, 540, 0, 0],
            ],
            gravity,
        )


def test_exact_geocalib_initialization_is_rejected_as_optimizer_stall() -> None:
    camera = [[1920, 1080, 1285.2, 1285.2, 960, 540, 0, 0]] * 3
    gravity = [[0, -1, 0]] * 3

    assert looks_like_geocalib_initialization(camera, gravity)
    camera[0] = [1920, 1080, 1900, 1900, 960, 540, 0, 0]
    assert not looks_like_geocalib_initialization(camera, gravity)
