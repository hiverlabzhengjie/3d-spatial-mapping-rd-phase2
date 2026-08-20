from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from spatial_mapping_phase2.p08_floor import (
    FLOOR_Z_METRES,
    FloorProcessingConfig,
    FloorSourceGeometry,
    P08FloorError,
    process_floor,
)


def _source(points: NDArray[Any]) -> FloorSourceGeometry:
    count = len(points)
    return FloorSourceGeometry(
        points=points,
        colors_rgb=np.arange(count * 3, dtype=np.uint8).reshape(count, 3),
        confidence=np.linspace(1.0, 2.0, count),
        source_pixel_count=np.ones(count, dtype=np.int32),
        source_camera_index=np.zeros(count, dtype=np.int16),
        camera_ids=("test-camera",),
    )


def test_floor_is_one_rectangle_at_z_zero_using_full_source_xy_box() -> None:
    source = _source(
        np.array(
            [
                [-2.0, 4.0, -0.7],
                [8.0, 4.0, 2.0],
                [8.0, 14.0, 0.2],
                [-2.0, 14.0, -0.1],
            ]
        )
    )
    result = process_floor(source, FloorProcessingConfig())
    np.testing.assert_array_equal(
        result.plane_vertices_xyz_metres,
        np.array(
            [[-2.0, 4.0, 0.0], [8.0, 4.0, 0.0], [8.0, 14.0, 0.0], [-2.0, 14.0, 0.0]]
        ),
    )
    np.testing.assert_array_equal(result.plane_triangle_indices, [[0, 1, 2], [0, 2, 3]])
    assert np.all(result.plane_vertices_xyz_metres[:, 2] == FLOOR_Z_METRES)
    assert result.plane_area_square_metres == 100.0


def test_original_points_and_colors_are_untouched_including_below_z_zero() -> None:
    points = np.array([[0.0, 0.0, -0.5], [1.0, 1.0, 1.0], [2.0, 3.0, -0.01]])
    source = _source(points)
    original_points = source.points.copy()
    original_colors = source.colors_rgb.copy()
    result = process_floor(source, FloorProcessingConfig())
    np.testing.assert_array_equal(result.source.points, original_points)
    np.testing.assert_array_equal(result.source.colors_rgb, original_colors)
    assert result.summary()["original_points_removed"] == 0
    assert result.summary()["original_point_colors_modified"] is False
    assert result.summary()["generated_point_count"] == 0


def test_floor_plane_is_deterministic_and_supports_configured_inset() -> None:
    source = _source(np.array([[0.0, 0.0, -1.0], [4.0, 6.0, 2.0]]))
    config = FloorProcessingConfig(inset_metres=0.5)
    first = process_floor(source, config)
    second = process_floor(source, config)
    np.testing.assert_array_equal(
        first.plane_vertices_xyz_metres, second.plane_vertices_xyz_metres
    )
    np.testing.assert_array_equal(
        first.plane_vertices_xyz_metres[:, :2],
        [[0.5, 0.5], [3.5, 0.5], [3.5, 5.5], [0.5, 5.5]],
    )


def test_invalid_policy_inset_and_collapsed_box_fail_closed() -> None:
    with pytest.raises(P08FloorError, match="boundary policy"):
        FloorProcessingConfig(boundary_policy="unbounded-plane")
    with pytest.raises(P08FloorError, match="non-negative"):
        FloorProcessingConfig(inset_metres=-0.1)
    with pytest.raises(P08FloorError, match="collapses"):
        process_floor(
            _source(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
            FloorProcessingConfig(inset_metres=0.6),
        )


def test_source_rejects_empty_non_finite_and_malformed_membership() -> None:
    with pytest.raises(P08FloorError, match="non-empty finite"):
        _source(np.empty((0, 3)))
    with pytest.raises(P08FloorError, match="non-empty finite"):
        _source(np.array([[0.0, 0.0, np.nan]]))
    with pytest.raises(P08FloorError, match="camera_index"):
        FloorSourceGeometry(
            points=np.zeros((1, 3)),
            colors_rgb=np.zeros((1, 3), dtype=np.uint8),
            confidence=np.ones(1),
            source_pixel_count=np.ones(1, dtype=np.int32),
            source_camera_index=np.ones(1, dtype=np.int16),
            camera_ids=("camera",),
        )
