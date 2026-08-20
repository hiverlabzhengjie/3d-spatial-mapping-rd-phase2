"""Deterministic authoritative floor plane for the frozen P07 facility geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

FACILITY_FRAME = "facility-world-x-plan-left-y-plan-down-z-up"
METRE_UNITS = "metres"
FLOOR_Z_METRES = 0.0
FLOOR_DERIVATIVE_AUTHORITY = "authoritative-facility-floor-plane-z-zero"
FLOOR_DERIVATIVE_USE = (
    "physical-and-geometric-floor-reference-within-source-world-frame-box"
)
FLOOR_PLANE_COLOR_RGBA = (48, 52, 58, 8)
FLOOR_PLANE_OPACITY = 0.025


class P08FloorError(ValueError):
    """Raised when a floor input, configuration, or result violates the P08 contract."""


@dataclass(frozen=True)
class FloorSourceGeometry:
    """Frozen P07 v2 combined geometry and aligned source attributes."""

    points: Array
    colors_rgb: Array
    confidence: Array
    source_pixel_count: Array
    source_camera_index: Array
    camera_ids: tuple[str, ...]
    frame_id: str = FACILITY_FRAME
    units: str = METRE_UNITS

    def __post_init__(self) -> None:
        if self.frame_id != FACILITY_FRAME or self.units != METRE_UNITS:
            raise P08FloorError(
                "floor processing requires facility-frame metre geometry"
            )
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise P08FloorError("source camera_ids must be non-empty and unique")
        points = _immutable_array(self.points, np.float64)
        colors = _immutable_array(self.colors_rgb, np.uint8)
        confidence = _immutable_array(self.confidence, np.float64)
        represented = _immutable_array(self.source_pixel_count, np.int32)
        membership = _immutable_array(self.source_camera_index, np.int16)
        count = len(points)
        if points.shape != (count, 3) or not count or not np.all(np.isfinite(points)):
            raise P08FloorError("source points must be a non-empty finite Nx3 array")
        if colors.shape != (count, 3):
            raise P08FloorError("source colors_rgb must be aligned Nx3 uint8")
        if confidence.shape != (count,) or not np.all(np.isfinite(confidence)):
            raise P08FloorError("source confidence must be an aligned finite vector")
        if represented.shape != (count,) or np.any(represented < 1):
            raise P08FloorError("source_pixel_count must be aligned and positive")
        if membership.shape != (count,) or np.any(membership < 0):
            raise P08FloorError("source_camera_index must be aligned and non-negative")
        if int(np.max(membership)) >= len(self.camera_ids):
            raise P08FloorError(
                "source_camera_index exceeds the declared camera roster"
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors_rgb", colors)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_pixel_count", represented)
        object.__setattr__(self, "source_camera_index", membership)

    @property
    def point_count(self) -> int:
        return int(len(self.points))

    @property
    def represented_source_pixel_count(self) -> int:
        return int(np.sum(self.source_pixel_count, dtype=np.int64))


@dataclass(frozen=True)
class FloorProcessingConfig:
    """Configuration for a single mathematical Z=0 plane inside the source XY box."""

    boundary_policy: str = "source-xy-axis-aligned-world-frame-box"
    inset_metres: float = 0.0

    def __post_init__(self) -> None:
        if self.boundary_policy != "source-xy-axis-aligned-world-frame-box":
            raise P08FloorError("unsupported floor boundary policy")
        if not np.isfinite(self.inset_metres) or self.inset_metres < 0:
            raise P08FloorError("inset_metres must be finite and non-negative")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "boundary_policy": self.boundary_policy,
            "inset_metres": self.inset_metres,
        }


@dataclass(frozen=True)
class FloorProcessingResult:
    """The untouched source plus one rectangular authoritative floor surface."""

    source: FloorSourceGeometry
    config: FloorProcessingConfig
    source_bounds_xyz_metres: Array
    plane_vertices_xyz_metres: Array
    plane_triangle_indices: Array
    authority: str = FLOOR_DERIVATIVE_AUTHORITY
    intended_use: str = FLOOR_DERIVATIVE_USE

    def __post_init__(self) -> None:
        if (
            self.authority != FLOOR_DERIVATIVE_AUTHORITY
            or self.intended_use != FLOOR_DERIVATIVE_USE
        ):
            raise P08FloorError("floor result authority labels changed")
        bounds = _immutable_array(self.source_bounds_xyz_metres, np.float64)
        vertices = _immutable_array(self.plane_vertices_xyz_metres, np.float64)
        triangles = _immutable_array(self.plane_triangle_indices, np.uint32)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise P08FloorError("source bounds must be finite min/max XYZ")
        if np.any(bounds[1] <= bounds[0]):
            raise P08FloorError("source bounds must have positive extent on every axis")
        if vertices.shape != (4, 3) or not np.all(np.isfinite(vertices)):
            raise P08FloorError("floor plane must have four finite XYZ vertices")
        if triangles.shape != (2, 3) or np.any(triangles > 3):
            raise P08FloorError("floor plane must have two valid triangles")
        if not np.all(vertices[:, 2] == FLOOR_Z_METRES):
            raise P08FloorError(
                "authoritative floor plane must remain exactly at facility Z=0"
            )
        expected_xy = np.array(
            [
                [
                    bounds[0, 0] + self.config.inset_metres,
                    bounds[0, 1] + self.config.inset_metres,
                ],
                [
                    bounds[1, 0] - self.config.inset_metres,
                    bounds[0, 1] + self.config.inset_metres,
                ],
                [
                    bounds[1, 0] - self.config.inset_metres,
                    bounds[1, 1] - self.config.inset_metres,
                ],
                [
                    bounds[0, 0] + self.config.inset_metres,
                    bounds[1, 1] - self.config.inset_metres,
                ],
            ]
        )
        if not np.array_equal(vertices[:, :2], expected_xy):
            raise P08FloorError(
                "floor plane vertices do not match the configured source XY box"
            )
        object.__setattr__(self, "source_bounds_xyz_metres", bounds)
        object.__setattr__(self, "plane_vertices_xyz_metres", vertices)
        object.__setattr__(self, "plane_triangle_indices", triangles)

    @property
    def plane_area_square_metres(self) -> float:
        extents = np.ptp(self.plane_vertices_xyz_metres[:, :2], axis=0)
        return float(extents[0] * extents[1])

    def summary(self) -> dict[str, Any]:
        return {
            "source_point_count": self.source.point_count,
            "source_represented_pixel_count": self.source.represented_source_pixel_count,
            "original_points_modified": False,
            "original_points_removed": 0,
            "original_point_colors_modified": False,
            "generated_point_count": 0,
            "floor_surface_kind": "mathematical-rectangle-mesh",
            "floor_plane_z_metres": FLOOR_Z_METRES,
            "floor_plane_vertex_count": 4,
            "floor_plane_triangle_count": 2,
            "floor_plane_area_square_metres": self.plane_area_square_metres,
            "source_bounds_xyz_metres": self.source_bounds_xyz_metres.tolist(),
            "floor_plane_bounds_xy_metres": [
                np.min(self.plane_vertices_xyz_metres[:, :2], axis=0).tolist(),
                np.max(self.plane_vertices_xyz_metres[:, :2], axis=0).tolist(),
            ],
            "frame_id": FACILITY_FRAME,
            "units": METRE_UNITS,
            "boundary_policy": self.config.boundary_policy,
            "authority": self.authority,
            "intended_use": self.intended_use,
        }


def process_floor(
    source: FloorSourceGeometry, config: FloorProcessingConfig
) -> FloorProcessingResult:
    """Create one solid mathematical Z=0 rectangle without altering any P07 point."""

    bounds = np.stack((np.min(source.points, axis=0), np.max(source.points, axis=0)))
    x_min = float(bounds[0, 0] + config.inset_metres)
    x_max = float(bounds[1, 0] - config.inset_metres)
    y_min = float(bounds[0, 1] + config.inset_metres)
    y_max = float(bounds[1, 1] - config.inset_metres)
    if x_min >= x_max or y_min >= y_max:
        raise P08FloorError("inset_metres collapses the source XY world-frame box")
    vertices = np.array(
        [
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return FloorProcessingResult(
        source=source,
        config=config,
        source_bounds_xyz_metres=bounds,
        plane_vertices_xyz_metres=vertices,
        plane_triangle_indices=triangles,
    )


def _immutable_array(value: Array, dtype: Any) -> Array:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result
