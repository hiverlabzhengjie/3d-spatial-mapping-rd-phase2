"""Reusable calibrated-camera visualization contracts for Rerun recordings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]


class RerunCameraVisualizationError(ValueError):
    """Raised when a camera visualization would misrepresent its source geometry."""


@dataclass(frozen=True)
class RerunCameraFrustum:
    """One immutable OpenCV-camera frustum placed by ``T_world_from_camera``.

    The camera convention is x-right, y-down, z-forward. Intrinsics are expressed in pixels for
    the exact RGB frame supplied here. ``image_plane_distance_metres`` affects display only.
    """

    camera_id: str
    T_world_from_camera: Array
    K_processed: Array
    frame_rgb: Array
    image_plane_distance_metres: float = 0.75
    axis_length_metres: float = 0.55

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or "/" in self.camera_id:
            raise RerunCameraVisualizationError("camera_id must be a non-empty entity-safe name")
        transform = np.asarray(self.T_world_from_camera, dtype=np.float64).copy()
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RerunCameraVisualizationError("T_world_from_camera must be a finite 4x4 matrix")
        if not np.array_equal(transform[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise RerunCameraVisualizationError("T_world_from_camera has an invalid bottom row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise RerunCameraVisualizationError("T_world_from_camera rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise RerunCameraVisualizationError("T_world_from_camera rotation is not proper")

        intrinsics = np.asarray(self.K_processed, dtype=np.float64).copy()
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise RerunCameraVisualizationError("K_processed must be a finite 3x3 matrix")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise RerunCameraVisualizationError("K_processed focal lengths must be positive")
        if not np.allclose(intrinsics[2], np.array([0.0, 0.0, 1.0]), atol=1e-9):
            raise RerunCameraVisualizationError("K_processed has an invalid homogeneous row")

        frame = np.asarray(self.frame_rgb).copy()
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RerunCameraVisualizationError("frame_rgb must be an HxWx3 uint8 RGB array")
        height, width = frame.shape[:2]
        principal_x = float(intrinsics[0, 2])
        principal_y = float(intrinsics[1, 2])
        if not (0.0 <= principal_x <= width and 0.0 <= principal_y <= height):
            raise RerunCameraVisualizationError(
                "K_processed principal point must lie within the processed frame"
            )
        if (
            not np.isfinite(self.image_plane_distance_metres)
            or self.image_plane_distance_metres <= 0
        ):
            raise RerunCameraVisualizationError("image_plane_distance_metres must be positive")
        if not np.isfinite(self.axis_length_metres) or self.axis_length_metres <= 0:
            raise RerunCameraVisualizationError("axis_length_metres must be positive")

        transform.setflags(write=False)
        intrinsics.setflags(write=False)
        frame.setflags(write=False)
        object.__setattr__(self, "T_world_from_camera", transform)
        object.__setattr__(self, "K_processed", intrinsics)
        object.__setattr__(self, "frame_rgb", frame)

    @property
    def resolution_xy(self) -> tuple[int, int]:
        """Return exact processed frame resolution as ``(width, height)``."""

        return int(self.frame_rgb.shape[1]), int(self.frame_rgb.shape[0])


def camera_entity_path(camera_root: str, camera_id: str) -> str:
    """Return one stable entity path without leading/trailing slash ambiguity."""

    normalized_root = camera_root.strip("/")
    if not normalized_root:
        raise RerunCameraVisualizationError("camera_root must be a non-empty entity path")
    if not camera_id.strip() or "/" in camera_id:
        raise RerunCameraVisualizationError("camera_id must be a non-empty entity-safe name")
    return f"{normalized_root}/{camera_id}"


def log_camera_frustum(
    rr: Any,
    camera_root: str,
    label_root: str,
    camera: RerunCameraFrustum,
    *,
    label_position_world: Array | None = None,
    label_text: str | None = None,
) -> tuple[str, str]:
    """Log a fixed transform, pinhole, image plane, axis triad and in-scene label to Rerun."""

    entity_path = camera_entity_path(camera_root, camera.camera_id)
    label_path = camera_entity_path(label_root, camera.camera_id)
    width, height = camera.resolution_xy
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=camera.T_world_from_camera[:3, 3],
            mat3x3=camera.T_world_from_camera[:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
            axis_length=camera.axis_length_metres,
        ),
        static=True,
    )
    rr.log(
        entity_path,
        rr.Pinhole(
            image_from_camera=camera.K_processed,
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=camera.image_plane_distance_metres,
        ),
        static=True,
    )
    rr.log(entity_path, rr.Image(camera.frame_rgb), static=True)
    label_position = (
        camera.T_world_from_camera[:3, 3]
        if label_position_world is None
        else np.asarray(label_position_world, dtype=np.float64)
    )
    if label_position.shape != (3,) or not np.isfinite(label_position).all():
        raise RerunCameraVisualizationError("label_position_world must be a finite XYZ vector")
    readable_label = label_text or f"{camera.camera_id} | fixed working pose"
    if not readable_label.strip():
        raise RerunCameraVisualizationError("label_text must be non-empty when provided")
    rr.log(
        label_path,
        rr.Points3D(
            [label_position],
            labels=[readable_label],
            radii=0.085,
            colors=[[255, 214, 64]],
            show_labels=True,
        ),
        static=True,
    )
    return f"/{entity_path}", f"/{label_path}"
