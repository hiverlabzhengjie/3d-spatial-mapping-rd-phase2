from __future__ import annotations

import numpy as np
import pytest

from spatial_mapping_phase2.rerun_camera_visualization import (
    RerunCameraFrustum,
    RerunCameraVisualizationError,
    camera_entity_path,
)


def _camera() -> RerunCameraFrustum:
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    intrinsics = np.array([[400.0, 0.0, 252.0], [0.0, 398.0, 140.0], [0.0, 0.0, 1.0]])
    frame = np.zeros((280, 504, 3), dtype=np.uint8)
    return RerunCameraFrustum("office-cam-01", transform, intrinsics, frame)


def test_camera_frustum_preserves_exact_pose_intrinsics_frame_and_resolution() -> None:
    camera = _camera()
    np.testing.assert_array_equal(camera.T_world_from_camera[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(camera.K_processed[0], [400.0, 0.0, 252.0])
    assert camera.resolution_xy == (504, 280)
    assert not camera.T_world_from_camera.flags.writeable
    assert not camera.K_processed.flags.writeable
    assert not camera.frame_rgb.flags.writeable
    assert (
        camera_entity_path("working/facility_geometry_v2/cameras", camera.camera_id)
        == "working/facility_geometry_v2/cameras/office-cam-01"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_id", "bad/id", "entity-safe"),
        ("T_world_from_camera", np.eye(3), "finite 4x4"),
        ("K_processed", np.eye(4), "finite 3x3"),
        ("frame_rgb", np.zeros((4, 4), dtype=np.uint8), "HxWx3 uint8"),
        ("image_plane_distance_metres", 0.0, "must be positive"),
    ],
)
def test_camera_frustum_rejects_malformed_visualization_inputs(
    field: str, value: object, message: str
) -> None:
    camera = _camera()
    values: dict[str, object] = {
        "camera_id": camera.camera_id,
        "T_world_from_camera": camera.T_world_from_camera,
        "K_processed": camera.K_processed,
        "frame_rgb": camera.frame_rgb,
        "image_plane_distance_metres": camera.image_plane_distance_metres,
    }
    values[field] = value
    with pytest.raises(RerunCameraVisualizationError, match=message):
        RerunCameraFrustum(**values)  # type: ignore[arg-type]


def test_camera_frustum_rejects_pose_or_intrinsic_misrepresentation() -> None:
    camera = _camera()
    improper = camera.T_world_from_camera.copy()
    improper[0, 0] = -1.0
    with pytest.raises(RerunCameraVisualizationError, match="not proper"):
        RerunCameraFrustum(camera.camera_id, improper, camera.K_processed, camera.frame_rgb)

    negative_focal = camera.K_processed.copy()
    negative_focal[0, 0] = -1.0
    with pytest.raises(RerunCameraVisualizationError, match="focal lengths"):
        RerunCameraFrustum(
            camera.camera_id, camera.T_world_from_camera, negative_focal, camera.frame_rgb
        )
