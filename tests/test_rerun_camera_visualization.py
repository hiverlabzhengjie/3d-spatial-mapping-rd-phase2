from __future__ import annotations

import numpy as np
import pytest

from spatial_mapping_phase2.rerun_camera_visualization import (
    RerunCameraFrustum,
    RerunCameraVisualizationError,
    camera_entity_path,
    log_camera_frustum,
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


class _FakeRerun:
    class TransformRelation:
        ParentFromChild = "parent-from-child"

    class ViewCoordinates:
        RDF = "rdf"

    def __init__(self) -> None:
        self.logs: list[tuple[str, object, bool]] = []

    @staticmethod
    def Transform3D(**values: object) -> tuple[str, dict[str, object]]:
        return "transform", values

    @staticmethod
    def Pinhole(**values: object) -> tuple[str, dict[str, object]]:
        return "pinhole", values

    @staticmethod
    def Image(value: object) -> tuple[str, object]:
        return "image", value

    @staticmethod
    def Points3D(
        *values: object, **named: object
    ) -> tuple[str, tuple[object, ...], dict[str, object]]:
        return "points", values, named

    def log(self, path: str, value: object, *, static: bool) -> None:
        self.logs.append((path, value, static))


def test_log_camera_frustum_uses_readable_world_label_without_changing_camera_data() -> None:
    camera = _camera()
    rerun = _FakeRerun()
    label_position = np.array([0.5, 2.0, 3.8])

    paths = log_camera_frustum(
        rerun,
        "review/cameras",
        "review/labels",
        camera,
        label_position_world=label_position,
        label_text="Camera 1 | calibrated fixed working pose",
    )

    assert paths == ("/review/cameras/office-cam-01", "/review/labels/office-cam-01")
    assert [path for path, _, _ in rerun.logs] == [
        "review/cameras/office-cam-01",
        "review/cameras/office-cam-01",
        "review/cameras/office-cam-01",
        "review/labels/office-cam-01",
    ]
    label = rerun.logs[-1][1]
    assert isinstance(label, tuple)
    np.testing.assert_array_equal(label[1][0][0], label_position)
    assert label[2]["labels"] == ["Camera 1 | calibrated fixed working pose"]
    assert label[2]["show_labels"] is True


@pytest.mark.parametrize("position", [[1.0, 2.0], [1.0, np.nan, 3.0]])
def test_log_camera_frustum_rejects_invalid_world_label_position(position: list[float]) -> None:
    with pytest.raises(RerunCameraVisualizationError, match="finite XYZ"):
        log_camera_frustum(
            _FakeRerun(),
            "review/cameras",
            "review/labels",
            _camera(),
            label_position_world=np.asarray(position),
        )
