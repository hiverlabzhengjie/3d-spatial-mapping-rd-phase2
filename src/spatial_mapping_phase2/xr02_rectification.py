"""XR02 fused native-to-processed rectification without changing P09 authority."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.p09_projection import (
    CameraProjectionCalibration,
    P09ProjectionError,
)


class FusedLiveFrameRectifier:
    """Apply distortion correction and calibrated resize in one remap.

    The output pinhole is still defined by the accepted ``K_processed``. This
    avoids constructing a full-resolution undistorted 1920x1080 intermediate
    for every live frame while leaving P09's accepted rectifier unchanged.
    """

    profile_id = "xr02-fused-native-to-k-processed-v1"

    def __init__(self, calibration: CameraProjectionCalibration) -> None:
        self.calibration = calibration
        coefficients = np.asarray(
            [calibration.simple_radial_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            calibration.K_native,
            coefficients,
            np.eye(3, dtype=np.float64),
            calibration.K_processed,
            calibration.processed_resolution_xy,
            cv2.CV_32FC1,
        )

    def rectify(self, native_frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        frame = np.asarray(native_frame_bgr)
        expected_shape = (
            self.calibration.native_resolution_xy[1],
            self.calibration.native_resolution_xy[0],
            3,
        )
        if frame.dtype != np.uint8 or frame.shape != expected_shape:
            raise P09ProjectionError("native frame must be uint8 1920x1080 BGR")
        processed = cv2.remap(
            frame,
            self._map_x,
            self._map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return np.ascontiguousarray(processed, dtype=np.uint8)
