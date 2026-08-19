from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from spatial_mapping_phase2.p04_intrinsic_fleet import CameraIntrinsicEstimate
from spatial_mapping_phase2.p04_pose_domain import CameraIntrinsics
from spatial_mapping_phase2.pre_p05_charuco import (
    CharucoBoardSpec,
    CharucoCheckpointError,
    CharucoObservation,
    FixedProfileEvaluation,
    NaturalCameraEvidence,
    NaturalFrameMetrics,
    calibrate_charuco_reference,
    detect_charuco_observation,
    evaluate_fixed_intrinsics,
    median_cluster_profile,
    rank_natural_camera_quality,
    robust_intrinsic_cluster,
    select_policy,
)

FloatArray = npt.NDArray[np.float64]


def _board() -> CharucoBoardSpec:
    return CharucoBoardSpec(
        "s01-charuco-6x8-30mm-5x5-100-v1", "DICT_5X5_100", 6, 8, 0.030, 0.022
    )


def _estimate(
    camera_id: str, focal: float, k1: float, cv: float = 0.01
) -> CameraIntrinsicEstimate:
    return CameraIntrinsicEstimate(
        camera_id,
        "stream-profile-v1",
        "simple_radial",
        1920,
        1080,
        focal,
        focal,
        960.0,
        540.0,
        (k1,),
        cv,
    )


def _observation(
    frame_id: str,
    intrinsics: CameraIntrinsics,
    rvec: FloatArray,
    tvec: FloatArray,
) -> CharucoObservation:
    points = np.asarray(_board().create_board().getChessboardCorners(), dtype=np.float64)
    matrix = np.array(
        [
            [intrinsics.fx_pixels, 0, intrinsics.cx_pixels],
            [0, intrinsics.fy_pixels, intrinsics.cy_pixels],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    distortion = np.array([*intrinsics.distortion, 0, 0, 0, 0], dtype=np.float64)[:5]
    image, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
    pixels = image.reshape(-1, 2)
    return CharucoObservation(
        frame_id,
        1920,
        1080,
        tuple(range(len(points))),
        tuple((float(item[0]), float(item[1])) for item in pixels),
        0.08,
        4,
        100.0,
        120.0,
        0.0,
        (),
    )


def test_validates_board_and_detects_rendered_observation() -> None:
    board = _board()
    rendered = board.create_board().generateImage((900, 1200), marginSize=30)
    observation = detect_charuco_observation("synthetic-board", rendered, board)
    assert observation.accepted
    assert len(observation.corner_ids) == board.maximum_charuco_corners
    assert observation.coverage_fraction > 0.4
    with pytest.raises(CharucoCheckpointError, match="marker length"):
        CharucoBoardSpec("bad", "DICT_5X5_100", 6, 8, 0.03, 0.04)


def test_quality_score_ranks_stable_distributed_camera_and_rejects_failed() -> None:
    good = NaturalFrameMetrics(300, 120, 9, 500.0, 0.99)
    weak = NaturalFrameMetrics(100, 30, 4, 100.0, 0.90)
    ranked = rank_natural_camera_quality(
        (
            NaturalCameraEvidence("office-cam-01", (weak, weak), 0.03, True, True),
            NaturalCameraEvidence("office-cam-02", (good, good), 0.001, True, True),
            NaturalCameraEvidence("office-cam-03", (good, weak), 0.01, False, True),
        )
    )
    assert ranked[0].camera_id == "office-cam-02"
    assert ranked[-1].camera_id == "office-cam-03"
    assert not ranked[-1].eligible


def test_robust_cluster_rejects_focal_outlier_and_uses_median() -> None:
    estimates = (
        _estimate("office-cam-01", 1274.0, -0.265),
        _estimate("office-cam-02", 1402.0, -0.279),
        _estimate("office-cam-03", 1367.0, -0.261),
        _estimate("office-cam-04", 1410.0, -0.281),
    )
    cluster = robust_intrinsic_cluster(estimates)
    assert tuple(item.camera_id for item in cluster) == (
        "office-cam-02",
        "office-cam-03",
        "office-cam-04",
    )
    profile = median_cluster_profile(cluster)
    assert profile.fx_pixels == pytest.approx(1402.0)
    assert profile.distortion == pytest.approx((-0.279,))
    incompatible = CameraIntrinsicEstimate(
        "office-cam-04",
        "stream-profile-v2",
        "simple_radial",
        1920,
        1080,
        1410,
        1410,
        960,
        540,
        (-0.28,),
        0.01,
    )
    with pytest.raises(CharucoCheckpointError, match="compatible"):
        robust_intrinsic_cluster((*estimates[:3], incompatible))


def test_fixed_intrinsic_reprojection_estimates_pose_without_refitting() -> None:
    intrinsics = CameraIntrinsics("simple_radial", 1920, 1080, 1350, 1350, 960, 540, (-0.2,))
    observations = tuple(
        _observation(
            f"view-{index}",
            intrinsics,
            np.array([0.05 * index, -0.1, 0.03], dtype=np.float64),
            np.array([0.02 * index, -0.08, 0.7 + 0.05 * index], dtype=np.float64),
        )
        for index in range(3)
    )
    result = evaluate_fixed_intrinsics("candidate", intrinsics, observations, _board())
    assert result.median_error_pixels < 1e-4
    assert result.maximum_error_pixels < 1e-3
    assert intrinsics.fx_pixels == 1350


def test_calibration_and_insufficient_board_evidence() -> None:
    intrinsics = CameraIntrinsics("simple_radial", 1920, 1080, 1350, 1345, 955, 545, (-0.18,))
    observations = tuple(
        _observation(
            f"view-{index}",
            intrinsics,
            np.array([0.04 * index, -0.08 + 0.02 * index, 0.02], dtype=np.float64),
            np.array([-0.03 + 0.01 * index, -0.08, 0.65 + 0.06 * index], dtype=np.float64),
        )
        for index in range(8)
    )
    reference = calibrate_charuco_reference(observations, _board())
    assert reference.rms_error_pixels < 1e-2
    assert reference.intrinsics.fx_pixels == pytest.approx(1350, rel=0.02)
    with pytest.raises(CharucoCheckpointError, match="at least 6"):
        calibrate_charuco_reference(observations[:5], _board())


def test_policy_selection_is_deterministic_and_prefers_robust_tie() -> None:
    def evaluation(label: str, value: float) -> FixedProfileEvaluation:
        return FixedProfileEvaluation(label, (value,), (value,), (value, value), ())

    assert select_policy(
        evaluation("a", 1.0),
        evaluation("b", 1.1),
        adequacy_limit_pixels=3,
        material_tie_pixels=0.2,
    ) == "candidate-b-robust-cluster"
    assert select_policy(
        evaluation("a", 0.5),
        evaluation("b", 1.0),
        adequacy_limit_pixels=3,
        material_tie_pixels=0.2,
    ) == "candidate-a-quality-selected"
    assert select_policy(
        evaluation("a", 5.0),
        evaluation("b", 4.0),
        adequacy_limit_pixels=3,
        material_tie_pixels=0.2,
    ) == "fleet-prior-with-camera-specific-rollback"
