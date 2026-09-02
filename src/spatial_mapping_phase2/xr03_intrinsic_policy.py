"""Lens-group-aware intrinsic candidate orchestration for future XR03 runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from spatial_mapping_phase2.p04_intrinsic_fleet import (
    CameraIntrinsicEstimate,
    IntrinsicFleetError,
    build_fleet_profiles,
)
from spatial_mapping_phase2.xr03_camera_policy import SceneCameraPolicy


class GroupedIntrinsicPolicyError(ValueError):
    """Raised when independent estimates cannot be evaluated under the active lens policy."""


def build_grouped_intrinsic_candidates(
    estimates: Sequence[CameraIntrinsicEstimate],
    policy: SceneCameraPolicy,
) -> dict[str, Any]:
    """Build leave-one-camera-out candidates without crossing a declared lens group.

    This function preserves D028/D033: one independent estimate per camera, exact model/profile/
    resolution compatibility, equal-camera votes, and at least three peer cameras after exclusion.
    It creates eligible candidates; it never forces assignment of a pooled candidate.
    """

    if not policy.lens_complete:
        raise GroupedIntrinsicPolicyError(
            "lens-model grouping must cover every enabled camera before intrinsic processing"
        )
    by_camera: dict[str, CameraIntrinsicEstimate] = {}
    for estimate in estimates:
        if estimate.camera_id in by_camera:
            raise GroupedIntrinsicPolicyError(
                "each enabled camera must contribute exactly one independent estimate"
            )
        by_camera[estimate.camera_id] = estimate
    if set(by_camera) != set(policy.camera_ids):
        missing = sorted(set(policy.camera_ids) - set(by_camera))
        extra = sorted(set(by_camera) - set(policy.camera_ids))
        raise GroupedIntrinsicPolicyError(
            f"intrinsic estimates do not match the policy roster; missing={missing}, extra={extra}"
        )

    group_results: list[dict[str, Any]] = []
    for group in policy.intrinsic_groups:
        group_estimates = tuple(by_camera[camera_id] for camera_id in group.camera_ids)
        targets: list[dict[str, Any]] = []
        for camera_id in group.camera_ids:
            try:
                candidates = build_fleet_profiles(group_estimates, exclude_camera_id=camera_id)
            except IntrinsicFleetError as error:
                targets.append(
                    {
                        "camera_id": camera_id,
                        "status": "insufficient-or-incompatible",
                        "reason": str(error),
                        "candidates": [],
                        "individual_estimate_retained": True,
                    }
                )
            else:
                targets.append(
                    {
                        "camera_id": camera_id,
                        "status": "candidates-ready",
                        "reason": None,
                        "candidates": [candidate.to_dict() for candidate in candidates],
                        "individual_estimate_retained": True,
                    }
                )
        group_results.append(
            {
                "group_id": group.group_id,
                "lens_model": group.lens_model,
                "camera_ids": list(group.camera_ids),
                "targets": targets,
            }
        )
    return {
        "schema_version": "xr03-grouped-intrinsic-candidates-v1",
        "camera_policy_sha256": policy.sha256,
        "camera_ids": list(policy.camera_ids),
        "assignment_policy": "eligible-candidates-only-no-forced-sharing",
        "groups": group_results,
    }
