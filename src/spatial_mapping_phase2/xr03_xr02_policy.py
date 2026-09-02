"""Convert XR03 pairwise overlap authority into the isolated XR02 dedup topology."""

from __future__ import annotations

from collections.abc import Sequence

from spatial_mapping_phase2.xr02_topology import SceneTopology
from spatial_mapping_phase2.xr03_camera_policy import CameraPolicyError, SceneCameraPolicy


def topology_from_camera_policy(
    policy: SceneCameraPolicy,
    *,
    transition_edges: Sequence[tuple[str, str]],
) -> SceneTopology:
    """Use explicit overlap pairs only; retain independently authorized transition edges."""

    if not policy.overlap_complete:
        raise CameraPolicyError(
            "view-overlap review must be complete before starting a new XR02 scene epoch"
        )
    return SceneTopology(
        overlap_edges=policy.overlap_edges,
        transition_edges=tuple(transition_edges),
    )
