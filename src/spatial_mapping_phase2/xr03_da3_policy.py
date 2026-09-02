"""Typed DA3 scene-cohort contract kept deliberately independent of view overlap."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SceneDa3PolicyError(ValueError):
    """Raised when a scene cohort cannot be submitted as one DA3 invocation."""


@dataclass(frozen=True, slots=True)
class SceneDa3Cohort:
    camera_ids: tuple[str, ...]
    camera_policy_sha256: str

    @classmethod
    def build(cls, camera_ids: Sequence[str], camera_policy_sha256: str) -> SceneDa3Cohort:
        return cls(tuple(camera_ids), camera_policy_sha256)

    def __post_init__(self) -> None:
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise SceneDa3PolicyError(
                "scene DA3 cohort must contain a non-empty unique camera roster"
            )
        if any(not camera_id.strip() for camera_id in self.camera_ids):
            raise SceneDa3PolicyError("scene DA3 cohort camera IDs must not be blank")
        if not _SHA256.fullmatch(self.camera_policy_sha256):
            raise SceneDa3PolicyError("scene DA3 cohort requires a lowercase policy SHA-256")

    @property
    def inference_mode(self) -> str:
        return (
            "joint-pose-conditioned-multi-view"
            if len(self.camera_ids) > 1
            else "pose-conditioned-single-view"
        )

    def cli_arguments(self) -> tuple[str, ...]:
        arguments: list[str] = []
        for camera_id in self.camera_ids:
            arguments.extend(("--camera-id", camera_id))
        arguments.extend(("--camera-policy-sha256", self.camera_policy_sha256))
        return tuple(arguments)

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_ids": list(self.camera_ids),
            "camera_policy_sha256": self.camera_policy_sha256,
            "cohort_policy": "all-enabled-cameras-per-scene-joint",
            "inference_mode": self.inference_mode,
            "view_overlap_influence": "none",
        }
