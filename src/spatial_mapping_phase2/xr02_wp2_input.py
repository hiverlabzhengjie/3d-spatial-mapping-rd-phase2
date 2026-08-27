"""Read-only adapter from immutable XR02 WP2 journals into WP3 observations."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_mapping_phase2.xr02_global_domain import (
    AssociationObservation,
    XR02AssociationContractError,
    canonical_sha256,
)
from spatial_mapping_phase2.xr02_journal import EmbeddingStore, verify_journal
from spatial_mapping_phase2.xr02_local_domain import EmbeddingReference


@dataclass(frozen=True, slots=True)
class WP2InputAudit:
    """Verified, read-only summary of one WP2 local-observation journal."""

    journal_path: str
    journal_records: int
    journal_final_sha256: str
    scene_context_sha256: str
    tracker_profile: str
    valid_observations: tuple[AssociationObservation, ...]
    rejected_projection_states: tuple[tuple[str, int], ...]
    missing_embedding_count: int
    observation_signature_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "journal_path": self.journal_path,
            "journal_records": self.journal_records,
            "journal_final_sha256": self.journal_final_sha256,
            "scene_context_sha256": self.scene_context_sha256,
            "tracker_profile": self.tracker_profile,
            "valid_observation_count": len(self.valid_observations),
            "rejected_projection_states": dict(self.rejected_projection_states),
            "missing_embedding_count": self.missing_embedding_count,
            "observation_signature_sha256": self.observation_signature_sha256,
            "cross_camera_timing_authority": (
                "not_claimed; WP2 camera clips were replayed sequentially and are "
                "schema inputs only"
            ),
        }


def load_wp2_journal_read_only(path: Path, expected_tracker_profile: str) -> WP2InputAudit:
    """Verify and consume a WP2 journal without changing it or its embedding store."""

    verification = verify_journal(path)
    run_root = path.parent
    embedding_store: EmbeddingStore | None = None
    observations: list[AssociationObservation] = []
    rejected: Counter[str] = Counter()
    missing_embedding_count = 0
    scene_contexts: set[str] = set()
    profiles: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _mapping(json.loads(line), f"journal line {line_number}")
            payload = _mapping(record.get("payload"), f"payload line {line_number}")
            if payload.get("schema") != "xr02.local_track_observation.v1":
                raise XR02AssociationContractError("WP2 observation schema changed")
            frame = _mapping(payload.get("frame"), "frame")
            track = _mapping(payload.get("track"), "track")
            scene_context = _string(frame, "scene_context_sha256")
            tracker_profile = _string(track, "tracker_profile")
            if track.get("scene_context_sha256") != scene_context:
                raise XR02AssociationContractError("WP2 frame and track contexts disagree")
            if frame.get("camera_id") != track.get("camera_id"):
                raise XR02AssociationContractError("WP2 frame and track cameras disagree")
            scene_contexts.add(scene_context)
            profiles.add(tracker_profile)

            projection_status = _string(payload, "projection_status")
            if projection_status != "valid":
                rejected[projection_status] += 1
                continue
            world_xy = payload.get("world_xy_metres")
            if not isinstance(world_xy, list) or len(world_xy) != 2:
                raise XR02AssociationContractError("valid WP2 projection lacks world XY")

            embedding_payload = payload.get("embedding")
            embedding_reference: EmbeddingReference | None = None
            embedding: tuple[float, ...] | None = None
            if embedding_payload is not None:
                data = _mapping(embedding_payload, "embedding")
                embedding_reference = EmbeddingReference(
                    sha256=_string(data, "sha256"),
                    model_id=_string(data, "model_id"),
                    dimension=_integer(data, "dimension"),
                    relative_path=_string(data, "relative_path"),
                )
                if embedding_store is None:
                    embedding_store = EmbeddingStore(run_root, embedding_reference.model_id)
                elif embedding_store.model_id != embedding_reference.model_id:
                    raise XR02AssociationContractError(
                        "WP2 embedding model changed within journal"
                    )
                embedding = tuple(
                    float(value) for value in embedding_store.load(embedding_reference)
                )
            else:
                missing_embedding_count += 1

            confidence = _number(payload, "confidence")
            crop_quality = _mapping(payload.get("crop_quality"), "crop_quality")
            visible_fraction = _number(crop_quality, "visible_fraction")
            quality_weight = max(1e-6, min(1.0, confidence * visible_fraction))
            frame_id = _string(frame, "frame_id")
            detection_index = _integer(payload, "detection_index")
            observations.append(
                AssociationObservation(
                    scene_context_sha256=scene_context,
                    observation_id=f"{frame_id}.d{detection_index}",
                    local_track_stable_id=_string(track, "stable_id"),
                    camera_id=_string(frame, "camera_id"),
                    tracker_profile=tracker_profile,
                    observed_monotonic_ns=_integer(frame, "acquisition_monotonic_ns"),
                    world_xy_metres=(float(world_xy[0]), float(world_xy[1])),
                    confidence=confidence,
                    quality_weight=quality_weight,
                    embedding_reference_sha256=(
                        None if embedding_reference is None else embedding_reference.sha256
                    ),
                    embedding=embedding,
                )
            )

    if profiles != {expected_tracker_profile}:
        raise XR02AssociationContractError(
            f"WP2 tracker profile mismatch: expected {expected_tracker_profile}, got {profiles}"
        )
    if len(scene_contexts) != 1:
        raise XR02AssociationContractError("WP2 journal must contain exactly one scene context")
    ordered = tuple(
        sorted(observations, key=lambda item: (item.observed_monotonic_ns, item.observation_id))
    )
    signature = canonical_sha256([item.as_dict() for item in ordered])
    return WP2InputAudit(
        journal_path=str(path.resolve()),
        journal_records=verification.records,
        journal_final_sha256=verification.final_sha256,
        scene_context_sha256=next(iter(scene_contexts)),
        tracker_profile=expected_tracker_profile,
        valid_observations=ordered,
        rejected_projection_states=tuple(sorted(rejected.items())),
        missing_embedding_count=missing_embedding_count,
        observation_signature_sha256=signature,
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise XR02AssociationContractError(f"{label} must be an object")
    return value


def _string(value: dict[str, Any], key: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str) or not selected:
        raise XR02AssociationContractError(f"{key} must be a non-empty string")
    return selected


def _integer(value: dict[str, Any], key: str) -> int:
    selected = value.get(key)
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise XR02AssociationContractError(f"{key} must be an integer")
    return selected


def _number(value: dict[str, Any], key: str) -> float:
    selected = value.get(key)
    if not isinstance(selected, int | float) or isinstance(selected, bool):
        raise XR02AssociationContractError(f"{key} must be numeric")
    return float(selected)
