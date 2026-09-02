"""Lightweight XR02 scene-topology contract without inference/runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from spatial_mapping_phase2.xr02_global_domain import XR02AssociationContractError


@dataclass(frozen=True, slots=True)
class SceneTopology:
    overlap_edges: tuple[tuple[str, str], ...]
    transition_edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for edges, label in (
            (self.overlap_edges, "overlap"),
            (self.transition_edges, "transition"),
        ):
            normalized = {_edge(left, right) for left, right in edges}
            if len(normalized) != len(edges):
                raise XR02AssociationContractError(f"{label} topology edges are duplicated")

    def overlaps(self, left: str, right: str) -> bool:
        return left != right and _edge(left, right) in {_edge(a, b) for a, b in self.overlap_edges}

    def can_transition(self, left: str, right: str) -> bool:
        return left == right or _edge(left, right) in {
            _edge(a, b) for a, b in self.transition_edges
        }


def _edge(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise XR02AssociationContractError("topology edges cannot self-reference")
    return (left, right) if left < right else (right, left)
