"""Validation for the bounded P00 supporting-stack smoke manifest."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REQUIRED_SUPPORTING_STACK_COMPONENTS = frozenset(
    {
        "geocalib",
        "opencv",
        "scipy",
        "open3d",
        "rerun",
        "pyav_ffmpeg",
        "pillow_heif",
        "pre_commit",
        "web",
    }
)


class SupportingStackContractError(ValueError):
    """Raised when a P00 supporting-stack smoke manifest is incomplete or failed."""


def validate_supporting_stack_result(result: Mapping[str, Any]) -> None:
    """Require a successful status record for every bounded P00 stack component."""
    if result.get("success") is not True:
        raise SupportingStackContractError("The supporting-stack smoke result is not successful.")
    components = result.get("components")
    if not isinstance(components, Mapping):
        raise SupportingStackContractError("The supporting-stack result must contain components.")
    missing = REQUIRED_SUPPORTING_STACK_COMPONENTS - components.keys()
    if missing:
        raise SupportingStackContractError(
            f"The supporting-stack result omits: {', '.join(sorted(missing))}."
        )
    failed = sorted(
        name
        for name in REQUIRED_SUPPORTING_STACK_COMPONENTS
        if not isinstance(components[name], Mapping) or components[name].get("success") is not True
    )
    if failed:
        raise SupportingStackContractError(
            f"The supporting-stack result records failed component(s): {', '.join(failed)}."
        )
