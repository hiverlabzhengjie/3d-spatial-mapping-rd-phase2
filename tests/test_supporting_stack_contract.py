from __future__ import annotations

import pytest

from spatial_mapping_phase2.supporting_stack_contract import (
    REQUIRED_SUPPORTING_STACK_COMPONENTS,
    SupportingStackContractError,
    validate_supporting_stack_result,
)


def _success_result() -> dict[str, object]:
    return {
        "success": True,
        "components": {
            name: {"success": True} for name in REQUIRED_SUPPORTING_STACK_COMPONENTS
        },
    }


def test_validate_supporting_stack_result_accepts_complete_success() -> None:
    validate_supporting_stack_result(_success_result())


def test_validate_supporting_stack_result_rejects_unsuccessful_manifest() -> None:
    result = _success_result()
    result["success"] = False

    with pytest.raises(SupportingStackContractError, match="not successful"):
        validate_supporting_stack_result(result)


def test_validate_supporting_stack_result_rejects_missing_component() -> None:
    result = _success_result()
    components = result["components"]
    assert isinstance(components, dict)
    components.pop("rerun")

    with pytest.raises(SupportingStackContractError, match="omits: rerun"):
        validate_supporting_stack_result(result)


def test_validate_supporting_stack_result_rejects_component_failure() -> None:
    result = _success_result()
    components = result["components"]
    assert isinstance(components, dict)
    components["open3d"] = {"success": False}

    with pytest.raises(SupportingStackContractError, match=r"failed component\(s\): open3d"):
        validate_supporting_stack_result(result)
