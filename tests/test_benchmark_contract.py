from dataclasses import replace

import pytest

from spatial_mapping_phase2.benchmark_contract import (
    BenchmarkCase,
    BenchmarkContractError,
    BenchmarkRunManifest,
)

SHA256 = "a" * 64


def valid_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="one-view-512p",
        view_count=1,
        width_px=512,
        height_px=512,
        batch_size=1,
        input_manifest_sha256=SHA256,
    )


def test_benchmark_case_accepts_required_one_two_and_three_view_counts() -> None:
    for view_count in (1, 2, 3):
        replace(valid_case(), view_count=view_count).validate()


def test_benchmark_case_rejects_uncontrolled_batch_and_view_count() -> None:
    with pytest.raises(BenchmarkContractError, match="view count"):
        replace(valid_case(), view_count=4).validate()
    with pytest.raises(BenchmarkContractError, match="batch size"):
        replace(valid_case(), batch_size=2).validate()


def test_run_manifest_requires_complete_provenance() -> None:
    manifest = BenchmarkRunManifest.create(
        runtime_id="native-windows-py311",
        code_revision="a1b2c3d",
        dependency_lock_sha256=SHA256,
        checkpoint_sha256=SHA256,
        case=valid_case(),
    )

    assert manifest.to_dict()
    assert manifest.case.view_count == 1


def test_run_manifest_rejects_missing_checkpoint_identity() -> None:
    manifest = BenchmarkRunManifest.create(
        runtime_id="native-windows-py311",
        code_revision="a1b2c3d",
        dependency_lock_sha256=SHA256,
        checkpoint_sha256="not-a-hash",
        case=valid_case(),
    )

    with pytest.raises(BenchmarkContractError, match="Checkpoint"):
        manifest.validate()
