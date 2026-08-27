from pathlib import Path

import pytest

from spatial_mapping_phase2.xr02_wp1 import (
    XR02WP1Error,
    batches,
    identify_file,
    summarize_ms,
)


def test_identify_file_matches_expected_hash(tmp_path: Path) -> None:
    source = tmp_path / "asset.bin"
    source.write_bytes(b"xr02")

    identity = identify_file(
        source,
        expected_sha256="5252ddffddd8a7f6b976778eeee1f4c70e1a789bab11b315630b719f9ecf321c",
    )

    assert identity.bytes == 4


def test_identify_file_rejects_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "asset.bin"
    source.write_bytes(b"xr02")

    with pytest.raises(XR02WP1Error, match="SHA-256 mismatch"):
        identify_file(source, expected_sha256="0" * 64)


def test_summarize_ms_uses_linear_p95() -> None:
    summary = summarize_ms([10.0, 20.0, 30.0, 40.0])

    assert summary.median_ms == 25.0
    assert summary.p95_ms == pytest.approx(38.5)
    assert summary.mean_ms == 25.0


@pytest.mark.parametrize("values", [[], [-1.0], [float("nan")]])
def test_summarize_ms_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(XR02WP1Error):
        summarize_ms(values)


def test_batches_are_contiguous_and_reject_invalid_size() -> None:
    assert list(batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    with pytest.raises(XR02WP1Error, match="batch_size"):
        list(batches([1], 0))
