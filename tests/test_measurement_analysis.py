import numpy as np
import pytest

from spatial_mapping_phase2.measurement_analysis import summarize_array_difference


def test_summarize_array_difference_reports_exact_match() -> None:
    result = summarize_array_difference(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    assert result == {
        "exact_equal": True,
        "element_count": 2,
        "different_element_count": 0,
        "max_abs_delta": 0.0,
        "mean_abs_delta": 0.0,
        "allclose_rtol_1e-5_atol_1e-6": True,
    }


def test_summarize_array_difference_reports_numerical_drift() -> None:
    result = summarize_array_difference(np.array([1.0, 2.0]), np.array([1.0, 2.1]))

    assert result["exact_equal"] is False
    assert result["different_element_count"] == 1
    assert result["max_abs_delta"] == pytest.approx(0.1)
    assert result["mean_abs_delta"] == pytest.approx(0.05)
    assert result["allclose_rtol_1e-5_atol_1e-6"] is False


def test_summarize_array_difference_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        summarize_array_difference(np.array([1.0]), np.array([[1.0]]))
