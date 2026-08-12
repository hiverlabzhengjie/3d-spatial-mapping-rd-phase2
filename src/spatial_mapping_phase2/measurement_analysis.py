"""Numerical summaries for retained P00 model-output repeatability evidence."""

from __future__ import annotations

from typing import Any


def summarize_array_difference(reference: Any, candidate: Any) -> dict[str, object]:
    """Return exact and numerical comparison indicators without setting acceptance criteria."""
    import numpy as np

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Repeatability arrays must have matching shapes: "
            f"{reference_array.shape} != {candidate_array.shape}."
        )
    absolute_delta = np.abs(reference_array - candidate_array)
    return {
        "exact_equal": bool(np.array_equal(reference_array, candidate_array)),
        "element_count": int(absolute_delta.size),
        "different_element_count": int(np.count_nonzero(absolute_delta)),
        "max_abs_delta": float(absolute_delta.max()),
        "mean_abs_delta": float(absolute_delta.mean()),
        "allclose_rtol_1e-5_atol_1e-6": bool(
            np.allclose(reference_array, candidate_array, rtol=1e-5, atol=1e-6)
        ),
    }

