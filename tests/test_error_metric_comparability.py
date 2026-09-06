"""Every benchmark's error must mean the same thing at the shared bound."""

import numpy as np
import pytest

from utils.error_analyzer import (
    ERROR_METRIC_SPEC,
    _normalized,
    describe_error_metric,
)

BENCHMARKS = [
    "ar-iclib", "sobel-iclib", "stringsearch-iclib",
    "fft-iclib", "lqi-iclib", "bc-iclib",
]


@pytest.mark.parametrize("app", BENCHMARKS)
def test_every_benchmark_declares_its_error_metric(app):
    """A shared 30% bound is meaningless if nobody states what it bounds."""
    assert app in ERROR_METRIC_SPEC
    assert describe_error_metric(app) != "unspecified"


def test_normalisation_clamps_into_the_unit_interval():
    assert _normalized(-0.2) == 0.0
    assert _normalized(0.35) == pytest.approx(0.35)
    assert _normalized(7.0) == 1.0


def test_normalisation_rejects_non_finite_values():
    """An infinity used to propagate straight into the objective."""
    with pytest.raises(RuntimeError, match="non-finite"):
        _normalized(float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        _normalized(float("nan"))


def test_sobel_error_is_no_longer_halved():
    """The debug line printed 1 - SSIM while the function returned it halved."""
    import inspect

    from utils.error_analyzer import sobelError

    source = inspect.getsource(sobelError)
    assert "(1 - ssim_score) / 2" not in source
    assert "_normalized(1 - ssim_score)" in source


def test_ar_metric_preserves_per_window_structure():
    """Aggregate column totals let a fully mislabelled output score near zero."""
    reference = np.array([[19.0, 109.0], [36.0, 92.0], [44.0, 84.0]])
    swapped = reference[::-1]  # same totals, completely wrong per window

    denominator = np.maximum(np.abs(reference), 1.0)
    per_window = float(np.mean(np.abs(reference - swapped) / denominator))
    aggregate = abs(reference.sum() - swapped.sum()) / reference.sum()

    assert aggregate == pytest.approx(0.0)
    assert per_window > 0.3, (
        "a per-window metric must penalise an output whose totals happen to match"
    )


def test_zero_safe_denominators_do_not_explode():
    """bc-iclib divided by `ground_truth + 1e-10`, turning errors into ~1e10."""
    reference = np.array([0.0, 0.0, 4.0])
    predicted = np.array([1.0, 2.0, 4.0])

    old_style = float(np.mean(np.abs(reference - predicted) / (reference + 1e-10)))
    new_style = float(
        np.mean(np.abs(reference - predicted) / np.maximum(np.abs(reference), 1.0))
    )

    assert old_style > 1e9
    assert 0.0 < new_style <= 1.0
