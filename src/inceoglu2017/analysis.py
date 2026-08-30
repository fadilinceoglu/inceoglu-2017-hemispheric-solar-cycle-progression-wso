"""Numerical primitives used by the Inceoglu et al. (2017) analysis.

The functions in this module reproduce the paper-specific MATLAB semantics
without retaining a MATLAB dependency. They are deliberately
independent of files and of the published table values: arrays enter, derived
arrays leave, and the validation targets live in :mod:`inceoglu2017.events`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t


DEFAULT_SMOOTHING_SPAN = 11
DEFAULT_CONFIDENCE_LEVEL = 0.99


def _finite_time_series(values: np.ndarray, name: str = "values") -> np.ndarray:
    """Return a finite float array whose first dimension is time."""

    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _validate_span(span: int, sample_count: int) -> int:
    if isinstance(span, bool) or not isinstance(span, (int, np.integer)):
        raise TypeError("span must be an integer")
    width = int(span)
    if width < 1 or width % 2 == 0:
        raise ValueError("span must be a positive odd integer")
    if width > sample_count:
        raise ValueError("span cannot exceed the number of samples")
    return width


def matlab_smooth(
    values: np.ndarray,
    span: int = DEFAULT_SMOOTHING_SPAN,
) -> np.ndarray:
    """Apply the paper's centered moving mean, including its edge rule.

    MATLAB's ``smooth(x, 11)`` uses 11 samples in the interior.  At each edge
    it uses the largest *centered odd* window that fits: 1, 3, 5, 7, 9, then
    11 samples.  A convolution with zero padding or a truncated asymmetric
    window therefore does not reproduce the paper result.

    One-dimensional input returns one-dimensional output.  For a two-
    dimensional array, rows are time samples and every column is smoothed
    independently.
    """

    array = _finite_time_series(values)
    width = _validate_span(span, array.shape[0])
    maximum_half_width = width // 2
    result = np.empty_like(array, dtype=float)

    for index in range(array.shape[0]):
        half_width = min(
            maximum_half_width,
            index,
            array.shape[0] - index - 1,
        )
        result[index] = np.mean(
            array[index - half_width : index + half_width + 1], axis=0
        )
    return result


@dataclass(frozen=True)
class SmoothedSeries:
    """Two-pass moving mean and the historical 99% confidence half-width."""

    first_pass: np.ndarray
    values: np.ndarray
    confidence_half_width: np.ndarray
    sample_counts: np.ndarray
    span: int
    confidence_level: float

    @property
    def lower(self) -> np.ndarray:
        """Lower pointwise confidence bound."""

        return self.values - self.confidence_half_width

    @property
    def upper(self) -> np.ndarray:
        """Upper pointwise confidence bound."""

        return self.values + self.confidence_half_width


def smooth_with_confidence(
    values: np.ndarray,
    span: int = DEFAULT_SMOOTHING_SPAN,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> SmoothedSeries:
    """Apply two smoothing passes and reproduce the pointwise confidence band.

    The plotted mean is ``smooth(smooth(x, 11), 11)``.  At each point, the
    standard error is calculated from the values in the corresponding second-
    pass window of the *once-smoothed* series.  The standard deviation uses
    ``n - 1`` degrees of freedom and the two-sided Student-t multiplier.  The
    one-sample endpoints have zero half-width by construction.
    """

    array = _finite_time_series(values)
    width = _validate_span(span, array.shape[0])
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    first_pass = matlab_smooth(array, width)
    second_pass = matlab_smooth(first_pass, width)
    half_width = np.zeros_like(second_pass, dtype=float)
    sample_counts = np.empty(array.shape[0], dtype=int)
    maximum_half_width = width // 2
    tail_probability = (1.0 - confidence) / 2.0

    for index in range(array.shape[0]):
        edge_half_width = min(
            maximum_half_width,
            index,
            array.shape[0] - index - 1,
        )
        window = first_pass[
            index - edge_half_width : index + edge_half_width + 1
        ]
        count = window.shape[0]
        sample_counts[index] = count
        if count > 1:
            standard_error = np.std(window, axis=0, ddof=1) / np.sqrt(count)
            half_width[index] = (
                t.ppf(1.0 - tail_probability, count - 1) * standard_error
            )

    return SmoothedSeries(
        first_pass=first_pass,
        values=second_pass,
        confidence_half_width=half_width,
        sample_counts=sample_counts,
        span=width,
        confidence_level=float(confidence),
    )


@dataclass(frozen=True)
class CrossCovariance:
    """Normalized cross-covariance over integer sample lags."""

    lags: np.ndarray
    coefficients: np.ndarray
    best_lag: int
    best_coefficient: float


def xcov_coeff(
    reference: np.ndarray,
    comparison: np.ndarray,
    max_lag: int | None = None,
) -> CrossCovariance:
    """Reproduce MATLAB ``xcov(reference, comparison, 'coeff')``.

    Both complete series are mean-centered once.  Every lag shares the fixed
    normalization ``sqrt(sum(x**2) * sum(y**2))``; only the numerator's
    overlap changes.  A positive lag means the reference series is delayed
    relative to the comparison series.  This convention is the one used for
    the published onset delays.
    """

    x = _finite_time_series(reference, "reference")
    y = _finite_time_series(comparison, "comparison")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("xcov_coeff requires one-dimensional series")
    if x.shape != y.shape:
        raise ValueError("reference and comparison must have the same length")

    largest_lag = x.size - 1
    if max_lag is None:
        requested_lag = largest_lag
    else:
        if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)):
            raise TypeError("max_lag must be an integer or None")
        requested_lag = int(max_lag)
        if not 0 <= requested_lag <= largest_lag:
            raise ValueError(f"max_lag must lie between 0 and {largest_lag}")

    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = np.sqrt(
        np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)
    )
    if denominator == 0.0:
        raise ValueError("cross-covariance is undefined for a constant series")

    lags = np.arange(-requested_lag, requested_lag + 1, dtype=int)
    coefficients = np.empty(lags.size, dtype=float)
    for position, lag in enumerate(lags):
        if lag < 0:
            numerator = np.dot(centered_x[:lag], centered_y[-lag:])
        elif lag > 0:
            numerator = np.dot(centered_x[lag:], centered_y[:-lag])
        else:
            numerator = np.dot(centered_x, centered_y)
        coefficients[position] = numerator / denominator

    best_position = int(np.argmax(coefficients))
    return CrossCovariance(
        lags=lags,
        coefficients=coefficients,
        best_lag=int(lags[best_position]),
        best_coefficient=float(coefficients[best_position]),
    )


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return the ordinary sample Pearson correlation of two finite vectors."""

    first = _finite_time_series(x, "x")
    second = _finite_time_series(y, "y")
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("pearson_correlation requires one-dimensional arrays")
    if first.shape != second.shape:
        raise ValueError("x and y must have the same length")
    if first.size < 2:
        raise ValueError("at least two paired samples are required")
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        raise ValueError("correlation is undefined for a constant sample")
    return float(np.corrcoef(first, second)[0, 1])


def fisher_bias_correction(correlation: float, sample_size: int) -> float:
    """Apply the small-sample correction quoted from Fisher (1915)."""

    r = float(correlation)
    if not np.isfinite(r) or not -1.0 <= r <= 1.0:
        raise ValueError("correlation must be finite and lie between -1 and 1")
    if isinstance(sample_size, bool) or not isinstance(
        sample_size, (int, np.integer)
    ):
        raise TypeError("sample_size must be an integer")
    n = int(sample_size)
    if n < 1:
        raise ValueError("sample_size must be positive")
    return float(r * (1.0 + (1.0 - r * r) / (2.0 * n)))


def adjusted_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson's r after the paper's Fisher small-sample correction."""

    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    r = pearson_correlation(first, second)
    return fisher_bias_correction(r, first.size)
