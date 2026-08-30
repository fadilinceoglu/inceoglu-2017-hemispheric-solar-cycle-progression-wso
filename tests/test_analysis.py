from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t

from inceoglu2017.analysis import (
    adjusted_pearson,
    fisher_bias_correction,
    matlab_smooth,
    smooth_with_confidence,
    xcov_coeff,
)
from inceoglu2017.events import (
    BAND_NAMES,
    EXPECTED_TABLE1_MAXIMUM_ROTATIONS,
    EXPECTED_TABLE1_MINIMUM_ROTATIONS,
    EXPECTED_FIRST_PEAK_ROTATIONS,
    EXPECTED_LAST_PEAK_ROTATIONS,
    EXPECTED_TABLE2_CORRELATIONS,
    ONE_YEAR_CARRINGTON_ROTATIONS,
    PUBLISHED_CORRELATION_TOLERANCE,
    PUBLISHED_TABLE1_CYCLE_LENGTHS,
    PUBLISHED_TABLE1_MAXIMUM_DATES,
    PUBLISHED_TABLE1_MINIMUM_DATES,
    PUBLISHED_TABLE2_CORRELATIONS,
    PUBLISHED_TABLE2_LAGS,
    PUBLISHED_TABLE3,
    PUBLISHED_TABLE3_S_0_15_PEAK_TOLERANCE,
    PUBLISHED_TABLE4,
    PUBLISHED_TABLE5,
    analyze_study,
    nearest_year_index,
    significant_peak_indices,
)


ROOT = Path(__file__).resolve().parents[1]
WSO_ENERGY = ROOT / "data" / "wso_energy_cr1642_2182.csv"


def test_matlab_smooth_uses_centered_odd_edge_windows() -> None:
    values = np.asarray([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    expected = np.asarray([0.0, 5.0 / 3.0, 6.0, 11.0, 18.0, 77.0 / 3.0, 36.0])

    np.testing.assert_allclose(matlab_smooth(values, span=5), expected)

    matrix = np.column_stack((values, 2.0 * values))
    smoothed = matlab_smooth(matrix, span=5)
    assert smoothed.shape == matrix.shape
    np.testing.assert_allclose(smoothed[:, 0], expected)
    np.testing.assert_allclose(smoothed[:, 1], 2.0 * expected)


def test_double_smoothing_confidence_uses_once_smoothed_windows() -> None:
    raw = np.asarray([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    result = smooth_with_confidence(raw, span=5, confidence=0.99)

    np.testing.assert_allclose(result.values, matlab_smooth(result.first_pass, 5))
    np.testing.assert_array_equal(result.sample_counts, [1, 3, 5, 5, 5, 3, 1])
    assert result.confidence_half_width[0] == 0.0
    assert result.confidence_half_width[-1] == 0.0

    center_window = result.first_pass[1:6]
    expected_half_width = (
        t.ppf(0.995, 4)
        * np.std(center_window, ddof=1)
        / np.sqrt(center_window.size)
    )
    assert result.confidence_half_width[3] == pytest.approx(expected_half_width)
    np.testing.assert_allclose(
        result.upper - result.lower, 2.0 * result.confidence_half_width
    )


def test_xcov_coeff_has_the_published_positive_lag_convention() -> None:
    comparison = np.asarray([1.0, -2.0, 3.0, 5.0, -1.0, 2.0, 0.0])
    # The same pattern in reference is delayed by two samples.
    reference = np.concatenate(([0.0, 0.0], comparison[:-2]))

    result = xcov_coeff(reference, comparison, max_lag=4)

    assert result.best_lag == 2
    assert result.best_coefficient == pytest.approx(0.9437939116158042)
    zero_lag = int(np.flatnonzero(result.lags == 0)[0])
    centered_reference = reference - np.mean(reference)
    centered_comparison = comparison - np.mean(comparison)
    expected_zero = np.dot(centered_reference, centered_comparison) / np.sqrt(
        np.dot(centered_reference, centered_reference)
        * np.dot(centered_comparison, centered_comparison)
    )
    assert result.coefficients[zero_lag] == pytest.approx(expected_zero)


def test_fisher_small_sample_correction_is_the_published_equation() -> None:
    assert fisher_bias_correction(0.5, 4) == pytest.approx(0.546875)
    assert adjusted_pearson([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]) == pytest.approx(1.0)


def test_secondary_peaks_require_height_and_prominence() -> None:
    # Index 0 is the primary endpoint, index 2 is a genuine secondary peak,
    # and the low index-4 peak is excluded by the 50% height rule.
    values = np.asarray([10.0, 0.0, 6.0, 0.0, 4.9, 0.0])

    primary, retained = significant_peak_indices(values, 0, values.size)

    assert primary == 0
    np.testing.assert_array_equal(retained, [0, 2])


def test_nearest_one_year_sample_is_thirteen_carrington_rotations() -> None:
    # A regular Carrington cadence is 27.2753 days, so 13 rotations are nearer
    # one year than either 12 or 14.
    dates = 1976.0 + np.arange(80) * 27.2753 / 365.2425

    assert nearest_year_index(dates, 40, -1) == 40 - ONE_YEAR_CARRINGTON_ROTATIONS
    assert nearest_year_index(dates, 40, 1) == 40 + ONE_YEAR_CARRINGTON_ROTATIONS


def _load_committed_energy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with WSO_ENERGY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"carrington_rotation", "analysis_decimal_year", *BAND_NAMES}
    if not rows or not required.issubset(rows[0]):
        missing = required - (set(rows[0]) if rows else set())
        raise AssertionError(f"WSO energy input is missing columns: {sorted(missing)}")
    rotations = np.asarray([int(row["carrington_rotation"]) for row in rows])
    dates = np.asarray([float(row["analysis_decimal_year"]) for row in rows])
    energy = np.asarray(
        [[float(row[band]) for band in BAND_NAMES] for row in rows], dtype=float
    )
    return rotations, dates, energy


@pytest.mark.skipif(
    not WSO_ENERGY.is_file(),
    reason="WSO energy input has not yet been prepared",
)
def test_committed_energy_reproduces_tables_1_to_5() -> None:
    rotations, dates, energy = _load_committed_energy()
    result = analyze_study(rotations, dates, energy)

    # Table 1: event rotations and every value at the paper's printed precision.
    np.testing.assert_array_equal(
        result.table1.minimum_rotations, EXPECTED_TABLE1_MINIMUM_ROTATIONS
    )
    np.testing.assert_array_equal(
        result.table1.maximum_rotations, EXPECTED_TABLE1_MAXIMUM_ROTATIONS
    )
    np.testing.assert_array_equal(
        rotations[result.events.first_peak_indices], EXPECTED_FIRST_PEAK_ROTATIONS
    )
    np.testing.assert_array_equal(
        rotations[result.events.last_peak_indices], EXPECTED_LAST_PEAK_ROTATIONS
    )
    np.testing.assert_array_equal(
        np.round(result.table1.minimum_dates, 1), PUBLISHED_TABLE1_MINIMUM_DATES
    )
    np.testing.assert_array_equal(
        np.round(result.table1.maximum_dates, 1), PUBLISHED_TABLE1_MAXIMUM_DATES
    )
    np.testing.assert_array_equal(
        np.round(result.table1.cycle_lengths, 1), PUBLISHED_TABLE1_CYCLE_LENGTHS
    )

    # The peak endpoints and every one-year selection are exact.
    np.testing.assert_array_equal(
        result.metrics.rise_sample_indices - result.events.first_peak_indices,
        -ONE_YEAR_CARRINGTON_ROTATIONS,
    )
    np.testing.assert_array_equal(
        result.metrics.decay_start_indices - result.events.last_peak_indices[:, :3],
        ONE_YEAR_CARRINGTON_ROTATIONS,
    )
    np.testing.assert_array_equal(
        result.metrics.decay_end_indices - result.events.minimum_indices[:, 1:],
        -ONE_YEAR_CARRINGTON_ROTATIONS,
    )

    # Table 2: full-precision calculated values, lags, and published rounding.
    np.testing.assert_allclose(
        result.table2.correlations,
        EXPECTED_TABLE2_CORRELATIONS,
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_array_equal(result.table2.lags, PUBLISHED_TABLE2_LAGS)
    np.testing.assert_array_equal(
        np.round(result.table2.correlations, 2), PUBLISHED_TABLE2_CORRELATIONS
    )

    # Tables 3--5 contain correlations based on only three or four points.  Use
    # the declared numerical-reproduction tolerance, apart from the one known
    # Table-3 typesetting/calculation inconsistency asserted immediately below.
    np.testing.assert_allclose(
        result.table3.peak_time_amplitude[[0, 1, 2, 4, 5]],
        PUBLISHED_TABLE3.peak_time_amplitude[[0, 1, 2, 4, 5]],
        rtol=0.0,
        atol=PUBLISHED_CORRELATION_TOLERANCE,
    )
    exceptional_delta = abs(
        result.table3.peak_time_amplitude[3]
        - PUBLISHED_TABLE3.peak_time_amplitude[3]
    )
    assert exceptional_delta > PUBLISHED_CORRELATION_TOLERANCE
    assert exceptional_delta <= PUBLISHED_TABLE3_S_0_15_PEAK_TOLERANCE
    np.testing.assert_allclose(
        result.table3.rise_time_amplitude,
        PUBLISHED_TABLE3.rise_time_amplitude,
        rtol=0.0,
        atol=PUBLISHED_CORRELATION_TOLERANCE,
    )
    np.testing.assert_allclose(
        result.table4.peak_time_amplitude,
        PUBLISHED_TABLE4.peak_time_amplitude,
        rtol=0.0,
        atol=PUBLISHED_CORRELATION_TOLERANCE,
    )
    np.testing.assert_allclose(
        result.table4.rise_time_amplitude,
        PUBLISHED_TABLE4.rise_time_amplitude,
        rtol=0.0,
        atol=PUBLISHED_CORRELATION_TOLERANCE,
    )
    np.testing.assert_allclose(
        result.table5.decay_peak_amplitude,
        PUBLISHED_TABLE5.decay_peak_amplitude,
        rtol=0.0,
        atol=PUBLISHED_CORRELATION_TOLERANCE,
    )
    np.testing.assert_allclose(
        result.table5.decay_rise_amplitude,
        PUBLISHED_TABLE5.decay_rise_amplitude,
        rtol=0.0,
        atol=PUBLISHED_CORRELATION_TOLERANCE,
    )
