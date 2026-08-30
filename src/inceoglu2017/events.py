"""Solar-cycle event definitions and Tables 1--5 for the 2017 study.

All table values are calculated from the six twice-smoothed WSO regional
energy series.  The published arrays at the end of this module are validation
targets only; no analysis function reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import find_peaks

from .analysis import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_SMOOTHING_SPAN,
    SmoothedSeries,
    adjusted_pearson,
    smooth_with_confidence,
    xcov_coeff,
)


BAND_NAMES = (
    "N_0_15",
    "N_15_30",
    "N_30_45",
    "S_0_15",
    "S_15_30",
    "S_30_45",
)
BAND_LABELS = {
    "N_0_15": "0--15 degrees N",
    "N_15_30": "15--30 degrees N",
    "N_30_45": "30--45 degrees N",
    "S_0_15": "0--15 degrees S",
    "S_15_30": "15--30 degrees S",
    "S_30_45": "30--45 degrees S",
}
CYCLES = (21, 22, 23, 24)
HEMISPHERES = ("N", "S")
HEMISPHERE_BAND_INDICES = {"N": (0, 1, 2), "S": (3, 4, 5)}

# Upper bounds are exclusive. These are the paper-specific transition intervals
# 1976.3--1977.4, 1984.5--1988.0, 1994.5--1998.3, and 2006.5--2010.0 on the
# study analysis clock. The extrema inside them are found independently
# for every latitude band.
DEFAULT_TRANSITION_WINDOWS_CR = (
    (1642, 1656),
    (1751, 1798),
    (1885, 1936),
    (2045, 2092),
)
ONE_YEAR_CARRINGTON_ROTATIONS = 13
SECONDARY_PEAK_HEIGHT_FRACTION = 0.50
SECONDARY_PEAK_PROMINENCE_FRACTION = 0.10


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _energy_matrix(values: np.ndarray, sample_count: int) -> np.ndarray:
    energy = np.asarray(values, dtype=float)
    expected = (sample_count, len(BAND_NAMES))
    if energy.shape != expected:
        raise ValueError(
            f"energy must have shape {expected} in BAND_NAMES order; got {energy.shape}"
        )
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy contains a non-finite value")
    return energy


def _rotation_vector(rotations: np.ndarray) -> np.ndarray:
    numeric = _finite_vector(rotations, "rotations")
    integer = numeric.astype(int)
    if not np.array_equal(numeric, integer):
        raise ValueError("rotations must contain integer Carrington rotations")
    if not np.all(np.diff(integer) == 1):
        raise ValueError("rotations must be strictly consecutive")
    return integer


@dataclass(frozen=True)
class CycleEvents:
    """Detected event indexes; every array has shape ``(6 bands, 4 cycles)``."""

    minimum_indices: np.ndarray
    primary_peak_indices: np.ndarray
    first_peak_indices: np.ndarray
    last_peak_indices: np.ndarray

    # Short aliases keep plotting code readable without weakening the explicit
    # stored field names used in tabular output and tests.
    @property
    def minima(self) -> np.ndarray:
        return self.minimum_indices

    @property
    def primary_peaks(self) -> np.ndarray:
        return self.primary_peak_indices

    @property
    def first_peaks(self) -> np.ndarray:
        return self.first_peak_indices

    @property
    def last_peaks(self) -> np.ndarray:
        return self.last_peak_indices


def significant_peak_indices(
    values: np.ndarray,
    start: int,
    stop: int,
    *,
    height_fraction: float = SECONDARY_PEAK_HEIGHT_FRACTION,
    prominence_fraction: float = SECONDARY_PEAK_PROMINENCE_FRACTION,
) -> tuple[int, np.ndarray]:
    """Return the primary and retained peak indexes in ``[start, stop)``.

    The paper defines a secondary peak as reaching at least 50% of the primary
    peak. The executable criterion also rejects shoulders whose local
    prominence is below 10% of the primary peak.  The global maximum is always
    retained, including when it lies at an interval endpoint (which
    :func:`scipy.signal.find_peaks` does not classify as a local peak).
    """

    series = _finite_vector(values, "values")
    if not 0 <= start < stop <= series.size:
        raise ValueError("require 0 <= start < stop <= len(values)")
    if not 0.0 < height_fraction <= 1.0:
        raise ValueError("height_fraction must lie in (0, 1]")
    if not 0.0 <= prominence_fraction <= 1.0:
        raise ValueError("prominence_fraction must lie in [0, 1]")

    segment = series[start:stop]
    primary = start + int(np.argmax(segment))
    primary_amplitude = series[primary]
    if primary_amplitude <= 0.0:
        raise ValueError("peak-fraction criteria require a positive primary peak")

    local_peaks, _ = find_peaks(
        segment,
        height=height_fraction * primary_amplitude,
        prominence=prominence_fraction * primary_amplitude,
    )
    retained = np.unique(np.append(local_peaks + start, primary)).astype(int)
    retained.sort()
    return primary, retained


def detect_cycle_events(
    rotations: np.ndarray,
    smoothed_energy: np.ndarray,
    transition_windows: Sequence[tuple[int, int]] = DEFAULT_TRANSITION_WINDOWS_CR,
) -> CycleEvents:
    """Find band-specific minima, primary peaks, and multiple-peak endpoints."""

    cr = _rotation_vector(rotations)
    energy = _energy_matrix(smoothed_energy, cr.size)
    windows = tuple((int(first), int(stop)) for first, stop in transition_windows)
    if len(windows) != len(CYCLES):
        raise ValueError(f"exactly {len(CYCLES)} transition windows are required")
    for position, (first, stop) in enumerate(windows):
        if first >= stop:
            raise ValueError("every transition window must have first < stop")
        if position and first < windows[position - 1][1]:
            raise ValueError("transition windows must be ordered and non-overlapping")

    minima = np.empty((len(BAND_NAMES), len(CYCLES)), dtype=int)
    for cycle_index, (first_cr, stop_cr) in enumerate(windows):
        candidates = np.flatnonzero((cr >= first_cr) & (cr < stop_cr))
        if candidates.size == 0:
            raise ValueError(
                f"transition window [{first_cr}, {stop_cr}) has no input rotations"
            )
        for band_index in range(len(BAND_NAMES)):
            minima[band_index, cycle_index] = candidates[
                int(np.argmin(energy[candidates, band_index]))
            ]

    primary = np.empty_like(minima)
    first_peak = np.empty_like(minima)
    last_peak = np.empty_like(minima)
    for band_index in range(len(BAND_NAMES)):
        for cycle_index in range(len(CYCLES)):
            start = int(minima[band_index, cycle_index])
            stop = (
                int(minima[band_index, cycle_index + 1])
                if cycle_index + 1 < len(CYCLES)
                else cr.size
            )
            peak, retained = significant_peak_indices(
                energy[:, band_index], start, stop
            )
            primary[band_index, cycle_index] = peak
            first_peak[band_index, cycle_index] = retained[0]
            last_peak[band_index, cycle_index] = retained[-1]

    return CycleEvents(
        minimum_indices=minima,
        primary_peak_indices=primary,
        first_peak_indices=first_peak,
        last_peak_indices=last_peak,
    )


def nearest_year_index(
    decimal_years: np.ndarray,
    event_index: int,
    direction: int,
) -> int:
    """Find the observed sample nearest one year before or after an event."""

    dates = _finite_vector(decimal_years, "decimal_years")
    if not np.all(np.diff(dates) > 0.0):
        raise ValueError("decimal_years must be strictly increasing")
    if not 0 <= event_index < dates.size:
        raise IndexError("event_index is outside decimal_years")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (before) or +1 (after)")

    if direction < 0:
        candidates = np.arange(0, event_index, dtype=int)
    else:
        candidates = np.arange(event_index + 1, dates.size, dtype=int)
    if candidates.size == 0:
        raise ValueError("there is no sample on the requested side of the event")
    target = dates[event_index] + float(direction)
    return int(candidates[np.argmin(np.abs(dates[candidates] - target))])


@dataclass(frozen=True)
class CycleMetrics:
    """Event values from which Tables 1 and 3--5 are calculated."""

    minimum_dates: np.ndarray
    primary_peak_dates: np.ndarray
    cycle_lengths: np.ndarray
    peak_times: np.ndarray
    peak_amplitudes: np.ndarray
    rise_sample_indices: np.ndarray
    rise_times: np.ndarray
    rise_amplitudes: np.ndarray
    decay_start_indices: np.ndarray
    decay_end_indices: np.ndarray
    decay_rates: np.ndarray


def calculate_cycle_metrics(
    decimal_years: np.ndarray,
    smoothed_energy: np.ndarray,
    events: CycleEvents,
) -> CycleMetrics:
    """Calculate rise, peak, cycle-length, and decay quantities from events."""

    dates = _finite_vector(decimal_years, "decimal_years")
    if not np.all(np.diff(dates) > 0.0):
        raise ValueError("decimal_years must be strictly increasing")
    energy = _energy_matrix(smoothed_energy, dates.size)
    expected_event_shape = (len(BAND_NAMES), len(CYCLES))
    for name, indexes in (
        ("minimum_indices", events.minimum_indices),
        ("primary_peak_indices", events.primary_peak_indices),
        ("first_peak_indices", events.first_peak_indices),
        ("last_peak_indices", events.last_peak_indices),
    ):
        if np.asarray(indexes).shape != expected_event_shape:
            raise ValueError(f"{name} must have shape {expected_event_shape}")
        if np.any(np.asarray(indexes) < 0) or np.any(np.asarray(indexes) >= dates.size):
            raise ValueError(f"{name} contains an out-of-range index")

    minima = np.asarray(events.minimum_indices, dtype=int)
    peaks = np.asarray(events.primary_peak_indices, dtype=int)
    first_peaks = np.asarray(events.first_peak_indices, dtype=int)
    last_peaks = np.asarray(events.last_peak_indices, dtype=int)
    minimum_dates = dates[minima]
    peak_dates = dates[peaks]
    cycle_lengths = np.diff(minimum_dates, axis=1)
    peak_times = peak_dates - minimum_dates
    peak_amplitudes = np.empty(expected_event_shape, dtype=float)
    rise_indexes = np.empty(expected_event_shape, dtype=int)
    rise_times = np.empty(expected_event_shape, dtype=float)
    rise_amplitudes = np.empty(expected_event_shape, dtype=float)

    for band_index in range(len(BAND_NAMES)):
        for cycle_index in range(len(CYCLES)):
            peak_index = int(peaks[band_index, cycle_index])
            rise_index = nearest_year_index(
                dates,
                int(first_peaks[band_index, cycle_index]),
                direction=-1,
            )
            peak_amplitudes[band_index, cycle_index] = energy[
                peak_index, band_index
            ]
            rise_indexes[band_index, cycle_index] = rise_index
            rise_times[band_index, cycle_index] = (
                dates[rise_index] - minimum_dates[band_index, cycle_index]
            )
            rise_amplitudes[band_index, cycle_index] = energy[
                rise_index, band_index
            ]

    # A following minimum is required, so decay rates exist for cycles 21--23.
    decay_shape = (len(BAND_NAMES), len(CYCLES) - 1)
    decay_start = np.empty(decay_shape, dtype=int)
    decay_end = np.empty(decay_shape, dtype=int)
    decay_rates = np.empty(decay_shape, dtype=float)
    for band_index in range(len(BAND_NAMES)):
        for cycle_index in range(len(CYCLES) - 1):
            start = nearest_year_index(
                dates,
                int(last_peaks[band_index, cycle_index]),
                direction=1,
            )
            stop = nearest_year_index(
                dates,
                int(minima[band_index, cycle_index + 1]),
                direction=-1,
            )
            if start >= stop:
                raise ValueError("decay interval has no positive duration")
            decay_start[band_index, cycle_index] = start
            decay_end[band_index, cycle_index] = stop
            # Report a positive decay rate: earlier energy minus later energy
            # divided by elapsed years.
            decay_rates[band_index, cycle_index] = (
                energy[start, band_index] - energy[stop, band_index]
            ) / (dates[stop] - dates[start])

    return CycleMetrics(
        minimum_dates=minimum_dates,
        primary_peak_dates=peak_dates,
        cycle_lengths=cycle_lengths,
        peak_times=peak_times,
        peak_amplitudes=peak_amplitudes,
        rise_sample_indices=rise_indexes,
        rise_times=rise_times,
        rise_amplitudes=rise_amplitudes,
        decay_start_indices=decay_start,
        decay_end_indices=decay_end,
        decay_rates=decay_rates,
    )


@dataclass(frozen=True)
class Table1:
    """Minima, primary maxima, and lengths for six bands and four cycles."""

    minimum_rotations: np.ndarray
    maximum_rotations: np.ndarray
    minimum_dates: np.ndarray
    maximum_dates: np.ndarray
    cycle_lengths: np.ndarray


@dataclass(frozen=True)
class Table2:
    """Low-latitude/reference cross-correlations for four band pairs."""

    reference_bands: tuple[str, ...]
    comparison_bands: tuple[str, ...]
    correlations: np.ndarray
    lags: np.ndarray


@dataclass(frozen=True)
class Table3:
    """Across-cycle adjusted correlations for each latitude band."""

    peak_time_amplitude: np.ndarray
    rise_time_amplitude: np.ndarray


@dataclass(frozen=True)
class Table4:
    """Across-latitude adjusted correlations for each hemisphere and cycle."""

    peak_time_amplitude: np.ndarray
    rise_time_amplitude: np.ndarray


@dataclass(frozen=True)
class Table5:
    """Adjusted decay-rate correlations for each latitude band."""

    decay_peak_amplitude: np.ndarray
    decay_rise_amplitude: np.ndarray


@dataclass(frozen=True)
class StudyTables:
    """The five numerical tables published by Inceoglu et al. (2017)."""

    table1: Table1
    table2: Table2
    table3: Table3
    table4: Table4
    table5: Table5


@dataclass(frozen=True)
class StudyAnalysis:
    """Complete downstream numerical state shared by tables and figures."""

    smoothed: SmoothedSeries
    events: CycleEvents
    metrics: CycleMetrics
    tables: StudyTables

    @property
    def table1(self) -> Table1:
        return self.tables.table1

    @property
    def table2(self) -> Table2:
        return self.tables.table2

    @property
    def table3(self) -> Table3:
        return self.tables.table3

    @property
    def table4(self) -> Table4:
        return self.tables.table4

    @property
    def table5(self) -> Table5:
        return self.tables.table5


def build_study_tables(
    rotations: np.ndarray,
    smoothed_energy: np.ndarray,
    events: CycleEvents,
    metrics: CycleMetrics,
) -> StudyTables:
    """Calculate Tables 1--5 from the event and metric arrays."""

    cr = _rotation_vector(rotations)
    energy = _energy_matrix(smoothed_energy, cr.size)

    table1 = Table1(
        minimum_rotations=cr[events.minimum_indices],
        maximum_rotations=cr[events.primary_peak_indices],
        minimum_dates=metrics.minimum_dates,
        maximum_dates=metrics.primary_peak_dates,
        cycle_lengths=metrics.cycle_lengths,
    )

    pair_indexes = ((0, 1), (0, 2), (3, 4), (3, 5))
    correlations = np.empty(len(pair_indexes), dtype=float)
    lags = np.empty(len(pair_indexes), dtype=int)
    for row, (reference_index, comparison_index) in enumerate(pair_indexes):
        result = xcov_coeff(
            energy[:, reference_index], energy[:, comparison_index]
        )
        correlations[row] = result.best_coefficient
        lags[row] = result.best_lag
    table2 = Table2(
        reference_bands=tuple(BAND_NAMES[first] for first, _ in pair_indexes),
        comparison_bands=tuple(BAND_NAMES[second] for _, second in pair_indexes),
        correlations=correlations,
        lags=lags,
    )

    table3 = Table3(
        peak_time_amplitude=np.asarray(
            [
                adjusted_pearson(
                    metrics.peak_times[band], metrics.peak_amplitudes[band]
                )
                for band in range(len(BAND_NAMES))
            ]
        ),
        rise_time_amplitude=np.asarray(
            [
                adjusted_pearson(
                    metrics.rise_times[band], metrics.rise_amplitudes[band]
                )
                for band in range(len(BAND_NAMES))
            ]
        ),
    )

    table4_peak = np.empty((len(HEMISPHERES), len(CYCLES)), dtype=float)
    table4_rise = np.empty_like(table4_peak)
    for hemisphere_index, hemisphere in enumerate(HEMISPHERES):
        band_indexes = np.asarray(HEMISPHERE_BAND_INDICES[hemisphere])
        for cycle_index in range(len(CYCLES)):
            table4_peak[hemisphere_index, cycle_index] = adjusted_pearson(
                metrics.peak_times[band_indexes, cycle_index],
                metrics.peak_amplitudes[band_indexes, cycle_index],
            )
            table4_rise[hemisphere_index, cycle_index] = adjusted_pearson(
                metrics.rise_times[band_indexes, cycle_index],
                metrics.rise_amplitudes[band_indexes, cycle_index],
            )
    table4 = Table4(
        peak_time_amplitude=table4_peak,
        rise_time_amplitude=table4_rise,
    )

    # Decay rates exist for cycles 21--23; compare them with amplitudes from
    # those same cycles rather than silently dropping a mismatched row later.
    decay_cycle_count = metrics.decay_rates.shape[1]
    table5 = Table5(
        decay_peak_amplitude=np.asarray(
            [
                adjusted_pearson(
                    metrics.decay_rates[band],
                    metrics.peak_amplitudes[band, :decay_cycle_count],
                )
                for band in range(len(BAND_NAMES))
            ]
        ),
        decay_rise_amplitude=np.asarray(
            [
                adjusted_pearson(
                    metrics.decay_rates[band],
                    metrics.rise_amplitudes[band, :decay_cycle_count],
                )
                for band in range(len(BAND_NAMES))
            ]
        ),
    )
    return StudyTables(table1, table2, table3, table4, table5)


def analyze_study(
    rotations: np.ndarray,
    analysis_decimal_years: np.ndarray,
    raw_energy: np.ndarray,
    *,
    smoothing_span: int = DEFAULT_SMOOTHING_SPAN,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> StudyAnalysis:
    """Run the complete deterministic downstream analysis on six raw series.

    ``analysis_decimal_years`` must be the explicit historical date column,
    not a conventional timestamp-derived decimal year.  Its preparation uses
    ``year + (month + day / 31 - 1) / 12`` and retains the authoritative UTC
    rotation start in a separate input column.
    """

    cr = _rotation_vector(rotations)
    dates = _finite_vector(analysis_decimal_years, "analysis_decimal_years")
    if dates.size != cr.size:
        raise ValueError(
            "rotations and analysis_decimal_years must have the same length"
        )
    if not np.all(np.diff(dates) > 0.0):
        raise ValueError("analysis_decimal_years must be strictly increasing")
    energy = _energy_matrix(raw_energy, cr.size)

    smoothed = smooth_with_confidence(
        energy, span=smoothing_span, confidence=confidence
    )
    events = detect_cycle_events(cr, smoothed.values)
    metrics = calculate_cycle_metrics(dates, smoothed.values, events)
    tables = build_study_tables(cr, smoothed.values, events, metrics)
    return StudyAnalysis(smoothed, events, metrics, tables)


# British spelling retained as a discoverable alias because the paper and its
# surrounding documentation use "analyse" throughout.
analyse_study = analyze_study


def reproduce_tables(
    rotations: np.ndarray,
    analysis_decimal_years: np.ndarray,
    raw_energy: np.ndarray,
) -> StudyTables:
    """Convenience wrapper returning only the five calculated tables."""

    return analyze_study(rotations, analysis_decimal_years, raw_energy).tables


# ---------------------------------------------------------------------------
# Published validation targets.  These constants are never consulted by the
# calculation above.  Tables 1 and 2 are exact at their printed precision;
# Tables 3--5 use an absolute validation tolerance of 0.011 because several
# entries in the article were rounded inconsistently at the second decimal.

EXPECTED_TABLE1_MINIMUM_ROTATIONS = np.asarray(
    [
        [1649, 1788, 1912, 2079],
        [1642, 1767, 1910, 2065],
        [1642, 1770, 1898, 2065],
        [1651, 1780, 1926, 2084],
        [1642, 1776, 1906, 2078],
        [1643, 1776, 1907, 2077],
    ],
    dtype=int,
)
EXPECTED_TABLE1_MAXIMUM_ROTATIONS = np.asarray(
    [
        [1708, 1838, 1985, 2164],
        [1682, 1815, 1951, 2116],
        [1673, 1814, 1948, 2115],
        [1712, 1844, 1985, 2156],
        [1697, 1841, 1984, 2157],
        [1697, 1813, 1952, 2128],
    ],
    dtype=int,
)
EXPECTED_FIRST_PEAK_ROTATIONS = np.asarray(
    [
        [1682, 1838, 1985, 2117],
        [1682, 1815, 1951, 2116],
        [1673, 1814, 1948, 2115],
        [1712, 1844, 1960, 2156],
        [1697, 1816, 1957, 2126],
        [1680, 1813, 1952, 2128],
    ],
    dtype=int,
)
EXPECTED_LAST_PEAK_ROTATIONS = np.asarray(
    [
        [1708, 1838, 1985, 2164],
        [1682, 1815, 1993, 2165],
        [1673, 1814, 1948, 2115],
        [1712, 1844, 1985, 2156],
        [1697, 1841, 1984, 2157],
        [1697, 1843, 1984, 2158],
    ],
    dtype=int,
)
PUBLISHED_TABLE1_MINIMUM_DATES = np.asarray(
    [
        [1976.9, 1987.3, 1996.6, 2009.0],
        [1976.4, 1985.7, 1996.4, 2008.0],
        [1976.4, 1986.0, 1995.5, 2008.0],
        [1977.1, 1986.7, 1997.6, 2009.4],
        [1976.4, 1986.4, 1996.1, 2009.0],
        [1976.5, 1986.4, 1996.2, 2008.9],
    ]
)
PUBLISHED_TABLE1_MAXIMUM_DATES = np.asarray(
    [
        [1981.3, 1991.0, 2002.0, 2015.4],
        [1979.4, 1989.3, 1999.5, 2011.8],
        [1978.7, 1989.3, 1999.3, 2011.7],
        [1981.6, 1991.5, 2002.0, 2014.8],
        [1980.5, 1991.3, 2001.9, 2014.9],
        [1980.5, 1989.2, 1999.6, 2012.7],
    ]
)
PUBLISHED_TABLE1_CYCLE_LENGTHS = np.asarray(
    [
        [10.4, 9.3, 12.5],
        [9.3, 10.7, 11.6],
        [9.6, 9.6, 12.5],
        [9.6, 10.9, 11.8],
        [10.0, 9.7, 12.8],
        [9.9, 9.8, 12.7],
    ]
)

EXPECTED_TABLE2_CORRELATIONS = np.asarray(
    [
        0.8605146183582548,
        0.7701182106230832,
        0.8343403094797389,
        0.7894881968154978,
    ]
)
PUBLISHED_TABLE2_CORRELATIONS = np.asarray([0.86, 0.77, 0.83, 0.79])
PUBLISHED_TABLE2_LAGS = np.asarray([10, 22, 6, 19], dtype=int)

PUBLISHED_TABLE3 = Table3(
    peak_time_amplitude=np.asarray([-0.78, -0.27, -0.47, -0.28, -0.60, -0.52]),
    rise_time_amplitude=np.asarray([0.76, -0.62, -0.22, 0.62, -0.09, -0.91]),
)
PUBLISHED_TABLE4 = Table4(
    peak_time_amplitude=np.asarray(
        [[0.93, 0.72, 0.40, 0.71], [0.90, 0.95, 0.87, 0.86]]
    ),
    rise_time_amplitude=np.asarray(
        [[0.99, 0.97, 0.76, 0.19], [1.00, 0.43, 0.86, 0.93]]
    ),
)
PUBLISHED_TABLE5 = Table5(
    decay_peak_amplitude=np.asarray([0.87, 0.97, 0.95, 0.95, 0.85, 0.99]),
    decay_rise_amplitude=np.asarray([-0.66, 0.93, 0.86, 0.97, 0.98, 0.91]),
)
PUBLISHED_CORRELATION_TOLERANCE = 0.011
# The article stores -0.2800 for this cell, while the calculation gives
# -0.262397 with the paper-specific decimal-date arithmetic. Keep both facts
# explicit and apply the dedicated comparison tolerance below.
PUBLISHED_TABLE3_S_0_15_PEAK_TOLERANCE = 0.018
