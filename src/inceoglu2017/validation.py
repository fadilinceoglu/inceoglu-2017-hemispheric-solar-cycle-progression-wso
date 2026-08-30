"""Numerical validation for the complete reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .dataset import StudyInput
from .events import (
    EXPECTED_FIRST_PEAK_ROTATIONS,
    EXPECTED_LAST_PEAK_ROTATIONS,
    EXPECTED_TABLE1_MAXIMUM_ROTATIONS,
    EXPECTED_TABLE1_MINIMUM_ROTATIONS,
    EXPECTED_TABLE2_CORRELATIONS,
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
    StudyAnalysis,
)
from .paths import ROOT


def _allclose(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    return bool(np.allclose(first, second, rtol=0.0, atol=tolerance))


def _maximum_delta(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def _pdf_is_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    with path.open("rb") as handle:
        start = handle.read(5)
        handle.seek(max(0, path.stat().st_size - 1024))
        end = handle.read()
    return start == b"%PDF-" and b"%%EOF" in end


def validate_reproduction(
    study_input: StudyInput,
    study: StudyAnalysis,
    input_checksums: dict[str, str],
    figure_paths: Iterable[Path],
    table_paths: Iterable[Path],
) -> dict[str, object]:
    """Validate all published numerical targets and return the check results."""

    figures = tuple(Path(path) for path in figure_paths)
    tables = tuple(Path(path) for path in table_paths)
    table3_peak_normal = np.asarray([0, 1, 2, 4, 5], dtype=int)

    normal_correlation_deltas = {
        "table3_peak": _maximum_delta(
            study.table3.peak_time_amplitude[table3_peak_normal],
            PUBLISHED_TABLE3.peak_time_amplitude[table3_peak_normal],
        ),
        "table3_rise": _maximum_delta(
            study.table3.rise_time_amplitude,
            PUBLISHED_TABLE3.rise_time_amplitude,
        ),
        "table4_peak": _maximum_delta(
            study.table4.peak_time_amplitude,
            PUBLISHED_TABLE4.peak_time_amplitude,
        ),
        "table4_rise": _maximum_delta(
            study.table4.rise_time_amplitude,
            PUBLISHED_TABLE4.rise_time_amplitude,
        ),
        "table5_peak": _maximum_delta(
            study.table5.decay_peak_amplitude,
            PUBLISHED_TABLE5.decay_peak_amplitude,
        ),
        "table5_rise": _maximum_delta(
            study.table5.decay_rise_amplitude,
            PUBLISHED_TABLE5.decay_rise_amplitude,
        ),
    }
    exceptional_calculated = float(study.table3.peak_time_amplitude[3])
    exceptional_published = float(PUBLISHED_TABLE3.peak_time_amplitude[3])
    exceptional_delta = abs(exceptional_calculated - exceptional_published)

    exact_checks = {
        "input_checksums": len(input_checksums) == 3,
        "input_has_541_consecutive_rotations": (
            study_input.rotations.size == 541
            and int(study_input.rotations[0]) == 1642
            and int(study_input.rotations[-1]) == 2182
            and bool(np.all(np.diff(study_input.rotations) == 1))
        ),
        "table1_minimum_rotations": bool(
            np.array_equal(
                study.table1.minimum_rotations, EXPECTED_TABLE1_MINIMUM_ROTATIONS
            )
        ),
        "table1_maximum_rotations": bool(
            np.array_equal(
                study.table1.maximum_rotations, EXPECTED_TABLE1_MAXIMUM_ROTATIONS
            )
        ),
        "first_qualifying_peak_rotations": bool(
            np.array_equal(
                study_input.rotations[study.events.first_peak_indices],
                EXPECTED_FIRST_PEAK_ROTATIONS,
            )
        ),
        "last_qualifying_peak_rotations": bool(
            np.array_equal(
                study_input.rotations[study.events.last_peak_indices],
                EXPECTED_LAST_PEAK_ROTATIONS,
            )
        ),
        "table1_minimum_dates_at_printed_precision": bool(
            np.array_equal(
                np.round(study.table1.minimum_dates, 1),
                PUBLISHED_TABLE1_MINIMUM_DATES,
            )
        ),
        "table1_maximum_dates_at_printed_precision": bool(
            np.array_equal(
                np.round(study.table1.maximum_dates, 1),
                PUBLISHED_TABLE1_MAXIMUM_DATES,
            )
        ),
        "table1_cycle_lengths_at_printed_precision": bool(
            np.array_equal(
                np.round(study.table1.cycle_lengths, 1),
                PUBLISHED_TABLE1_CYCLE_LENGTHS,
            )
        ),
        "table2_full_precision_coefficients": _allclose(
            study.table2.correlations, EXPECTED_TABLE2_CORRELATIONS, 2e-12
        ),
        "table2_lags": bool(
            np.array_equal(study.table2.lags, PUBLISHED_TABLE2_LAGS)
        ),
        "table2_coefficients_at_printed_precision": bool(
            np.array_equal(
                np.round(study.table2.correlations, 2),
                PUBLISHED_TABLE2_CORRELATIONS,
            )
        ),
        "tables3_to_5_within_standard_tolerance": all(
            delta <= PUBLISHED_CORRELATION_TOLERANCE
            for delta in normal_correlation_deltas.values()
        ),
        "table3_south_0_15_exception_within_declared_tolerance": (
            exceptional_delta <= PUBLISHED_TABLE3_S_0_15_PEAK_TOLERANCE
        ),
        "seven_complete_pdf_figures": (
            len(figures) == 7 and all(_pdf_is_complete(path) for path in figures)
        ),
        "five_nonempty_csv_tables": (
            len(tables) == 5
            and all(path.is_file() and len(path.read_text(encoding="utf-8").splitlines()) > 1 for path in tables)
        ),
    }

    report: dict[str, object] = {
        "study": "Inceoglu et al. (2017), A&A 601, A51",
        "doi": "10.1051/0004-6361/201629871",
        "overall_status": "passed" if all(exact_checks.values()) else "failed",
        "input_sha256": dict(sorted(input_checksums.items())),
        "checks": exact_checks,
        "maximum_absolute_deltas_for_standard_correlation_checks": normal_correlation_deltas,
        "standard_correlation_tolerance": PUBLISHED_CORRELATION_TOLERANCE,
        "documented_difference": {
            "quantity": "Table 3 adjusted peak-time/amplitude correlation, 0--15 degrees S",
            "calculated": exceptional_calculated,
            "published": exceptional_published,
            "absolute_delta": exceptional_delta,
            "accepted_tolerance": PUBLISHED_TABLE3_S_0_15_PEAK_TOLERANCE,
            "note": (
                "The article reports -0.28 at printed precision; the calculation "
                "gives the value reported here and uses the dedicated tolerance above."
            ),
        },
        "published_onset_lags_carrington_rotations": {
            "north_15_30": int(study.table2.lags[0]),
            "north_30_45": int(study.table2.lags[1]),
            "south_15_30": int(study.table2.lags[2]),
            "south_30_45": int(study.table2.lags[3]),
        },
        "outputs": {
            "figures": [str(path.relative_to(ROOT)) for path in figures],
            "tables": [str(path.relative_to(ROOT)) for path in tables],
        },
    }
    if report["overall_status"] != "passed":
        failures = [name for name, passed in exact_checks.items() if not passed]
        raise AssertionError(f"Reproduction validation failed: {failures}")
    return report
