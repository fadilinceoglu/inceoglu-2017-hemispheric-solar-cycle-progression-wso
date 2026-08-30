"""Write the five recalculated published tables as stable CSV files."""

from __future__ import annotations

from pathlib import Path

from .events import BAND_LABELS, BAND_NAMES, CYCLES, HEMISPHERES, StudyAnalysis
from .io import write_csv
from .paths import OUTPUT_TABLES


TABLE_FILENAMES = {
    1: "table1_cycle_timing.csv",
    2: "table2_cross_correlations.csv",
    3: "table3_waldmeier_by_latitude.csv",
    4: "table4_waldmeier_by_cycle.csv",
    5: "table5_decay_rate_correlations.csv",
}


def _one_decimal(value: float) -> str:
    return f"{float(value):.1f}"


def _six_decimals(value: float) -> str:
    return f"{float(value):.6f}"


def write_study_tables(
    study: StudyAnalysis,
    output_directory: Path = OUTPUT_TABLES,
) -> dict[int, Path]:
    """Write Tables 1--5 from calculated arrays, never reference targets."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}

    table1_fields = ["latitude_band"]
    for cycle_index, cycle in enumerate(CYCLES):
        table1_fields.extend(
            [
                f"cycle_{cycle}_minimum_decimal_year",
                f"cycle_{cycle}_maximum_decimal_year",
            ]
        )
        if cycle_index < len(CYCLES) - 1:
            table1_fields.append(f"cycle_{cycle}_length_years")
    table1_rows: list[dict[str, object]] = []
    for band_index, band in enumerate(BAND_NAMES):
        row: dict[str, object] = {"latitude_band": BAND_LABELS[band]}
        for cycle_index, cycle in enumerate(CYCLES):
            row[f"cycle_{cycle}_minimum_decimal_year"] = _one_decimal(
                study.table1.minimum_dates[band_index, cycle_index]
            )
            row[f"cycle_{cycle}_maximum_decimal_year"] = _one_decimal(
                study.table1.maximum_dates[band_index, cycle_index]
            )
            if cycle_index < len(CYCLES) - 1:
                row[f"cycle_{cycle}_length_years"] = _one_decimal(
                    study.table1.cycle_lengths[band_index, cycle_index]
                )
        table1_rows.append(row)
    path = destination / TABLE_FILENAMES[1]
    write_csv(path, table1_fields, table1_rows)
    written[1] = path

    table2_rows = [
        {
            "reference_latitude_band": BAND_LABELS[reference],
            "comparison_latitude_band": BAND_LABELS[comparison],
            "normalized_cross_correlation": _six_decimals(correlation),
            "lag_carrington_rotations": int(lag),
        }
        for reference, comparison, correlation, lag in zip(
            study.table2.reference_bands,
            study.table2.comparison_bands,
            study.table2.correlations,
            study.table2.lags,
        )
    ]
    path = destination / TABLE_FILENAMES[2]
    write_csv(
        path,
        [
            "reference_latitude_band",
            "comparison_latitude_band",
            "normalized_cross_correlation",
            "lag_carrington_rotations",
        ],
        table2_rows,
    )
    written[2] = path

    table3_rows = [
        {
            "latitude_band": BAND_LABELS[band],
            "adjusted_peak_time_amplitude_correlation": _six_decimals(
                study.table3.peak_time_amplitude[index]
            ),
            "adjusted_rise_time_amplitude_correlation": _six_decimals(
                study.table3.rise_time_amplitude[index]
            ),
        }
        for index, band in enumerate(BAND_NAMES)
    ]
    path = destination / TABLE_FILENAMES[3]
    write_csv(
        path,
        [
            "latitude_band",
            "adjusted_peak_time_amplitude_correlation",
            "adjusted_rise_time_amplitude_correlation",
        ],
        table3_rows,
    )
    written[3] = path

    table4_rows = []
    for hemisphere_index, hemisphere in enumerate(HEMISPHERES):
        for cycle_index, cycle in enumerate(CYCLES):
            table4_rows.append(
                {
                    "hemisphere": hemisphere,
                    "solar_cycle": cycle,
                    "adjusted_peak_time_amplitude_correlation": _six_decimals(
                        study.table4.peak_time_amplitude[
                            hemisphere_index, cycle_index
                        ]
                    ),
                    "adjusted_rise_time_amplitude_correlation": _six_decimals(
                        study.table4.rise_time_amplitude[
                            hemisphere_index, cycle_index
                        ]
                    ),
                }
            )
    path = destination / TABLE_FILENAMES[4]
    write_csv(
        path,
        [
            "hemisphere",
            "solar_cycle",
            "adjusted_peak_time_amplitude_correlation",
            "adjusted_rise_time_amplitude_correlation",
        ],
        table4_rows,
    )
    written[4] = path

    table5_rows = [
        {
            "latitude_band": BAND_LABELS[band],
            "adjusted_decay_peak_amplitude_correlation": _six_decimals(
                study.table5.decay_peak_amplitude[index]
            ),
            "adjusted_decay_rise_amplitude_correlation": _six_decimals(
                study.table5.decay_rise_amplitude[index]
            ),
        }
        for index, band in enumerate(BAND_NAMES)
    ]
    path = destination / TABLE_FILENAMES[5]
    write_csv(
        path,
        [
            "latitude_band",
            "adjusted_decay_peak_amplitude_correlation",
            "adjusted_decay_rise_amplitude_correlation",
        ],
        table5_rows,
    )
    written[5] = path
    return written
