"""Load and validate the single clean numerical input used by the study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import BAND_NAMES
from .io import read_csv
from .paths import FIRST_CARRINGTON_ROTATION, LAST_CARRINGTON_ROTATION, WSO_ENERGY


GLOBAL_NAMES = ("FD", "NH", "SH")


@dataclass(frozen=True)
class StudyInput:
    """Carrington coordinates and the nine regional magnetic-energy series."""

    rotations: np.ndarray
    analysis_decimal_years: np.ndarray
    global_energy: np.ndarray
    band_energy: np.ndarray


def load_study_input(path: Path = WSO_ENERGY) -> StudyInput:
    """Load the checksum-verified WSO energy CSV with strict structure."""

    source = Path(path)
    rows = read_csv(source)
    expected_rows = LAST_CARRINGTON_ROTATION - FIRST_CARRINGTON_ROTATION + 1
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} WSO rotations; got {len(rows)}")
    required = {
        "carrington_rotation",
        "rotation_start_utc",
        "analysis_decimal_year",
        *GLOBAL_NAMES,
        *BAND_NAMES,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"WSO energy CSV lacks columns: {sorted(missing)}")

    try:
        rotations = np.asarray(
            [int(row["carrington_rotation"]) for row in rows], dtype=int
        )
        years = np.asarray(
            [float(row["analysis_decimal_year"]) for row in rows], dtype=float
        )
        global_energy = np.asarray(
            [[float(row[name]) for name in GLOBAL_NAMES] for row in rows], dtype=float
        )
        band_energy = np.asarray(
            [[float(row[name]) for name in BAND_NAMES] for row in rows], dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value in {source}") from error

    expected_rotations = np.arange(
        FIRST_CARRINGTON_ROTATION, LAST_CARRINGTON_ROTATION + 1, dtype=int
    )
    if not np.array_equal(rotations, expected_rotations):
        raise ValueError("WSO input must contain consecutive CR 1642--2182 exactly once")
    if not np.all(np.diff(years) > 0.0):
        raise ValueError("Historical decimal years must increase strictly")
    if not np.all(np.isfinite(global_energy)) or not np.all(np.isfinite(band_energy)):
        raise ValueError("WSO input contains a non-finite energy")
    if np.any(global_energy < 0.0) or np.any(band_energy < 0.0):
        raise ValueError("Magnetic modal energies cannot be negative")
    return StudyInput(rotations, years, global_energy, band_energy)
