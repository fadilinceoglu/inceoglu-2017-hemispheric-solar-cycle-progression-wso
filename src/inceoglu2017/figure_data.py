"""Read the WSO-derived numerical inputs used by Figures 1 and 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_csv
from .paths import WSO_FIGURE_1_MAPS, WSO_MEAN_SPECTRUM
from .wso import WSO_LATITUDE_BINS, WSO_LONGITUDE_BINS


FIGURE_1_ROTATIONS: tuple[int, int] = (2149, 1917)


@dataclass(frozen=True)
class WsoContextMap:
    """One filled WSO map on its native longitude-by-sine-latitude grid."""

    carrington_rotation: int
    longitude_degrees: np.ndarray
    sine_latitude: np.ndarray
    latitude_degrees: np.ndarray
    field_microtesla: np.ndarray


def load_figure_1_maps(
    path: Path = WSO_FIGURE_1_MAPS,
) -> dict[int, WsoContextMap]:
    """Load the two native-grid WSO maps selected for Figure 1."""

    source = Path(path)
    rows = read_csv(source)
    if not rows:
        raise ValueError(f"Figure 1 map CSV is empty: {source}")
    required = {
        "carrington_rotation",
        "longitude_index",
        "carrington_longitude_degree",
        "latitude_index",
        "sine_latitude",
        "latitude_degree",
        "field_microtesla",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Figure 1 map CSV lacks columns: {sorted(missing)}")

    maps: dict[int, WsoContextMap] = {}
    for rotation in FIGURE_1_ROTATIONS:
        field = np.full((WSO_LONGITUDE_BINS, WSO_LATITUDE_BINS), np.nan)
        longitude = np.full(WSO_LONGITUDE_BINS, np.nan)
        sine_latitude = np.full(WSO_LATITUDE_BINS, np.nan)
        latitude = np.full(WSO_LATITUDE_BINS, np.nan)
        seen = np.zeros(field.shape, dtype=bool)
        for row in rows:
            try:
                row_rotation = int(row["carrington_rotation"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Figure 1 map CSV has an invalid Carrington rotation: {source}"
                ) from error
            if row_rotation != rotation:
                continue
            try:
                longitude_index = int(row["longitude_index"])
                latitude_index = int(row["latitude_index"])
                longitude_value = float(row["carrington_longitude_degree"])
                sine_value = float(row["sine_latitude"])
                latitude_value = float(row["latitude_degree"])
                field_value = float(row["field_microtesla"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Figure 1 map CSV has an invalid numeric value: {source}"
                ) from error
            if not (
                0 <= longitude_index < WSO_LONGITUDE_BINS
                and 0 <= latitude_index < WSO_LATITUDE_BINS
            ):
                raise ValueError("Figure 1 map CSV contains an out-of-range grid index")
            if seen[longitude_index, latitude_index]:
                raise ValueError("Figure 1 map CSV contains a duplicate grid cell")
            seen[longitude_index, latitude_index] = True
            field[longitude_index, latitude_index] = field_value
            if np.isfinite(longitude[longitude_index]) and not np.isclose(
                longitude[longitude_index], longitude_value
            ):
                raise ValueError("Figure 1 map CSV has inconsistent longitudes")
            if np.isfinite(sine_latitude[latitude_index]) and not np.isclose(
                sine_latitude[latitude_index], sine_value
            ):
                raise ValueError("Figure 1 map CSV has inconsistent sine latitudes")
            if np.isfinite(latitude[latitude_index]) and not np.isclose(
                latitude[latitude_index], latitude_value
            ):
                raise ValueError("Figure 1 map CSV has inconsistent latitudes")
            longitude[longitude_index] = longitude_value
            sine_latitude[latitude_index] = sine_value
            latitude[latitude_index] = latitude_value

        if not np.all(seen):
            raise ValueError(
                f"Figure 1 map CSV does not contain a complete CR {rotation} grid"
            )
        if not (
            np.all(np.isfinite(field))
            and np.all(np.diff(longitude) > 0.0)
            and np.all(np.diff(sine_latitude) > 0.0)
            and np.all(np.diff(latitude) > 0.0)
        ):
            raise ValueError(f"Figure 1 map CSV has an invalid CR {rotation} grid")
        maps[rotation] = WsoContextMap(
            carrington_rotation=rotation,
            longitude_degrees=longitude,
            sine_latitude=sine_latitude,
            latitude_degrees=latitude,
            field_microtesla=field,
        )

    unexpected = {
        int(row["carrington_rotation"]) for row in rows
    } - set(FIGURE_1_ROTATIONS)
    if unexpected:
        raise ValueError(
            f"Figure 1 map CSV contains unexpected rotations: {sorted(unexpected)}"
        )
    return maps


def load_mean_spectrum(
    path: Path = WSO_MEAN_SPECTRUM,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the time-mean, all-order harmonic energy for degrees 1--60."""

    source = Path(path)
    rows = read_csv(source)
    if not rows:
        raise ValueError(f"Mean-spectrum CSV is empty: {source}")
    required = {"degree", "mean_energy_gauss2", "rotation_count"}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Mean-spectrum CSV lacks columns: {sorted(missing)}")
    try:
        degrees = np.asarray([int(row["degree"]) for row in rows], dtype=int)
        energy = np.asarray(
            [float(row["mean_energy_gauss2"]) for row in rows], dtype=float
        )
        counts = np.asarray(
            [int(row["rotation_count"]) for row in rows], dtype=int
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Mean-spectrum CSV has an invalid value: {source}") from error
    if not np.array_equal(degrees, np.arange(1, 61, dtype=int)):
        raise ValueError("Mean-spectrum CSV must contain degrees 1 through 60 once")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("Mean-spectrum energies must be finite and positive")
    if not np.all(counts == 541):
        raise ValueError("Mean-spectrum values must average all 541 rotations")
    return degrees, energy


__all__ = [
    "FIGURE_1_ROTATIONS",
    "WsoContextMap",
    "load_figure_1_maps",
    "load_mean_spectrum",
]
