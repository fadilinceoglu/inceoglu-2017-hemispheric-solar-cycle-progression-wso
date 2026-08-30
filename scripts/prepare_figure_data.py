#!/usr/bin/env python3
"""Derive the transparent numerical inputs used to plot Figures 1 and 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inceoglu2017.figure_data import FIGURE_1_ROTATIONS  # noqa: E402
from inceoglu2017.io import sha256_file, write_csv  # noqa: E402
from inceoglu2017.paths import (  # noqa: E402
    FIRST_CARRINGTON_ROTATION,
    LAST_CARRINGTON_ROTATION,
    WSO_FIGURE_1_MAPS,
    WSO_MAPS,
    WSO_MEAN_SPECTRUM,
)
from inceoglu2017.wso import (  # noqa: E402
    MAX_SPHERICAL_HARMONIC_DEGREE,
    degree_energy_spectrum,
    read_wso_filled_map,
    remap_line_of_sight_to_gauss,
    spherical_harmonic_coefficients,
)


def _number(value: float) -> str:
    """Serialize enough digits to round-trip an IEEE-754 double."""

    return format(float(value), ".17g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=WSO_MAPS)
    parser.add_argument("--map-output", type=Path, default=WSO_FIGURE_1_MAPS)
    parser.add_argument("--spectrum-output", type=Path, default=WSO_MEAN_SPECTRUM)
    args = parser.parse_args()

    spectrum_sum = np.zeros(MAX_SPHERICAL_HARMONIC_DEGREE + 1, dtype=float)
    selected_maps: dict[int, np.ndarray] = {}
    rotation_count = LAST_CARRINGTON_ROTATION - FIRST_CARRINGTON_ROTATION + 1
    for rotation in range(FIRST_CARRINGTON_ROTATION, LAST_CARRINGTON_ROTATION + 1):
        path = args.input_dir / f"WSO.{rotation}.F.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing WSO map {path}. Run scripts/download_wso_maps.py first "
                "or pass --input-dir."
            )
        source = read_wso_filled_map(path, target_rotation=rotation)
        if rotation in FIGURE_1_ROTATIONS:
            selected_maps[rotation] = source
        prepared = remap_line_of_sight_to_gauss(source)
        coefficients = spherical_harmonic_coefficients(
            prepared.field_gauss,
            lmax=MAX_SPHERICAL_HARMONIC_DEGREE,
        )
        spectrum_sum += degree_energy_spectrum(coefficients)
        if rotation == FIRST_CARRINGTON_ROTATION or rotation % 50 == 0:
            print(f"Prepared Figure 2 spectrum through CR {rotation}", flush=True)

    if set(selected_maps) != set(FIGURE_1_ROTATIONS):
        raise AssertionError("The selected Figure 1 rotations were not prepared")

    longitude = np.arange(1, 73, dtype=float) * 5.0
    sine_latitude = np.arange(-14.5, 15.0, dtype=float) / 15.0
    latitude = np.degrees(np.arcsin(sine_latitude))
    map_rows: list[dict[str, object]] = []
    for rotation in FIGURE_1_ROTATIONS:
        field = selected_maps[rotation]
        for longitude_index, longitude_value in enumerate(longitude):
            for latitude_index, (sine_value, latitude_value) in enumerate(
                zip(sine_latitude, latitude)
            ):
                map_rows.append(
                    {
                        "carrington_rotation": rotation,
                        "longitude_index": longitude_index,
                        "carrington_longitude_degree": _number(longitude_value),
                        "latitude_index": latitude_index,
                        "sine_latitude": _number(sine_value),
                        "latitude_degree": _number(latitude_value),
                        "field_microtesla": _number(
                            field[longitude_index, latitude_index]
                        ),
                    }
                )
    write_csv(
        args.map_output,
        [
            "carrington_rotation",
            "longitude_index",
            "carrington_longitude_degree",
            "latitude_index",
            "sine_latitude",
            "latitude_degree",
            "field_microtesla",
        ],
        map_rows,
    )

    mean_spectrum = spectrum_sum / rotation_count
    spectrum_rows = [
        {
            "degree": degree,
            "mean_energy_gauss2": _number(mean_spectrum[degree]),
            "rotation_count": rotation_count,
        }
        for degree in range(1, MAX_SPHERICAL_HARMONIC_DEGREE + 1)
    ]
    write_csv(
        args.spectrum_output,
        ["degree", "mean_energy_gauss2", "rotation_count"],
        spectrum_rows,
    )
    print(
        f"Wrote {args.map_output} (SHA-256 {sha256_file(args.map_output)})",
        flush=True,
    )
    print(
        f"Wrote {args.spectrum_output} "
        f"(SHA-256 {sha256_file(args.spectrum_output)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
