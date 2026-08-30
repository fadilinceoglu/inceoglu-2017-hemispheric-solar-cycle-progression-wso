from __future__ import annotations

import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from scripts.download_wso_maps import validate_map
from inceoglu2017.wso import (
    LEGENDRE_LATITUDE_BINS,
    LEGENDRE_LONGITUDE_BINS,
    carrington_decimal_year,
    carrington_rotation_start,
    degree_energy_spectrum,
    geometric_region_masks,
    harmonic_energy,
    study_region_masks,
    study_decimal_year,
    integrated_spatial_energy,
    parse_wso_filled_map,
    read_wso_filled_map,
    reconstruct_from_coefficients,
    remap_line_of_sight_to_gauss,
    rotation_energy_products,
    spherical_harmonic_coefficients,
)


def _synthetic_filled_map_text(
    rotation: int = 1642,
    include_seam: bool = True,
) -> tuple[str, np.ndarray]:
    source_rows = np.arange(72 * 30, dtype=float).reshape(72, 30)
    lines = [
        "",
        "filled_synop: 30 data points in equal steps of sine latitude",
        "",
    ]

    def append_record(cr: int, longitude: int, values: np.ndarray) -> None:
        label = f"CT{cr}:{longitude:03d}".ljust(18)
        lines.append(label + "".join(f"{value:9.3f}" for value in values[:6]))
        for start in (6, 14, 22):
            lines.append(
                "".join(f"{value:9.3f}" for value in values[start : start + 8])
            )

    for row_index, longitude in enumerate(range(360, 0, -5)):
        append_record(rotation, longitude, source_rows[row_index])
    if include_seam:
        append_record(rotation + 1, 360, np.full(30, -999.0))
    return "\n".join(lines) + "\n", source_rows


def test_parser_selects_exact_rotation_and_excludes_next_rotation_seam(
    tmp_path: Path,
) -> None:
    text, source = _synthetic_filled_map_text()
    path = tmp_path / "WSO.1642.F.txt"
    path.write_text(text, encoding="ascii")

    parsed = read_wso_filled_map(path)

    np.testing.assert_array_equal(parsed, source[::-1, ::-1])
    assert not np.any(parsed == -999.0)
    validate_map(path.read_bytes(), 1642)


def test_parser_does_not_substitute_an_adjacent_rotation() -> None:
    text, _ = _synthetic_filled_map_text(rotation=1643)

    with pytest.raises(ValueError, match="target longitudes"):
        parse_wso_filled_map(text, target_rotation=1642)


def test_parser_rejects_duplicate_or_unexpected_records() -> None:
    text, _ = _synthetic_filled_map_text()
    duplicate = text.replace("CT1642:355", "CT1642:360", 1)
    with pytest.raises(ValueError, match="target longitudes"):
        parse_wso_filled_map(duplicate, target_rotation=1642)

    unexpected = text + text[text.index("CT1642:360") : text.index("CT1642:355")]
    with pytest.raises(ValueError, match="target longitudes"):
        parse_wso_filled_map(unexpected, target_rotation=1642)


def test_parser_rejects_an_extra_fixed_width_field() -> None:
    text, _ = _synthetic_filled_map_text()
    lines = text.splitlines()
    record = next(i for i, line in enumerate(lines) if line.startswith("CT1642:360"))
    lines[record] += f"{0.0:9.3f}"

    with pytest.raises(ValueError, match="trailing field"):
        parse_wso_filled_map("\n".join(lines) + "\n", target_rotation=1642)


def test_orthonormal_transform_recovers_a_known_y10_mode() -> None:
    cosine_colatitude, _ = np.polynomial.legendre.leggauss(
        LEGENDRE_LATITUDE_BINS
    )
    y10 = np.sqrt(3.0 / (4.0 * np.pi)) * cosine_colatitude
    field = np.broadcast_to(
        y10[None, :], (LEGENDRE_LONGITUDE_BINS, LEGENDRE_LATITUDE_BINS)
    ).copy()

    coefficients = spherical_harmonic_coefficients(field, lmax=4)

    assert coefficients[1, 0] == pytest.approx(1.0, abs=2e-14)
    residual = coefficients.copy()
    residual[1, 0] = 0.0
    assert np.max(np.abs(residual)) < 2e-14
    np.testing.assert_allclose(
        degree_energy_spectrum(coefficients),
        [0.0, 1.0, 0.0, 0.0, 0.0],
        atol=4e-14,
    )


def test_transform_reconstruction_and_parseval_agree() -> None:
    coefficients = np.zeros((9, 9), dtype=complex)
    coefficients[1, 0] = 0.7
    coefficients[2, 1] = 0.2 - 0.35j
    coefficients[5, 3] = -0.1 + 0.4j
    coefficients[8, 7] = 0.17 - 0.08j

    field = reconstruct_from_coefficients(coefficients)
    recovered = spherical_harmonic_coefficients(field, lmax=8)

    np.testing.assert_allclose(recovered, coefficients, atol=2e-13, rtol=2e-13)
    full_mask = np.ones(LEGENDRE_LATITUDE_BINS, dtype=bool)
    assert integrated_spatial_energy(field, full_mask) == pytest.approx(
        harmonic_energy(coefficients, minimum_degree=0), rel=3e-13, abs=3e-13
    )


def test_full_disk_regional_constructions_share_the_same_causal_input() -> None:
    longitude = np.arange(72, dtype=float)[:, None]
    sine_latitude = (np.arange(30, dtype=float)[None, :] - 14.5) / 15.0
    synthetic_los = (
        40.0 * np.cos(2.0 * np.pi * longitude / 72.0)
        + 30.0 * sine_latitude
        + 7.0 * np.sin(6.0 * np.pi * longitude / 72.0) * sine_latitude
    )

    products = rotation_energy_products(
        synthetic_los, spectrum_lmax=15, regional_lmax=15
    )

    expected = float(np.sum(products.spectrum[1:16]))
    assert products.masked_retransform["FD"] == pytest.approx(expected)
    assert products.lowpass_spatial["FD"] == pytest.approx(
        expected, rel=2e-13, abs=2e-13
    )


def test_paper_regions_partition_each_hemisphere_through_45_degrees() -> None:
    dummy = remap_line_of_sight_to_gauss(np.ones((72, 30)))
    masks = geometric_region_masks(dummy.latitude_degrees)

    assert np.all(~(masks["north"] & masks["south"]))
    assert np.all(masks["north"] | masks["south"])
    assert not np.any(masks["north_00_15"] & masks["north_15_30"])
    assert not np.any(masks["south_00_15"] & masks["south_15_30"])


def test_study_masks_preserve_inclusive_shared_boundary_rows() -> None:
    masks = study_region_masks()

    assert np.flatnonzero(masks["N_0_15"]).tolist() == list(range(30, 36))
    assert np.flatnonzero(masks["N_15_30"]).tolist() == list(range(35, 41))
    assert np.flatnonzero(masks["N_30_45"]).tolist() == list(range(40, 46))
    assert np.flatnonzero(masks["S_0_15"]).tolist() == list(range(24, 30))
    assert np.flatnonzero(masks["S_15_30"]).tolist() == list(range(19, 25))
    assert np.flatnonzero(masks["S_30_45"]).tolist() == list(range(14, 20))
    assert np.flatnonzero(masks["N_0_15"] & masks["N_15_30"]).tolist() == [35]
    assert np.flatnonzero(masks["S_0_15"] & masks["S_15_30"]).tolist() == [24]


def test_carrington_dates_reproduce_the_paper_boundaries() -> None:
    assert carrington_rotation_start(1642).date() == date(1976, 5, 27)
    assert carrington_rotation_start(2182).date() == date(2016, 9, 23)
    assert carrington_decimal_year(1642) == pytest.approx(1976.403462568306)
    assert carrington_decimal_year(2182) == pytest.approx(2016.7276756830602)
    assert study_decimal_year(carrington_rotation_start(1814)) == pytest.approx(
        1989.252688172043
    )


def test_preparation_cli_defaults_to_the_final_nine_regional_series(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "maps"
    input_directory.mkdir()
    text, _ = _synthetic_filled_map_text(rotation=1642)
    (input_directory / "WSO.1642.F.txt").write_text(text, encoding="ascii")
    output = tmp_path / "prepared.csv"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_wso_data.py"),
            "--input-dir",
            str(input_directory),
            "--output",
            str(output),
            "--first-cr",
            "1642",
            "--last-cr",
            "1642",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "carrington_rotation",
            "rotation_start_utc",
            "analysis_decimal_year",
            "FD",
            "NH",
            "SH",
            "N_0_15",
            "N_15_30",
            "N_30_45",
            "S_0_15",
            "S_15_30",
            "S_30_45",
        ]
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["carrington_rotation"] == "1642"
