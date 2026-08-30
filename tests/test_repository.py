from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from inceoglu2017.dataset import load_study_input
from inceoglu2017.events import analyze_study
from inceoglu2017.figure_data import load_figure_1_maps, load_mean_spectrum
from inceoglu2017.figures import ordinary_least_squares, propagated_decay_confidence
from inceoglu2017.provenance import EXPECTED_SHA256, verify_repository_inputs
from inceoglu2017.tables import TABLE_FILENAMES, write_study_tables


def test_committed_reproduction_inputs_have_immutable_checksums() -> None:
    verified = verify_repository_inputs()

    assert verified == {
        name: digest for name, (_, digest) in EXPECTED_SHA256.items()
    }


def test_single_clean_input_has_the_complete_study_interval() -> None:
    source = load_study_input()

    assert source.rotations.shape == (541,)
    assert source.global_energy.shape == (541, 3)
    assert source.band_energy.shape == (541, 6)
    assert source.rotations[[0, -1]].tolist() == [1642, 2182]


def test_figure_1_and_2_inputs_are_complete_numerical_products() -> None:
    context_maps = load_figure_1_maps()
    degrees, mean_energy = load_mean_spectrum()

    assert set(context_maps) == {1917, 2149}
    assert all(item.field_microtesla.shape == (72, 30) for item in context_maps.values())
    assert all(np.all(np.isfinite(item.field_microtesla)) for item in context_maps.values())
    np.testing.assert_array_equal(degrees, np.arange(1, 61))
    assert np.all(np.isfinite(mean_energy))
    assert np.all(mean_energy > 0.0)
    assert mean_energy[0] == pytest.approx(6.9499866547014593)
    assert mean_energy[-1] == pytest.approx(0.02209773883336234)


def test_table_writer_uses_calculated_study_values(tmp_path: Path) -> None:
    source = load_study_input()
    study = analyze_study(
        source.rotations, source.analysis_decimal_years, source.band_energy
    )

    written = write_study_tables(study, tmp_path)

    assert {path.name for path in written.values()} == set(TABLE_FILENAMES.values())
    with written[1].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert rows[0]["cycle_21_minimum_decimal_year"] == "1976.9"
    assert rows[0]["cycle_24_maximum_decimal_year"] == "2015.4"
    with written[2].open(encoding="utf-8", newline="") as handle:
        cross_rows = list(csv.DictReader(handle))
    assert [int(row["lag_carrington_rotations"]) for row in cross_rows] == [
        10,
        22,
        6,
        19,
    ]


def test_plotting_regression_and_decay_uncertainty_are_causal() -> None:
    slope, intercept = ordinary_least_squares([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert slope == 2.0
    assert intercept == 1.0

    years = np.arange(8, dtype=float)
    confidence = np.ones((8, 6), dtype=float)
    starts = np.tile([[1, 2]], (6, 1))
    ends = np.tile([[3, 6]], (6, 1))
    propagated = propagated_decay_confidence(years, confidence, starts, ends)
    np.testing.assert_allclose(
        propagated,
        np.tile([[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 4.0]], (6, 1)),
    )
