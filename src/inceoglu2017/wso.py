"""Prepare WSO synoptic maps and reproduce their harmonic energy products.

The paper calculation used the PFSS package in SolarSoft. This module
implements the relevant photospheric operations directly: strict map parsing,
line-of-sight to radial conversion, interpolation onto a 60-point
Gauss--Legendre latitude grid, and an orthonormal spherical-harmonic transform.
No PFSS extrapolation is required because the paper uses the photospheric
coefficients themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import gammaln, lpmv


WSO_LONGITUDE_BINS = 72
WSO_LATITUDE_BINS = 30
LEGENDRE_LATITUDE_BINS = 60
LEGENDRE_LONGITUDE_BINS = 2 * LEGENDRE_LATITUDE_BINS
MAX_SPHERICAL_HARMONIC_DEGREE = 60
PAPER_LARGE_SCALE_MAX_DEGREE = 15
MICROTESLA_TO_GAUSS = 0.01

# Carrington rotation 1 began at JD 2398167.4, conventionally represented by
# this UTC epoch.  A synodic rotation is 27.2753 days.  The resulting CR 1642
# and CR 2182 starts reproduce the dates stated in the paper (27 May 1976 and
# 23 September 2016).
CARRINGTON_ROTATION_ONE_START = datetime(1853, 11, 9, 21, 36, tzinfo=timezone.utc)
CARRINGTON_ROTATION_DAYS = 27.2753

_FILE_ROTATION = re.compile(r"^WSO\.(\d{4})\.F\.txt$")
_CARRINGTON_LINE = re.compile(r"^CT(\d+):(\d{3})")

GEOMETRIC_REGION_BOUNDS: dict[str, tuple[float, float]] = {
    "full_disk": (-90.0, 90.0),
    "north": (0.0, 90.0),
    "south": (-90.0, 0.0),
    "north_00_15": (0.0, 15.0),
    "north_15_30": (15.0, 30.0),
    "north_30_45": (30.0, 45.0),
    "south_00_15": (-15.0, 0.0),
    "south_15_30": (-30.0, -15.0),
    "south_30_45": (-45.0, -30.0),
}

# Paper-specific IDL range endpoints are inclusive, so adjacent 15-degree bands
# intentionally share one Gauss--Legendre latitude row. Python code must
# therefore use ``end + 1``.
STUDY_REGION_ROWS: dict[str, tuple[int, int]] = {
    "FD": (0, 59),
    "NH": (30, 59),
    "SH": (0, 29),
    "N_0_15": (30, 35),
    "N_15_30": (35, 40),
    "N_30_45": (40, 45),
    "S_0_15": (24, 29),
    "S_15_30": (19, 24),
    "S_30_45": (14, 19),
}


@dataclass(frozen=True)
class GaussLegendreMap:
    """Radial magnetic field sampled on the transform quadrature grid."""

    field_gauss: np.ndarray
    longitude_degrees: np.ndarray
    cosine_colatitude: np.ndarray
    latitude_degrees: np.ndarray
    latitude_weights: np.ndarray


@dataclass(frozen=True)
class RotationEnergyProducts:
    """Independent harmonic products used to compare regional formulations."""

    spectrum: np.ndarray
    masked_retransform: dict[str, float]
    lowpass_spatial: dict[str, float]


def infer_rotation(path: Path) -> int:
    """Extract the four-digit Carrington rotation from an official filename."""

    match = _FILE_ROTATION.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Expected filename WSO.####.F.txt; got {Path(path).name!r}")
    return int(match.group(1))


def _fixed_width_floats(line: str, count: int) -> list[float]:
    """Parse adjacent Fortran ``F9.3`` fields, including joined negatives."""

    required = 9 * count
    if len(line) < required:
        raise ValueError(
            f"Expected {count} fixed-width F9.3 fields ({required} characters); "
            f"got {len(line)}"
        )
    if line[required:].strip():
        raise ValueError("Unexpected trailing field in a WSO filled synoptic map")
    try:
        return [float(line[9 * index : 9 * (index + 1)]) for index in range(count)]
    except ValueError as error:
        raise ValueError("Invalid numeric field in a WSO filled synoptic map") from error


def parse_wso_filled_map(text: str, target_rotation: int) -> np.ndarray:
    """Parse one WSO filled map and return increasing longitude and latitude.

    An official file contains 72 records belonging to ``target_rotation`` and
    one longitude-360 seam record belonging to the next rotation.  Selection
    is by rotation label, not by row position, so the seam can never be
    mistaken for an observation in the target map.
    """

    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("WSO filled synoptic maps must contain ASCII text") from error

    lines = text.splitlines()
    target_rows: list[list[float]] = []
    target_longitudes: list[int] = []
    other_records: list[tuple[int, int]] = []

    index = 0
    while index < len(lines):
        match = _CARRINGTON_LINE.match(lines[index])
        if match is None:
            index += 1
            continue
        rotation = int(match.group(1))
        longitude = int(match.group(2))
        if index + 3 >= len(lines):
            raise ValueError(f"CR {target_rotation}: truncated Carrington record")

        expected_label = f"CT{rotation}:{longitude:03d}".ljust(18)
        if lines[index][:18] != expected_label:
            raise ValueError(f"CR {target_rotation}: malformed Carrington label")
        values = (
            _fixed_width_floats(lines[index][18:], 6)
            + _fixed_width_floats(lines[index + 1], 8)
            + _fixed_width_floats(lines[index + 2], 8)
            + _fixed_width_floats(lines[index + 3], 8)
        )
        if rotation == target_rotation:
            target_longitudes.append(longitude)
            target_rows.append(values)
        else:
            other_records.append((rotation, longitude))
        index += 4

    expected_longitudes = list(range(360, 0, -5))
    if target_longitudes != expected_longitudes:
        raise ValueError(
            f"CR {target_rotation}: expected target longitudes "
            "360, 355, ..., 5 exactly once"
        )
    if other_records not in ([], [(target_rotation + 1, 360)]):
        raise ValueError(
            f"CR {target_rotation}: expected only an optional next-rotation "
            f"longitude-360 seam; got {other_records}"
        )

    data = np.asarray(target_rows, dtype=float)
    expected_shape = (WSO_LONGITUDE_BINS, WSO_LATITUDE_BINS)
    if data.shape != expected_shape:
        raise ValueError(f"CR {target_rotation}: expected {expected_shape}; got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"CR {target_rotation}: map contains a non-finite value")

    # WSO stores longitude 360..5 and latitude north..south.  The preparation
    # grid uses increasing axes, so both dimensions are reversed exactly once.
    return data[::-1, ::-1]


def read_wso_filled_map(path: Path, target_rotation: int | None = None) -> np.ndarray:
    """Read a strict official ``WSO.####.F.txt`` file."""

    source = Path(path)
    rotation = infer_rotation(source) if target_rotation is None else target_rotation
    return parse_wso_filled_map(source.read_text(encoding="ascii"), rotation)


def _fractional_index(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return clamped fractional indexes used by the paper's remapping."""

    return np.interp(
        values,
        grid,
        np.arange(grid.size, dtype=float),
        left=0.0,
        right=float(grid.size - 1),
    )


def _bilinear_grid(
    values: np.ndarray,
    longitude_index: np.ndarray,
    latitude_index: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a longitude-by-latitude array on a product grid."""

    x0 = np.floor(longitude_index).astype(int)
    x1 = np.minimum(x0 + 1, values.shape[0] - 1)
    y0 = np.floor(latitude_index).astype(int)
    y1 = np.minimum(y0 + 1, values.shape[1] - 1)
    dx = longitude_index - x0
    dy = latitude_index - y0
    return (
        (1.0 - dx)[:, None]
        * (1.0 - dy)[None, :]
        * values[x0[:, None], y0[None, :]]
        + dx[:, None]
        * (1.0 - dy)[None, :]
        * values[x1[:, None], y0[None, :]]
        + (1.0 - dx)[:, None]
        * dy[None, :]
        * values[x0[:, None], y1[None, :]]
        + dx[:, None] * dy[None, :] * values[x1[:, None], y1[None, :]]
    )


def remap_line_of_sight_to_gauss(
    line_of_sight_microtesla: np.ndarray,
) -> GaussLegendreMap:
    """Convert one 72x30 WSO line-of-sight map to a radial 120x60 grid."""

    field = np.asarray(line_of_sight_microtesla, dtype=float)
    expected_shape = (WSO_LONGITUDE_BINS, WSO_LATITUDE_BINS)
    if field.shape != expected_shape:
        raise ValueError(f"Expected a {expected_shape} WSO map; got {field.shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError("WSO map contains a non-finite value")

    input_sine_latitude = np.arange(-14.5, 15.0, 1.0) / 15.0
    radial_microtesla = field / np.sqrt(1.0 - input_sine_latitude**2)[None, :]

    cosine_colatitude, latitude_weights = np.polynomial.legendre.leggauss(
        LEGENDRE_LATITUDE_BINS
    )
    input_latitude = np.degrees(np.arcsin(input_sine_latitude))
    output_latitude = np.degrees(np.arcsin(cosine_colatitude))
    input_longitude = np.arange(WSO_LONGITUDE_BINS, dtype=float) * 5.0 + 2.5
    output_longitude = (
        np.arange(LEGENDRE_LONGITUDE_BINS, dtype=float)
        * 360.0
        / LEGENDRE_LONGITUDE_BINS
    )
    radial_gauss = (
        _bilinear_grid(
            radial_microtesla,
            _fractional_index(input_longitude, output_longitude),
            _fractional_index(input_latitude, output_latitude),
        )
        * MICROTESLA_TO_GAUSS
    )
    return GaussLegendreMap(
        field_gauss=radial_gauss,
        longitude_degrees=output_longitude,
        cosine_colatitude=cosine_colatitude,
        latitude_degrees=output_latitude,
        latitude_weights=latitude_weights,
    )


def _validate_gauss_grid(field_gauss: np.ndarray) -> np.ndarray:
    field = np.asarray(field_gauss, dtype=float)
    expected_shape = (LEGENDRE_LONGITUDE_BINS, LEGENDRE_LATITUDE_BINS)
    if field.shape != expected_shape:
        raise ValueError(f"Expected a {expected_shape} Gauss grid; got {field.shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError("Gauss grid contains a non-finite value")
    return field


def _validate_lmax(lmax: int) -> None:
    if not 0 <= lmax <= MAX_SPHERICAL_HARMONIC_DEGREE:
        raise ValueError(
            f"lmax must be between 0 and {MAX_SPHERICAL_HARMONIC_DEGREE}"
        )


def orthonormal_legendre_basis(
    lmax: int,
    cosine_colatitude: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return normalized associated Legendre functions for each nonnegative m.

    For each ``m``, the returned array has rows ``l=m..lmax``.  Together with
    ``exp(i*m*longitude)``, the normalization is that of complex orthonormal
    spherical harmonics, including the Condon--Shortley phase used by SciPy.
    """

    x = np.asarray(cosine_colatitude, dtype=float)
    if x.ndim != 1 or x.size != LEGENDRE_LATITUDE_BINS:
        raise ValueError(
            f"Expected {LEGENDRE_LATITUDE_BINS} Gauss latitude nodes; got {x.shape}"
        )
    _validate_lmax(lmax)

    matrices: list[np.ndarray] = []
    for order in range(lmax + 1):
        rows: list[np.ndarray] = []
        for degree in range(order, lmax + 1):
            log_normalization = 0.5 * (
                np.log(2.0 * degree + 1.0)
                - np.log(4.0 * np.pi)
                + gammaln(degree - order + 1.0)
                - gammaln(degree + order + 1.0)
            )
            rows.append(
                np.exp(log_normalization) * lpmv(order, degree, x)
            )
        matrices.append(np.asarray(rows))
    return tuple(matrices)


@lru_cache(maxsize=None)
def _fixed_orthonormal_legendre_basis(lmax: int) -> tuple[np.ndarray, ...]:
    """Cache bases because every Carrington rotation uses the same grid."""

    cosine_colatitude, _ = np.polynomial.legendre.leggauss(
        LEGENDRE_LATITUDE_BINS
    )
    return orthonormal_legendre_basis(lmax, cosine_colatitude)


def spherical_harmonic_coefficients(
    field_gauss: np.ndarray,
    lmax: int = MAX_SPHERICAL_HARMONIC_DEGREE,
) -> np.ndarray:
    """Calculate standard complex coefficients ``a[l,m]`` for ``m >= 0``."""

    field = _validate_gauss_grid(field_gauss)
    _validate_lmax(lmax)
    _, latitude_weights = np.polynomial.legendre.leggauss(
        LEGENDRE_LATITUDE_BINS
    )
    basis = _fixed_orthonormal_legendre_basis(lmax)
    longitude_fourier = np.fft.fft(field, axis=0) / LEGENDRE_LONGITUDE_BINS
    sphere_weights = latitude_weights * (2.0 * np.pi)

    coefficients = np.zeros((lmax + 1, lmax + 1), dtype=complex)
    for order, functions in enumerate(basis):
        coefficients[order:, order] = np.einsum(
            "ln,n->l",
            functions * sphere_weights[None, :],
            longitude_fourier[order],
        )
    return coefficients


def degree_energy_spectrum(coefficients: np.ndarray) -> np.ndarray:
    """Return ``sum_m |a_lm|^2`` for a real field at every degree.

    Only nonnegative orders are stored.  Positive-order power is multiplied by
    two to account for the conjugate negative-order coefficient, matching the
    historical IDL ``0.5 * |2*a_lm|^2`` convention.
    """

    values = np.asarray(coefficients, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Coefficients must be a square [degree, order] array")
    spectrum = np.abs(values[:, 0]) ** 2
    if values.shape[1] > 1:
        spectrum = spectrum + 2.0 * np.sum(np.abs(values[:, 1:]) ** 2, axis=1)
    return np.asarray(spectrum, dtype=float)


def harmonic_energy(
    coefficients: np.ndarray,
    minimum_degree: int = 1,
    maximum_degree: int | None = None,
) -> float:
    """Sum the energy spectrum over an inclusive degree interval."""

    spectrum = degree_energy_spectrum(coefficients)
    upper = spectrum.size - 1 if maximum_degree is None else maximum_degree
    if not 0 <= minimum_degree <= upper < spectrum.size:
        raise ValueError("Invalid spherical-harmonic degree interval")
    return float(np.sum(spectrum[minimum_degree : upper + 1]))


def reconstruct_from_coefficients(
    coefficients: np.ndarray,
    minimum_degree: int = 0,
) -> np.ndarray:
    """Reconstruct a real 120x60 grid from nonnegative-order coefficients."""

    values = np.asarray(coefficients, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Coefficients must be a square [degree, order] array")
    lmax = values.shape[0] - 1
    if not 0 <= minimum_degree <= lmax:
        raise ValueError("minimum_degree is outside the coefficient range")

    basis = _fixed_orthonormal_legendre_basis(lmax)
    longitude_fourier = np.zeros(
        (LEGENDRE_LONGITUDE_BINS, LEGENDRE_LATITUDE_BINS), dtype=complex
    )
    for order, functions in enumerate(basis):
        first_degree = max(order, minimum_degree)
        if first_degree > lmax:
            continue
        first_row = first_degree - order
        longitude_fourier[order] = np.einsum(
            "l,ln->n",
            values[first_degree:, order],
            functions[first_row:],
        )
        if order > 0:
            longitude_fourier[-order] = np.conjugate(longitude_fourier[order])
    return np.fft.ifft(
        longitude_fourier * LEGENDRE_LONGITUDE_BINS, axis=0
    ).real


def latitude_band_mask(
    latitude_degrees: np.ndarray,
    lower_degrees: float,
    upper_degrees: float,
) -> np.ndarray:
    """Select a signed latitude interval on the Gauss grid."""

    latitude = np.asarray(latitude_degrees, dtype=float)
    if latitude.ndim != 1 or latitude.size != LEGENDRE_LATITUDE_BINS:
        raise ValueError(
            f"Expected {LEGENDRE_LATITUDE_BINS} latitude values; got {latitude.shape}"
        )
    if not -90.0 <= lower_degrees < upper_degrees <= 90.0:
        raise ValueError("Latitude bounds must satisfy -90 <= lower < upper <= 90")
    # Gauss nodes lie strictly inside (-90, 90) and none fall on the paper's
    # 0/15/30/45-degree boundaries, so adjacent half-open bands are disjoint.
    return (latitude >= lower_degrees) & (latitude < upper_degrees)


def geometric_region_masks(latitude_degrees: np.ndarray) -> dict[str, np.ndarray]:
    """Build disjoint masks from nominal signed latitude boundaries.

    These masks are useful for checking sensitivity to literal angular
    boundaries. They are not the inclusive-row masks used for the paper's
    regional series; use :func:`study_region_masks` for reproduction.
    """

    return {
        name: latitude_band_mask(latitude_degrees, lower, upper)
        for name, (lower, upper) in GEOMETRIC_REGION_BOUNDS.items()
    }


def study_region_masks() -> dict[str, np.ndarray]:
    """Return the nine exact inclusive-row masks used by the study."""

    masks: dict[str, np.ndarray] = {}
    for name, (first_row, last_row) in STUDY_REGION_ROWS.items():
        mask = np.zeros(LEGENDRE_LATITUDE_BINS, dtype=bool)
        mask[first_row : last_row + 1] = True
        masks[name] = mask
    return masks


def _validate_latitude_mask(mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (LEGENDRE_LATITUDE_BINS,):
        raise ValueError(
            f"Expected a {LEGENDRE_LATITUDE_BINS}-element latitude mask; "
            f"got {selected.shape}"
        )
    if not np.any(selected):
        raise ValueError("Latitude mask selects no Gauss nodes")
    return selected


def masked_retransform_energy(
    field_gauss: np.ndarray,
    latitude_mask: np.ndarray,
    lmax: int = PAPER_LARGE_SCALE_MAX_DEGREE,
    minimum_degree: int = 1,
) -> float:
    """Zero the field outside a region, retransform it, and sum mode energy."""

    field = _validate_gauss_grid(field_gauss)
    _validate_lmax(lmax)
    selected = _validate_latitude_mask(latitude_mask)
    masked = field * selected[None, :]
    coefficients = spherical_harmonic_coefficients(masked, lmax=lmax)
    return harmonic_energy(coefficients, minimum_degree, lmax)


def integrated_spatial_energy(
    field_gauss: np.ndarray,
    latitude_mask: np.ndarray,
) -> float:
    """Integrate ``B^2`` over a selected latitude interval and all longitudes."""

    field = _validate_gauss_grid(field_gauss)
    selected = _validate_latitude_mask(latitude_mask)
    _, latitude_weights = np.polynomial.legendre.leggauss(LEGENDRE_LATITUDE_BINS)
    longitude_mean_square = np.mean(field**2, axis=0)
    return float(
        np.sum(longitude_mean_square[selected] * latitude_weights[selected])
        * (2.0 * np.pi)
    )


def lowpass_spatial_energy(
    field_gauss: np.ndarray,
    latitude_mask: np.ndarray,
    lmax: int = PAPER_LARGE_SCALE_MAX_DEGREE,
    minimum_degree: int = 1,
) -> float:
    """Low-pass the full field, then integrate its energy inside a region."""

    coefficients = spherical_harmonic_coefficients(field_gauss, lmax=lmax)
    lowpass = reconstruct_from_coefficients(
        coefficients, minimum_degree=minimum_degree
    )
    return integrated_spatial_energy(lowpass, latitude_mask)


def rotation_energy_products(
    line_of_sight_microtesla: np.ndarray,
    regions: Mapping[str, np.ndarray] | None = None,
    spectrum_lmax: int = MAX_SPHERICAL_HARMONIC_DEGREE,
    regional_lmax: int = PAPER_LARGE_SCALE_MAX_DEGREE,
) -> RotationEnergyProducts:
    """Calculate the full spectrum and both candidate regional definitions."""

    _validate_lmax(spectrum_lmax)
    _validate_lmax(regional_lmax)
    if regional_lmax > spectrum_lmax:
        raise ValueError("regional_lmax cannot exceed spectrum_lmax")

    prepared = remap_line_of_sight_to_gauss(line_of_sight_microtesla)
    masks = (
        study_region_masks()
        if regions is None
        else {name: _validate_latitude_mask(mask) for name, mask in regions.items()}
    )

    full_coefficients = spherical_harmonic_coefficients(
        prepared.field_gauss, lmax=spectrum_lmax
    )
    spectrum = degree_energy_spectrum(full_coefficients)

    regional_coefficients = full_coefficients[: regional_lmax + 1, : regional_lmax + 1]
    lowpass = reconstruct_from_coefficients(regional_coefficients, minimum_degree=1)
    masked_retransform: dict[str, float] = {}
    lowpass_spatial: dict[str, float] = {}
    for name, mask in masks.items():
        if np.all(mask):
            masked_retransform[name] = float(np.sum(spectrum[1 : regional_lmax + 1]))
        else:
            masked_retransform[name] = masked_retransform_energy(
                prepared.field_gauss, mask, lmax=regional_lmax, minimum_degree=1
            )
        lowpass_spatial[name] = integrated_spatial_energy(lowpass, mask)

    return RotationEnergyProducts(
        spectrum=spectrum,
        masked_retransform=masked_retransform,
        lowpass_spatial=lowpass_spatial,
    )


def carrington_rotation_start(rotation: int) -> datetime:
    """Return the conventional UTC start of a Carrington rotation."""

    if rotation < 1:
        raise ValueError("Carrington rotation numbers start at 1")
    return CARRINGTON_ROTATION_ONE_START + timedelta(
        days=CARRINGTON_ROTATION_DAYS * (rotation - 1)
    )


def decimal_year(moment: datetime) -> float:
    """Convert a timezone-aware datetime to a calendar-length decimal year."""

    if moment.tzinfo is None:
        raise ValueError("decimal_year requires a timezone-aware datetime")
    start = datetime(moment.year, 1, 1, tzinfo=moment.tzinfo)
    end = datetime(moment.year + 1, 1, 1, tzinfo=moment.tzinfo)
    return moment.year + (moment - start) / (end - start)


def study_decimal_year(moment: datetime) -> float:
    """Return the decimal-year convention used by the study.

    The paper-specific calculation represents each calendar month as one
    twelfth of a year and the day as ``day / 31`` of that month. It intentionally
    ignores time of day and unequal month lengths.  Keeping this function
    separate from :func:`decimal_year` makes the study analysis clock
    explicit while preserving a conventional conversion for other uses.
    """

    if moment.tzinfo is None:
        raise ValueError("study_decimal_year requires a timezone-aware datetime")
    return moment.year + (moment.month + moment.day / 31.0 - 1.0) / 12.0


def carrington_decimal_year(rotation: int) -> float:
    """Return the decimal year at the start of a Carrington rotation."""

    return decimal_year(carrington_rotation_start(rotation))
