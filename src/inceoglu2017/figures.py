"""Recalculate the paper's seven figures from numerical inputs.

The plotting functions in this module are deliberately separated from file
I/O.  Each ``make_figure_*`` function consumes already-calculated numerical
products and returns one Matplotlib ``Figure``.  Consequently the scientific
relationships can be tested without writing files, while
``write_recalculated_figures`` is the single stateful orchestration boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .figure_data import (
    FIGURE_1_ROTATIONS,
    WsoContextMap,
    load_figure_1_maps,
    load_mean_spectrum,
)
from .io import read_csv
from .paths import (
    OUTPUT_FIGURES,
    WSO_ENERGY,
    WSO_FIGURE_1_MAPS,
    WSO_MEAN_SPECTRUM,
)


PUBLISHED_BAND_NAMES: tuple[str, ...] = (
    "N_0_15",
    "N_15_30",
    "N_30_45",
    "S_0_15",
    "S_15_30",
    "S_30_45",
)
PUBLISHED_BAND_LABELS: tuple[str, ...] = (
    r"0-15$^{\circ}$ N",
    r"15-30$^{\circ}$ N",
    r"30-45$^{\circ}$ N",
    r"0-15$^{\circ}$ S",
    r"15-30$^{\circ}$ S",
    r"30-45$^{\circ}$ S",
)
PUBLISHED_CYCLES: tuple[int, ...] = (21, 22, 23, 24)

_GLOBAL_NAMES = ("FD", "NH", "SH")
_GLOBAL_LABELS = ("Full disk", "Northern hemisphere", "Southern hemisphere")
_LATITUDE_COLORS = ("#b517b8", "#2450e6", "#18a83b")
_RELATIONSHIP_HANDLES = (
    Line2D(
        [],
        [],
        color="#252525",
        marker="o",
        linestyle="-",
        label="Peak amplitude",
    ),
    Line2D(
        [],
        [],
        color="#252525",
        marker="x",
        linestyle="--",
        label="Rise amplitude",
    ),
)
_FIXED_PDF_DATE = datetime(2017, 5, 1, tzinfo=timezone.utc)


def make_figure_1(context_maps: dict[int, WsoContextMap]) -> Figure:
    """Build Figure 1 from the two selected native-grid WSO maps."""

    if set(context_maps) != set(FIGURE_1_ROTATIONS):
        raise ValueError(
            f"Figure 1 requires WSO maps {sorted(FIGURE_1_ROTATIONS)}"
        )
    figure, axes = _subplots(2, 1, figsize=(10.0, 7.3), sharex=True, sharey=True)
    titles = {
        2149: "Solar-cycle 24 maximum - April 2014",
        1917: "Solar-cycle 23 minimum - December 1996",
    }
    positive_levels = np.asarray([100.0, 500.0, 1000.0, 2000.0])
    negative_levels = -positive_levels[::-1]

    for panel, rotation in enumerate(FIGURE_1_ROTATIONS):
        axis = axes[panel, 0]
        context = context_maps[rotation]
        longitude, field = _cyclic_context_field(context)
        latitude = np.asarray(context.latitude_degrees, dtype=float)
        minimum = float(np.min(field))
        maximum = float(np.max(field))
        available_negative = negative_levels[
            (negative_levels >= minimum) & (negative_levels <= maximum)
        ]
        available_positive = positive_levels[
            (positive_levels >= minimum) & (positive_levels <= maximum)
        ]
        if available_negative.size:
            axis.contour(
                longitude,
                latitude,
                field.T,
                levels=available_negative,
                colors="#c62828",
                linestyles="dashed",
                linewidths=0.9,
            )
        if minimum <= 0.0 <= maximum:
            axis.contour(
                longitude,
                latitude,
                field.T,
                levels=[0.0],
                colors="#202020",
                linewidths=1.0,
            )
        if available_positive.size:
            axis.contour(
                longitude,
                latitude,
                field.T,
                levels=available_positive,
                colors="#244db5",
                linestyles="solid",
                linewidths=0.9,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.55, alpha=0.55)
        axis.set_xlim(0.0, 360.0)
        axis.set_ylim(-75.0, 75.0)
        axis.set_yticks([-60.0, -30.0, 0.0, 30.0, 60.0])
        axis.set_ylabel("Heliographic latitude (deg)")
        axis.set_title(f"({chr(ord('a') + panel)}) CR {rotation}: {titles[rotation]}")
        axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.55)
        axis.tick_params(direction="out", length=3.0, width=0.7)

    axes[-1, 0].set_xticks(np.arange(0.0, 361.0, 60.0))
    axes[-1, 0].set_xlabel("Carrington longitude (deg)")
    figure.suptitle("WSO photospheric magnetic-field distributions")
    figure.legend(
        handles=(
            Line2D([], [], color="#244db5", label="Positive field"),
            Line2D([], [], color="#202020", label="Neutral line"),
            Line2D(
                [],
                [],
                color="#c62828",
                linestyle="dashed",
                label="Negative field",
            ),
        ),
        loc="outside lower center",
        ncols=3,
        frameon=False,
        title="Contours at 0 and +/-100, 500, 1000, 2000 microtesla",
    )
    return figure


def make_figure_2(
    degrees: Sequence[int] | np.ndarray,
    mean_energy: Sequence[float] | np.ndarray,
) -> Figure:
    """Build Figure 2 from the CR 1642--2182 time-mean energy spectrum."""

    degree = np.asarray(degrees, dtype=int).reshape(-1)
    energy = np.asarray(mean_energy, dtype=float).reshape(-1)
    if not np.array_equal(degree, np.arange(1, 61, dtype=int)):
        raise ValueError("Figure 2 requires spherical-harmonic degrees 1--60")
    if energy.shape != degree.shape or not np.all(np.isfinite(energy)):
        raise ValueError("Figure 2 energies must be finite and match the degrees")
    if np.any(energy <= 0.0):
        raise ValueError("Figure 2 energies must be positive for logarithmic axes")

    figure, axes = _subplots(1, 1, figsize=(6.8, 5.8))
    axis = axes[0, 0]
    axis.loglog(
        degree,
        energy,
        color="#d52020",
        marker="D",
        markerfacecolor="none",
        markeredgewidth=0.9,
        markersize=4.4,
        linewidth=1.35,
    )
    axis.axvline(
        15.0,
        color="#4d4d4d",
        linestyle="--",
        linewidth=0.9,
        label="Analysis limit (degree 15)",
    )
    axis.set_xlabel("Spherical-harmonic degree")
    axis.set_ylabel(r"Mean degree energy (G$^2$)")
    axis.set_title("Time-averaged WSO harmonic-energy spectrum, CR 1642-2182")
    axis.grid(True, which="both", color="#d2d2d2", linewidth=0.55, alpha=0.7)
    axis.legend(loc="lower left", frameon=False)
    axis.tick_params(direction="out", length=3.0, width=0.7)
    return figure


def _cyclic_context_field(context: WsoContextMap) -> tuple[np.ndarray, np.ndarray]:
    """Put Carrington longitude 360 at zero and close the periodic seam."""

    longitude = np.mod(np.asarray(context.longitude_degrees, dtype=float), 360.0)
    field = np.asarray(context.field_microtesla, dtype=float)
    if field.shape != (longitude.size, np.asarray(context.latitude_degrees).size):
        raise ValueError("Figure 1 map coordinates do not match its field grid")
    order = np.argsort(longitude)
    ordered_longitude = longitude[order]
    ordered_field = field[order]
    if not np.array_equal(ordered_longitude, np.arange(0.0, 360.0, 5.0)):
        raise ValueError("Figure 1 requires the native five-degree WSO longitude grid")
    return (
        np.append(ordered_longitude, 360.0),
        np.vstack((ordered_field, ordered_field[0])),
    )


def ordinary_least_squares(
    predictor: Sequence[float] | np.ndarray,
    response: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Return the OLS slope and intercept for finite paired observations.

    A fit is undefined unless there are at least two observations at distinct
    predictor values.  Rejecting that condition is preferable to drawing an
    arbitrary line for a singular design matrix.
    """

    x = np.asarray(predictor, dtype=float).reshape(-1)
    y = np.asarray(response, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Predictor and response must have the same shape")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        raise ValueError("OLS requires at least two finite observation pairs")

    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("OLS requires at least two distinct predictor values")
    slope = float(np.dot(centered, y - np.mean(y)) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def confidence_at_indices(
    confidence_half_width: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Select per-band confidence half-widths at event sample indices.

    ``confidence_half_width`` has time along rows and the six latitude bands
    along columns.  ``sample_indices`` has one row per band.  The explicit
    orientation check prevents accidentally taking the confidence interval
    from a different latitude, which would produce plausible-looking but
    causally unrelated error bars.
    """

    confidence = _matrix("confidence_half_width", confidence_half_width, columns=6)
    indices = _integer_matrix("sample_indices", sample_indices, rows=6)
    if np.any(indices < 0) or np.any(indices >= confidence.shape[0]):
        raise ValueError("Event sample index is outside the confidence series")
    band_indices = np.arange(6, dtype=int)[:, None]
    return confidence[indices, band_indices]


def propagated_decay_confidence(
    decimal_year: Sequence[float] | np.ndarray,
    confidence_half_width: np.ndarray,
    decay_start_indices: np.ndarray,
    decay_end_indices: np.ndarray,
) -> np.ndarray:
    """Propagate endpoint energy uncertainty into decay-rate uncertainty.

    The decay rate is an energy difference divided by elapsed time.  Treating
    the two endpoint estimates as independent therefore gives
    ``hypot(h_start, h_end) / delta_time``.  This is the vertical uncertainty
    used in Figure 7; amplitude uncertainty belongs to its horizontal variable
    and is not substituted for decay-rate uncertainty.
    """

    years = _vector("decimal_year", decimal_year)
    confidence = _matrix(
        "confidence_half_width",
        confidence_half_width,
        rows=years.size,
        columns=6,
    )
    starts = _integer_matrix("decay_start_indices", decay_start_indices, rows=6)
    ends = _integer_matrix(
        "decay_end_indices",
        decay_end_indices,
        rows=6,
        columns=starts.shape[1],
    )
    if np.any(starts < 0) or np.any(ends < 0):
        raise ValueError("Decay endpoint indices must be non-negative")
    if np.any(starts >= years.size) or np.any(ends >= years.size):
        raise ValueError("Decay endpoint index is outside the time series")

    start_error = confidence_at_indices(confidence, starts)
    end_error = confidence_at_indices(confidence, ends)
    elapsed = np.abs(years[ends] - years[starts])
    if np.any(~np.isfinite(elapsed)) or np.any(elapsed <= 0.0):
        raise ValueError("Decay endpoints must define a positive elapsed time")
    return np.hypot(start_error, end_error) / elapsed


def make_figure_3(
    decimal_year: Sequence[float] | np.ndarray,
    smoothed_energy: np.ndarray,
    confidence_half_width: np.ndarray,
) -> Figure:
    """Build recalculated Figure 3 for full-disk and hemispheric energy."""

    years, values, confidence = _time_series_inputs(
        decimal_year,
        smoothed_energy,
        confidence_half_width,
        columns=3,
    )
    figure, axes = _subplots(3, 1, figsize=(8.0, 8.8), sharex=True)
    line_styles = ("-", "--", "-.")

    for index, axis in enumerate(axes[:, 0]):
        color = "#222222"
        lower = values[:, index] - confidence[:, index]
        upper = values[:, index] + confidence[:, index]
        axis.fill_between(
            years,
            lower,
            upper,
            color="#8f8f8f",
            alpha=0.38,
            linewidth=0.0,
            label="99% t confidence band" if index == 0 else None,
        )
        axis.plot(
            years,
            values[:, index],
            color=color,
            linestyle=line_styles[index],
            linewidth=1.45,
            label=_GLOBAL_LABELS[index],
        )
        _finish_axis(axis)
        axis.set_ylim(bottom=0.0)
        axis.text(
            0.015,
            0.91,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        axis.legend(loc="upper right", frameon=False)

    axes[-1, 0].set_xlabel("Decimal year")
    figure.supylabel(r"Large-scale magnetic-energy proxy (G$^2$)")
    figure.suptitle("Full-disk and hemispheric solar-cycle progression")
    return figure


def make_figure_4(
    decimal_year: Sequence[float] | np.ndarray,
    smoothed_energy: np.ndarray,
    confidence_half_width: np.ndarray,
) -> Figure:
    """Build recalculated Figure 4 for the six 15-degree latitude bands."""

    years, values, confidence = _time_series_inputs(
        decimal_year,
        smoothed_energy,
        confidence_half_width,
        columns=6,
    )
    figure, axes = _subplots(
        3,
        2,
        figsize=(10.4, 8.1),
        sharex=True,
        sharey="row",
    )

    panel = 0
    for latitude_index in range(3):
        for hemisphere_index, band_index in enumerate(
            (latitude_index, latitude_index + 3)
        ):
            axis = axes[latitude_index, hemisphere_index]
            color = _LATITUDE_COLORS[latitude_index]
            axis.fill_between(
                years,
                values[:, band_index] - confidence[:, band_index],
                values[:, band_index] + confidence[:, band_index],
                color=color,
                alpha=0.25,
                linewidth=0.0,
            )
            axis.plot(years, values[:, band_index], color=color, linewidth=1.4)
            _finish_axis(axis)
            axis.set_title(PUBLISHED_BAND_LABELS[band_index], loc="right")
            axis.text(
                0.015,
                0.91,
                f"({chr(ord('a') + panel)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
            panel += 1

        # Both hemispheres must contribute to the shared row scale before its
        # physical lower bound is fixed.  Setting this inside the panel loop
        # disables autoscaling and can clip the second hemisphere.
        axes[latitude_index, 0].set_ylim(bottom=0.0)

    axes[-1, 0].set_xlabel("Decimal year")
    axes[-1, 1].set_xlabel("Decimal year")
    figure.supylabel(r"Large-scale magnetic-energy proxy (G$^2$)")
    figure.suptitle("Solar-cycle progression in 15-degree latitude bands (99% t CI)")
    return figure


def make_figure_5(
    metrics: Any,
    events: Any,
    confidence_half_width: np.ndarray,
) -> Figure:
    """Build recalculated Figure 5, grouping four cycles by latitude band."""

    peak_times = _metric(metrics, "peak_times", (6, 4))
    peak_amplitudes = _metric(metrics, "peak_amplitudes", (6, 4))
    rise_times = _metric(metrics, "rise_times", (6, 4))
    rise_amplitudes = _metric(metrics, "rise_amplitudes", (6, 4))
    peak_indices = _integer_matrix(
        "events.primary_peaks",
        getattr(events, "primary_peaks"),
        rows=6,
        columns=4,
    )
    rise_indices = _integer_matrix(
        "metrics.rise_sample_indices",
        getattr(metrics, "rise_sample_indices"),
        rows=6,
        columns=4,
    )
    peak_error = confidence_at_indices(confidence_half_width, peak_indices)
    rise_error = confidence_at_indices(confidence_half_width, rise_indices)

    figure, axes = _subplots(
        3,
        2,
        figsize=(9.0, 9.2),
        sharex="col",
        sharey="row",
    )
    panel = 0
    for latitude_index in range(3):
        for hemisphere_index, band_index in enumerate(
            (latitude_index, latitude_index + 3)
        ):
            axis = axes[latitude_index, hemisphere_index]
            color = _LATITUDE_COLORS[latitude_index]
            _relationship_with_fit(
                axis,
                peak_times[band_index],
                peak_amplitudes[band_index],
                peak_error[band_index],
                color=color,
                marker="o",
                linestyle="-",
            )
            _relationship_with_fit(
                axis,
                rise_times[band_index],
                rise_amplitudes[band_index],
                rise_error[band_index],
                color=color,
                marker="x",
                linestyle="--",
            )
            _finish_axis(axis)
            axis.set_title(PUBLISHED_BAND_LABELS[band_index], loc="right")
            axis.text(
                0.015,
                0.91,
                f"({chr(ord('a') + panel)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
            panel += 1

        axes[latitude_index, 0].set_ylim(bottom=0.0)

    # Columns share x scales, so postpone the lower bound until every latitude
    # row has contributed its observations.
    axes[-1, 0].set_xlim(left=0.0)
    axes[-1, 1].set_xlim(left=0.0)

    axes[-1, 0].set_xlabel("Time from cycle minimum (years)")
    axes[-1, 1].set_xlabel("Time from cycle minimum (years)")
    figure.supylabel(r"Magnetic-energy amplitude (G$^2$)")
    figure.suptitle("Cycle amplitude-time relationships by latitude")
    figure.legend(
        handles=_RELATIONSHIP_HANDLES,
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    return figure


def make_figure_6(
    metrics: Any,
    events: Any,
    confidence_half_width: np.ndarray,
) -> Figure:
    """Build recalculated Figure 6, grouping three latitudes within each cycle."""

    peak_times = _metric(metrics, "peak_times", (6, 4))
    peak_amplitudes = _metric(metrics, "peak_amplitudes", (6, 4))
    rise_times = _metric(metrics, "rise_times", (6, 4))
    rise_amplitudes = _metric(metrics, "rise_amplitudes", (6, 4))
    peak_indices = _integer_matrix(
        "events.primary_peaks",
        getattr(events, "primary_peaks"),
        rows=6,
        columns=4,
    )
    rise_indices = _integer_matrix(
        "metrics.rise_sample_indices",
        getattr(metrics, "rise_sample_indices"),
        rows=6,
        columns=4,
    )
    peak_error = confidence_at_indices(confidence_half_width, peak_indices)
    rise_error = confidence_at_indices(confidence_half_width, rise_indices)

    figure, axes = _subplots(
        4,
        2,
        figsize=(9.0, 10.8),
        sharex=True,
        sharey="row",
    )
    panel = 0
    for cycle_index, cycle in enumerate(PUBLISHED_CYCLES):
        for hemisphere_index, band_slice in enumerate((slice(0, 3), slice(3, 6))):
            axis = axes[cycle_index, hemisphere_index]
            colors = _LATITUDE_COLORS
            _multicolor_relationship_with_fit(
                axis,
                peak_times[band_slice, cycle_index],
                peak_amplitudes[band_slice, cycle_index],
                peak_error[band_slice, cycle_index],
                colors=colors,
                marker="o",
                linestyle="-",
            )
            _multicolor_relationship_with_fit(
                axis,
                rise_times[band_slice, cycle_index],
                rise_amplitudes[band_slice, cycle_index],
                rise_error[band_slice, cycle_index],
                colors=colors,
                marker="x",
                linestyle="--",
            )
            _finish_axis(axis)
            hemisphere = "Northern" if hemisphere_index == 0 else "Southern"
            axis.set_title(f"Cycle {cycle} - {hemisphere} hemisphere", loc="left")
            axis.text(
                0.98,
                0.91,
                f"({chr(ord('a') + panel)})",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
            )
            panel += 1

        axes[cycle_index, 0].set_ylim(bottom=0.0)

    # All eight panels share the time axis.  Fixing it earlier would freeze the
    # range before later cycles and hemispheres had been plotted.
    axes[-1, 0].set_xlim(left=0.0)

    axes[-1, 0].set_xlabel("Time from cycle minimum (years)")
    axes[-1, 1].set_xlabel("Time from cycle minimum (years)")
    figure.supylabel(r"Magnetic-energy amplitude (G$^2$)")
    figure.suptitle("Within-cycle amplitude-time relationships across latitude")
    latitude_handles = tuple(
        Line2D([], [], color=color, marker="o", linestyle="none", label=label)
        for color, label in zip(
            _LATITUDE_COLORS,
            ("0-15 deg", "15-30 deg", "30-45 deg"),
        )
    )
    figure.legend(
        handles=_RELATIONSHIP_HANDLES + latitude_handles,
        loc="outside lower center",
        ncols=5,
        frameon=False,
    )
    return figure


def make_figure_7(
    metrics: Any,
    decay_confidence_half_width: np.ndarray,
) -> Figure:
    """Build recalculated Figure 7 for cycles 21--23 decay relationships."""

    peak_amplitudes = _metric(metrics, "peak_amplitudes", (6, 4))[:, :3]
    rise_amplitudes = _metric(metrics, "rise_amplitudes", (6, 4))[:, :3]
    decay_rates = _metric(metrics, "decay_rates", (6, 3))
    decay_error = _matrix(
        "decay_confidence_half_width",
        decay_confidence_half_width,
        rows=6,
        columns=3,
    )
    if np.any(decay_error < 0.0):
        raise ValueError("Decay-rate confidence half-widths cannot be negative")

    figure, axes = _subplots(
        3,
        2,
        figsize=(9.0, 9.2),
        sharex="row",
        sharey="row",
    )
    panel = 0
    for latitude_index in range(3):
        for hemisphere_index, band_index in enumerate(
            (latitude_index, latitude_index + 3)
        ):
            axis = axes[latitude_index, hemisphere_index]
            color = _LATITUDE_COLORS[latitude_index]
            _relationship_with_fit(
                axis,
                peak_amplitudes[band_index],
                decay_rates[band_index],
                decay_error[band_index],
                color=color,
                marker="o",
                linestyle="-",
            )
            _relationship_with_fit(
                axis,
                rise_amplitudes[band_index],
                decay_rates[band_index],
                decay_error[band_index],
                color=color,
                marker="x",
                linestyle="--",
            )
            _finish_axis(axis)
            axis.set_title(PUBLISHED_BAND_LABELS[band_index], loc="right")
            axis.text(
                0.015,
                0.91,
                f"({chr(ord('a') + panel)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
            panel += 1

        axes[latitude_index, 0].set_xlim(left=0.0)
        axes[latitude_index, 0].set_ylim(bottom=0.0)

    axes[-1, 0].set_xlabel(r"Magnetic-energy amplitude (G$^2$)")
    axes[-1, 1].set_xlabel(r"Magnetic-energy amplitude (G$^2$)")
    figure.supylabel(r"Decay rate (G$^2$ year$^{-1}$)")
    figure.suptitle("Cycles 21-23 decay-rate relationships")
    figure.legend(
        handles=_RELATIONSHIP_HANDLES,
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    return figure


def write_recalculated_figures(
    output_directory: Path = OUTPUT_FIGURES,
    data_path: Path = WSO_ENERGY,
    figure_1_data_path: Path = WSO_FIGURE_1_MAPS,
    spectrum_path: Path = WSO_MEAN_SPECTRUM,
) -> dict[int, Path]:
    """Write deterministic, single-page PDFs for recalculated Figures 1--7.

    All inputs are committed WSO-derived numerical CSV files. Figures 1 and 2
    use the two context maps and time-mean degree spectrum, respectively. The
    energy series supplies Figures 3--7 and their shared event analysis.
    """

    # Local imports keep the pure plotting API usable independently of the
    # event-detection implementation and avoid a module-level dependency cycle.
    from .analysis import smooth_with_confidence
    from .events import analyse_study

    source = Path(data_path)
    rotations, years, global_energy, band_energy = _read_energy_products(source)
    context_maps = load_figure_1_maps(figure_1_data_path)
    spectrum_degrees, mean_spectrum = load_mean_spectrum(spectrum_path)

    global_smoothed = [
        smooth_with_confidence(global_energy[:, column], span=11, confidence=0.99)
        for column in range(global_energy.shape[1])
    ]
    global_values = np.column_stack([series.values for series in global_smoothed])
    global_confidence = np.column_stack(
        [series.confidence_half_width for series in global_smoothed]
    )

    study = analyse_study(rotations, years, band_energy)
    decay_confidence = propagated_decay_confidence(
        years,
        study.smoothed.confidence_half_width,
        study.metrics.decay_start_indices,
        study.metrics.decay_end_indices,
    )
    figures = {
        1: make_figure_1(context_maps),
        2: make_figure_2(spectrum_degrees, mean_spectrum),
        3: make_figure_3(years, global_values, global_confidence),
        4: make_figure_4(
            years,
            study.smoothed.values,
            study.smoothed.confidence_half_width,
        ),
        5: make_figure_5(
            study.metrics,
            study.events,
            study.smoothed.confidence_half_width,
        ),
        6: make_figure_6(
            study.metrics,
            study.events,
            study.smoothed.confidence_half_width,
        ),
        7: make_figure_7(study.metrics, decay_confidence),
    }

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    try:
        for number, figure in figures.items():
            path = destination / f"figure{number}.pdf"
            _write_pdf(figure, path, number)
            written[number] = path
    finally:
        for figure in figures.values():
            figure.clear()
    return written


def _read_energy_products(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"WSO energy CSV is empty: {path}")
    required = {
        "carrington_rotation",
        "analysis_decimal_year",
        *_GLOBAL_NAMES,
        *PUBLISHED_BAND_NAMES,
    }
    missing = required - rows[0].keys()
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
            [[float(row[name]) for name in _GLOBAL_NAMES] for row in rows],
            dtype=float,
        )
        band_energy = np.asarray(
            [[float(row[name]) for name in PUBLISHED_BAND_NAMES] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"WSO energy CSV contains an invalid numeric value: {path}") from error

    if not np.all(np.isfinite(years)):
        raise ValueError("Decimal years must all be finite")
    if not np.all(np.isfinite(global_energy)) or not np.all(np.isfinite(band_energy)):
        raise ValueError("Energy values must all be finite")
    if np.any(np.diff(rotations) <= 0) or np.any(np.diff(years) <= 0.0):
        raise ValueError("Carrington rotations and decimal years must increase strictly")
    return rotations, years, global_energy, band_energy


def _write_pdf(figure: Figure, destination: Path, number: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    metadata = {
        "Title": f"Inceoglu et al. (2017) recalculated Figure {number}",
        "Author": "Inceoglu et al. (2017) reproduction",
        "Subject": "Recalculated from committed WSO-derived numerical inputs",
        "Keywords": "solar cycle, magnetic energy, WSO, reproducibility",
        "Creator": "inceoglu2017.figures",
        "Producer": "Matplotlib PDF backend",
        "CreationDate": _FIXED_PDF_DATE,
        "ModDate": _FIXED_PDF_DATE,
    }
    figure.savefig(
        temporary,
        format="pdf",
        metadata=metadata,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    temporary.replace(destination)


def _subplots(
    rows: int,
    columns: int,
    *,
    figsize: tuple[float, float],
    sharex: bool | str = False,
    sharey: bool | str = False,
) -> tuple[Figure, np.ndarray]:
    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(
        rows,
        columns,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    return figure, np.asarray(axes, dtype=object)


def _finish_axis(axis: Any) -> None:
    axis.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.7)
    axis.tick_params(direction="out", length=3.0, width=0.7)
    axis.margins(x=0.02)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _relationship_with_fit(
    axis: Any,
    predictor: np.ndarray,
    response: np.ndarray,
    response_error: np.ndarray,
    *,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    x, y, error = _finite_relationship(predictor, response, response_error)
    if x.size == 0:
        return
    axis.errorbar(
        x,
        y,
        yerr=error,
        color=color,
        ecolor=color,
        marker=marker,
        linestyle="none",
        markersize=5.0,
        markeredgewidth=1.0,
        capsize=2.0,
        elinewidth=0.8,
        zorder=3,
    )
    _draw_fit(axis, x, y, color=color, linestyle=linestyle)


def _multicolor_relationship_with_fit(
    axis: Any,
    predictor: np.ndarray,
    response: np.ndarray,
    response_error: np.ndarray,
    *,
    colors: Sequence[str],
    marker: str,
    linestyle: str,
) -> None:
    x = np.asarray(predictor, dtype=float).reshape(-1)
    y = np.asarray(response, dtype=float).reshape(-1)
    error = np.asarray(response_error, dtype=float).reshape(-1)
    color_array = np.asarray(tuple(colors), dtype=object)
    if not (x.shape == y.shape == error.shape == color_array.shape):
        raise ValueError("Within-cycle relationship arrays must share one shape")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(error) & (error >= 0.0)
    x = x[finite]
    y = y[finite]
    error = error[finite]
    color_array = color_array[finite]
    for x_value, y_value, error_value, color in zip(x, y, error, color_array):
        axis.errorbar(
            [x_value],
            [y_value],
            yerr=[error_value],
            color=str(color),
            ecolor=str(color),
            marker=marker,
            linestyle="none",
            markersize=5.0,
            markeredgewidth=1.0,
            capsize=2.0,
            elinewidth=0.8,
            zorder=3,
        )
    _draw_fit(axis, x, y, color="#252525", linestyle=linestyle)


def _draw_fit(
    axis: Any,
    predictor: np.ndarray,
    response: np.ndarray,
    *,
    color: str,
    linestyle: str,
) -> None:
    try:
        slope, intercept = ordinary_least_squares(predictor, response)
    except ValueError:
        return
    domain = np.asarray([np.min(predictor), np.max(predictor)], dtype=float)
    axis.plot(
        domain,
        slope * domain + intercept,
        color=color,
        linestyle=linestyle,
        linewidth=1.15,
        zorder=2,
    )


def _finite_relationship(
    predictor: np.ndarray,
    response: np.ndarray,
    response_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(predictor, dtype=float).reshape(-1)
    y = np.asarray(response, dtype=float).reshape(-1)
    error = np.asarray(response_error, dtype=float).reshape(-1)
    if not (x.shape == y.shape == error.shape):
        raise ValueError("Relationship and uncertainty arrays must share one shape")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(error) & (error >= 0.0)
    return x[finite], y[finite], error[finite]


def _time_series_inputs(
    decimal_year: Sequence[float] | np.ndarray,
    smoothed_energy: np.ndarray,
    confidence_half_width: np.ndarray,
    *,
    columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = _vector("decimal_year", decimal_year)
    values = _matrix(
        "smoothed_energy", smoothed_energy, rows=years.size, columns=columns
    )
    confidence = _matrix(
        "confidence_half_width",
        confidence_half_width,
        rows=years.size,
        columns=columns,
    )
    if np.any(np.diff(years) <= 0.0):
        raise ValueError("Decimal years must increase strictly")
    if np.any(confidence < 0.0):
        raise ValueError("Confidence half-widths cannot be negative")
    return years, values, confidence


def _metric(metrics: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        value = getattr(metrics, name)
    except AttributeError as error:
        raise ValueError(f"Cycle metrics lack required field {name!r}") from error
    return _matrix(f"metrics.{name}", value, rows=shape[0], columns=shape[1])


def _vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matrix(
    name: str,
    values: np.ndarray,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows; got {array.shape[0]}")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns; got {array.shape[1]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_matrix(
    name: str,
    values: np.ndarray,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        try:
            numeric = raw.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain integer sample indices") from error
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"{name} must contain integer sample indices")
    array = raw.astype(int)
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows; got {array.shape[0]}")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns; got {array.shape[1]}")
    return array


__all__ = [
    "PUBLISHED_BAND_LABELS",
    "PUBLISHED_BAND_NAMES",
    "PUBLISHED_CYCLES",
    "confidence_at_indices",
    "make_figure_1",
    "make_figure_2",
    "make_figure_3",
    "make_figure_4",
    "make_figure_5",
    "make_figure_6",
    "make_figure_7",
    "ordinary_least_squares",
    "propagated_decay_confidence",
    "write_recalculated_figures",
]
