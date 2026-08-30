# Reproduction notes

## Numerical input and spherical harmonics

The downstream cycle analysis consumes `data/wso_energy_cr1642_2182.csv`. That file can be regenerated from the public WSO filled synoptic maps with `scripts/download_wso_maps.py` followed by `scripts/prepare_wso_data.py`. The same downloaded maps regenerate the Figure 1 and 2 numerical inputs with `scripts/prepare_figure_data.py`.

The paper’s large-scale magnetic energy is the sum over spherical-harmonic degrees 1–15 and all azimuthal orders. For the regional series, the study calculation first sets the field outside the requested latitude rows to zero and then repeats the transform. The implementation uses standard complex orthonormal harmonics: power at order zero is `|a_l0|²`, while positive orders contribute `2|a_lm|²` to include their negative-order conjugates.

The final latitude masks were inclusive array slices. In zero-based Python indices they are:

| Series | Inclusive Gauss–Legendre rows |
| --- | ---: |
| Full disc | 0–59 |
| North / south | 30–59 / 0–29 |
| North 0–15 / 15–30 / 30–45 degrees | 30–35 / 35–40 / 40–45 |
| South 0–15 / 15–30 / 30–45 degrees | 24–29 / 19–24 / 14–19 |

The shared endpoint row between adjacent bands is intentional.

## Smoothing and confidence intervals

The series are smoothed twice with an 11-point centered moving average. At an endpoint, the MATLAB-compatible window contracts symmetrically to 1, 3, 5, 7, or 9 points before reaching 11 points in the interior.

The plotted center is the second moving average. Its 99% confidence half-width is calculated from the first-pass values within that same second-pass window:

```text
h = t(0.995, n - 1) × sample_standard_deviation / sqrt(n)
```

The one-point endpoint window has zero half-width. This reproduces the confidence envelopes in the final time-series figures.

## Cycle measurements

The transition windows used to locate the four cycle minima are 1976.3–1977.4, 1984.5–1988.0, 1994.5–1998.3, and 2006.5–2010.0. Dates use the study calculation’s explicit decimal convention, `year + (month + day/31 - 1)/12`. A cycle’s primary maximum is the largest second-pass value between successive minima, or between the last minimum and the end of the series for cycle 24.

Secondary maxima must reach at least 50% of the primary maximum, following the method stated in the article. The executable criterion also requires 10% of the primary amplitude as the minimum prominence separating distinct peaks. This makes the selection rule deterministic.

- Peak time is the interval from the cycle minimum to its primary maximum.
- Rise time is the interval from the minimum to one year before the first qualifying peak.
- Rise amplitude is the smoothed energy one year before that first peak.
- Decay rate is the positive two-point slope from one year after the last qualifying peak to one year before the following minimum, and is available for cycles 21–23.

One year is represented by the nearest whole number of Carrington rotations, 13.

## Correlations

The onset-delay calculation applies MATLAB’s coefficient-normalized cross-covariance to the twice-smoothed, whole-series demeaned pairs. The denominator is fixed at the full-series root sum of squares for every lag. This yields the published delays of 10 and 22 rotations in the north and 6 and 19 in the south.

Ordinary Pearson coefficients are corrected for small-sample bias with the Fisher approximation printed in the paper:

```text
rho = r × [1 + (1 - r²) / (2n)]
```

The sample size is four for the peak/rise relationships and three for the decay relationships. The generated tables retain unrounded values. All ordinary comparisons with the article’s two-decimal entries use an absolute tolerance of 0.011. One Table 3 entry reports −0.28 for the 0–15 degree southern peak-time relationship, while the calculation gives −0.262397; the dedicated validation tolerance for that entry is 0.018.

## Figures and validation

Figures 1 and 2 are plotted from committed numerical WSO products: the native grids for Carrington rotations 2149 and 1917, and the degree spectrum averaged over all 541 rotations. Figure 2 sums modal energy over every azimuthal order before averaging each degree over time.

Figures 3–7 are plotted from the committed WSO-derived series and recalculated event measurements. No manuscript artwork is stored or used as plotting data. Every output uses PDF for figures and CSV for tables.
