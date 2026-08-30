# Inceoglu 2017: hemispheric solar-cycle progression from WSO data

This repository reproduces the numerical analysis in Inceoglu et al. (2017). The study used Wilcox Solar Observatory (WSO) filled synoptic magnetic-field maps for Carrington rotations 1642–2182 to follow solar cycles 21–24 on the full disc, in the two hemispheres, and in six 15-degree latitude bands. It examined cycle overlap, latitude-dependent onset delays, the Waldmeier relationships, and the relation between cycle-decay rates and amplitudes.

The Python preparation converts the WSO line-of-sight maps to radial magnetic field, performs the spherical-harmonic transform, and sums the energies over degrees 1–15. The reproduction then applies two successive 11-point moving averages, calculates the 99% confidence intervals and event measurements, and regenerates Tables 1–5 and Figures 1–7. Figure 1 is drawn from the numerical grids for Carrington rotations 2149 and 1917; Figure 2 is drawn from the degree-by-degree energy spectrum averaged over all 541 rotations.

No values digitized from a published plot are used to create the recalculated figures or tables.

## Citation

F. Inceoglu, R. Simoniello, M. F. Knudsen, and C. Karoff, “Hemispheric progression of solar cycles in solar magnetic field data and its relation to the solar dynamo models,” *Astronomy & Astrophysics* **601**, A51 (2017). [https://doi.org/10.1051/0004-6361/201629871](https://doi.org/10.1051/0004-6361/201629871)

## Reproduce

Create an environment from the exact pins in `requirements.lock`, then run from a fresh clone:

```bash
python3 scripts/reproduce.py
```

That one command verifies the three committed WSO-derived numerical inputs, creates all seven PDF figures and five CSV tables under `outputs/`, checks the numerical results against the article, and runs the test suite from a fresh clone.

To retrieve and verify the authoritative WSO maps and regenerate the numerical inputs separately:

```bash
python3 scripts/download_wso_maps.py
python3 scripts/prepare_wso_data.py
python3 scripts/prepare_figure_data.py
```

The complete 541 source responses are written only to the ignored `.cache/wso/` directory. The downloader validates every response and requires the expected checksum manifest for the complete CR 1642–2182 interval. The two preparation commands deterministically rebuild the nine analysis series, the two Figure 1 numerical grids, and the Figure 2 mean spectrum.

## Methods and results

The regional series use the paper-specific zero-mask-and-retransform calculation. Within each Carrington rotation, the field outside a selected latitude interval is set to zero, the masked field is transformed again, and modal energies are summed for degrees 1–15. The adjacent latitude bands intentionally share their boundary rows.

Successive centered 11-point moving averages give the plotted cycle-scale signals. The 99% confidence interval at each point is calculated from the first-pass values in the second-pass window using the Student-t distribution. Cycle minima, primary and qualifying secondary maxima, rise amplitudes, and decay rates are then determined from those smoothed series using the definitions in the article.

The reproduction yields the published onset sequence: the 30–45 degree bands lead the 0–15 degree bands by 22 Carrington rotations in the north and 19 in the south. It also reproduces the reported latitude- and cycle-dependent adjusted correlations and the strong positive relationship between decay rate and peak amplitude. Numerical comparisons use the unrounded recalculation; published two-decimal table entries are treated as rounded reference values.

Further implementation details are in [reproduction notes](docs/reproduction_notes.md), and the source, checksum, and licensing boundary are in [data provenance](docs/data_provenance.md).

## Repository contents

- `data/wso_energy_cr1642_2182.csv`: the nine WSO-derived series used by the cycle analysis.
- `data/wso_figure1_maps_cr1917_2149.csv`: the two numerical WSO grids plotted in Figure 1.
- `data/wso_mean_spectrum_l01_60.csv`: the 60-point time-mean spectrum plotted in Figure 2.
- `src/inceoglu2017/`: Python implementation of the preparation and published analyses.
- `scripts/reproduce.py`: complete local reproduction and validation command.
- `scripts/download_wso_maps.py`, `scripts/prepare_wso_data.py`, and `scripts/prepare_figure_data.py`: optional authoritative-source retrieval and preparation.
- `outputs/`: seven PDF figures and five CSV tables.
- `tests/`: numerical, preparation, and integration tests.

Original software and repository documentation are licensed under the [MIT License](LICENSE). That grant does not cover WSO observations, derived numerical inputs, or generated research outputs except where [`DATA_NOTICE.md`](DATA_NOTICE.md) expressly says otherwise.
