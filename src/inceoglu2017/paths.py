"""Repository paths and immutable WSO preparation constants."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
WSO_CACHE = CACHE / "wso"
WSO_MAPS = WSO_CACHE / "filled_synoptic_maps"
WSO_MANIFEST = WSO_CACHE / "manifest.json"
WSO_ENERGY = DATA / "wso_energy_cr1642_2182.csv"
WSO_FIGURE_1_MAPS = DATA / "wso_figure1_maps_cr1917_2149.csv"
WSO_MEAN_SPECTRUM = DATA / "wso_mean_spectrum_l01_60.csv"

OUTPUTS = ROOT / "outputs"
OUTPUT_FIGURES = OUTPUTS / "figures"
OUTPUT_TABLES = OUTPUTS / "tables"

FIRST_CARRINGTON_ROTATION = 1642
LAST_CARRINGTON_ROTATION = 2182
WSO_URL_TEMPLATE = "http://wso.stanford.edu/synoptic/WSO.{cr}.F.txt"
