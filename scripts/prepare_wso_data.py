#!/usr/bin/env python3
"""Derive harmonic-energy series from downloaded WSO filled synoptic maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inceoglu2017.io import sha256_file, write_csv  # noqa: E402
from inceoglu2017.paths import (  # noqa: E402
    FIRST_CARRINGTON_ROTATION,
    LAST_CARRINGTON_ROTATION,
    WSO_ENERGY,
    WSO_MAPS,
)
from inceoglu2017.wso import (  # noqa: E402
    STUDY_REGION_ROWS,
    MAX_SPHERICAL_HARMONIC_DEGREE,
    PAPER_LARGE_SCALE_MAX_DEGREE,
    carrington_rotation_start,
    study_decimal_year,
    read_wso_filled_map,
    rotation_energy_products,
)


def _number(value: float) -> str:
    """Serialize enough digits to round-trip an IEEE-754 double."""

    return format(float(value), ".17g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=WSO_MAPS)
    parser.add_argument("--output", type=Path, default=WSO_ENERGY)
    parser.add_argument("--first-cr", type=int, default=FIRST_CARRINGTON_ROTATION)
    parser.add_argument("--last-cr", type=int, default=LAST_CARRINGTON_ROTATION)
    parser.add_argument(
        "--regional-method",
        choices=("both", "masked-retransform", "lowpass-spatial"),
        default="masked-retransform",
        help=(
            "The default writes the paper-specific zero-mask/retransform "
            "series; alternatives remain available for sensitivity checks."
        ),
    )
    parser.add_argument(
        "--include-spectrum",
        action="store_true",
        help="Also write the l=1..spectrum-lmax full-disk spectrum used for Fig. 2.",
    )
    parser.add_argument(
        "--spectrum-lmax", type=int, default=MAX_SPHERICAL_HARMONIC_DEGREE
    )
    parser.add_argument(
        "--regional-lmax", type=int, default=PAPER_LARGE_SCALE_MAX_DEGREE
    )
    args = parser.parse_args()

    if not (
        FIRST_CARRINGTON_ROTATION
        <= args.first_cr
        <= args.last_cr
        <= LAST_CARRINGTON_ROTATION
    ):
        raise ValueError(
            f"Require {FIRST_CARRINGTON_ROTATION} <= first CR <= last CR "
            f"<= {LAST_CARRINGTON_ROTATION}"
        )
    if not (
        1 <= args.regional_lmax <= args.spectrum_lmax <= MAX_SPHERICAL_HARMONIC_DEGREE
    ):
        raise ValueError(
            "Require 1 <= regional lmax <= spectrum lmax <= "
            f"{MAX_SPHERICAL_HARMONIC_DEGREE}"
        )

    region_names = list(STUDY_REGION_ROWS)
    fieldnames = [
        "carrington_rotation",
        "rotation_start_utc",
        "analysis_decimal_year",
    ]
    if args.include_spectrum:
        fieldnames.extend(
            f"energy_l{degree:02d}_gauss2"
            for degree in range(1, args.spectrum_lmax + 1)
        )
    if args.regional_method in ("both", "masked-retransform"):
        fieldnames.extend(region_names)
    if args.regional_method in ("both", "lowpass-spatial"):
        fieldnames.extend(
            f"lowpass_spatial_{name}_l01_{args.regional_lmax:02d}_gauss2"
            for name in region_names
        )

    rows: list[dict[str, object]] = []
    for rotation in range(args.first_cr, args.last_cr + 1):
        path = args.input_dir / f"WSO.{rotation}.F.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing WSO map {path}. Run scripts/download_wso_maps.py first "
                "or pass --input-dir."
            )
        source = read_wso_filled_map(path, target_rotation=rotation)
        products = rotation_energy_products(
            source,
            spectrum_lmax=args.spectrum_lmax,
            regional_lmax=args.regional_lmax,
        )
        start = carrington_rotation_start(rotation)
        row: dict[str, object] = {
            "carrington_rotation": rotation,
            "rotation_start_utc": start.isoformat(),
            "analysis_decimal_year": _number(study_decimal_year(start)),
        }
        if args.include_spectrum:
            row.update(
                {
                    f"energy_l{degree:02d}_gauss2": _number(products.spectrum[degree])
                    for degree in range(1, args.spectrum_lmax + 1)
                }
            )
        if args.regional_method in ("both", "masked-retransform"):
            row.update(
                {
                    name: _number(products.masked_retransform[name])
                    for name in region_names
                }
            )
        if args.regional_method in ("both", "lowpass-spatial"):
            row.update(
                {
                    f"lowpass_spatial_{name}_l01_{args.regional_lmax:02d}_gauss2": _number(
                        products.lowpass_spatial[name]
                    )
                    for name in region_names
                }
            )
        rows.append(row)
        if rotation == args.first_cr or rotation % 50 == 0 or rotation == args.last_cr:
            print(f"Prepared CR {rotation}", flush=True)

    write_csv(args.output, fieldnames, rows)
    print(
        f"Wrote {len(rows)} rotations to {args.output} "
        f"(SHA-256 {sha256_file(args.output)})"
    )


if __name__ == "__main__":
    main()
