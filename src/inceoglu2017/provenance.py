"""Immutable provenance checks for committed reproduction inputs."""

from __future__ import annotations

from pathlib import Path

from .io import sha256_file
from .paths import WSO_ENERGY, WSO_FIGURE_1_MAPS, WSO_MEAN_SPECTRUM


EXPECTED_SHA256: dict[str, tuple[Path, str]] = {
    "wso_energy": (
        WSO_ENERGY,
        "777962c8070ab8b1f9dc991b4d80cdfe262209c23aa184c15b2b47fb0b716ba6",
    ),
    "figure_1_maps": (
        WSO_FIGURE_1_MAPS,
        "05101315ce4bd1b1c052b5076f159bab6b04c24949a315c2dd00a6732da5a11b",
    ),
    "figure_2_mean_spectrum": (
        WSO_MEAN_SPECTRUM,
        "ae9dd5a13047a489b5dd04ab919ee4a8614d0fbdd6d03b43287c64b3b9cc3245",
    ),
}


def verify_repository_inputs() -> dict[str, str]:
    """Verify every committed numerical input and return its digest."""

    verified: dict[str, str] = {}
    for name, (path, expected) in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required reproduction input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )
        verified[name] = actual
    return verified
