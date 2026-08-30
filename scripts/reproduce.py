#!/usr/bin/env python3
"""Reproduce and validate all published numerical outputs with one command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / "src"))

from inceoglu2017.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    products = run_pipeline()
    figures = products["figures"]
    tables = products["tables"]
    checks = products["checks"]
    print(
        f"Generated {len(figures)} PDF figures and {len(tables)} CSV tables.",
        flush=True,
    )
    print(f"Numerical checks: {checks['overall_status']}.", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        check=True,
    )
    print("All tests passed.")


if __name__ == "__main__":
    main()
