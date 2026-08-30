"""One-pass orchestration for all published tables, figures, and validation."""

from __future__ import annotations

from .dataset import load_study_input
from .events import analyze_study
from .figures import write_recalculated_figures
from .provenance import verify_repository_inputs
from .tables import write_study_tables
from .validation import validate_reproduction


def run_pipeline() -> dict[str, object]:
    """Generate all seven figures and five tables, then validate the results."""

    checksums = verify_repository_inputs()
    source = load_study_input()
    study = analyze_study(
        source.rotations,
        source.analysis_decimal_years,
        source.band_energy,
    )

    figures = write_recalculated_figures()
    tables = write_study_tables(study)

    checks = validate_reproduction(
        source,
        study,
        checksums,
        [figures[number] for number in range(1, 8)],
        [tables[number] for number in range(1, 6)],
    )
    return {
        "figures": figures,
        "tables": tables,
        "checks": checks,
    }
