"""HRC analysis utilities for parameter space exploration."""

from .parameter_scan import (
    scan_parameter_space,
    ParameterScanResult,
    PointClassification,
)
from .summary import (
    compare_potentials,
    print_layman_summary,
    generate_report,
    quick_potential_check,
    PotentialSummary,
    POTENTIAL_DESCRIPTIONS,
)

__all__ = [
    "scan_parameter_space",
    "ParameterScanResult",
    "PointClassification",
    # Summary functions
    "compare_potentials",
    "print_layman_summary",
    "generate_report",
    "quick_potential_check",
    "PotentialSummary",
    "POTENTIAL_DESCRIPTIONS",
]
