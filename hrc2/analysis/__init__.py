"""Analysis module for HRC 2.0.

This module provides tools for exploring the parameter space of
general scalar-tensor models, including:
- Xi-tradeoff scans for different coupling families
- Model-space exploration
- Comparison across coupling types
"""

from .xi_tradeoff import (
    XiTradeoffResultHRC2,
    scan_xi_tradeoff_hrc2,
    find_critical_xi_hrc2,
    compare_coupling_families,
    print_xi_tradeoff_summary_hrc2,
)

__all__ = [
    "XiTradeoffResultHRC2",
    "scan_xi_tradeoff_hrc2",
    "find_critical_xi_hrc2",
    "compare_coupling_families",
    "print_xi_tradeoff_summary_hrc2",
]
