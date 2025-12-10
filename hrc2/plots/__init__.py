"""Plotting module for HRC 2.0.

This module provides visualization tools for:
- Xi-tradeoff analysis for different coupling families
- Comparison plots across coupling types
- Evolution diagnostics
"""

from .xi_tradeoff import (
    plot_xi_tradeoff_hrc2,
    plot_coupling_comparison,
    create_hrc2_summary_figure,
)

__all__ = [
    "plot_xi_tradeoff_hrc2",
    "plot_coupling_comparison",
    "create_hrc2_summary_figure",
]
