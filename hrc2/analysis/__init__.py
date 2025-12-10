"""Analysis module for HRC 2.0.

This module provides tools for exploring the parameter space of
general scalar-tensor models, including:
- Xi-tradeoff scans for different coupling families
- Model-space exploration
- Comparison across coupling types
"""

from .xi_tradeoff import (
    XiTradeoffResultHRC2,
    SinglePointResult,
    scan_xi_tradeoff_hrc2,
    find_critical_xi_hrc2,
    compare_coupling_families,
    print_xi_tradeoff_summary_hrc2,
    evaluate_model_point,
    run_xi_tradeoff_parallel,
    run_xi_tradeoff_serial,
    rebuild_xi_tradeoff_result,
    save_partial_results,
)

from .interface_class import (
    HorizonMemoryClassExport,
    HorizonMemoryClassInterface,
    export_to_class_format,
    prepare_class_export,
)

__all__ = [
    "XiTradeoffResultHRC2",
    "SinglePointResult",
    "scan_xi_tradeoff_hrc2",
    "find_critical_xi_hrc2",
    "compare_coupling_families",
    "print_xi_tradeoff_summary_hrc2",
    "evaluate_model_point",
    "run_xi_tradeoff_parallel",
    "run_xi_tradeoff_serial",
    "rebuild_xi_tradeoff_result",
    "save_partial_results",
    # CLASS interface
    "HorizonMemoryClassExport",
    "HorizonMemoryClassInterface",
    "export_to_class_format",
    "prepare_class_export",
]
