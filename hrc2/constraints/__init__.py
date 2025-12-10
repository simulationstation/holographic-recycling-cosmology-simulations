"""Constraints module for HRC 2.0.

This module provides:
- Advanced stability diagnostics (ghost-free, gradient stability, DK condition)
- BBN, PPN, and stellar constraints adapted for general scalar-tensor theories
"""

from .stability import (
    StabilityResult,
    check_ghost_condition,
    check_gradient_stability,
    check_dolgov_kawasaki,
    check_all_stability,
    compute_scalar_sound_speed_squared,
)

from .observational import (
    HRC2ConstraintResult,
    check_bbn_constraint_hrc2,
    check_ppn_constraints_hrc2,
    check_stellar_constraints_hrc2,
    check_all_constraints_hrc2,
)

__all__ = [
    # Stability
    "StabilityResult",
    "check_ghost_condition",
    "check_gradient_stability",
    "check_dolgov_kawasaki",
    "check_all_stability",
    "compute_scalar_sound_speed_squared",
    # Observational
    "HRC2ConstraintResult",
    "check_bbn_constraint_hrc2",
    "check_ppn_constraints_hrc2",
    "check_stellar_constraints_hrc2",
    "check_all_constraints_hrc2",
]
