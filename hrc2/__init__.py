"""HRC 2.0: General Scalar-Tensor Framework.

This package implements a general scalar-tensor cosmology framework
with advanced couplings F(phi), kinetic terms Z(phi), and potentials V(phi).

The general action is:
    S = integral d^4x sqrt(-g) [F(phi)R/2 - Z(phi)(dphi)^2/2 - V(phi)] + S_matter

Key differences from HRC 1.x (linear coupling):
- General non-minimal coupling F(phi) instead of just xi*phi*R
- Support for multiple coupling families (linear, quadratic, exponential)
- Advanced stability diagnostics (ghost-free, gradient stability, DK condition)
- Unified constraint pipeline

Coupling families:
- LINEAR: F = M_pl^2 - alpha*phi (reproduces HRC 1.x behavior)
- QUADRATIC: F = M_pl^2 * (1 + xi*phi^2)
- EXPONENTIAL: F = M_pl^2 * exp(beta*phi/M_pl)
"""

from .theory import (
    CouplingFamily,
    KineticFamily,
    PotentialType,
    ScalarTensorModel,
    HRC2Parameters,
    create_model,
)

from .background import (
    BackgroundSolution,
    BackgroundCosmology,
)

from .effective_gravity import (
    compute_Geff,
    compute_Geff_ratio,
)

__version__ = "2.0.0"

__all__ = [
    # Theory
    "CouplingFamily",
    "KineticFamily",
    "PotentialType",
    "ScalarTensorModel",
    "HRC2Parameters",
    "create_model",
    # Background
    "BackgroundSolution",
    "BackgroundCosmology",
    # Effective gravity
    "compute_Geff",
    "compute_Geff_ratio",
]
