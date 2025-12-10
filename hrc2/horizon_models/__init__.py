"""Horizon-Memory Dark Energy Refinement Models.

This package implements four refinement pathways for horizon-memory cosmology
to address the CMB distance tension while maintaining late-time H0 effects:

Refinement A (T06A): Adaptive Memory Kernel
    τ_hor(a) = τ0 * (a/a0)^p
    - Faster relaxation at early times (reduce D_A error)
    - Normal relaxation at late times (keep H(z) modification)

Refinement B (T06B): 2-Component Memory Fluid
    dM1/dln(a) = (S_norm - M1) / tau1
    dM2/dln(a) = (M1 - M2) / tau2
    rho_hor = lambda1*M1 + lambda2*M2
    - M1 tracks horizon entropy
    - M2 lags behind for smoother evolution

Refinement C (T06C): Early-Time Suppression Window
    rho_hor(a) → rho_hor(a) * (1 - exp(-(a/a_supp)^n_supp))
    - Removes unwanted early-time contribution
    - Preserves late-time dynamics

Refinement D (T06D): Dynamical Equation-of-State Modifier
    w_eff(a) = w_base(a) + delta_w / (1 + (a/a_w)^m)
    - Bulk of phantom behavior near z~0-2
    - Revert to w ≈ -1 by z ~ 5-10
"""

from .base import (
    HorizonMemoryModel,
    HorizonMemoryParameters,
    HorizonMemoryResult,
    RefinementType,
)

from .refinement_a import AdaptiveMemoryKernel
from .refinement_b import TwoComponentMemory
from .refinement_c import EarlyTimeSuppression
from .refinement_d import DynamicalEoSModifier

__all__ = [
    # Base classes
    "HorizonMemoryModel",
    "HorizonMemoryParameters",
    "HorizonMemoryResult",
    "RefinementType",
    # Refinement implementations
    "AdaptiveMemoryKernel",
    "TwoComponentMemory",
    "EarlyTimeSuppression",
    "DynamicalEoSModifier",
]
