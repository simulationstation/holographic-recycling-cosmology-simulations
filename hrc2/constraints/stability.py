"""Advanced stability diagnostics for HRC 2.0.

This module implements stability checks for scalar-tensor theories:

1. No-ghost condition: Kinetic term coefficient must be positive
   - F > 0 (graviton ghost-free)
   - Z > 0 (scalar ghost-free)
   - Effective kinetic coefficient Q_s > 0 for scalar perturbations

2. Dolgov-Kawasaki condition: F'' >= 0 for certain coupling families
   - Avoids tachyonic curvature instabilities in f(R) limit
   - For exponential coupling, this is automatically satisfied

3. Gradient stability: c_s^2 > 0 for scalar perturbations
   - Sound speed squared must be positive to avoid gradient instabilities

4. F(phi) bounds: F > safety_margin to avoid strong coupling
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..theory import ScalarTensorModel
    from ..background import BackgroundSolution


@dataclass
class StabilityResult:
    """Result of stability analysis.

    Attributes:
        is_stable: Overall stability (all conditions satisfied)
        ghost_free: No ghost instabilities
        gradient_stable: Positive sound speed squared
        dk_stable: Dolgov-Kawasaki condition satisfied
        F_positive: F(phi) > 0 everywhere
        Z_positive: Z(phi) > 0 everywhere

        F_min: Minimum F value encountered
        cs2_min: Minimum sound speed squared
        d2F_min: Minimum F'' value

        failure_z: Redshift where first instability occurred (if any)
        failure_reason: Description of instability
    """
    is_stable: bool
    ghost_free: bool
    gradient_stable: bool
    dk_stable: bool
    F_positive: bool
    Z_positive: bool

    F_min: float = 1.0
    cs2_min: float = 1.0
    d2F_min: float = 0.0

    failure_z: Optional[float] = None
    failure_reason: Optional[str] = None


def check_ghost_condition(
    phi: float,
    model: "ScalarTensorModel",
    H: float = 1.0,
    phi_dot: float = 0.0,
) -> Tuple[bool, float]:
    """Check no-ghost condition for scalar perturbations.

    The effective kinetic coefficient for the scalar is:
        Q_s = Z * F / (F + 3*(dF/dphi)^2 / (2*Z))

    For canonical kinetic (Z=1), this simplifies.
    Ghost-free requires Q_s > 0, which needs F > 0 and Z > 0.

    A more refined condition includes the mixing with gravity:
        Q_s ~ (2*F*Z + 3*(F')^2) / F

    For stability, Q_s > 0.

    Args:
        phi: Field value
        model: ScalarTensorModel instance
        H: Hubble parameter (for refinement)
        phi_dot: Field velocity (for refinement)

    Returns:
        Tuple of (is_ghost_free, Q_s_value)
    """
    F = model.F(phi)
    Z = model.Z(phi)
    dF = model.dF_dphi(phi)

    if F <= 0 or Z <= 0:
        return False, 0.0

    # Effective kinetic coefficient
    # Simple approximation: Q_s ~ Z + 3*(F')^2 / (2*F)
    Q_s = Z + 3.0 * dF**2 / (2.0 * F)

    return Q_s > 0, Q_s


def check_gradient_stability(
    phi: float,
    model: "ScalarTensorModel",
    H: float = 1.0,
) -> Tuple[bool, float]:
    """Check gradient stability (positive sound speed squared).

    The scalar sound speed squared in scalar-tensor theory is approximately:
        c_s^2 ~ 1 for canonical kinetic term (Z=1)

    For non-canonical kinetic terms:
        c_s^2 = 1 / (1 + 2*Z'/(Z*...) * ...)

    For canonical Z=1 case, c_s^2 = 1 always.

    More refined expressions involve time derivatives.
    Here we use a simplified diagnostic.

    Args:
        phi: Field value
        model: ScalarTensorModel instance
        H: Hubble parameter

    Returns:
        Tuple of (is_gradient_stable, cs2_value)
    """
    Z = model.Z(phi)
    dZ = model.dZ_dphi(phi)
    F = model.F(phi)
    dF = model.dF_dphi(phi)

    if Z <= 0 or F <= 0:
        return False, 0.0

    # For canonical kinetic term (Z=1, dZ=0), c_s^2 = 1
    if abs(dZ) < 1e-15:
        cs2 = 1.0
    else:
        # Non-canonical case - simplified expression
        # c_s^2 ~ Z / (Z + phi_dot^2 * Z' / ...)
        # For now, assume approximately 1 unless Z becomes problematic
        cs2 = 1.0

    # Additional check: effective mass squared for phi
    # m_eff^2 = V'' - F'' * R / 2 at background level
    # For stability, want m_eff^2 > 0 or at least not too negative

    return cs2 > 0, cs2


def compute_scalar_sound_speed_squared(
    phi: float,
    phi_dot: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute scalar perturbation sound speed squared.

    For a general scalar-tensor theory with canonical kinetic term (Z=1):
        c_s^2 = 1

    For k-essence type theories:
        c_s^2 = P_X / (P_X + 2*X*P_XX)

    where X = -(dphi)^2/2 and P(X,phi) is the kinetic function.

    Args:
        phi: Field value
        phi_dot: Field velocity
        model: ScalarTensorModel instance

    Returns:
        Sound speed squared c_s^2
    """
    Z = model.Z(phi)
    dZ = model.dZ_dphi(phi)

    if Z <= 0:
        return 0.0

    # For canonical Z=1, c_s^2 = 1
    if abs(dZ) < 1e-15:
        return 1.0

    # For Z(phi) != 1 but still quadratic in (dphi)^2:
    # L = -Z(phi)/2 * (dphi)^2 - V(phi)
    # This is still canonical in the sense c_s^2 = 1

    # Non-trivial c_s^2 arises from:
    # L = P(X, phi) where P is nonlinear in X = -(dphi)^2/2

    # For now, return 1 for the canonical and simple non-canonical cases
    return 1.0


def check_dolgov_kawasaki(
    phi: float,
    model: "ScalarTensorModel",
) -> Tuple[bool, float]:
    """Check Dolgov-Kawasaki stability condition.

    For f(R) theories (which are scalar-tensor in disguise), the condition
    F'' >= 0 (or d^2F/dR^2 >= 0 in f(R) notation) is required to avoid
    tachyonic instabilities at high curvature.

    For general scalar-tensor with F(phi), this translates to a constraint
    on how rapidly F can vary with phi.

    Strictly speaking, DK condition is:
        d^2f/dR^2 > 0 for f(R) theories

    For F(phi) coupling, a related condition is that F'' shouldn't be
    too negative, which would cause rapid phi evolution.

    Args:
        phi: Field value
        model: ScalarTensorModel instance

    Returns:
        Tuple of (is_dk_stable, d2F_value)
    """
    d2F = model.d2F_dphi2(phi)

    # For quadratic coupling F = M_pl^2*(1 + xi*phi^2):
    #   d2F = 2*xi*M_pl^2 > 0 for xi > 0

    # For exponential coupling F = M_pl^2*exp(beta*phi):
    #   d2F = beta^2*exp(beta*phi) > 0 always

    # For linear coupling F = M_pl^2 - alpha*phi:
    #   d2F = 0 (neutral)

    # Conservative: require d2F >= -epsilon for stability
    # Very negative d2F can lead to runaway behavior

    # Threshold for DK violation
    dk_threshold = -0.1  # Allow slight negative d2F

    is_dk_stable = d2F >= dk_threshold

    return is_dk_stable, d2F


def check_all_stability(
    phi: float,
    model: "ScalarTensorModel",
    H: float = 1.0,
    phi_dot: float = 0.0,
    safety_margin: float = 0.05,
) -> StabilityResult:
    """Check all stability conditions at a given field value.

    Args:
        phi: Field value
        model: ScalarTensorModel instance
        H: Hubble parameter
        phi_dot: Field velocity
        safety_margin: Minimum F/M_pl^2 ratio

    Returns:
        StabilityResult with all diagnostics
    """
    F = model.F(phi)
    Z = model.Z(phi)

    # Basic positivity checks
    F_positive = F > safety_margin
    Z_positive = Z > 0

    # Ghost condition
    ghost_free, Q_s = check_ghost_condition(phi, model, H, phi_dot)

    # Gradient stability
    gradient_stable, cs2 = check_gradient_stability(phi, model, H)

    # Dolgov-Kawasaki
    dk_stable, d2F = check_dolgov_kawasaki(phi, model)

    # Overall stability
    is_stable = F_positive and Z_positive and ghost_free and gradient_stable and dk_stable

    # Determine failure reason
    failure_reason = None
    if not is_stable:
        if not F_positive:
            failure_reason = f"F(phi) = {F:.4f} < safety margin"
        elif not Z_positive:
            failure_reason = f"Z(phi) = {Z:.4f} <= 0 (ghost)"
        elif not ghost_free:
            failure_reason = f"Ghost instability: Q_s = {Q_s:.4f}"
        elif not gradient_stable:
            failure_reason = f"Gradient instability: c_s^2 = {cs2:.4f}"
        elif not dk_stable:
            failure_reason = f"DK instability: F'' = {d2F:.4f}"

    return StabilityResult(
        is_stable=is_stable,
        ghost_free=ghost_free,
        gradient_stable=gradient_stable,
        dk_stable=dk_stable,
        F_positive=F_positive,
        Z_positive=Z_positive,
        F_min=F,
        cs2_min=cs2,
        d2F_min=d2F,
        failure_reason=failure_reason,
    )


def check_stability_along_trajectory(
    solution: "BackgroundSolution",
    model: "ScalarTensorModel",
    safety_margin: float = 0.05,
) -> StabilityResult:
    """Check stability conditions along entire evolution trajectory.

    Args:
        solution: BackgroundSolution from integration
        model: ScalarTensorModel instance
        safety_margin: Minimum F/M_pl^2 ratio

    Returns:
        StabilityResult summarizing trajectory stability
    """
    if not solution.success:
        return StabilityResult(
            is_stable=False,
            ghost_free=False,
            gradient_stable=False,
            dk_stable=False,
            F_positive=False,
            Z_positive=False,
            failure_reason="Integration failed",
        )

    # Track minimum/maximum values along trajectory
    F_min = float('inf')
    Z_min = float('inf')
    cs2_min = float('inf')
    d2F_min = float('inf')

    first_failure_z = None
    first_failure_reason = None

    all_ghost_free = True
    all_gradient_stable = True
    all_dk_stable = True
    all_F_positive = True
    all_Z_positive = True

    for i, (z, phi, phi_dot, H) in enumerate(zip(
        solution.z, solution.phi, solution.phi_dot, solution.H
    )):
        if np.isnan(phi) or np.isnan(H):
            continue

        result = check_all_stability(phi, model, H, phi_dot, safety_margin)

        # Track minimums
        F_min = min(F_min, result.F_min)
        cs2_min = min(cs2_min, result.cs2_min)
        d2F_min = min(d2F_min, result.d2F_min)
        Z_min = min(Z_min, model.Z(phi))

        # Track failures
        if not result.is_stable and first_failure_z is None:
            first_failure_z = z
            first_failure_reason = result.failure_reason

        all_ghost_free = all_ghost_free and result.ghost_free
        all_gradient_stable = all_gradient_stable and result.gradient_stable
        all_dk_stable = all_dk_stable and result.dk_stable
        all_F_positive = all_F_positive and result.F_positive
        all_Z_positive = all_Z_positive and result.Z_positive

    is_stable = (all_ghost_free and all_gradient_stable and all_dk_stable
                 and all_F_positive and all_Z_positive)

    return StabilityResult(
        is_stable=is_stable,
        ghost_free=all_ghost_free,
        gradient_stable=all_gradient_stable,
        dk_stable=all_dk_stable,
        F_positive=all_F_positive,
        Z_positive=all_Z_positive,
        F_min=F_min,
        cs2_min=cs2_min,
        d2F_min=d2F_min,
        failure_z=first_failure_z,
        failure_reason=first_failure_reason,
    )
