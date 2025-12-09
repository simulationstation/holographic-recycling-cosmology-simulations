"""Parameterized Post-Newtonian (PPN) constraints on HRC.

Solar system tests provide extremely precise constraints on deviations
from General Relativity. The key constraints for scalar-tensor theories are:

1. Ġ/G constraint from lunar laser ranging: |Ġ/G| < 1.5 × 10⁻¹² yr⁻¹
2. Nordtvedt effect (η_N): |η_N| < 4.4 × 10⁻⁴
3. Shapiro delay (γ-1): |γ-1| < 2.3 × 10⁻⁵

For HRC with G_eff = G/(1-8πξφ):
- Ġ_eff/G_eff = 8πξφ̇/(1-8πξφ)
- In the solar system, φ must be slowly varying
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from ..utils.config import HRCParameters
from ..utils.constants import SI_UNITS
from ..background import BackgroundSolution


@dataclass
class PPNConstraint:
    """Result of a PPN constraint check."""

    name: str
    allowed: bool
    value: float
    bound: float
    sigma_margin: float
    message: str


def _G_dot_bound() -> Tuple[float, float]:
    """Return Ġ/G bound from lunar laser ranging.

    Constraint: |Ġ/G| < 1.5 × 10⁻¹² yr⁻¹
                       = 4.75 × 10⁻²⁰ s⁻¹

    In units of H₀ (H₀ ≈ 2.3 × 10⁻¹⁸ s⁻¹):
    |Ġ/G| < 0.02 H₀

    Returns:
        Tuple of (bound in H₀ units, 1σ uncertainty)
    """
    # Bound: 1.5e-12 yr^-1 = 4.75e-20 s^-1
    # H0 = 70 km/s/Mpc = 2.27e-18 s^-1
    bound_H0_units = 4.75e-20 / 2.27e-18
    return bound_H0_units, bound_H0_units * 0.5


def _nordtvedt_bound() -> Tuple[float, float]:
    """Return Nordtvedt parameter bound.

    The Nordtvedt effect measures the difference in gravitational
    acceleration between bodies of different gravitational binding energy.

    For scalar-tensor theories:
    η_N = 4β - γ - 3 ≈ (ξ φ)² for HRC

    Bound: |η_N| < 4.4 × 10⁻⁴

    Returns:
        Tuple of (bound, 1σ uncertainty)
    """
    return 4.4e-4, 1e-4


def _gamma_minus_one_bound() -> Tuple[float, float]:
    """Return Shapiro delay (γ-1) bound.

    γ parametrizes the space curvature produced by unit rest mass.
    GR predicts γ = 1 exactly.

    For scalar-tensor theories with non-minimal coupling:
    γ - 1 ≈ -2ξ²φ² / (1 + ξφ)² for small coupling

    Bound from Cassini: |γ-1| < 2.3 × 10⁻⁵

    Returns:
        Tuple of (bound, 1σ uncertainty)
    """
    return 2.3e-5, 1e-5


def compute_G_dot_over_G(
    phi: float,
    phi_dot: float,
    params: HRCParameters,
) -> float:
    """Compute Ġ_eff/G_eff.

    Ġ_eff/G_eff = 8πξφ̇ G_eff/G = 8πξφ̇/(1-8πξφ)

    Args:
        phi: Scalar field value
        phi_dot: Scalar field velocity (in H₀ units)
        params: HRC parameters

    Returns:
        Ġ_eff/G_eff in H₀ units
    """
    xi_8pi = 8 * np.pi * params.xi
    denominator = 1.0 - xi_8pi * phi

    if abs(denominator) < 1e-10:
        return np.inf

    return xi_8pi * phi_dot / denominator


def compute_ppn_gamma(
    phi: float,
    params: HRCParameters,
) -> float:
    """Compute PPN γ parameter.

    For scalar-tensor theories, γ deviates from 1 due to the scalar
    field contribution to light deflection.

    γ = (1 + ω(φ))/(2 + ω(φ))

    where ω is the Brans-Dicke parameter. For our theory:
    ω_eff ≈ 1/(16πξ²φ²) for small ξφ

    This gives:
    γ - 1 ≈ -2ξ²φ² for small ξφ

    Args:
        phi: Scalar field value
        params: HRC parameters

    Returns:
        γ parameter (should be ≈ 1)
    """
    xi_phi = params.xi * phi

    # For small ξφ
    if abs(xi_phi) < 0.1:
        gamma = 1.0 - 2 * (params.xi * phi) ** 2
    else:
        # More accurate formula for larger ξφ
        omega_eff = 1.0 / (16 * np.pi * (params.xi * phi) ** 2) if abs(xi_phi) > 1e-10 else np.inf
        gamma = (1 + omega_eff) / (2 + omega_eff) if omega_eff < np.inf else 1.0

    return gamma


def compute_nordtvedt_parameter(
    phi: float,
    params: HRCParameters,
) -> float:
    """Compute Nordtvedt parameter η_N.

    η_N = 4β - γ - 3

    For scalar-tensor theories:
    η_N ≈ (ξφ)² for small ξφ

    Args:
        phi: Scalar field value
        params: HRC parameters

    Returns:
        Nordtvedt parameter η_N
    """
    xi_phi = params.xi * phi
    return xi_phi**2


def check_G_dot_constraint(
    phi: float,
    phi_dot: float,
    params: HRCParameters,
) -> PPNConstraint:
    """Check Ġ/G constraint from lunar laser ranging.

    Args:
        phi: Scalar field value today
        phi_dot: Scalar field velocity today (in H₀ units)
        params: HRC parameters

    Returns:
        PPNConstraint result
    """
    bound, sigma = _G_dot_bound()
    G_dot_over_G = compute_G_dot_over_G(phi, phi_dot, params)
    value = abs(G_dot_over_G)

    allowed = value < bound
    sigma_margin = (bound - value) / sigma if sigma > 0 else np.inf

    if allowed:
        message = f"|Ġ/G| = {value:.2e} H₀ < {bound:.2e} H₀ ({sigma_margin:.1f}σ margin)"
    else:
        message = f"|Ġ/G| = {value:.2e} H₀ > {bound:.2e} H₀ VIOLATED"

    return PPNConstraint(
        name="G_dot",
        allowed=allowed,
        value=value,
        bound=bound,
        sigma_margin=sigma_margin,
        message=message,
    )


def check_gamma_constraint(
    phi: float,
    params: HRCParameters,
) -> PPNConstraint:
    """Check Shapiro delay (γ-1) constraint.

    Args:
        phi: Scalar field value
        params: HRC parameters

    Returns:
        PPNConstraint result
    """
    bound, sigma = _gamma_minus_one_bound()
    gamma = compute_ppn_gamma(phi, params)
    value = abs(gamma - 1.0)

    allowed = value < bound
    sigma_margin = (bound - value) / sigma if sigma > 0 else np.inf

    if allowed:
        message = f"|γ-1| = {value:.2e} < {bound:.2e} ({sigma_margin:.1f}σ margin)"
    else:
        message = f"|γ-1| = {value:.2e} > {bound:.2e} VIOLATED"

    return PPNConstraint(
        name="gamma",
        allowed=allowed,
        value=value,
        bound=bound,
        sigma_margin=sigma_margin,
        message=message,
    )


def check_nordtvedt_constraint(
    phi: float,
    params: HRCParameters,
) -> PPNConstraint:
    """Check Nordtvedt effect constraint.

    Args:
        phi: Scalar field value
        params: HRC parameters

    Returns:
        PPNConstraint result
    """
    bound, sigma = _nordtvedt_bound()
    eta_N = compute_nordtvedt_parameter(phi, params)
    value = abs(eta_N)

    allowed = value < bound
    sigma_margin = (bound - value) / sigma if sigma > 0 else np.inf

    if allowed:
        message = f"|η_N| = {value:.2e} < {bound:.2e} ({sigma_margin:.1f}σ margin)"
    else:
        message = f"|η_N| = {value:.2e} > {bound:.2e} VIOLATED"

    return PPNConstraint(
        name="nordtvedt",
        allowed=allowed,
        value=value,
        bound=bound,
        sigma_margin=sigma_margin,
        message=message,
    )


def check_ppn_constraints(
    solution: Optional[BackgroundSolution] = None,
    phi_0: Optional[float] = None,
    phi_dot_0: Optional[float] = None,
    params: Optional[HRCParameters] = None,
    verbose: bool = False,
) -> Tuple[bool, List[PPNConstraint]]:
    """Check all PPN constraints.

    Args:
        solution: Background solution (preferred)
        phi_0: Present scalar field value (alternative)
        phi_dot_0: Present scalar field velocity (alternative)
        params: HRC parameters
        verbose: Print results

    Returns:
        Tuple of (all_passed, list of PPNConstraint)
    """
    if solution is not None:
        phi_0 = solution.phi[0]
        phi_dot_0 = solution.phi_dot[0]
        params = params  # Use provided params
    elif phi_0 is None or params is None:
        raise ValueError("Must provide solution or (phi_0, params)")

    if phi_dot_0 is None:
        phi_dot_0 = 0.0

    results = [
        check_G_dot_constraint(phi_0, phi_dot_0, params),
        check_gamma_constraint(phi_0, params),
        check_nordtvedt_constraint(phi_0, params),
    ]

    all_passed = all(r.allowed for r in results)

    if verbose:
        print("\n=== PPN Constraint Checks ===")
        for r in results:
            status = "✓" if r.allowed else "✗"
            print(f"{status} {r.name}: {r.message}")

    return all_passed, results
