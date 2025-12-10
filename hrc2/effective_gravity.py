"""Effective gravitational coupling for HRC 2.0.

In scalar-tensor theories, the effective Newton's constant is:
    G_eff(phi) = G_N / F(phi)

where F(phi) is the non-minimal coupling function.

More precisely, for cosmological perturbations:
    G_eff = G_N * (2*F + 4*(dF/dphi)^2 / Z) / (2*F^2)

but for background evolution and simple estimates:
    G_eff ~ G_N / F(phi)
"""

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .theory import ScalarTensorModel

# Newton's constant in Planck units (G_N * M_pl^2 = 1/(8*pi))
G_N = 1.0 / (8.0 * np.pi)  # In units where M_pl = 1


def compute_Geff(phi: float, model: "ScalarTensorModel") -> float:
    """Compute effective gravitational coupling G_eff(phi).

    For scalar-tensor gravity:
        G_eff = G_N / F(phi)

    Args:
        phi: Scalar field value
        model: ScalarTensorModel instance

    Returns:
        G_eff in units of G_N
    """
    F = model.F(phi)

    if F <= 0:
        return float('inf')

    return G_N / F


def compute_Geff_ratio(phi: float, model: "ScalarTensorModel") -> float:
    """Compute G_eff / G_N ratio.

    This is more useful for diagnostics as it's dimensionless:
        G_eff / G_N = 1 / F(phi)

    In GR limit (F = M_pl^2 = 1 in our units), this returns 1.

    Args:
        phi: Scalar field value
        model: ScalarTensorModel instance

    Returns:
        G_eff/G_N ratio (should be ~1 for viable models)
    """
    F = model.F(phi)

    if F <= 0:
        return float('inf')

    # F is in units of M_pl^2 = 1, so G_eff/G_N = 1/F
    return 1.0 / F


def compute_Geff_derivative(phi: float, model: "ScalarTensorModel") -> float:
    """Compute d(G_eff/G_N)/dphi.

    d(G_eff/G_N)/dphi = -F'(phi) / F(phi)^2

    Args:
        phi: Scalar field value
        model: ScalarTensorModel instance

    Returns:
        Derivative of G_eff ratio w.r.t. phi
    """
    F = model.F(phi)
    dF = model.dF_dphi(phi)

    if F <= 0:
        return float('inf')

    return -dF / (F**2)


def compute_delta_G_over_G(
    phi_0: float,
    phi_rec: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute fractional change in G_eff between two field values.

    Delta G / G = (G_eff(phi_0) - G_eff(phi_rec)) / G_eff(phi_rec)
                = F(phi_rec) / F(phi_0) - 1

    Args:
        phi_0: Field value at z=0
        phi_rec: Field value at z_rec
        model: ScalarTensorModel instance

    Returns:
        Fractional G_eff change
    """
    F_0 = model.F(phi_0)
    F_rec = model.F(phi_rec)

    if F_0 <= 0 or F_rec <= 0:
        return float('inf')

    # G_eff ~ 1/F, so Delta G/G = F_rec/F_0 - 1
    return F_rec / F_0 - 1


def compute_Gdot_over_G(
    phi: float,
    phi_dot: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute time derivative of G_eff.

    (dG/dt) / G = -F'/F * (dphi/dt)

    This is constrained by lunar laser ranging to be < 10^-13 / yr.

    Args:
        phi: Field value
        phi_dot: Field velocity dphi/dt
        model: ScalarTensorModel instance

    Returns:
        (dG_eff/dt) / G_eff
    """
    F = model.F(phi)
    dF = model.dF_dphi(phi)

    if F <= 0:
        return float('inf')

    return -dF / F * phi_dot


def is_Geff_valid(
    phi: float,
    model: "ScalarTensorModel",
    safety_margin: float = 0.05,
    max_ratio: float = 20.0,
) -> bool:
    """Check if G_eff is in valid range.

    Args:
        phi: Field value
        model: ScalarTensorModel instance
        safety_margin: Minimum allowed G_eff/G_N ratio
        max_ratio: Maximum allowed G_eff/G_N ratio

    Returns:
        True if G_eff is in valid range
    """
    ratio = compute_Geff_ratio(phi, model)

    if np.isinf(ratio) or np.isnan(ratio):
        return False

    return safety_margin < ratio < max_ratio
