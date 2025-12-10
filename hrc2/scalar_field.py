"""Scalar field dynamics for HRC 2.0.

The scalar field equation in FLRW background is:
    Z(phi) * (phi'' + 3H*phi') + dV/dphi - 3*F'*(H' + 2H^2) = 0

where:
    - phi' = dphi/dt
    - H = Hubble parameter
    - F' = dF/dphi
    - V' = dV/dphi

This module provides functions to compute the scalar field acceleration
and energy density contributions.
"""

from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from .theory import ScalarTensorModel, HRC2Parameters


def compute_phi_acceleration(
    phi: float,
    phi_dot: float,
    H: float,
    H_dot: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute scalar field acceleration phi_ddot.

    From the scalar field equation:
        Z*phi'' + Z*3H*phi' + dV/dphi - 3*F'*(H' + 2H^2) = 0

    Solving for phi'':
        phi'' = (3*F'*(H' + 2H^2) - Z*3H*phi' - dV/dphi) / Z

    Args:
        phi: Field value
        phi_dot: Field velocity dphi/dt
        H: Hubble parameter
        H_dot: dH/dt
        model: ScalarTensorModel instance

    Returns:
        phi_ddot = d^2phi/dt^2
    """
    Z = model.Z(phi)
    dF = model.dF_dphi(phi)
    dV = model.dV_dphi(phi)

    if Z <= 0:
        return float('inf')

    # Scalar field equation source term
    source = 3.0 * dF * (H_dot + 2.0 * H**2)

    # Friction term
    friction = 3.0 * H * phi_dot * Z

    # phi_ddot = (source - friction - dV) / Z
    phi_ddot = (source - friction - dV) / Z

    return phi_ddot


def compute_scalar_energy_density(
    phi: float,
    phi_dot: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute scalar field energy density.

    rho_phi = Z(phi) * phi_dot^2 / 2 + V(phi)

    Args:
        phi: Field value
        phi_dot: Field velocity dphi/dt
        model: ScalarTensorModel instance

    Returns:
        Scalar field energy density
    """
    Z = model.Z(phi)
    V = model.V(phi)

    return 0.5 * Z * phi_dot**2 + V


def compute_scalar_pressure(
    phi: float,
    phi_dot: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute scalar field pressure.

    P_phi = Z(phi) * phi_dot^2 / 2 - V(phi)

    Args:
        phi: Field value
        phi_dot: Field velocity dphi/dt
        model: ScalarTensorModel instance

    Returns:
        Scalar field pressure
    """
    Z = model.Z(phi)
    V = model.V(phi)

    return 0.5 * Z * phi_dot**2 - V


def compute_effective_eos(
    phi: float,
    phi_dot: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute scalar field effective equation of state w_phi.

    w_phi = P_phi / rho_phi = (Z*phi_dot^2/2 - V) / (Z*phi_dot^2/2 + V)

    Args:
        phi: Field value
        phi_dot: Field velocity dphi/dt
        model: ScalarTensorModel instance

    Returns:
        w_phi (between -1 for pure potential and +1 for pure kinetic)
    """
    rho = compute_scalar_energy_density(phi, phi_dot, model)
    P = compute_scalar_pressure(phi, phi_dot, model)

    if rho == 0:
        return -1.0  # Pure cosmological constant

    return P / rho


def compute_F_contribution_to_friedmann(
    phi: float,
    phi_dot: float,
    H: float,
    model: "ScalarTensorModel",
) -> Tuple[float, float]:
    """Compute F(phi) time-variation contribution to Friedmann equation.

    The Friedmann equation in scalar-tensor gravity is:
        3*F*H^2 = rho_m + rho_r + rho_phi_eff

    where rho_phi_eff includes extra terms from F(phi) variation:
        rho_phi_eff = Z*phi_dot^2/2 + V - 3*H*F'*phi_dot

    Args:
        phi: Field value
        phi_dot: Field velocity dphi/dt
        H: Hubble parameter
        model: ScalarTensorModel instance

    Returns:
        Tuple of (rho_phi_standard, rho_F_variation)
    """
    Z = model.Z(phi)
    V = model.V(phi)
    dF = model.dF_dphi(phi)

    # Standard scalar field contribution
    rho_phi_std = 0.5 * Z * phi_dot**2 + V

    # F-variation contribution (coupling to H)
    rho_F_var = -3.0 * H * dF * phi_dot

    return rho_phi_std, rho_F_var


def compute_effective_scalar_density(
    phi: float,
    phi_dot: float,
    H: float,
    model: "ScalarTensorModel",
) -> float:
    """Compute total effective scalar field density for Friedmann equation.

    rho_phi_eff = Z*phi_dot^2/2 + V - 3*H*F'*phi_dot

    Args:
        phi: Field value
        phi_dot: Field velocity
        H: Hubble parameter
        model: ScalarTensorModel instance

    Returns:
        Effective scalar field energy density
    """
    rho_std, rho_F = compute_F_contribution_to_friedmann(phi, phi_dot, H, model)
    return rho_std + rho_F
