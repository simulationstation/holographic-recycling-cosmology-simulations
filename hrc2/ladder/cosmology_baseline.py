#!/usr/bin/env python3
"""
Cosmology baseline for SN Ia distance ladder systematics simulation.

Provides the "true" cosmology and distance modulus calculations.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad


@dataclass
class TrueCosmology:
    """Container for the true underlying cosmology."""
    H0: float = 67.5       # km/s/Mpc
    Omega_m: float = 0.315
    Omega_L: float = 0.685
    # Flat LCDM with w = -1


def E_of_z(z: float, cosmo: TrueCosmology) -> float:
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0.

    For flat LCDM: E(z) = sqrt(Omega_m*(1+z)^3 + Omega_L)
    """
    return np.sqrt(cosmo.Omega_m * (1.0 + z)**3 + cosmo.Omega_L)


def comoving_distance(z: float, cosmo: TrueCosmology) -> float:
    """
    Comoving distance D_C(z) in Mpc.

    D_C = (c/H0) * integral_0^z dz'/E(z')
    """
    c_km_s = 299792.458  # km/s

    def integrand(zp):
        return 1.0 / E_of_z(zp, cosmo)

    result, _ = quad(integrand, 0, z)
    return (c_km_s / cosmo.H0) * result


def luminosity_distance(z: float, cosmo: TrueCosmology) -> float:
    """
    Luminosity distance D_L(z) in Mpc.

    For flat universe: D_L = (1 + z) * D_C
    """
    D_C = comoving_distance(z, cosmo)
    return (1.0 + z) * D_C


def mu_of_z(z: float, cosmo: TrueCosmology) -> float:
    """
    Distance modulus mu(z) = 5*log10(D_L / 10 pc).

    Args:
        z: Redshift
        cosmo: Cosmology parameters

    Returns:
        Distance modulus in magnitudes
    """
    if z <= 0:
        return 0.0

    D_L_Mpc = luminosity_distance(z, cosmo)
    D_L_pc = D_L_Mpc * 1.0e6
    return 5.0 * np.log10(D_L_pc / 10.0)


def mu_of_z_array(z_array: np.ndarray, cosmo: TrueCosmology) -> np.ndarray:
    """Vectorized version of mu_of_z."""
    return np.array([mu_of_z(z, cosmo) for z in z_array])
